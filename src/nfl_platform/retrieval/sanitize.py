"""Sanitisation of untrusted retrieved text.

Everything a search provider returns originated outside this system. The
structural defence is that retrieved text only ever reaches the model inside a
tool-result payload, never as a system or user message, and that no tool can
fetch a URL the model chose. This module is the layer beneath that: it removes
the markup and control characters that make injection easier to hide, bounds the
size of what is passed on, and flags text that reads like an instruction.

The pattern list is a tripwire, not a security boundary. Indirect prompt
injection is unsolved; what actually contains it here is that every tool is
read-only, so there is no dangerous action for injected text to reach.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width and bidirectional-override characters. Legitimate prose does not
# need them and they are a standard way to hide text from a human reviewer while
# leaving it visible to a model.
_INVISIBLE = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_DANGEROUS_URI = re.compile(r"\b(?:javascript|data|vbscript|file)\s*:", re.IGNORECASE)
# Markdown/HTML links collapse to their text: the label is what a reader sees,
# and the target is provenance that belongs in a metadata field, not in prose.
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]{0,200})\]\([^)]{0,500}\)")
_WHITESPACE = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES = re.compile(r"\n{3,}")

# Phrases that attempt to redirect the model rather than inform it.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
        r"(?:instructions?|prompts?|messages?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"disregard\s+(?:all\s+|the\s+|your\s+)?(?:previous|prior|above|system)\s*"
        r"(?:instructions?|prompts?|rules?)?",
        re.IGNORECASE,
    ),
    re.compile(r"forget\s+(?:everything|all)\s+(?:above|before|you)", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(?:a|an|the)\b", re.IGNORECASE),
    re.compile(r"\bnew\s+(?:system\s+)?(?:instructions?|prompt|rules?)\s*:", re.IGNORECASE),
    re.compile(r"^\s*(?:system|assistant|developer)\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<\|\s*im_(?:start|end)\s*\|>", re.IGNORECASE),
    re.compile(r"\[\s*(?:system|INST)\s*\]", re.IGNORECASE),
    re.compile(
        r"(?:reveal|repeat|print|output|show)\s+(?:me\s+)?(?:your|the)\s+"
        r"(?:system\s+)?(?:prompt|instructions?)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:you\s+must|now)\s+call\s+the\s+\w+\s+tool", re.IGNORECASE),
    re.compile(r"\boverride\s+(?:your\s+)?(?:safety|system|previous)\b", re.IGNORECASE),
)

# The same phrases with every non-alphanumeric character removed. Stripping HTML
# leaves a space behind, so "ig<b></b>nore all previous instructions" survives
# the pattern list above as "ig nore all previous instructions". Matching a
# compacted copy of the text closes both the markup-splitting and the
# space-splitting evasions in one step.
_COMPACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore(?:all|any)?(?:previous|prior|above|earlier)(?:instruction|prompt|rule)"),
    re.compile(r"disregard(?:all|the|your)?(?:previous|prior|above|system)"),
    re.compile(r"forget(?:everything|all)(?:above|before)"),
    re.compile(r"youarenow(?:a|an|the)"),
    re.compile(r"new(?:system)?(?:instruction|prompt|rule)s?:"),
    re.compile(r"(?:reveal|repeat|print|output|show)(?:me)?(?:your|the)(?:system)?prompt"),
    re.compile(r"override(?:your)?(?:safety|system|previous)"),
)
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9:]+")

TRUNCATION_MARKER = " […]"


def strip_markup(text: str) -> str:
    """Remove HTML, scripts, dangerous URI schemes and link targets."""
    cleaned = _SCRIPT_OR_STYLE.sub(" ", text)
    cleaned = _HTML_COMMENT.sub(" ", cleaned)
    cleaned = _HTML_TAG.sub(" ", cleaned)
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned)
    return _DANGEROUS_URI.sub("", cleaned)


def normalise_text(text: str) -> str:
    """Collapse untrusted text into plain, inspectable prose."""
    if not text:
        return ""
    # NFKC folds look-alike glyphs so a pattern cannot be evaded with fullwidth
    # or mathematical alphanumeric characters.
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _INVISIBLE.sub("", cleaned)
    cleaned = strip_markup(cleaned)
    cleaned = "".join(
        char for char in cleaned if char in "\n\t" or not unicodedata.category(char).startswith("C")
    )
    cleaned = _WHITESPACE.sub(" ", cleaned)
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()


def truncate(text: str, max_chars: int) -> str:
    """Cap length at a word boundary, marking that the text was shortened."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    boundary = clipped.rfind(" ")
    if boundary > max_chars * 0.6:
        clipped = clipped[:boundary]
    return clipped.rstrip() + TRUNCATION_MARKER


def _compact(text: str) -> str:
    return _NON_ALPHANUMERIC.sub("", text.lower())


def injection_matches(text: str) -> list[str]:
    """Names of the injection patterns this text triggers."""
    matched = [pattern.pattern[:60] for pattern in _INJECTION_PATTERNS if pattern.search(text)]
    compact = _compact(text)
    matched.extend(pattern.pattern[:60] for pattern in _COMPACT_PATTERNS if pattern.search(compact))
    return list(dict.fromkeys(matched))


def looks_like_injection(text: str) -> bool:
    """Whether the text contains anything resembling an instruction to the model."""
    if any(pattern.search(text) for pattern in _INJECTION_PATTERNS):
        return True
    compact = _compact(text)
    return any(pattern.search(compact) for pattern in _COMPACT_PATTERNS)


def sanitise(text: str, *, max_chars: int) -> tuple[str, bool]:
    """Normalise, truncate, and report whether the text reads like an instruction.

    Detection runs on the normalised text so that markup and invisible
    characters cannot be used to break a pattern apart.
    """
    cleaned = normalise_text(text)
    flagged = looks_like_injection(cleaned)
    return truncate(cleaned, max_chars), flagged
