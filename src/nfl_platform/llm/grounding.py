"""Grounding and citation checks.

Prompting a model not to invent statistics is necessary but not sufficient, so
this compares what the answer claims against what the tools actually returned.

It is a detector, not a gate. At runtime, findings are attached to the trace and
nothing is blocked -- regenerating costs a second full run for a signal that is
occasionally a false positive. The tests treat a finding as a failure, which is
where the real enforcement lives.

## Why numbers are partitioned by source

Before retrieval existed, every tool result was deterministic analytics output,
so "this number appears in a tool result" and "this number came from the
analytics layer" were the same statement. They stop being the same once a tool
can return a thousand characters of news prose.

A web result carries dozens of incidental numbers -- yardages for other players,
years, ages, scores -- drawn from much the same distribution as the numbers a
model would invent. Pooling them would mean a fabricated "312 yards" is
certified as grounded because some unrelated article happened to mention 312.
The check would appear to pass more often while actually detecting less, which
is the worst way for a safety mechanism to fail.

So numbers are sorted into three buckets:

* *grounded* -- present in analytics output, or in the user's own question
* *from unstructured sources* -- present only in retrieved web or knowledge-base
  text; permitted, but recorded, and expected to be attributed
* *ungrounded* -- present nowhere; the fabrication signal

``is_grounded`` keeps its original meaning. A run that calls no retrieval tools
produces exactly the report it would have produced before this existed.

Known limitation: a number the model derives correctly but should not have
derived (summing two tool values, say) is flagged, which is the intent -- the
prompt forbids arithmetic precisely so that every stated number is traceable.

This is coincidence-resistant, not true provenance. Tracking which specific
result each value came from is a larger change and belongs with the
observability work in a later milestone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from nfl_platform.llm.results import ToolResult
from nfl_platform.schemas.evidence import SourceKind

# Standalone numbers only. The surrounding character classes keep the components
# of GSIS ids ("00-0033873") and ISO dates ("2024-09-15") out of the comparison,
# since those are identifiers rather than claims. A comma only counts as a
# thousands separator when three digits follow it, so "September 23, 2026"
# yields 23 and 2026 rather than the literal "23,".
_NUMBER = re.compile(r"(?<![-\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(%?)(?![-\w])")
_TOLERANCE = 1e-6

# Dates carried in retrieval metadata (published_at, retrieved_at, updated).
# A model that writes "published September 23, 2026" is quoting provenance, not
# inventing a statistic, so the components count as attributable evidence.
# Digit lookarounds rather than \b: in "2026-09-23T22:00:00" there is no word
# boundary between the day and the "T".
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")

_URL = re.compile(r"https?://[^\s<>\"'\)\]]+", re.IGNORECASE)
# Bare domains the model might cite without a scheme, e.g. "per espn.com".
_BARE_DOMAIN = re.compile(
    r"(?<![\w./@-])((?:[a-z0-9][a-z0-9-]*\.)+(?:com|org|net|gov|edu|io))(?![\w-])",
    re.IGNORECASE,
)

# Claims of having consulted a channel. A model that writes "according to the
# knowledge base" without having called it is fabricating provenance, which is a
# worse failure than answering from memory openly: it borrows the authority of a
# source it never read.
_ATTRIBUTION_CLAIMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "knowledge_base",
        re.compile(
            r"\b(?:according to|per|from|in|based on|states in|sourced from)\s+"
            r"(?:the|our|this)?\s*knowledge\s*base\b",
            re.IGNORECASE,
        ),
    ),
    (
        "knowledge_base",
        re.compile(r"\bthe knowledge base (?:says|states|defines|explains)\b", re.IGNORECASE),
    ),
    (
        "web",
        re.compile(
            r"\b(?:according to|per|based on)\s+(?:recent\s+|the\s+latest\s+)?"
            r"(?:web\s+)?(?:search results|reports|reporting|articles)\b",
            re.IGNORECASE,
        ),
    ),
)


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _decimals(text: str) -> int:
    _, _, fraction = text.replace(",", "").partition(".")
    return len(fraction)


def extract_numbers(text: str) -> list[tuple[str, float, bool]]:
    """Return (literal, value, is_percent) for every standalone number."""
    found: list[tuple[str, float, bool]] = []
    for match in _NUMBER.finditer(text):
        value = _to_float(match.group(1))
        if value is not None:
            found.append((match.group(0), value, bool(match.group(2))))
    return found


def _walk(value: Any, sink: set[float], *, dates: bool = False) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        sink.add(float(value))
    elif isinstance(value, str):
        for _, number, _ in extract_numbers(value):
            sink.add(number)
        if dates:
            for match in _ISO_DATE.finditer(value):
                sink.update(float(part) for part in match.groups())
    elif isinstance(value, dict):
        for item in value.values():
            _walk(item, sink, dates=dates)
    elif isinstance(value, list):
        for item in value:
            _walk(item, sink, dates=dates)


def available_numbers(results: Iterable[ToolResult], *, dates: bool = False) -> set[float]:
    """Every numeric value the given results put in front of the model."""
    sink: set[float] = set()
    for result in results:
        _walk(result.as_dict(), sink, dates=dates)
    return sink


def partition_by_source(results: Iterable[ToolResult]) -> tuple[set[float], set[float]]:
    """Split available numbers into (structured, unstructured)."""
    structured: list[ToolResult] = []
    unstructured: list[ToolResult] = []
    for result in results:
        if result.source_kind is SourceKind.STRUCTURED:
            structured.append(result)
        else:
            unstructured.append(result)
    # Date components count only on the retrieval side, where they are the
    # publication provenance an answer is expected to cite.
    return available_numbers(structured), available_numbers(unstructured, dates=True)


def _is_supported(literal: str, value: float, is_percent: bool, allowed: set[float]) -> bool:
    places = _decimals(literal)
    # Half a unit in the last stated decimal place. Comparing against a rounded
    # candidate instead would reject correct answers: Python rounds ties to even,
    # so round(129.25, 1) is 129.2, while "129.3" is the conventional rendering.
    margin = 0.5 * (10**-places) + _TOLERANCE
    for candidate in allowed:
        if abs(candidate - value) <= margin:
            return True
        # Shares are stored as 0-1 fractions but usually spoken as percentages.
        if (
            (is_percent or value > 1)
            and 0.0 <= candidate <= 1.0
            and abs(candidate * 100 - value) <= margin
        ):
            return True
    return False


@dataclass(frozen=True)
class GroundingReport:
    """Where each number in an answer could be traced to."""

    checked: tuple[str, ...]
    ungrounded: tuple[str, ...]
    from_unstructured: tuple[str, ...] = ()

    @property
    def is_grounded(self) -> bool:
        """True when no number is untraceable.

        Numbers sourced from retrieved text do not break grounding -- they are
        reported separately so attribution can be checked -- but a number found
        nowhere at all does.
        """
        return not self.ungrounded


@dataclass(frozen=True)
class CitationReport:
    """Sources named in an answer that were never retrieved."""

    cited: tuple[str, ...] = ()
    fabricated: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.fabricated


def check_grounding(
    answer: str, results: Sequence[ToolResult], question: str = ""
) -> GroundingReport:
    """Compare an answer's numbers against what the tools returned.

    Numbers from the question count as grounded: echoing back the season or week
    the user asked about is not a fabricated statistic.
    """
    structured, unstructured = partition_by_source(results)
    allowed = set(structured)
    allowed.update(number for _, number, _ in extract_numbers(question))

    checked: list[str] = []
    ungrounded: list[str] = []
    from_unstructured: list[str] = []

    for literal, value, is_percent in extract_numbers(answer):
        checked.append(literal)
        if _is_supported(literal, value, is_percent, allowed):
            continue
        if _is_supported(literal, value, is_percent, unstructured):
            from_unstructured.append(literal)
        else:
            ungrounded.append(literal)

    return GroundingReport(
        checked=tuple(checked),
        ungrounded=tuple(dict.fromkeys(ungrounded)),
        from_unstructured=tuple(dict.fromkeys(from_unstructured)),
    )


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_dicts(item)


def _retrieved_sources(results: Iterable[ToolResult]) -> tuple[set[str], set[str]]:
    """URLs and domains that actually came back from a retrieval tool."""
    urls: set[str] = set()
    domains: set[str] = set()
    for result in results:
        if result.source_kind is SourceKind.STRUCTURED or not result.data:
            continue
        for item in _iter_dicts(result.data):
            url = item.get("url")
            if isinstance(url, str):
                urls.add(url.rstrip("/").lower())
            domain = item.get("domain")
            if isinstance(domain, str):
                domains.add(domain.lower().removeprefix("www."))
    return urls, domains


def check_citations(answer: str, results: Sequence[ToolResult]) -> CitationReport:
    """Flag any source the answer credits but never retrieved.

    The textual counterpart of the numeric check, covering two failures. A model
    that invents a plausible URL is doing the same thing as one that invents a
    plausible statistic, and it is harder to notice because a link looks like
    evidence. A model that writes "according to the knowledge base" without
    having called it is worse still: it claims an authority it never consulted.
    """
    urls, domains = _retrieved_sources(results)
    consulted = {
        str(result.source_kind)
        for result in results
        if result.ok and result.source_kind is not SourceKind.STRUCTURED
    }

    cited: list[str] = []
    fabricated: list[str] = []

    for match in _URL.finditer(answer):
        raw = match.group(0).rstrip(".,;:").rstrip("/")
        cited.append(raw)
        normalised = raw.lower()
        if normalised in urls or any(url.startswith(normalised) for url in urls):
            continue
        if any(normalised.startswith(url) for url in urls):
            continue
        fabricated.append(raw)

    for match in _BARE_DOMAIN.finditer(answer):
        host = match.group(1).lower().removeprefix("www.")
        if any(host == known or host.endswith(f".{known}") for known in domains):
            continue
        # A domain already covered by a validated URL is not a separate claim.
        if any(host in url for url in urls):
            continue
        cited.append(host)
        fabricated.append(host)

    for kind, pattern in _ATTRIBUTION_CLAIMS:
        if not pattern.search(answer):
            continue
        label = f"claimed {kind.replace('_', ' ')} source"
        cited.append(label)
        if kind not in consulted:
            fabricated.append(label)

    return CitationReport(
        cited=tuple(dict.fromkeys(cited)),
        fabricated=tuple(dict.fromkeys(fabricated)),
    )


def collect_sources(results: Sequence[ToolResult]) -> list[dict[str, Any]]:
    """Build the citation list from what the tools returned.

    Derived, never authored by the model, so the assistant cannot cite a source
    it did not retrieve.
    """
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    for result in results:
        if not result.ok or not result.data:
            continue

        if result.source_kind is SourceKind.STRUCTURED:
            key = f"structured:{result.tool}"
            if key not in seen:
                seen.add(key)
                sources.append({"kind": "structured", "label": result.tool})
            continue

        for item in _iter_dicts(result.data):
            if result.source_kind is SourceKind.WEB and isinstance(item.get("url"), str):
                key = f"web:{item['url']}"
                if key in seen:
                    continue
                seen.add(key)
                sources.append(
                    {
                        "kind": "web",
                        "label": str(item.get("title") or item["url"]),
                        "url": item["url"],
                        "domain": item.get("domain"),
                        "published_at": item.get("published_at"),
                        "retrieved_at": item.get("retrieved_at"),
                    }
                )
            elif result.source_kind is SourceKind.KNOWLEDGE_BASE and isinstance(
                item.get("document_id"), str
            ):
                key = f"kb:{item['document_id']}"
                if key in seen:
                    continue
                seen.add(key)
                sources.append(
                    {
                        "kind": "knowledge_base",
                        "label": str(item.get("title") or item["document_id"]),
                        "document_id": item["document_id"],
                        "updated": item.get("updated"),
                    }
                )
    return sources
