"""Versioned system prompts.

Prompts are files, not string literals, so a change to the model's instructions
shows up as a reviewable diff. The version is part of the filename and is
recorded in every run trace, which is what makes "this answer came from v1"
answerable later.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_VERSION = "analyst_v1"


@dataclass(frozen=True)
class Prompt:
    """A versioned system prompt."""

    version: str
    text: str


@lru_cache(maxsize=8)
def load_prompt(version: str = DEFAULT_PROMPT_VERSION) -> Prompt:
    """Read a prompt from disk by version name."""
    path = PROMPT_DIR / f"{version}.md"
    if not path.is_file():
        available = ", ".join(available_versions()) or "none"
        raise FileNotFoundError(f"Unknown prompt version {version!r}. Available: {available}.")
    return Prompt(version=version, text=path.read_text(encoding="utf-8").strip())


def available_versions() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in PROMPT_DIR.glob("*.md")))


__all__ = ["DEFAULT_PROMPT_VERSION", "Prompt", "available_versions", "load_prompt"]
