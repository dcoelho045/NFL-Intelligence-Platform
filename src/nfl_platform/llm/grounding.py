"""Numeric grounding check.

Prompting a model not to invent statistics is necessary but not sufficient, so
this compares every number in the final answer against the numbers the tools
actually returned.

It is a detector, not a gate. At runtime, findings are attached to the trace and
nothing is blocked -- regenerating costs a second full run for a signal that is
occasionally a false positive. The tests treat a finding as a failure, which is
where the real enforcement lives.

Known limitation: a number the model derives correctly but should not have
derived (summing two tool values, say) is flagged, which is the intent -- the
prompt forbids arithmetic precisely so that every stated number is traceable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from nfl_platform.llm.results import ToolResult

# Standalone numbers only. The surrounding character classes keep the components
# of GSIS ids ("00-0033873") and ISO dates ("2024-09-15") out of the comparison,
# since those are identifiers rather than claims.
_NUMBER = re.compile(r"(?<![-\w.])(\d[\d,]*(?:\.\d+)?)(%?)(?![-\w])")
_TOLERANCE = 1e-6


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


def _walk(value: Any, sink: set[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        sink.add(float(value))
    elif isinstance(value, str):
        for _, number, _ in extract_numbers(value):
            sink.add(number)
    elif isinstance(value, dict):
        for item in value.values():
            _walk(item, sink)
    elif isinstance(value, list):
        for item in value:
            _walk(item, sink)


def available_numbers(results: Iterable[ToolResult]) -> set[float]:
    """Every numeric value the tools put in front of the model."""
    sink: set[float] = set()
    for result in results:
        _walk(result.as_dict(), sink)
    return sink


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
    """Which numbers in an answer could not be traced to a tool result."""

    checked: tuple[str, ...]
    ungrounded: tuple[str, ...]

    @property
    def is_grounded(self) -> bool:
        return not self.ungrounded


def check_grounding(
    answer: str, results: Sequence[ToolResult], question: str = ""
) -> GroundingReport:
    """Compare an answer's numbers against what the tools returned.

    Numbers from the question count as available: echoing back the season or week
    the user asked about is not a fabricated statistic.
    """
    allowed = available_numbers(results)
    allowed.update(number for _, number, _ in extract_numbers(question))
    checked: list[str] = []
    ungrounded: list[str] = []
    for literal, value, is_percent in extract_numbers(answer):
        checked.append(literal)
        if not _is_supported(literal, value, is_percent, allowed):
            ungrounded.append(literal)
    return GroundingReport(checked=tuple(checked), ungrounded=tuple(dict.fromkeys(ungrounded)))
