"""The envelope every tool returns, and the DTO-to-JSON policy behind it.

Two concerns live here that are easy to get wrong:

*Truncation.* Game logs and injury reports are unbounded. Sending all of one to
the model is expensive, and sending a silent subset is dangerous -- a model shown
four of seventeen games will happily describe them as "the season". Lists are
capped and the cap is announced in the payload, so partial data can never be
mistaken for complete data.

*Null suppression.* ``exclude_none`` strips the large number of fields that do
not apply to a given position (a receiver has no ``completions``). This cuts
tokens substantially and stops the model reading ``null`` as ``zero``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from nfl_platform.llm.errors import ToolErrorDetail
from nfl_platform.schemas.evidence import SourceKind

# Rows of any list field handed to the model. Enough to see a trend, not enough
# to dominate the context window.
MAX_LIST_ITEMS = 10


def _truncate(value: Any, notes: list[str], path: str = "") -> Any:
    """Recursively cap list lengths, recording what was shortened."""
    if isinstance(value, dict):
        return {
            key: _truncate(item, notes, f"{path}.{key}" if path else key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if len(value) > MAX_LIST_ITEMS:
            notes.append(
                f"'{path}' shows the first {MAX_LIST_ITEMS} of {len(value)} entries; "
                "totals and averages still cover the full window."
            )
            value = value[:MAX_LIST_ITEMS]
        return [_truncate(item, notes, path) for item in value]
    return value


def serialise_dto(dto: BaseModel) -> tuple[dict[str, Any], list[str]]:
    """Convert a service DTO into a compact JSON-ready payload plus notes."""
    payload = dto.model_dump(mode="json", exclude_none=True)
    notes: list[str] = []
    return _truncate(payload, notes), notes


@dataclass
class ToolResult:
    """What one tool execution produced, ready to become a tool message."""

    tool: str
    ok: bool
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    # Retrieval results must not be able to ground a statistic, so the grounding
    # check needs to know which channel produced each payload.
    source_kind: SourceKind = SourceKind.STRUCTURED

    @classmethod
    def success(
        cls,
        tool: str,
        data: dict[str, Any],
        notes: list[str] | None = None,
        *,
        source_kind: SourceKind = SourceKind.STRUCTURED,
    ) -> ToolResult:
        return cls(tool=tool, ok=True, data=data, notes=notes or [], source_kind=source_kind)

    @classmethod
    def from_dto(
        cls,
        tool: str,
        dto: BaseModel,
        *,
        source_kind: SourceKind = SourceKind.STRUCTURED,
    ) -> ToolResult:
        payload, notes = serialise_dto(dto)
        return cls.success(tool, payload, notes, source_kind=source_kind)

    @classmethod
    def failure(cls, tool: str, detail: ToolErrorDetail) -> ToolResult:
        return cls(tool=tool, ok=False, error=detail.as_dict())

    @property
    def error_code(self) -> str | None:
        return None if self.error is None else str(self.error.get("code"))

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"tool": self.tool, "ok": self.ok}
        if self.data is not None:
            payload["data"] = self.data
        if self.error is not None:
            payload["error"] = self.error
        if self.notes:
            payload["notes"] = self.notes
        return payload

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"), default=str)
