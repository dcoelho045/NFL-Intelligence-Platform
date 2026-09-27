"""Reference-data tool.

Exists so the model can check which teams and seasons actually exist instead of
assuming. Assumed team abbreviations are a common source of confidently wrong
answers, particularly for relocated franchises.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field

from nfl_platform.llm.results import ToolResult
from nfl_platform.llm.tools.base import Tool, ToolArgs
from nfl_platform.llm.tools.common import SeasonsField
from nfl_platform.services.container import ServiceContainer


class ReferenceDataArgs(ToolArgs):
    resource: Literal["teams", "seasons", "all"] = Field(
        default="all", description="Which reference list to return."
    )
    seasons: SeasonsField = None


class ListReferenceDataTool(Tool[ReferenceDataArgs]):
    name: ClassVar[str] = "list_reference_data"
    description: ClassVar[str] = (
        "List valid team abbreviations, the seasons that have data, and the current season. "
        "Use this to confirm a team abbreviation or find out what 'this season' means before "
        "assuming either."
    )
    args_model: ClassVar[type[ToolArgs]] = ReferenceDataArgs

    def invoke(self, container: ServiceContainer, args: ReferenceDataArgs) -> ToolResult:
        payload: dict[str, Any] = {}
        if args.resource in ("teams", "all"):
            payload["teams"] = container.meta.teams(args.seasons)
        if args.resource in ("seasons", "all"):
            payload["seasons"] = container.meta.seasons()
            payload["current_season"] = container.meta.current_season()
        # Returned untruncated on purpose: a partial list of valid teams would be
        # worse than none, because the model would treat absence as invalidity.
        return ToolResult.success(self.name, payload)
