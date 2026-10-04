"""Tool registration, schema export, and dispatch.

This is where a model's request becomes a service call, and it is the only place
a tool failure is allowed to become an exception-free result. Everything the
model can get wrong -- unknown tool name, malformed JSON, bad argument types,
unknown player -- comes back as a structured payload it can read and retry from.

Genuine bugs are not swallowed silently: they are logged with a traceback and
returned as a generic error with no internals attached.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import ValidationError

from nfl_platform.config.logging import get_logger
from nfl_platform.exceptions import UnknownToolError
from nfl_platform.llm.errors import ToolErrorDetail, describe_exception
from nfl_platform.llm.results import ToolResult
from nfl_platform.llm.tools.base import Tool
from nfl_platform.llm.tools.fantasy import (
    CompareFantasyPlayersTool,
    GetFantasyMatchupContextTool,
    GetFantasyProductionTool,
    GetFantasyUsageTrendTool,
)
from nfl_platform.llm.tools.meta import ListReferenceDataTool
from nfl_platform.llm.tools.players import (
    ComparePlayersTool,
    GetPlayerGameLogTool,
    GetPlayerInjuriesTool,
    GetPlayerProfileTool,
    GetPlayerStatsTool,
    GetPlayerUsageTool,
    SearchPlayersTool,
)
from nfl_platform.llm.tools.predictions import (
    CompareFantasyProjectionsTool,
    PredictGameTool,
    PredictSlateTool,
    ProjectFantasyPointsTool,
)
from nfl_platform.llm.tools.research import SearchKnowledgeBaseTool, SearchWebTool
from nfl_platform.llm.tools.teams import (
    GetMatchupTool,
    GetRecentGamesTool,
    GetTeamFormTool,
    GetTeamInjuriesTool,
    GetTeamStatsTool,
)
from nfl_platform.schemas.evidence import SourceKind
from nfl_platform.services.container import ServiceContainer

logger = get_logger(__name__)

DEFAULT_TOOL_TYPES: tuple[type[Tool[Any]], ...] = (
    SearchPlayersTool,
    GetPlayerProfileTool,
    GetPlayerStatsTool,
    GetPlayerGameLogTool,
    GetPlayerUsageTool,
    GetPlayerInjuriesTool,
    ComparePlayersTool,
    GetTeamStatsTool,
    GetTeamFormTool,
    GetRecentGamesTool,
    GetTeamInjuriesTool,
    GetMatchupTool,
    GetFantasyProductionTool,
    GetFantasyUsageTrendTool,
    GetFantasyMatchupContextTool,
    CompareFantasyPlayersTool,
    ListReferenceDataTool,
    SearchWebTool,
    SearchKnowledgeBaseTool,
    PredictGameTool,
    PredictSlateTool,
    ProjectFantasyPointsTool,
    CompareFantasyProjectionsTool,
)


class ToolRegistry:
    """Holds the tools and turns model requests into results."""

    def __init__(self, tools: Iterable[Tool[Any]]) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name {tool.name!r}.")
            self._tools[tool.name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def get(self, name: str) -> Tool[Any]:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(name, self.names) from None

    def schemas(self) -> list[dict[str, Any]]:
        """Provider-facing definitions for every registered tool."""
        return [tool.json_schema() for tool in self._tools.values()]

    def execute(
        self, name: str, arguments: str | dict[str, Any], container: ServiceContainer
    ) -> ToolResult:
        """Validate and run one tool call. Never raises for recoverable failures."""
        try:
            tool = self.get(name)
        except UnknownToolError as exc:
            return ToolResult.failure(name, describe_exception(exc))

        parsed = self._parse_arguments(arguments)
        if isinstance(parsed, ToolErrorDetail):
            return ToolResult.failure(name, parsed)

        try:
            args = tool.args_model.model_validate(parsed)
        except ValidationError as exc:
            return ToolResult.failure(name, self._invalid_arguments(tool, exc))

        try:
            return tool.invoke(container, args)
        except Exception as exc:
            detail = describe_exception(exc)
            if detail.code in ("tool_error", "internal_error"):
                logger.exception("Tool %s failed unexpectedly", name)
            else:
                logger.info(
                    "Tool %s returned a handled error",
                    name,
                    extra={"tool": name, "error_code": detail.code},
                )
            return ToolResult.failure(name, detail)

    def source_kind(self, name: str) -> SourceKind:
        """Which evidence channel a tool speaks for, defaulting to structured."""
        tool = self._tools.get(name)
        return SourceKind.STRUCTURED if tool is None else tool.source_kind

    @staticmethod
    def _parse_arguments(arguments: str | dict[str, Any]) -> dict[str, Any] | ToolErrorDetail:
        if isinstance(arguments, dict):
            return arguments
        text = (arguments or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return ToolErrorDetail(
                "malformed_arguments",
                f"Arguments were not valid JSON: {exc.msg}.",
                recovery="Send the arguments again as a single valid JSON object.",
            )
        if not isinstance(parsed, dict):
            return ToolErrorDetail(
                "malformed_arguments",
                "Arguments must be a JSON object, not a bare value or list.",
                recovery="Wrap the arguments in a JSON object keyed by parameter name.",
            )
        return parsed

    @staticmethod
    def _invalid_arguments(tool: Tool[Any], exc: ValidationError) -> ToolErrorDetail:
        """Turn Pydantic's report into field-level guidance plus the true schema."""
        problems = [
            {
                "field": ".".join(str(part) for part in error["loc"]) or "(root)",
                "problem": error["msg"],
            }
            for error in exc.errors()
        ]
        return ToolErrorDetail(
            "invalid_arguments",
            f"Arguments for {tool.name} failed validation.",
            recovery=(
                "Fix the fields listed in problems and call the tool again. "
                "expected_parameters shows the accepted schema."
            ),
            context={
                "problems": problems,
                "expected_parameters": tool.json_schema()["function"]["parameters"],
            },
        )


def build_default_registry(extra: Sequence[Tool[Any]] = ()) -> ToolRegistry:
    """The tool set the assistant ships with."""
    return ToolRegistry([tool_type() for tool_type in DEFAULT_TOOL_TYPES] + list(extra))
