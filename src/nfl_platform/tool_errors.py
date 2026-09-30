"""Domain exception to tool-error mapping.

The counterpart of :mod:`nfl_platform.api.errors`, using the same error
vocabulary so an HTTP caller, the assistant and an MCP client see one taxonomy.
The difference is the destination: HTTP maps to a status code for a human, this
maps to a structured payload for a machine.

Each entry carries a ``recovery`` hint. That hint is the single highest-value
field here -- it is what turns "Player name is ambiguous" from a dead end into a
second, correct tool call.

This module is adapter-neutral on purpose. ``llm/`` and ``mcp_server/`` both
need it and must not import each other, so it sits beside
:mod:`nfl_platform.exceptions` rather than inside either adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nfl_platform.exceptions import (
    AmbiguousPlayerError,
    DatasetNotAvailableError,
    EmbeddingError,
    InvalidGameCountError,
    InvalidSeasonError,
    InvalidWeekError,
    KnowledgeDocumentNotFoundError,
    KnowledgeIndexStaleError,
    KnowledgeIndexUnavailableError,
    NflPlatformError,
    PlayerNotFoundError,
    SchemaValidationError,
    TeamNotFoundError,
    UnknownDatasetError,
    UnknownToolError,
    UnresolvedPlayersError,
    WebSearchBudgetExceededError,
    WebSearchError,
    WebSearchNotConfiguredError,
)

# How many disambiguation candidates to show the model. Enough to choose from,
# few enough not to flood the context.
MAX_CANDIDATES = 10


class ToolErrorDetail:
    """A structured failure the model is expected to read and act on."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        recovery: str,
        context: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        self.code = code
        self.message = message
        self.recovery = recovery
        self.context = context or {}
        self.retryable = retryable

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "recovery": self.recovery,
            "retryable": self.retryable,
        }
        if self.context:
            payload["context"] = self.context
        return payload


def _ambiguous_player(exc: AmbiguousPlayerError) -> ToolErrorDetail:
    candidates = [
        {"player_id": player_id, "description": description}
        for player_id, description in exc.candidates[:MAX_CANDIDATES]
    ]
    return ToolErrorDetail(
        "ambiguous_player",
        str(exc),
        recovery=(
            "Several players share this name. Pick the player_id from candidates that "
            "matches the user's context (team, position, era) and call the tool again. "
            "If the context does not identify one, ask the user which player they mean."
        ),
        context={
            "query": exc.query,
            "candidates": candidates,
            "total_candidates": len(exc.candidates),
        },
    )


def _player_not_found(exc: PlayerNotFoundError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "player_not_found",
        str(exc),
        recovery=(
            "Call search_players to find the correct identifier, or widen the seasons "
            "argument. Do not substitute a different player."
        ),
        context={"player_id": exc.player_id, "scope": exc.scope},
    )


def _unresolved_players(exc: UnresolvedPlayersError) -> ToolErrorDetail:
    """Every unidentifiable name at once, so one retry fixes the whole call."""
    return ToolErrorDetail(
        "unresolved_players",
        str(exc),
        recovery=(
            "Replace every name listed in ambiguous with the matching player_id from its "
            "candidates, and drop or correct anything in not_found. Fix them all in one "
            "retry rather than one at a time."
        ),
        context={
            "ambiguous": [
                {
                    "query": query,
                    "candidates": [
                        {"player_id": player_id, "description": description}
                        for player_id, description in candidates[:MAX_CANDIDATES]
                    ],
                }
                for query, candidates in exc.ambiguous
            ],
            "not_found": list(exc.missing),
        },
    )


def _team_not_found(exc: TeamNotFoundError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "team_not_found",
        str(exc),
        recovery=(
            "Use a valid team abbreviation from valid_teams. Call list_reference_data "
            "if you need the full list. Note that relocated franchises change "
            "abbreviation by season (for example OAK before LV)."
        ),
        context={"team": exc.team, "valid_teams": list(exc.known)},
    )


def _invalid_season(exc: InvalidSeasonError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "invalid_season",
        str(exc),
        recovery="Supply a season within the supported range, or omit it to use the default.",
        context={"season": repr(exc.season)},
    )


def _invalid_week(exc: InvalidWeekError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "invalid_week",
        str(exc),
        recovery="Supply a week between 1 and 22, or omit it to cover every week.",
        context={"week": repr(exc.week)},
    )


def _invalid_game_count(exc: InvalidGameCountError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "invalid_game_count",
        str(exc),
        recovery="last_n_games must be a positive integer, or null to include every game.",
        context={"value": repr(exc.value)},
    )


def _unknown_tool(exc: UnknownToolError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "unknown_tool",
        str(exc),
        recovery="Call one of the available tools listed in available_tools.",
        context={"available_tools": list(exc.known)},
    )


def _dataset_unavailable(exc: DatasetNotAvailableError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "data_source_unavailable",
        str(exc),
        recovery=(
            "The underlying data source is unreachable. Do not estimate the value. "
            "Tell the user this information is temporarily unavailable."
        ),
        context={"dataset": exc.dataset},
        retryable=False,
    )


def _unknown_dataset(exc: UnknownDatasetError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "unknown_dataset",
        str(exc),
        recovery="Use a different tool; this dataset is not available.",
        retryable=False,
    )


def _schema_validation(exc: SchemaValidationError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "schema_validation_failed",
        str(exc),
        recovery=(
            "The upstream data changed shape and this tool cannot return trustworthy "
            "values. Report the gap rather than estimating."
        ),
        context={"dataset": exc.dataset},
        retryable=False,
    )


def _value_error(exc: ValueError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "invalid_request",
        str(exc),
        recovery="Correct the arguments and call the tool again.",
    )


def _web_search_not_configured(exc: WebSearchNotConfiguredError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "web_search_not_configured",
        str(exc),
        recovery=(
            "Web retrieval is switched off on this deployment. This is a configuration "
            "state, not a search that found nothing: it is not evidence that no current "
            "information exists. Do not retry and do not substitute remembered news. "
            "Answer from the analytics tools if the question allows it, and tell the user "
            "that current external information could not be checked."
        ),
        retryable=False,
    )


def _web_search_failed(exc: WebSearchError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "web_search_error",
        str(exc),
        recovery=(
            "The search provider could not be reached. Do not invent news or injury status. "
            "Answer with what the analytics tools provide and say the current information "
            "could not be retrieved."
        ),
        context={"provider": exc.provider},
        retryable=exc.retryable,
    )


def _web_search_budget(exc: WebSearchBudgetExceededError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "web_search_budget_exhausted",
        str(exc),
        recovery=(
            "No further web searches are available for this question. Answer using the "
            "evidence already retrieved and state what remains unknown."
        ),
        context={"limit": exc.limit},
        retryable=False,
    )


def _knowledge_index_unavailable(exc: KnowledgeIndexUnavailableError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "knowledge_base_unavailable",
        str(exc),
        recovery=(
            "The local knowledge base is not built on this deployment. Do not retry. "
            "You may explain general football terminology from your own knowledge, but "
            "say plainly that it is not sourced from the knowledge base."
        ),
        retryable=False,
    )


def _knowledge_index_stale(exc: KnowledgeIndexStaleError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "knowledge_base_unavailable",
        str(exc),
        recovery=(
            "The knowledge base needs rebuilding and cannot be searched. Do not retry. "
            "Tell the user this capability is unavailable."
        ),
        retryable=False,
    )


def _embedding_failed(exc: EmbeddingError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "knowledge_base_unavailable",
        str(exc),
        recovery=(
            "The knowledge base could not process the query. Do not retry the same search. "
            "Answer from other evidence and note the gap."
        ),
        context={"provider": exc.provider},
        retryable=False,
    )


def _knowledge_document_not_found(exc: KnowledgeDocumentNotFoundError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "knowledge_document_not_found",
        str(exc),
        recovery=(
            "Use one of the identifiers in known_documents. Document ids are not file "
            "paths and cannot be constructed."
        ),
        context={"document_id": exc.document_id, "known_documents": list(exc.known)},
        retryable=False,
    )


def _platform_error(exc: NflPlatformError) -> ToolErrorDetail:
    return ToolErrorDetail(
        "internal_error",
        str(exc),
        recovery="This tool cannot answer. Try another tool or report the gap.",
        retryable=False,
    )


# Ordered specific-to-general; the first isinstance match wins.
_HANDLERS: tuple[tuple[type[Exception], Callable[[Any], ToolErrorDetail]], ...] = (
    (AmbiguousPlayerError, _ambiguous_player),
    (UnresolvedPlayersError, _unresolved_players),
    (PlayerNotFoundError, _player_not_found),
    (TeamNotFoundError, _team_not_found),
    (InvalidSeasonError, _invalid_season),
    (InvalidWeekError, _invalid_week),
    (InvalidGameCountError, _invalid_game_count),
    (UnknownToolError, _unknown_tool),
    (DatasetNotAvailableError, _dataset_unavailable),
    (UnknownDatasetError, _unknown_dataset),
    (SchemaValidationError, _schema_validation),
    (WebSearchNotConfiguredError, _web_search_not_configured),
    (WebSearchBudgetExceededError, _web_search_budget),
    (WebSearchError, _web_search_failed),
    (KnowledgeIndexStaleError, _knowledge_index_stale),
    (KnowledgeIndexUnavailableError, _knowledge_index_unavailable),
    (KnowledgeDocumentNotFoundError, _knowledge_document_not_found),
    (EmbeddingError, _embedding_failed),
    (NflPlatformError, _platform_error),
    (ValueError, _value_error),
)


def describe_exception(exc: Exception) -> ToolErrorDetail:
    """Translate an exception into a payload the model can act on.

    Anything unrecognised becomes a generic error with no internals attached --
    tracebacks and file paths must never reach a prompt.
    """
    for exception_type, handler in _HANDLERS:
        if isinstance(exc, exception_type):
            return handler(exc)
    return ToolErrorDetail(
        "tool_error",
        "The tool failed unexpectedly.",
        recovery="Try a different tool, or tell the user the information is unavailable.",
        retryable=False,
    )
