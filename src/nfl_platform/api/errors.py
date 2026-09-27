"""Domain exception to HTTP status mapping.

Registered once on the application so routers contain no try/except blocks and
every failure returns the same :class:`ErrorResponse` shape.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from nfl_platform.config.logging import get_logger
from nfl_platform.exceptions import (
    AmbiguousPlayerError,
    DatasetNotAvailableError,
    InvalidGameCountError,
    InvalidSeasonError,
    InvalidWeekError,
    LlmConfigurationError,
    LlmProviderError,
    NflPlatformError,
    PlayerNotFoundError,
    SchemaValidationError,
    TeamNotFoundError,
    UnknownDatasetError,
    UnresolvedPlayersError,
)
from nfl_platform.schemas.common import ErrorResponse

logger = get_logger(__name__)

# Starlette renamed its 422 constant; the literal avoids depending on either spelling.
HTTP_422_UNPROCESSABLE = 422


def _response(
    status_code: int, error: str, message: str, context: dict[str, Any] | None = None
) -> JSONResponse:
    body = ErrorResponse(error=error, message=message, context=context)
    return JSONResponse(status_code=status_code, content=body.model_dump(exclude_none=True))


def _player_not_found(_: Request, exc: PlayerNotFoundError) -> JSONResponse:
    return _response(
        status.HTTP_404_NOT_FOUND,
        "player_not_found",
        str(exc),
        {"player_id": exc.player_id, "scope": exc.scope},
    )


def _ambiguous_player(_: Request, exc: AmbiguousPlayerError) -> JSONResponse:
    """409: the request is valid but cannot be fulfilled without disambiguation.

    The candidates are returned so the caller -- or an LLM -- can choose rather
    than guess.
    """
    return _response(
        status.HTTP_409_CONFLICT,
        "ambiguous_player",
        str(exc),
        {
            "query": exc.query,
            "candidates": [
                {"player_id": player_id, "description": description}
                for player_id, description in exc.candidates
            ],
        },
    )


def _unresolved_players(_: Request, exc: UnresolvedPlayersError) -> JSONResponse:
    """409: a batch request where several names need disambiguating at once."""
    return _response(
        status.HTTP_409_CONFLICT,
        "unresolved_players",
        str(exc),
        {
            "ambiguous": [
                {
                    "query": query,
                    "candidates": [
                        {"player_id": player_id, "description": description}
                        for player_id, description in candidates
                    ],
                }
                for query, candidates in exc.ambiguous
            ],
            "not_found": list(exc.missing),
        },
    )


def _team_not_found(_: Request, exc: TeamNotFoundError) -> JSONResponse:
    return _response(
        status.HTTP_404_NOT_FOUND,
        "team_not_found",
        str(exc),
        {"team": exc.team, "valid_teams": list(exc.known)},
    )


def _invalid_season(_: Request, exc: InvalidSeasonError) -> JSONResponse:
    return _response(HTTP_422_UNPROCESSABLE, "invalid_season", str(exc), {"season": exc.season})


def _invalid_week(_: Request, exc: InvalidWeekError) -> JSONResponse:
    return _response(HTTP_422_UNPROCESSABLE, "invalid_week", str(exc), {"week": exc.week})


def _invalid_game_count(_: Request, exc: InvalidGameCountError) -> JSONResponse:
    return _response(
        HTTP_422_UNPROCESSABLE,
        "invalid_game_count",
        str(exc),
        {"value": exc.value},
    )


def _unknown_dataset(_: Request, exc: UnknownDatasetError) -> JSONResponse:
    return _response(status.HTTP_400_BAD_REQUEST, "unknown_dataset", str(exc))


def _dataset_unavailable(_: Request, exc: DatasetNotAvailableError) -> JSONResponse:
    """503: nflverse is unreachable. Transient and retryable, not the caller's fault."""
    logger.warning(
        "Upstream dataset unavailable",
        extra={"dataset": exc.dataset, "reason": exc.reason},
    )
    return _response(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "data_source_unavailable",
        str(exc),
        {"dataset": exc.dataset},
    )


def _schema_validation(_: Request, exc: SchemaValidationError) -> JSONResponse:
    """500: upstream changed shape and our contract is stale. Our problem, not the caller's."""
    logger.error(
        "Upstream schema drift detected",
        extra={"dataset": exc.dataset, "missing_columns": sorted(exc.missing_columns)},
    )
    return _response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "schema_validation_failed",
        str(exc),
        {"dataset": exc.dataset, "missing_columns": sorted(exc.missing_columns)},
    )


def _value_error(_: Request, exc: ValueError) -> JSONResponse:
    """Service-level argument errors, e.g. comparing fewer than two players."""
    return _response(HTTP_422_UNPROCESSABLE, "invalid_request", str(exc))


def _llm_not_configured(_: Request, exc: LlmConfigurationError) -> JSONResponse:
    """503: the analytics endpoints still work; only the assistant is unavailable."""
    logger.warning("Assistant requested without provider configuration")
    return _response(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "llm_not_configured",
        str(exc),
        {"hint": "Set NFL_LLM_API_KEY or OPENAI_API_KEY to enable the assistant."},
    )


def _llm_provider_error(_: Request, exc: LlmProviderError) -> JSONResponse:
    """502: we are reachable, the upstream model is not."""
    logger.error("LLM provider failed", extra={"retryable": exc.retryable})
    return _response(
        status.HTTP_502_BAD_GATEWAY,
        "llm_provider_error",
        str(exc),
        {"retryable": exc.retryable},
    )


def _platform_error(_: Request, exc: NflPlatformError) -> JSONResponse:
    """Catch-all for domain errors without a more specific handler."""
    logger.error("Unhandled platform error", exc_info=exc)
    return _response(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(exc))


# Ordered specific-to-general; FastAPI dispatches on the exact class then walks
# the MRO, so the base-class entry only catches what is not listed above it.
_HANDLERS: tuple[tuple[type[Exception], Callable[..., JSONResponse]], ...] = (
    (PlayerNotFoundError, _player_not_found),
    (AmbiguousPlayerError, _ambiguous_player),
    (UnresolvedPlayersError, _unresolved_players),
    (TeamNotFoundError, _team_not_found),
    (InvalidSeasonError, _invalid_season),
    (InvalidWeekError, _invalid_week),
    (InvalidGameCountError, _invalid_game_count),
    (UnknownDatasetError, _unknown_dataset),
    (DatasetNotAvailableError, _dataset_unavailable),
    (SchemaValidationError, _schema_validation),
    (LlmConfigurationError, _llm_not_configured),
    (LlmProviderError, _llm_provider_error),
    (NflPlatformError, _platform_error),
    (ValueError, _value_error),
)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every domain exception handler to ``app``."""
    for exception_type, handler in _HANDLERS:
        app.add_exception_handler(exception_type, handler)


def log_level_for(status_code: int) -> int:
    """Server faults are errors; client mistakes are not."""
    return logging.ERROR if status_code >= 500 else logging.INFO
