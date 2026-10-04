"""FastAPI application factory.

The HTTP layer is an *adapter*. It validates input, resolves a service, and
serialises the result. All business logic lives in ``nfl_platform.services``,
which later milestones call in-process rather than over HTTP.

Endpoints are declared with ``def`` rather than ``async def`` on purpose: the
data layer does blocking network and CPU-bound Polars work, so Starlette runs
these in a threadpool instead of stalling the event loop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from nfl_platform import __version__
from nfl_platform.api.errors import register_exception_handlers
from nfl_platform.api.routers import (
    assistant,
    fantasy,
    health,
    matchups,
    meta,
    players,
    predictions,
    teams,
)
from nfl_platform.config.logging import configure_logging, get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.llm.agent import NflAnalystAgent
from nfl_platform.services.container import ServiceContainer

logger = get_logger(__name__)

API_PREFIX = "/api/v1"

DESCRIPTION = """
Deterministic NFL analytics over [nflverse](https://github.com/nflverse) data.

Every number returned by the analytics endpoints is computed from source data by
deterministic functions. No language model is involved, and none is required to
use them.

The `/assistant` endpoints add a tool-calling LLM on top. It does not compute
anything itself -- it selects and calls the same analytics services, and its
answers carry the list of tool calls that produced them.

**Conventions**

* `last_n_games` counts *games with data*, not calendar weeks, so bye weeks and
  inactive weeks never silently shrink a sample.
* Defensive statistics are prefixed `allowed_`, so "what a team gave up" cannot
  be confused with "what a team gained".
* Player endpoints accept a GSIS id or a name. Ambiguous names return **409**
  with the candidate players rather than a guess.
* Matchup analysis is descriptive. It contains no prediction.
* `/predictions` is the only part of this API where a number refers to a game
  that has not been played. Those values come from versioned ML artifacts, carry
  the model that produced them, and are never mixed into the analytics responses.
"""

TAGS_METADATA = [
    {"name": "health", "description": "Liveness and readiness probes."},
    {"name": "players", "description": "Player identity, production, usage, and injuries."},
    {"name": "teams", "description": "Team offence, defence, form, and splits."},
    {"name": "matchups", "description": "Descriptive two-team comparison."},
    {"name": "fantasy", "description": "Fantasy production, trends, and matchup context."},
    {"name": "meta", "description": "Valid teams and seasons."},
    {
        "name": "predictions",
        "description": (
            "Machine-learning predictions. Estimates about unplayed games, produced by "
            "versioned model artifacts, never by a language model. Requires trained "
            "models to be installed."
        ),
    },
    {
        "name": "assistant",
        "description": "Tool-calling LLM assistant. Requires a configured provider key.",
    },
]


def create_app(
    container: ServiceContainer | None = None,
    settings: Settings | None = None,
    agent: NflAnalystAgent | None = None,
) -> FastAPI:
    """Build the application.

    Args:
        container: Pre-built services. Tests pass a fixture-backed container so
            the suite runs entirely offline.
        settings: Configuration override, mainly for tests.
        agent: Pre-built assistant. Tests pass one driven by a scripted model;
            in production it is built on first use so the API still starts
            without a provider key.
    """
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One container per process: its loader memoises datasets, so the cache
        # stays warm across requests instead of refetching per call.
        app.state.container = container or ServiceContainer(settings=resolved_settings)
        app.state.settings = resolved_settings
        app.state.agent = agent
        logger.info("API started", extra={"version": __version__})
        yield
        logger.info("API stopped")

    app = FastAPI(
        title="NFL Intelligence Platform API",
        description=DESCRIPTION,
        version=__version__,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    for module in (players, teams, matchups, fantasy, meta, predictions, assistant):
        app.include_router(module.router, prefix=API_PREFIX)

    return app


app = create_app()
