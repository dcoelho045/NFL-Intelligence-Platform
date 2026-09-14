"""FastAPI dependency providers and shared query parameters.

Services are resolved from application state, so tests can swap the whole
container with ``app.dependency_overrides`` and run entirely offline.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Path, Query, Request

from nfl_platform.schemas.common import SeasonType
from nfl_platform.services.container import ServiceContainer
from nfl_platform.services.directory import PlayerDirectoryService
from nfl_platform.services.fantasy import FantasyAnalyticsService
from nfl_platform.services.matchups import MatchupAnalyticsService
from nfl_platform.services.meta import MetaService
from nfl_platform.services.players import PlayerAnalyticsService
from nfl_platform.services.teams import TeamAnalyticsService

MAX_LAST_N_GAMES = 50


def get_container(request: Request) -> ServiceContainer:
    """Return the process-wide container held on application state."""
    container: ServiceContainer = request.app.state.container
    return container


ContainerDep = Annotated[ServiceContainer, Depends(get_container)]


def get_players(container: ContainerDep) -> PlayerAnalyticsService:
    return container.players


def get_teams(container: ContainerDep) -> TeamAnalyticsService:
    return container.teams


def get_matchups(container: ContainerDep) -> MatchupAnalyticsService:
    return container.matchups


def get_fantasy(container: ContainerDep) -> FantasyAnalyticsService:
    return container.fantasy


def get_directory(container: ContainerDep) -> PlayerDirectoryService:
    return container.directory


def get_meta(container: ContainerDep) -> MetaService:
    return container.meta


PlayersDep = Annotated[PlayerAnalyticsService, Depends(get_players)]
TeamsDep = Annotated[TeamAnalyticsService, Depends(get_teams)]
MatchupsDep = Annotated[MatchupAnalyticsService, Depends(get_matchups)]
FantasyDep = Annotated[FantasyAnalyticsService, Depends(get_fantasy)]
DirectoryDep = Annotated[PlayerDirectoryService, Depends(get_directory)]
MetaDep = Annotated[MetaService, Depends(get_meta)]

# --- shared query parameters -------------------------------------------------
# Range checking beyond these bounds is left to the domain layer, so the rules
# live in exactly one place.

SeasonsQuery = Annotated[
    list[int] | None,
    Query(
        description="Seasons to cover. Repeat the parameter for several. "
        "Omit to use the configured default season."
    ),
]

WeeksQuery = Annotated[
    list[int] | None,
    Query(description="Explicit week numbers. Applied before any last_n_games limit."),
]

LastNGamesQuery = Annotated[
    int | None,
    Query(
        ge=1,
        le=MAX_LAST_N_GAMES,
        description="Most recent games to include. Counts games with data, not calendar weeks.",
    ),
]

SeasonTypeQuery = Annotated[
    SeasonType,
    Query(description="REG, POST, or ALL. Defaults to ALL so playoff games are not dropped."),
]

PlayerIdPath = Annotated[
    str,
    Path(description="GSIS player id such as '00-0033873', or a name that resolves uniquely."),
]

TeamPath = Annotated[str, Path(description="Team abbreviation such as 'KC'.")]
