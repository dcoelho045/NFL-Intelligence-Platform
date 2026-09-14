"""Player endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from nfl_platform.api.dependencies import (
    DirectoryDep,
    LastNGamesQuery,
    PlayerIdPath,
    PlayersDep,
    SeasonsQuery,
    SeasonTypeQuery,
    WeeksQuery,
)
from nfl_platform.schemas.common import SeasonType
from nfl_platform.schemas.player import (
    PlayerComparison,
    PlayerGameLog,
    PlayerInjuryReport,
    PlayerProfile,
    PlayerSearchResult,
    PlayerSeasonStats,
    PlayerUsage,
)
from nfl_platform.schemas.requests import PlayerComparisonRequest

router = APIRouter(prefix="/players", tags=["players"])


# Declared before /{player_id} so "search" and "compare" are not captured as ids.
@router.get("/search", response_model=PlayerSearchResult, summary="Search players by name")
def search_players(
    directory: DirectoryDep,
    name: str = Query(description="Full or partial player name."),
    position: str | None = Query(default=None, description="Narrow by position, e.g. 'QB'."),
    team: str | None = Query(default=None, description="Narrow by team abbreviation."),
    season: int | None = Query(default=None, description="Narrow to players active that season."),
    limit: int = Query(default=10, ge=1, le=50),
) -> PlayerSearchResult:
    """Resolve a name to candidate players.

    Around 7.5% of nflverse players share a display name, so this returns every
    match with a disambiguator rather than guessing one.
    """
    return directory.search(name, position=position, team=team, season=season, limit=limit)


@router.post("/compare", response_model=PlayerComparison, summary="Compare players")
def compare_players(request: PlayerComparisonRequest, players: PlayersDep) -> PlayerComparison:
    """Compare players over one shared window."""
    return players.compare(
        request.player_ids,
        seasons=request.seasons,
        last_n_games=request.last_n_games,
        season_type=request.season_type,
    )


@router.get("/{player_id}", response_model=PlayerProfile, summary="Player profile")
def player_profile(player_id: PlayerIdPath, directory: DirectoryDep) -> PlayerProfile:
    """Identity and career detail."""
    return directory.profile(directory.resolve(player_id))


@router.get(
    "/{player_id}/stats/recent",
    response_model=PlayerSeasonStats,
    summary="Recent performance",
)
def recent_performance(
    player_id: PlayerIdPath,
    players: PlayersDep,
    seasons: SeasonsQuery = None,
    last_n_games: LastNGamesQuery = 4,
    season_type: SeasonTypeQuery = SeasonType.ALL,
) -> PlayerSeasonStats:
    """Totals and per-game averages over the most recent games."""
    return players.recent_performance(
        player_id, seasons=seasons, last_n_games=last_n_games, season_type=season_type
    )


@router.get(
    "/{player_id}/stats/season",
    response_model=PlayerSeasonStats,
    summary="Season performance",
)
def season_performance(
    player_id: PlayerIdPath,
    players: PlayersDep,
    seasons: SeasonsQuery = None,
    season_type: SeasonTypeQuery = SeasonType.REGULAR,
) -> PlayerSeasonStats:
    """Totals and per-game averages across a whole season."""
    return players.season_performance(player_id, seasons=seasons, season_type=season_type)


@router.get("/{player_id}/usage", response_model=PlayerUsage, summary="Usage and snap share")
def player_usage(
    player_id: PlayerIdPath,
    players: PlayersDep,
    seasons: SeasonsQuery = None,
    last_n_games: LastNGamesQuery = 4,
    season_type: SeasonTypeQuery = SeasonType.ALL,
    include_snap_share: bool = Query(
        default=True, description="Join snap counts via the PFR identifier crosswalk."
    ),
) -> PlayerUsage:
    """Opportunity metrics: snaps, targets, carries, and shares."""
    return players.usage(
        player_id,
        seasons=seasons,
        last_n_games=last_n_games,
        season_type=season_type,
        include_snap_share=include_snap_share,
    )


@router.get("/{player_id}/games", response_model=PlayerGameLog, summary="Game-by-game log")
def player_game_log(
    player_id: PlayerIdPath,
    players: PlayersDep,
    seasons: SeasonsQuery = None,
    last_n_games: LastNGamesQuery = 4,
    weeks: WeeksQuery = None,
    season_type: SeasonTypeQuery = SeasonType.ALL,
) -> PlayerGameLog:
    """Per-game production, newest first."""
    return players.game_log(
        player_id,
        seasons=seasons,
        last_n_games=last_n_games,
        weeks=weeks,
        season_type=season_type,
    )


@router.get("/{player_id}/injuries", response_model=PlayerInjuryReport, summary="Injury report")
def player_injuries(
    player_id: PlayerIdPath,
    players: PlayersDep,
    seasons: SeasonsQuery = None,
    weeks: WeeksQuery = None,
    latest_only: bool = Query(default=False, description="Return only the most recent entry."),
) -> PlayerInjuryReport:
    """Injury-report history. An empty list means no injury was reported."""
    return players.injury_report(player_id, seasons=seasons, weeks=weeks, latest_only=latest_only)
