"""Fantasy endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from nfl_platform.api.dependencies import (
    FantasyDep,
    LastNGamesQuery,
    PlayerIdPath,
    SeasonsQuery,
    SeasonTypeQuery,
)
from nfl_platform.schemas.common import SeasonType
from nfl_platform.schemas.fantasy import (
    FantasyComparison,
    FantasyMatchupContext,
    FantasyProduction,
    FantasyScoring,
    FantasyUsageTrend,
)
from nfl_platform.schemas.requests import FantasyComparisonRequest

router = APIRouter(prefix="/fantasy", tags=["fantasy"])


@router.post("/compare", response_model=FantasyComparison, summary="Compare fantasy production")
def compare_fantasy(request: FantasyComparisonRequest, fantasy: FantasyDep) -> FantasyComparison:
    """Compare players over one shared window."""
    return fantasy.compare(
        request.player_ids,
        seasons=request.seasons,
        last_n_games=request.last_n_games,
        scoring=request.scoring,
        season_type=request.season_type,
    )


@router.get(
    "/players/{player_id}/production",
    response_model=FantasyProduction,
    summary="Recent fantasy production",
)
def fantasy_production(
    player_id: PlayerIdPath,
    fantasy: FantasyDep,
    seasons: SeasonsQuery = None,
    last_n_games: LastNGamesQuery = 4,
    scoring: FantasyScoring = Query(default=FantasyScoring.PPR),
    season_type: SeasonTypeQuery = SeasonType.ALL,
) -> FantasyProduction:
    """Week-by-week scoring with totals, extremes, and a consistency measure."""
    return fantasy.production(
        player_id,
        seasons=seasons,
        last_n_games=last_n_games,
        scoring=scoring,
        season_type=season_type,
    )


@router.get(
    "/players/{player_id}/trends",
    response_model=FantasyUsageTrend,
    summary="Usage trend",
)
def fantasy_trends(
    player_id: PlayerIdPath,
    fantasy: FantasyDep,
    seasons: SeasonsQuery = None,
    last_n_games: LastNGamesQuery = 8,
    season_type: SeasonTypeQuery = SeasonType.ALL,
) -> FantasyUsageTrend:
    """Direction of opportunity across the window.

    Describes what already happened. It is not a projection, and the direction is
    withheld entirely when the sample is too small to mean anything.
    """
    return fantasy.usage_trend(
        player_id, seasons=seasons, last_n_games=last_n_games, season_type=season_type
    )


@router.get(
    "/players/{player_id}/matchup",
    response_model=FantasyMatchupContext,
    summary="Matchup context",
)
def fantasy_matchup(
    player_id: PlayerIdPath,
    fantasy: FantasyDep,
    opponent: str = Query(description="Opponent team abbreviation."),
    seasons: SeasonsQuery = None,
    last_n_games: LastNGamesQuery = 4,
    season_type: SeasonTypeQuery = SeasonType.ALL,
) -> FantasyMatchupContext:
    """What the opponent's defence has allowed recently."""
    return fantasy.matchup_context(
        player_id,
        opponent=opponent,
        seasons=seasons,
        last_n_games=last_n_games,
        season_type=season_type,
    )
