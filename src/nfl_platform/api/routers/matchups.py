"""Matchup endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Query

from nfl_platform.api.dependencies import (
    LastNGamesQuery,
    MatchupsDep,
    SeasonsQuery,
    SeasonTypeQuery,
)
from nfl_platform.schemas.common import SeasonType
from nfl_platform.schemas.matchup import MatchupAnalysis

router = APIRouter(prefix="/matchups", tags=["matchups"])


@router.get("", response_model=MatchupAnalysis, summary="Analyse a matchup")
def analyze_matchup(
    matchups: MatchupsDep,
    team_a: str = Query(description="First team abbreviation."),
    team_b: str = Query(description="Second team abbreviation."),
    seasons: SeasonsQuery = None,
    week: int | None = Query(default=None, description="Anchor to a scheduled week."),
    last_n_games: LastNGamesQuery = 4,
    season_type: SeasonTypeQuery = SeasonType.ALL,
) -> MatchupAnalysis:
    """Side-by-side descriptive comparison of two teams.

    Descriptive only: no win probability or projected score. Prediction arrives
    in a later milestone with an explicitly predictive endpoint.
    """
    return matchups.analyze(
        team_a,
        team_b,
        seasons=seasons,
        week=week,
        last_n_games=last_n_games,
        season_type=season_type,
    )
