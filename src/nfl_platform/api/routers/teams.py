"""Team endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from nfl_platform.api.dependencies import (
    LastNGamesQuery,
    SeasonsQuery,
    SeasonTypeQuery,
    TeamPath,
    TeamsDep,
    WeeksQuery,
)
from nfl_platform.schemas.common import SeasonType
from nfl_platform.schemas.team import (
    TeamDefenseSummary,
    TeamForm,
    TeamGameLog,
    TeamHomeAwaySplits,
    TeamInjuryReport,
    TeamOffenseSummary,
)

router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("/{team}/offense", response_model=TeamOffenseSummary, summary="Offensive production")
def team_offense(
    team: TeamPath,
    teams: TeamsDep,
    seasons: SeasonsQuery = None,
    last_n_games: LastNGamesQuery = 4,
    weeks: WeeksQuery = None,
    season_type: SeasonTypeQuery = SeasonType.ALL,
) -> TeamOffenseSummary:
    """What the team's offence produced."""
    return teams.offense(
        team,
        seasons=seasons,
        last_n_games=last_n_games,
        weeks=weeks,
        season_type=season_type,
    )


@router.get("/{team}/defense", response_model=TeamDefenseSummary, summary="Defensive performance")
def team_defense(
    team: TeamPath,
    teams: TeamsDep,
    seasons: SeasonsQuery = None,
    last_n_games: LastNGamesQuery = 4,
    weeks: WeeksQuery = None,
    season_type: SeasonTypeQuery = SeasonType.ALL,
) -> TeamDefenseSummary:
    """What the team's defence allowed. All statistics are prefixed `allowed_`."""
    return teams.defense(
        team,
        seasons=seasons,
        last_n_games=last_n_games,
        weeks=weeks,
        season_type=season_type,
    )


@router.get("/{team}/form", response_model=TeamForm, summary="Recent form")
def team_form(
    team: TeamPath,
    teams: TeamsDep,
    seasons: SeasonsQuery = None,
    last_n_games: LastNGamesQuery = 4,
    season_type: SeasonTypeQuery = SeasonType.ALL,
) -> TeamForm:
    """Won-lost record and scoring over the most recent games."""
    return teams.recent_form(
        team, seasons=seasons, last_n_games=last_n_games, season_type=season_type
    )


@router.get("/{team}/splits", response_model=TeamHomeAwaySplits, summary="Home/away splits")
def team_splits(
    team: TeamPath,
    teams: TeamsDep,
    seasons: SeasonsQuery = None,
    last_n_games: LastNGamesQuery = None,
    season_type: SeasonTypeQuery = SeasonType.ALL,
) -> TeamHomeAwaySplits:
    """Form at home versus on the road. `last_n_games` applies within each split."""
    return teams.home_away_splits(
        team, seasons=seasons, last_n_games=last_n_games, season_type=season_type
    )


@router.get("/{team}/games", response_model=TeamGameLog, summary="Game log")
def team_games(
    team: TeamPath,
    teams: TeamsDep,
    seasons: SeasonsQuery = None,
    last_n_games: LastNGamesQuery = 4,
    season_type: SeasonTypeQuery = SeasonType.ALL,
    completed_only: bool = Query(default=True, description="Exclude games without a final score."),
) -> TeamGameLog:
    """Games from the team's own perspective, newest first."""
    return teams.game_log(
        team,
        seasons=seasons,
        last_n_games=last_n_games,
        season_type=season_type,
        completed_only=completed_only,
    )


@router.get("/{team}/injuries", response_model=TeamInjuryReport, summary="Team injury report")
def team_injuries(
    team: TeamPath,
    teams: TeamsDep,
    seasons: SeasonsQuery = None,
    week: int | None = Query(default=None, description="Restrict to a single week."),
) -> TeamInjuryReport:
    """Every reported injury for the team."""
    return teams.injury_report(team, seasons=seasons, week=week)
