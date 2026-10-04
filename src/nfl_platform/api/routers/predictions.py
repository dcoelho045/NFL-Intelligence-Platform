"""Prediction endpoints.

Separate from ``/matchups`` on purpose. The matchup endpoint is descriptive and
says so in its docstring; these endpoints are the only place in the HTTP surface
where a number refers to a game that has not been played, and keeping them under
their own prefix makes that boundary visible in a URL rather than in prose.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Query

from nfl_platform.api.dependencies import PredictionsDep
from nfl_platform.schemas.prediction import (
    FantasyProjection,
    FantasyProjectionComparison,
    GamePrediction,
    GamePredictionSlate,
    ModelRegistryStatus,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])

MAX_COMPARED = 6


@router.get("/models", response_model=ModelRegistryStatus, summary="Installed models")
def model_status(predictions: PredictionsDep) -> ModelRegistryStatus:
    """Which prediction models are installed, at which version, trained on what.

    Returns 200 even when nothing is installed; ``available`` and ``notes`` carry
    that fact. A status endpoint that fails when the thing it reports on is
    missing cannot do its job.
    """
    return predictions.model_status()


@router.get("/game", response_model=GamePrediction, summary="Predict one game")
def predict_game(
    predictions: PredictionsDep,
    home_team: str = Query(description="Home team abbreviation, e.g. 'KC'."),
    away_team: str = Query(description="Away team abbreviation, e.g. 'BUF'."),
    season: int | None = Query(default=None, description="Defaults to the current season."),
    week: int | None = Query(
        default=None,
        description="Defaults to the first scheduled meeting of these teams that season.",
    ),
) -> GamePrediction:
    """Model probability that the home team wins.

    A model estimate, not a measured statistic. Betting lines are not inputs;
    ``market_home_win_probability`` is shown for reference where a line exists.
    """
    return predictions.predict_game(home_team, away_team, season=season, week=week)


@router.get(
    "/slate/{season}/{week}",
    response_model=GamePredictionSlate,
    summary="Predict a week's games",
)
def predict_slate(predictions: PredictionsDep, season: int, week: int) -> GamePredictionSlate:
    """Every game in one week the model can score.

    Games without enough history are listed in ``skipped`` with the reason
    rather than silently omitted.
    """
    return predictions.predict_slate(season, week)


@router.get(
    "/fantasy/{player_id}",
    response_model=FantasyProjection,
    summary="Project one player's fantasy points",
)
def project_fantasy_points(
    predictions: PredictionsDep,
    player_id: str,
    season: int | None = Query(default=None),
    week: int | None = Query(default=None),
) -> FantasyProjection:
    """Projected PPR points, **conditional on the player playing**.

    The projection contains no estimate of the chance the player is inactive.
    Check injury status separately before acting on it.
    """
    return predictions.project_fantasy_points(player_id, season=season, week=week)


@router.post(
    "/fantasy/compare",
    response_model=FantasyProjectionComparison,
    summary="Compare fantasy projections",
)
def compare_projections(
    predictions: PredictionsDep,
    players: list[str] = Body(
        embed=True, min_length=2, max_length=MAX_COMPARED, description="Player ids or names."
    ),
    season: int | None = Body(default=None, embed=True),
    week: int | None = Body(default=None, embed=True),
) -> FantasyProjectionComparison:
    """Project several players for one week and name the highest.

    Players the model cannot project become caveats rather than failing the
    whole request.
    """
    return predictions.compare_projections(players, season=season, week=week)
