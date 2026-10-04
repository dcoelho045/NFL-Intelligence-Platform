"""Prediction tools.

The only tools on this server whose numbers refer to games that have not been
played. They sit behind the same service the REST API and the in-process agent
use, so the three surfaces cannot disagree about a probability -- which is the
whole reason MCP is an interface layer here rather than a second implementation.

``DETERMINISTIC`` annotations are correct and worth justifying. Given a frozen
artifact and a fixed data snapshot these calls are pure: the same arguments
return the same probability every time. "Deterministic" is a statement about
repeatability, not about certainty, and the descriptions carry the uncertainty.
"""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.server import MCPServer
from pydantic import Field

from nfl_platform.config.settings import Settings
from nfl_platform.mcp_server.errors import tool_errors, unavailable
from nfl_platform.mcp_server.tools.players import DETERMINISTIC
from nfl_platform.schemas.fields import (
    ComparedPlayersField,
    PlayerField,
    SeasonField,
    TeamField,
    WeekField,
)
from nfl_platform.schemas.prediction import (
    FantasyProjection,
    FantasyProjectionComparison,
    GamePrediction,
    GamePredictionSlate,
)
from nfl_platform.services.container import ServiceContainer

MAX_COMPARED = 6


def register_prediction_tools(
    mcp: MCPServer, container: ServiceContainer, settings: Settings
) -> None:
    """Register the prediction tools against one service container."""

    def _require_enabled() -> None:
        if settings.ml_enable_predictions:
            return
        raise unavailable(
            "model_not_available",
            "Prediction models are switched off on this deployment.",
            recovery=(
                "Do not estimate a probability or a projection yourself. Answer from the "
                "deterministic analytics tools where the question allows it, and say that "
                "no model prediction is available."
            ),
        )

    @mcp.tool(title="Predict a game", annotations=DETERMINISTIC)
    def predict_game(
        home_team: TeamField,
        away_team: TeamField,
        season: SeasonField = None,
        week: WeekField = None,
    ) -> GamePrediction:
        """Model probability that the home team wins a specific game.

        Use this for any question about who will win or who is favoured. Do not
        state a win probability you worked out yourself.

        This is a model estimate about a game that has not been played, not a
        measured statistic. Report it as an estimate and name the model version
        in `model_info`. Betting lines are not inputs to the model;
        `market_home_win_probability` is a separate reference point where a line
        was published, and the two must not be presented as the same number.
        """
        _require_enabled()
        with tool_errors():
            return container.predictions.predict_game(
                home_team, away_team, season=season, week=week
            )

    @mcp.tool(title="Predict a week's games", annotations=DETERMINISTIC)
    def predict_slate(
        season: Annotated[int, Field(description="Season to predict.")],
        week: Annotated[int, Field(ge=1, le=22, description="Week to predict.")],
    ) -> GamePredictionSlate:
        """Model probabilities for every game in one week.

        Games the model cannot score appear in `skipped` with the reason. That
        list is part of the answer: a shorter slate than the schedule is not
        evidence that the missing games are not being played.
        """
        _require_enabled()
        with tool_errors():
            return container.predictions.predict_slate(season, week)

    @mcp.tool(title="Project fantasy points", annotations=DETERMINISTIC)
    def project_fantasy_points(
        player: PlayerField,
        season: SeasonField = None,
        week: WeekField = None,
        scoring: Annotated[
            Literal["ppr"],
            Field(description="Only PPR is modelled in this version."),
        ] = "ppr",
    ) -> FantasyProjection:
        """Projected PPR fantasy points for one player in one week.

        **Conditional on the player playing.** The model was trained only on
        games the player took a snap in, so this is production given that he
        plays and contains no estimate of the chance he is inactive. State that
        condition whenever you quote the number, and check `get_player_injuries`
        before treating it as a recommendation.

        For points a player has already scored, use `get_fantasy_production`
        instead: that is measured, this is predicted.
        """
        _require_enabled()
        with tool_errors():
            return container.predictions.project_fantasy_points(player, season=season, week=week)

    @mcp.tool(title="Compare fantasy projections", annotations=DETERMINISTIC)
    def compare_fantasy_projections(
        players: ComparedPlayersField,
        season: SeasonField = None,
        week: WeekField = None,
        scoring: Annotated[Literal["ppr"], Field()] = "ppr",
    ) -> FantasyProjectionComparison:
        """Rank several players' projections for one week, for a start/sit answer.

        Quote `margin` alongside the ranking. When the gap is inside the model's
        typical error the honest answer is that it is close, not that one player
        is better. Players who could not be projected appear in `caveats` rather
        than being dropped silently.
        """
        _require_enabled()
        with tool_errors():
            return container.predictions.compare_projections(
                list(players), season=season, week=week
            )
