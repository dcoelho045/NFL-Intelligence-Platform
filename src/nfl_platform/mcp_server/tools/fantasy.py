"""Fantasy football tools."""

from __future__ import annotations

from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field

from nfl_platform.mcp_server.errors import tool_errors
from nfl_platform.mcp_server.tools.players import DETERMINISTIC
from nfl_platform.schemas.common import SeasonType
from nfl_platform.schemas.fantasy import FantasyComparison, FantasyProduction, FantasyScoring
from nfl_platform.schemas.fields import (
    ComparedPlayersField,
    LastNGamesField,
    PlayerField,
    SeasonsField,
    SeasonTypeField,
)
from nfl_platform.services.container import ServiceContainer

ScoringField = Annotated[
    FantasyScoring,
    Field(
        default=FantasyScoring.PPR,
        description="Scoring format: 'ppr' awards a point per reception, 'standard' does not.",
    ),
]


def register_fantasy_tools(mcp: MCPServer, container: ServiceContainer) -> None:
    """Register the fantasy tools against one service container."""

    @mcp.tool(title="Fantasy production", annotations=DETERMINISTIC)
    def get_fantasy_production(
        player: PlayerField,
        seasons: SeasonsField = None,
        last_n_games: LastNGamesField = 4,
        scoring: ScoringField = FantasyScoring.PPR,
        season_type: SeasonTypeField = SeasonType.ALL,
    ) -> FantasyProduction:
        """Get a player's week-by-week fantasy points.

        Includes total, average, best and worst weeks and a standard deviation. The
        standard deviation describes past consistency; it is not a projection, and
        this server produces no projections.
        """
        with tool_errors():
            return container.fantasy.production(
                player,
                seasons=seasons,
                last_n_games=last_n_games,
                scoring=scoring,
                season_type=season_type,
            )

    @mcp.tool(title="Compare fantasy players", annotations=DETERMINISTIC)
    def compare_fantasy_players(
        players: ComparedPlayersField,
        seasons: SeasonsField = None,
        last_n_games: LastNGamesField = 4,
        scoring: ScoringField = FantasyScoring.PPR,
        season_type: SeasonTypeField = SeasonType.ALL,
    ) -> FantasyComparison:
        """Compare two to six players' fantasy production over the same window.

        This is the right tool for "should I start A or B" questions. It reports
        what each player has already produced; the start/sit judgement is the
        caller's.
        """
        with tool_errors():
            return container.fantasy.compare(
                players,
                seasons=seasons,
                last_n_games=last_n_games,
                scoring=scoring,
                season_type=season_type,
            )
