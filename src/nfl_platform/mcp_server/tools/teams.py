"""Team and matchup tools."""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.server import MCPServer
from pydantic import Field

from nfl_platform.mcp_server.errors import tool_errors
from nfl_platform.mcp_server.tools.players import DETERMINISTIC
from nfl_platform.schemas.common import SeasonType
from nfl_platform.schemas.fields import (
    LastNGamesField,
    SeasonsField,
    SeasonTypeField,
    TeamField,
    WeekField,
    WeeksField,
)
from nfl_platform.schemas.matchup import MatchupAnalysis
from nfl_platform.schemas.team import (
    TeamDefenseSummary,
    TeamForm,
    TeamInjuryReport,
    TeamOffenseSummary,
)
from nfl_platform.services.container import ServiceContainer


def register_team_tools(mcp: MCPServer, container: ServiceContainer) -> None:
    """Register the team-facing tools against one service container."""

    @mcp.tool(title="Team statistics", annotations=DETERMINISTIC)
    def get_team_stats(
        team: TeamField,
        side: Annotated[
            Literal["offense", "defense"],
            Field(
                description=(
                    "'offense' is what the team produced; 'defense' is what it allowed "
                    "to opponents."
                )
            ),
        ] = "offense",
        seasons: SeasonsField = None,
        last_n_games: LastNGamesField = 4,
        weeks: WeeksField = None,
        season_type: SeasonTypeField = SeasonType.ALL,
    ) -> TeamOffenseSummary | TeamDefenseSummary:
        """Get team passing/rushing yards, touchdowns and EPA.

        Either produced by the offense or allowed by the defense. Use side='defense'
        for questions about how a team defends.
        """
        with tool_errors():
            if side == "defense":
                return container.teams.defense(
                    team,
                    seasons=seasons,
                    last_n_games=last_n_games,
                    weeks=weeks,
                    season_type=season_type,
                )
            return container.teams.offense(
                team,
                seasons=seasons,
                last_n_games=last_n_games,
                weeks=weeks,
                season_type=season_type,
            )

    @mcp.tool(title="Team form", annotations=DETERMINISTIC)
    def get_team_form(
        team: TeamField,
        split: Annotated[
            Literal["all", "home", "away"],
            Field(description="Restrict the record to home games, away games, or neither."),
        ] = "all",
        seasons: SeasonsField = None,
        last_n_games: LastNGamesField = 4,
        season_type: SeasonTypeField = SeasonType.ALL,
    ) -> TeamForm:
        """Get a team's won-lost record, scoring averages and recent results sequence.

        Use split='home' or 'away' for home/away performance questions.
        """
        with tool_errors():
            if split == "all":
                return container.teams.recent_form(
                    team, seasons=seasons, last_n_games=last_n_games, season_type=season_type
                )
            splits = container.teams.home_away_splits(
                team, seasons=seasons, last_n_games=last_n_games, season_type=season_type
            )
            return splits.home if split == "home" else splits.away

    @mcp.tool(title="Team injury report", annotations=DETERMINISTIC)
    def get_team_injuries(
        team: TeamField,
        seasons: SeasonsField = None,
        week: WeekField = None,
    ) -> TeamInjuryReport:
        """Get every reported injury for a team in a window.

        Includes player names and report status. An empty list means nothing was
        reported for that window, which is a historical record rather than a
        guarantee about current availability.
        """
        with tool_errors():
            return container.teams.injury_report(team, seasons=seasons, week=week)

    @mcp.tool(title="Team matchup", annotations=DETERMINISTIC)
    def get_matchup(
        team_a: TeamField,
        team_b: TeamField,
        seasons: SeasonsField = None,
        week: WeekField = None,
        last_n_games: LastNGamesField = 4,
        season_type: SeasonTypeField = SeasonType.ALL,
    ) -> MatchupAnalysis:
        """Compare two teams side by side.

        Offense, defense, recent form, the scheduled fixture and head-to-head
        history. This is descriptive data only and contains no prediction or win
        probability: any judgement about who is likely to win is the caller's
        reasoning over these numbers and must be presented as such.
        """
        with tool_errors():
            return container.matchups.analyze(
                team_a,
                team_b,
                seasons=seasons,
                week=week,
                last_n_games=last_n_games,
                season_type=season_type,
            )
