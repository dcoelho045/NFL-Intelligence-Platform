"""Player tools.

Each function is an adapter: validate through the schema, call one service
method, return the DTO the service already produced. If one of these ever needs
an ``if`` that changes a number, that logic belongs in ``services/`` where it can
be tested without a protocol in the way.

The DTO returned is also the published output schema, so an MCP client knows the
shape of a player's statistics before it ever calls the tool.
"""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from nfl_platform.domain import EARLIEST_SEASON
from nfl_platform.mcp_server.errors import tool_errors
from nfl_platform.schemas.common import SeasonType
from nfl_platform.schemas.fields import (
    ComparedPlayersField,
    LastNGamesField,
    PlayerField,
    SeasonsField,
    SeasonTypeField,
    WeeksField,
)
from nfl_platform.schemas.player import (
    PlayerComparison,
    PlayerGameLog,
    PlayerInjuryReport,
    PlayerProfile,
    PlayerSearchResult,
    PlayerSeasonStats,
    PlayerUsage,
)
from nfl_platform.services.container import ServiceContainer

# Deterministic statistics read from nflverse: nothing is written, and the
# answer never depends on anything outside the loaded data.
DETERMINISTIC = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def register_player_tools(mcp: MCPServer, container: ServiceContainer) -> None:
    """Register the player-facing tools against one service container."""

    @mcp.tool(title="Search players", annotations=DETERMINISTIC)
    def search_players(
        name: Annotated[
            str,
            Field(
                min_length=1,
                max_length=100,
                description="Full or partial player name to look up.",
            ),
        ],
        position: Annotated[
            str | None,
            Field(max_length=4, description="Optional position filter, for example 'QB', 'WR'."),
        ] = None,
        team: Annotated[
            str | None,
            Field(min_length=2, max_length=4, description="Optional team abbreviation filter."),
        ] = None,
        season: Annotated[
            int | None,
            Field(ge=EARLIEST_SEASON, description="Optional season the player was active."),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=25, description="Maximum candidates to return.")] = 10,
    ) -> PlayerSearchResult:
        """Find players by name and get their GSIS ids.

        Use this first when a name might match more than one player, and use it to
        recover from an ambiguous_player error.
        """
        with tool_errors():
            return container.directory.search(
                name, position=position, team=team, season=season, limit=limit
            )

    @mcp.tool(title="Player profile", annotations=DETERMINISTIC)
    def get_player_profile(player: PlayerField) -> PlayerProfile:
        """Get biographical and career details for one player.

        Position, current team, draft information, college and experience.
        Contains no game statistics.
        """
        with tool_errors():
            return container.directory.profile(container.directory.resolve(player))

    @mcp.tool(title="Player statistics", annotations=DETERMINISTIC)
    def get_player_stats(
        player: PlayerField,
        window: Annotated[
            Literal["recent", "season"],
            Field(
                description=(
                    "'recent' aggregates the last N games; 'season' aggregates entire "
                    "seasons and ignores last_n_games."
                )
            ),
        ] = "recent",
        seasons: SeasonsField = None,
        last_n_games: LastNGamesField = 4,
        season_type: SeasonTypeField = SeasonType.ALL,
    ) -> PlayerSeasonStats:
        """Get passing, rushing and receiving totals plus per-game averages for one player.

        This is the authoritative source for questions like "how many yards did X
        throw for". Every figure is computed from the nflverse record, not estimated.
        """
        with tool_errors():
            if window == "season":
                return container.players.season_performance(
                    player, seasons=seasons, season_type=season_type
                )
            return container.players.recent_performance(
                player, seasons=seasons, last_n_games=last_n_games, season_type=season_type
            )

    @mcp.tool(title="Player game log", annotations=DETERMINISTIC)
    def get_player_game_log(
        player: PlayerField,
        seasons: SeasonsField = None,
        last_n_games: LastNGamesField = 4,
        weeks: WeeksField = None,
        season_type: SeasonTypeField = SeasonType.ALL,
    ) -> PlayerGameLog:
        """Get game-by-game statistics for one player, newest first.

        Use when the question is about specific weeks, consistency, or week-to-week
        variation rather than totals.
        """
        with tool_errors():
            return container.players.game_log(
                player,
                seasons=seasons,
                last_n_games=last_n_games,
                weeks=weeks,
                season_type=season_type,
            )

    @mcp.tool(title="Player usage", annotations=DETERMINISTIC)
    def get_player_usage(
        player: PlayerField,
        seasons: SeasonsField = None,
        last_n_games: LastNGamesField = 4,
        season_type: SeasonTypeField = SeasonType.ALL,
    ) -> PlayerUsage:
        """Get opportunity metrics for one player.

        Snap share, target share, air yards share, WOPR, targets and carries. Use
        for questions about role, workload or involvement.
        """
        with tool_errors():
            return container.players.usage(
                player, seasons=seasons, last_n_games=last_n_games, season_type=season_type
            )

    @mcp.tool(title="Player injury report", annotations=DETERMINISTIC)
    def get_player_injuries(
        player: PlayerField,
        seasons: SeasonsField = None,
        weeks: WeeksField = None,
        latest_only: Annotated[
            bool, Field(description="True returns only the most recent injury report entry.")
        ] = False,
    ) -> PlayerInjuryReport:
        """Get injury report history for one player.

        An empty entries list means no injury was reported in the window. That is a
        fact about the player, not a failure, and it is a historical record rather
        than a current status.
        """
        with tool_errors():
            return container.players.injury_report(
                player, seasons=seasons, weeks=weeks, latest_only=latest_only
            )

    @mcp.tool(title="Compare players", annotations=DETERMINISTIC)
    def compare_players(
        players: ComparedPlayersField,
        seasons: SeasonsField = None,
        last_n_games: LastNGamesField = 4,
        season_type: SeasonTypeField = SeasonType.ALL,
    ) -> PlayerComparison:
        """Compare two to six players' statistics over the same window in one call.

        Prefer this over several get_player_stats calls, because the shared window is
        guaranteed identical and a comparison across different game counts misleads.
        """
        with tool_errors():
            return container.players.compare(
                players, seasons=seasons, last_n_games=last_n_games, season_type=season_type
            )
