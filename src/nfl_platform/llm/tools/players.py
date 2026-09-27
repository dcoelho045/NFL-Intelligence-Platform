"""Player-facing tools, including identity resolution.

``search_players`` matters more than its size suggests: it is the recovery path
from :class:`AmbiguousPlayerError`, and roughly 7.5% of nflverse players share a
display name with someone else.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from nfl_platform.llm.results import ToolResult
from nfl_platform.llm.tools.base import Tool, ToolArgs
from nfl_platform.llm.tools.common import (
    LastNGamesField,
    PlayerField,
    SeasonsField,
    SeasonTypeField,
    WeeksField,
)
from nfl_platform.schemas.common import SeasonType
from nfl_platform.services.container import ServiceContainer

MAX_COMPARISON_PLAYERS = 6


class SearchPlayersArgs(ToolArgs):
    name: str = Field(description="Full or partial player name to look up.", min_length=1)
    position: str | None = Field(
        default=None, description="Optional position filter, for example 'QB', 'WR'."
    )
    team: str | None = Field(default=None, description="Optional team abbreviation filter.")
    season: int | None = Field(default=None, description="Optional season the player was active.")
    limit: int = Field(default=10, ge=1, le=25, description="Maximum candidates to return.")


class SearchPlayersTool(Tool[SearchPlayersArgs]):
    name: ClassVar[str] = "search_players"
    description: ClassVar[str] = (
        "Find players by name and get their GSIS ids. Use this first when a name might match "
        "more than one player, and use it to recover from an ambiguous_player error."
    )
    args_model: ClassVar[type[ToolArgs]] = SearchPlayersArgs

    def invoke(self, container: ServiceContainer, args: SearchPlayersArgs) -> ToolResult:
        result = container.directory.search(
            args.name,
            position=args.position,
            team=args.team,
            season=args.season,
            limit=args.limit,
        )
        return ToolResult.from_dto(self.name, result)


class PlayerProfileArgs(ToolArgs):
    player: PlayerField


class GetPlayerProfileTool(Tool[PlayerProfileArgs]):
    name: ClassVar[str] = "get_player_profile"
    description: ClassVar[str] = (
        "Get biographical and career details for one player: position, current team, draft "
        "information, college, experience. Contains no game statistics."
    )
    args_model: ClassVar[type[ToolArgs]] = PlayerProfileArgs

    def invoke(self, container: ServiceContainer, args: PlayerProfileArgs) -> ToolResult:
        player_id = container.directory.resolve(args.player)
        return ToolResult.from_dto(self.name, container.directory.profile(player_id))


class PlayerStatsArgs(ToolArgs):
    player: PlayerField
    window: Literal["recent", "season"] = Field(
        default="recent",
        description=(
            "'recent' aggregates the last N games; 'season' aggregates entire seasons and "
            "ignores last_n_games."
        ),
    )
    seasons: SeasonsField = None
    last_n_games: LastNGamesField = 4
    season_type: SeasonTypeField = SeasonType.ALL


class GetPlayerStatsTool(Tool[PlayerStatsArgs]):
    name: ClassVar[str] = "get_player_stats"
    description: ClassVar[str] = (
        "Get passing, rushing and receiving totals plus per-game averages for one player. "
        "This is the default tool for questions like 'how many yards did X throw for'."
    )
    args_model: ClassVar[type[ToolArgs]] = PlayerStatsArgs

    def invoke(self, container: ServiceContainer, args: PlayerStatsArgs) -> ToolResult:
        if args.window == "season":
            stats = container.players.season_performance(
                args.player,
                seasons=args.seasons,
                season_type=args.season_type,
            )
        else:
            stats = container.players.recent_performance(
                args.player,
                seasons=args.seasons,
                last_n_games=args.last_n_games,
                season_type=args.season_type,
            )
        return ToolResult.from_dto(self.name, stats)


class PlayerGameLogArgs(ToolArgs):
    player: PlayerField
    seasons: SeasonsField = None
    last_n_games: LastNGamesField = 4
    weeks: WeeksField = None
    season_type: SeasonTypeField = SeasonType.ALL


class GetPlayerGameLogTool(Tool[PlayerGameLogArgs]):
    name: ClassVar[str] = "get_player_game_log"
    description: ClassVar[str] = (
        "Get game-by-game statistics for one player, newest first. Use when the question is "
        "about specific weeks, consistency, or week-to-week variation rather than totals."
    )
    args_model: ClassVar[type[ToolArgs]] = PlayerGameLogArgs

    def invoke(self, container: ServiceContainer, args: PlayerGameLogArgs) -> ToolResult:
        log = container.players.game_log(
            args.player,
            seasons=args.seasons,
            last_n_games=args.last_n_games,
            weeks=args.weeks,
            season_type=args.season_type,
        )
        return ToolResult.from_dto(self.name, log)


class PlayerUsageArgs(ToolArgs):
    player: PlayerField
    seasons: SeasonsField = None
    last_n_games: LastNGamesField = 4
    season_type: SeasonTypeField = SeasonType.ALL


class GetPlayerUsageTool(Tool[PlayerUsageArgs]):
    name: ClassVar[str] = "get_player_usage"
    description: ClassVar[str] = (
        "Get opportunity metrics for one player: snap share, target share, air yards share, "
        "WOPR, targets and carries. Use for questions about role, workload or involvement."
    )
    args_model: ClassVar[type[ToolArgs]] = PlayerUsageArgs

    def invoke(self, container: ServiceContainer, args: PlayerUsageArgs) -> ToolResult:
        usage = container.players.usage(
            args.player,
            seasons=args.seasons,
            last_n_games=args.last_n_games,
            season_type=args.season_type,
        )
        return ToolResult.from_dto(self.name, usage)


class PlayerInjuriesArgs(ToolArgs):
    player: PlayerField
    seasons: SeasonsField = None
    weeks: WeeksField = None
    latest_only: bool = Field(
        default=False, description="True returns only the most recent injury report entry."
    )


class GetPlayerInjuriesTool(Tool[PlayerInjuriesArgs]):
    name: ClassVar[str] = "get_player_injuries"
    description: ClassVar[str] = (
        "Get injury report history for one player. An empty entries list means no injury was "
        "reported, which is a fact about the player, not a failure."
    )
    args_model: ClassVar[type[ToolArgs]] = PlayerInjuriesArgs

    def invoke(self, container: ServiceContainer, args: PlayerInjuriesArgs) -> ToolResult:
        report = container.players.injury_report(
            args.player,
            seasons=args.seasons,
            weeks=args.weeks,
            latest_only=args.latest_only,
        )
        return ToolResult.from_dto(self.name, report)


class ComparePlayersArgs(ToolArgs):
    players: list[str] = Field(
        min_length=2,
        max_length=MAX_COMPARISON_PLAYERS,
        description="Two to six player names or GSIS ids to compare over one shared window.",
    )
    seasons: SeasonsField = None
    last_n_games: LastNGamesField = 4
    season_type: SeasonTypeField = SeasonType.ALL


class ComparePlayersTool(Tool[ComparePlayersArgs]):
    name: ClassVar[str] = "compare_players"
    description: ClassVar[str] = (
        "Compare two to six players' statistics over the same window in one call. Prefer this "
        "over several get_player_stats calls so the comparison window is guaranteed identical."
    )
    args_model: ClassVar[type[ToolArgs]] = ComparePlayersArgs

    def invoke(self, container: ServiceContainer, args: ComparePlayersArgs) -> ToolResult:
        comparison = container.players.compare(
            args.players,
            seasons=args.seasons,
            last_n_games=args.last_n_games,
            season_type=args.season_type,
        )
        return ToolResult.from_dto(self.name, comparison)
