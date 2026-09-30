"""Fantasy football tools."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from nfl_platform.llm.results import ToolResult
from nfl_platform.llm.tools.base import Tool, ToolArgs
from nfl_platform.llm.tools.common import (
    ComparedPlayersField,
    LastNGamesField,
    PlayerField,
    SeasonsField,
    SeasonTypeField,
    TeamField,
)
from nfl_platform.schemas.common import SeasonType
from nfl_platform.schemas.fantasy import FantasyScoring
from nfl_platform.services.container import ServiceContainer

ScoringField = Field(
    default=FantasyScoring.PPR,
    description="Scoring format: 'ppr' awards a point per reception, 'standard' does not.",
)


class FantasyProductionArgs(ToolArgs):
    player: PlayerField
    seasons: SeasonsField = None
    last_n_games: LastNGamesField = 4
    scoring: FantasyScoring = ScoringField
    season_type: SeasonTypeField = SeasonType.ALL


class GetFantasyProductionTool(Tool[FantasyProductionArgs]):
    name: ClassVar[str] = "get_fantasy_production"
    description: ClassVar[str] = (
        "Get a player's week-by-week fantasy points with total, average, best and worst weeks "
        "and a standard deviation. The standard deviation describes past consistency; it is "
        "not a projection."
    )
    args_model: ClassVar[type[ToolArgs]] = FantasyProductionArgs

    def invoke(self, container: ServiceContainer, args: FantasyProductionArgs) -> ToolResult:
        production = container.fantasy.production(
            args.player,
            seasons=args.seasons,
            last_n_games=args.last_n_games,
            scoring=args.scoring,
            season_type=args.season_type,
        )
        return ToolResult.from_dto(self.name, production)


class FantasyUsageTrendArgs(ToolArgs):
    player: PlayerField
    seasons: SeasonsField = None
    last_n_games: LastNGamesField = 8
    season_type: SeasonTypeField = SeasonType.ALL


class GetFantasyUsageTrendTool(Tool[FantasyUsageTrendArgs]):
    name: ClassVar[str] = "get_fantasy_usage_trend"
    description: ClassVar[str] = (
        "Compare a player's target share and snap share in the earlier half of a window against "
        "the recent half, to show whether their role is rising, falling or stable. Describes "
        "what already happened; it does not forecast."
    )
    args_model: ClassVar[type[ToolArgs]] = FantasyUsageTrendArgs

    def invoke(self, container: ServiceContainer, args: FantasyUsageTrendArgs) -> ToolResult:
        trend = container.fantasy.usage_trend(
            args.player,
            seasons=args.seasons,
            last_n_games=args.last_n_games,
            season_type=args.season_type,
        )
        return ToolResult.from_dto(self.name, trend)


class FantasyMatchupContextArgs(ToolArgs):
    player: PlayerField
    opponent: TeamField
    seasons: SeasonsField = None
    last_n_games: LastNGamesField = 4
    season_type: SeasonTypeField = SeasonType.ALL


class GetFantasyMatchupContextTool(Tool[FantasyMatchupContextArgs]):
    name: ClassVar[str] = "get_fantasy_matchup_context"
    description: ClassVar[str] = (
        "Get what an upcoming opponent's defense has recently allowed, paired with the player's "
        "identity. Use for start/sit questions where the opponent matters."
    )
    args_model: ClassVar[type[ToolArgs]] = FantasyMatchupContextArgs

    def invoke(self, container: ServiceContainer, args: FantasyMatchupContextArgs) -> ToolResult:
        context = container.fantasy.matchup_context(
            args.player,
            opponent=args.opponent,
            seasons=args.seasons,
            last_n_games=args.last_n_games,
            season_type=args.season_type,
        )
        return ToolResult.from_dto(self.name, context)


class CompareFantasyPlayersArgs(ToolArgs):
    players: ComparedPlayersField
    seasons: SeasonsField = None
    last_n_games: LastNGamesField = 4
    scoring: FantasyScoring = ScoringField
    season_type: SeasonTypeField = SeasonType.ALL


class CompareFantasyPlayersTool(Tool[CompareFantasyPlayersArgs]):
    name: ClassVar[str] = "compare_fantasy_players"
    description: ClassVar[str] = (
        "Compare two to six players' fantasy production over the same window. This is the right "
        "tool for 'should I start A or B' questions."
    )
    args_model: ClassVar[type[ToolArgs]] = CompareFantasyPlayersArgs

    def invoke(self, container: ServiceContainer, args: CompareFantasyPlayersArgs) -> ToolResult:
        comparison = container.fantasy.compare(
            args.players,
            seasons=args.seasons,
            last_n_games=args.last_n_games,
            scoring=args.scoring,
            season_type=args.season_type,
        )
        return ToolResult.from_dto(self.name, comparison)
