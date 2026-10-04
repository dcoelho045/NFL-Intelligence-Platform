"""Prediction tools.

The only tools in the catalogue whose output is an estimate rather than a
measurement, which is why they are the only ones declaring
``source_kind = SourceKind.MODEL``. That is not decoration: the grounding check
reads it to decide whether a number in an answer is traceable, and the citation
builder reads it to credit the model artifact rather than the tool name.

The descriptions carry one instruction each, aimed at the failure these tools
make possible -- a model that has a win probability available and writes its own
instead, or that reports a projection without the condition attached to it.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from nfl_platform.llm.results import ToolResult
from nfl_platform.llm.tools.base import Tool, ToolArgs
from nfl_platform.schemas.evidence import SourceKind
from nfl_platform.schemas.fields import (
    ComparedPlayersField,
    PlayerField,
    SeasonField,
    TeamField,
    WeekField,
)
from nfl_platform.services.container import ServiceContainer


class PredictGameArgs(ToolArgs):
    home_team: TeamField
    away_team: TeamField
    season: SeasonField = None
    week: WeekField = None


class PredictGameTool(Tool[PredictGameArgs]):
    name: ClassVar[str] = "predict_game"
    description: ClassVar[str] = (
        "Get the model's probability that the home team wins a specific game. Use this "
        "for any question about who will win, who is favoured, or how likely a result "
        "is. Never state a win probability you worked out yourself -- if this tool "
        "cannot answer, say so. The result is a model estimate about an unplayed game, "
        "not a measured statistic; report it as such and quote the model version. "
        "Betting lines are not inputs to the model: market_home_win_probability is a "
        "separate reference point, and the two should not be conflated."
    )
    args_model: ClassVar[type[ToolArgs]] = PredictGameArgs
    source_kind: ClassVar[SourceKind] = SourceKind.MODEL

    def invoke(self, container: ServiceContainer, args: PredictGameArgs) -> ToolResult:
        prediction = container.predictions.predict_game(
            args.home_team, args.away_team, season=args.season, week=args.week
        )
        return ToolResult.from_dto(self.name, prediction, source_kind=self.source_kind)


class PredictSlateArgs(ToolArgs):
    season: int = Field(description="Season to predict.")
    week: int = Field(ge=1, le=22, description="Week to predict.")


class PredictSlateTool(Tool[PredictSlateArgs]):
    name: ClassVar[str] = "predict_slate"
    description: ClassVar[str] = (
        "Get the model's win probability for every game in one week. Use this for "
        "'who wins this week' questions rather than calling predict_game repeatedly. "
        "Games the model cannot score appear in 'skipped' with the reason; report that "
        "list, because a shorter slate than the schedule is not evidence those games "
        "are not being played."
    )
    args_model: ClassVar[type[ToolArgs]] = PredictSlateArgs
    source_kind: ClassVar[SourceKind] = SourceKind.MODEL

    def invoke(self, container: ServiceContainer, args: PredictSlateArgs) -> ToolResult:
        slate = container.predictions.predict_slate(args.season, args.week)
        return ToolResult.from_dto(self.name, slate, source_kind=self.source_kind)


class ProjectFantasyArgs(ToolArgs):
    player: PlayerField
    season: SeasonField = None
    week: WeekField = None
    scoring: Literal["ppr"] = Field(
        default="ppr",
        description="Only PPR is modelled in this version.",
    )


class ProjectFantasyPointsTool(Tool[ProjectFantasyArgs]):
    name: ClassVar[str] = "project_fantasy_points"
    description: ClassVar[str] = (
        "Get the model's projected PPR fantasy points for one player in one week. Use "
        "this for any 'how many points will X score' question. Never estimate a "
        "projection yourself. The projection is CONDITIONAL ON THE PLAYER PLAYING: it "
        "contains no estimate of the chance he is inactive, so always state that "
        "condition and suggest checking injury status with get_player_injuries. For "
        "what a player has already scored, use get_fantasy_production instead -- that "
        "is measured, this is predicted."
    )
    args_model: ClassVar[type[ToolArgs]] = ProjectFantasyArgs
    source_kind: ClassVar[SourceKind] = SourceKind.MODEL

    def invoke(self, container: ServiceContainer, args: ProjectFantasyArgs) -> ToolResult:
        projection = container.predictions.project_fantasy_points(
            args.player, season=args.season, week=args.week
        )
        return ToolResult.from_dto(self.name, projection, source_kind=self.source_kind)


class CompareProjectionsArgs(ToolArgs):
    players: ComparedPlayersField
    season: SeasonField = None
    week: WeekField = None
    scoring: Literal["ppr"] = Field(default="ppr")


class CompareFantasyProjectionsTool(Tool[CompareProjectionsArgs]):
    name: ClassVar[str] = "compare_fantasy_projections"
    description: ClassVar[str] = (
        "Project several players for the same week and rank them. This is the tool for "
        "start/sit questions. Report the margin as well as the order: when the gap is "
        "inside the model's typical error the honest answer is that it is close, not "
        "that one player is better. Every projection is conditional on the player "
        "playing."
    )
    args_model: ClassVar[type[ToolArgs]] = CompareProjectionsArgs
    source_kind: ClassVar[SourceKind] = SourceKind.MODEL

    def invoke(self, container: ServiceContainer, args: CompareProjectionsArgs) -> ToolResult:
        comparison = container.predictions.compare_projections(
            list(args.players), season=args.season, week=args.week
        )
        return ToolResult.from_dto(self.name, comparison, source_kind=self.source_kind)
