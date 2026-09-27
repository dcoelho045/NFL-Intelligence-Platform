"""Team and matchup tools."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from nfl_platform.llm.results import ToolResult
from nfl_platform.llm.tools.base import Tool, ToolArgs
from nfl_platform.llm.tools.common import (
    LastNGamesField,
    SeasonsField,
    SeasonTypeField,
    TeamField,
    WeeksField,
)
from nfl_platform.schemas.common import SeasonType
from nfl_platform.services.container import ServiceContainer


class TeamStatsArgs(ToolArgs):
    team: TeamField
    side: Literal["offense", "defense"] = Field(
        default="offense",
        description=(
            "'offense' is what the team produced; 'defense' is what it allowed to opponents."
        ),
    )
    seasons: SeasonsField = None
    last_n_games: LastNGamesField = 4
    weeks: WeeksField = None
    season_type: SeasonTypeField = SeasonType.ALL


class GetTeamStatsTool(Tool[TeamStatsArgs]):
    name: ClassVar[str] = "get_team_stats"
    description: ClassVar[str] = (
        "Get team passing/rushing yards, touchdowns and EPA, either produced by the offense or "
        "allowed by the defense. Use side='defense' for questions about how a team defends."
    )
    args_model: ClassVar[type[ToolArgs]] = TeamStatsArgs

    def invoke(self, container: ServiceContainer, args: TeamStatsArgs) -> ToolResult:
        method = container.teams.defense if args.side == "defense" else container.teams.offense
        summary = method(
            args.team,
            seasons=args.seasons,
            last_n_games=args.last_n_games,
            weeks=args.weeks,
            season_type=args.season_type,
        )
        return ToolResult.from_dto(self.name, summary)


class TeamFormArgs(ToolArgs):
    team: TeamField
    split: Literal["all", "home", "away"] = Field(
        default="all", description="Restrict the record to home games, away games, or neither."
    )
    seasons: SeasonsField = None
    last_n_games: LastNGamesField = 4
    season_type: SeasonTypeField = SeasonType.ALL


class GetTeamFormTool(Tool[TeamFormArgs]):
    name: ClassVar[str] = "get_team_form"
    description: ClassVar[str] = (
        "Get a team's won-lost record, scoring averages and recent results sequence. "
        "Use split='home' or 'away' for home/away performance questions."
    )
    args_model: ClassVar[type[ToolArgs]] = TeamFormArgs

    def invoke(self, container: ServiceContainer, args: TeamFormArgs) -> ToolResult:
        if args.split == "all":
            form = container.teams.recent_form(
                args.team,
                seasons=args.seasons,
                last_n_games=args.last_n_games,
                season_type=args.season_type,
            )
        else:
            splits = container.teams.home_away_splits(
                args.team,
                seasons=args.seasons,
                last_n_games=args.last_n_games,
                season_type=args.season_type,
            )
            form = splits.home if args.split == "home" else splits.away
        return ToolResult.from_dto(self.name, form)


class RecentGamesArgs(ToolArgs):
    team: TeamField
    seasons: SeasonsField = None
    last_n_games: LastNGamesField = 4
    season_type: SeasonTypeField = SeasonType.ALL


class GetRecentGamesTool(Tool[RecentGamesArgs]):
    name: ClassVar[str] = "get_recent_games"
    description: ClassVar[str] = (
        "Get a team's game-by-game results with opponent, scores and outcome, newest first. "
        "Use for questions about specific games or streaks."
    )
    args_model: ClassVar[type[ToolArgs]] = RecentGamesArgs

    def invoke(self, container: ServiceContainer, args: RecentGamesArgs) -> ToolResult:
        log = container.teams.game_log(
            args.team,
            seasons=args.seasons,
            last_n_games=args.last_n_games,
            season_type=args.season_type,
        )
        return ToolResult.from_dto(self.name, log)


class TeamInjuriesArgs(ToolArgs):
    team: TeamField
    seasons: SeasonsField = None
    week: int | None = Field(
        default=None, ge=1, le=22, description="A single week number. Omit for every week."
    )


class GetTeamInjuriesTool(Tool[TeamInjuriesArgs]):
    name: ClassVar[str] = "get_team_injuries"
    description: ClassVar[str] = (
        "Get every reported injury for a team in a window, with player names and report status."
    )
    args_model: ClassVar[type[ToolArgs]] = TeamInjuriesArgs

    def invoke(self, container: ServiceContainer, args: TeamInjuriesArgs) -> ToolResult:
        report = container.teams.injury_report(args.team, seasons=args.seasons, week=args.week)
        return ToolResult.from_dto(self.name, report)


class MatchupArgs(ToolArgs):
    team_a: TeamField
    team_b: TeamField
    seasons: SeasonsField = None
    week: int | None = Field(
        default=None, ge=1, le=22, description="Week of the scheduled fixture, if asking about one."
    )
    last_n_games: LastNGamesField = 4
    season_type: SeasonTypeField = SeasonType.ALL


class GetMatchupTool(Tool[MatchupArgs]):
    name: ClassVar[str] = "get_matchup"
    description: ClassVar[str] = (
        "Compare two teams side by side: offense, defense, recent form, the scheduled fixture "
        "and head-to-head history. This is descriptive data only and contains no prediction or "
        "win probability -- any judgement about who is likely to win is your reasoning over "
        "these numbers, and must be presented as such."
    )
    args_model: ClassVar[type[ToolArgs]] = MatchupArgs

    def invoke(self, container: ServiceContainer, args: MatchupArgs) -> ToolResult:
        analysis = container.matchups.analyze(
            args.team_a,
            args.team_b,
            seasons=args.seasons,
            week=args.week,
            last_n_games=args.last_n_games,
            season_type=args.season_type,
        )
        return ToolResult.from_dto(self.name, analysis)
