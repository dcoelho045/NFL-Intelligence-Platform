"""Team analytics service.

Offence, defence, recent form, home/away splits, game logs, and injuries.
Home/away splitting is the one genuinely new computation in this milestone; the
rest composes Milestone 1 functions.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from nfl_platform.analytics.games import get_recent_games, summarize_recent_games
from nfl_platform.analytics.injuries import get_team_injuries
from nfl_platform.analytics.teams import get_team_defense, get_team_stats, summarize_team_stats
from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.domain import SeasonInput, normalize_team
from nfl_platform.schemas.common import PlayerRef, SeasonType
from nfl_platform.schemas.team import (
    TeamDefenseSummary,
    TeamForm,
    TeamGameLog,
    TeamGameResult,
    TeamHomeAwaySplits,
    TeamInjuryEntry,
    TeamInjuryReport,
    TeamOffenseSummary,
    TeamSplit,
)
from nfl_platform.services.base import AnalyticsService
from nfl_platform.services.players import PlayerAnalyticsService


class TeamAnalyticsService(AnalyticsService):
    """Team-level views over the deterministic analytics layer."""

    def __init__(
        self, loader: NflDataLoader, players: PlayerAnalyticsService | None = None
    ) -> None:
        super().__init__(loader)
        self._players = players or PlayerAnalyticsService(loader)

    # --- offence / defence ----------------------------------------------------

    def offense(
        self,
        team: str,
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = 4,
        weeks: int | Iterable[int] | None = None,
        season_type: SeasonType = SeasonType.ALL,
    ) -> TeamOffenseSummary:
        """What the team's offence produced."""
        normalized = normalize_team(team)
        frame = get_team_stats(
            normalized,
            loader=self._loader,
            seasons=seasons,
            last_n_games=last_n_games,
            weeks=weeks,
            season_type=self.season_type_filter(season_type),
        )
        summary = summarize_team_stats(frame)
        passing = self.optional_float(self.cell(summary, "avg_passing_yards"))
        rushing = self.optional_float(self.cell(summary, "avg_rushing_yards"))
        passing_tds = self.optional_float(self.cell(summary, "avg_passing_tds"))
        rushing_tds = self.optional_float(self.cell(summary, "avg_rushing_tds"))
        turnovers = self._turnovers(frame, prefix="")

        return TeamOffenseSummary(
            team=normalized,
            window=self.build_stat_window(
                seasons, season_type, frame.height, last_n_games=last_n_games, weeks=weeks
            ),
            split=TeamSplit.ALL,
            total_passing_yards=self.optional_int(self.cell(summary, "total_passing_yards")),
            total_rushing_yards=self.optional_int(self.cell(summary, "total_rushing_yards")),
            total_passing_tds=self.optional_int(self.cell(summary, "total_passing_tds")),
            total_rushing_tds=self.optional_int(self.cell(summary, "total_rushing_tds")),
            total_turnovers=turnovers,
            avg_passing_yards=passing,
            avg_rushing_yards=rushing,
            avg_total_yards=self._sum_optional(passing, rushing),
            avg_passing_tds=passing_tds,
            avg_rushing_tds=rushing_tds,
            avg_total_tds=self._sum_optional(passing_tds, rushing_tds),
            avg_turnovers=self._turnovers_per_game(frame, prefix=""),
            avg_passing_epa=self.column_mean(frame, "passing_epa"),
            avg_rushing_epa=self.column_mean(frame, "rushing_epa"),
        )

    def defense(
        self,
        team: str,
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = 4,
        weeks: int | Iterable[int] | None = None,
        season_type: SeasonType = SeasonType.ALL,
    ) -> TeamDefenseSummary:
        """What the team's defence allowed, derived from opponent rows."""
        normalized = normalize_team(team)
        frame = get_team_defense(
            normalized,
            loader=self._loader,
            seasons=seasons,
            last_n_games=last_n_games,
            weeks=weeks,
            season_type=self.season_type_filter(season_type),
        )
        summary = summarize_team_stats(frame, prefix="allowed_")
        passing = self.optional_float(self.cell(summary, "avg_allowed_passing_yards"))
        rushing = self.optional_float(self.cell(summary, "avg_allowed_rushing_yards"))
        passing_tds = self.optional_float(self.cell(summary, "avg_allowed_passing_tds"))
        rushing_tds = self.optional_float(self.cell(summary, "avg_allowed_rushing_tds"))

        return TeamDefenseSummary(
            team=normalized,
            window=self.build_stat_window(
                seasons, season_type, frame.height, last_n_games=last_n_games, weeks=weeks
            ),
            split=TeamSplit.ALL,
            total_allowed_passing_yards=self.optional_int(
                self.cell(summary, "total_allowed_passing_yards")
            ),
            total_allowed_rushing_yards=self.optional_int(
                self.cell(summary, "total_allowed_rushing_yards")
            ),
            total_allowed_passing_tds=self.optional_int(
                self.cell(summary, "total_allowed_passing_tds")
            ),
            total_allowed_rushing_tds=self.optional_int(
                self.cell(summary, "total_allowed_rushing_tds")
            ),
            avg_allowed_passing_yards=passing,
            avg_allowed_rushing_yards=rushing,
            avg_allowed_total_yards=self._sum_optional(passing, rushing),
            avg_allowed_passing_tds=self.optional_float(
                self.cell(summary, "avg_allowed_passing_tds")
            ),
            avg_allowed_rushing_tds=self.optional_float(
                self.cell(summary, "avg_allowed_rushing_tds")
            ),
            avg_allowed_total_tds=self._sum_optional(passing_tds, rushing_tds),
            avg_allowed_passing_epa=self.column_mean(frame, "allowed_passing_epa"),
            avg_allowed_rushing_epa=self.column_mean(frame, "allowed_rushing_epa"),
        )

    def _turnovers(self, frame: pl.DataFrame, *, prefix: str) -> int | None:
        """Interceptions thrown plus fumbles lost."""
        parts = [
            self.column_sum(frame, f"{prefix}passing_interceptions"),
            self.column_sum(frame, f"{prefix}rushing_fumbles_lost"),
            self.column_sum(frame, f"{prefix}receiving_fumbles_lost"),
        ]
        present = [value for value in parts if value is not None]
        return int(sum(present)) if present else None

    def _turnovers_per_game(self, frame: pl.DataFrame, *, prefix: str) -> float | None:
        """Summing per-column means avoids dividing by a game count that may be zero."""
        parts = [
            self.column_mean(frame, f"{prefix}passing_interceptions"),
            self.column_mean(frame, f"{prefix}rushing_fumbles_lost"),
            self.column_mean(frame, f"{prefix}receiving_fumbles_lost"),
        ]
        return self._sum_optional(*parts)

    @staticmethod
    def _sum_optional(*values: float | None) -> float | None:
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    # --- games and form -------------------------------------------------------

    def game_log(
        self,
        team: str,
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = 4,
        season_type: SeasonType = SeasonType.ALL,
        completed_only: bool = True,
    ) -> TeamGameLog:
        """The team's games, newest first, from its own perspective."""
        normalized = normalize_team(team)
        frame = self._games_frame(
            normalized,
            seasons=seasons,
            last_n_games=last_n_games,
            season_type=season_type,
            completed_only=completed_only,
        )
        return TeamGameLog(
            team=normalized,
            window=self.build_stat_window(
                seasons, season_type, frame.height, last_n_games=last_n_games
            ),
            games=[self._to_game_result(row) for row in frame.iter_rows(named=True)],
        )

    def recent_form(
        self,
        team: str,
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = 4,
        season_type: SeasonType = SeasonType.ALL,
    ) -> TeamForm:
        """Won-lost record and scoring over the most recent games."""
        normalized = normalize_team(team)
        frame = self._games_frame(
            normalized, seasons=seasons, last_n_games=last_n_games, season_type=season_type
        )
        return self._to_form(
            normalized,
            frame,
            split=TeamSplit.ALL,
            seasons=seasons,
            season_type=season_type,
            last_n_games=last_n_games,
        )

    def home_away_splits(
        self,
        team: str,
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = None,
        season_type: SeasonType = SeasonType.ALL,
    ) -> TeamHomeAwaySplits:
        """Form at home versus on the road.

        ``last_n_games`` is applied *after* splitting, so it means "last N home
        games" and "last N away games" rather than N games split unevenly.
        """
        normalized = normalize_team(team)
        all_games = self._games_frame(
            normalized, seasons=seasons, last_n_games=None, season_type=season_type
        )
        home = all_games.filter(pl.col("is_home"))
        away = all_games.filter(~pl.col("is_home"))
        if last_n_games is not None:
            home = home.head(last_n_games)
            away = away.head(last_n_games)

        return TeamHomeAwaySplits(
            team=normalized,
            window=self.build_season_window(seasons, season_type),
            home=self._to_form(
                normalized,
                home,
                split=TeamSplit.HOME,
                seasons=seasons,
                season_type=season_type,
                last_n_games=last_n_games,
            ),
            away=self._to_form(
                normalized,
                away,
                split=TeamSplit.AWAY,
                seasons=seasons,
                season_type=season_type,
                last_n_games=last_n_games,
            ),
        )

    def _games_frame(
        self,
        team: str,
        *,
        seasons: SeasonInput,
        last_n_games: int | None,
        season_type: SeasonType,
        completed_only: bool = True,
    ) -> pl.DataFrame:
        return get_recent_games(
            team,
            loader=self._loader,
            seasons=seasons,
            last_n_games=last_n_games,
            season_type=self.season_type_filter(season_type),
            completed_only=completed_only,
        )

    def _to_form(
        self,
        team: str,
        frame: pl.DataFrame,
        *,
        split: TeamSplit,
        seasons: SeasonInput,
        season_type: SeasonType,
        last_n_games: int | None,
    ) -> TeamForm:
        summary = summarize_recent_games(frame)
        outcomes = (
            [str(value) for value in frame.get_column("outcome").to_list() if value is not None]
            if "outcome" in frame.columns
            else []
        )
        return TeamForm(
            team=team,
            window=self.build_stat_window(
                seasons, season_type, frame.height, last_n_games=last_n_games
            ),
            split=split,
            wins=self.optional_int(self.cell(summary, "wins")) or 0,
            losses=self.optional_int(self.cell(summary, "losses")) or 0,
            ties=self.optional_int(self.cell(summary, "ties")) or 0,
            avg_points_for=self.optional_float(self.cell(summary, "avg_points_for")),
            avg_points_against=self.optional_float(self.cell(summary, "avg_points_against")),
            avg_point_differential=self.optional_float(
                self.cell(summary, "avg_point_differential")
            ),
            results=outcomes,
        )

    def _to_game_result(self, row: dict[str, object]) -> TeamGameResult:
        return TeamGameResult(
            game_id=self.row_str(row, "game_id") or "",
            season=self.row_int(row, "season") or 0,
            week=self.row_int(row, "week") or 0,
            game_type=self.row_str(row, "game_type"),
            gameday=self.row_str(row, "gameday"),
            opponent=self.row_str(row, "opponent") or "",
            is_home=bool(row.get("is_home")),
            points_for=self.row_int(row, "points_for"),
            points_against=self.row_int(row, "points_against"),
            point_differential=self.row_int(row, "point_differential"),
            outcome=self.row_str(row, "outcome"),
        )

    # --- injuries -------------------------------------------------------------

    def injury_report(
        self,
        team: str,
        *,
        seasons: SeasonInput = None,
        week: int | None = None,
    ) -> TeamInjuryReport:
        """Every reported injury for the team in the window."""
        normalized = normalize_team(team)
        frame = get_team_injuries(normalized, loader=self._loader, seasons=seasons, week=week)
        entries = [
            TeamInjuryEntry(
                player=PlayerRef(
                    player_id=self.row_str(row, "gsis_id") or "",
                    name=self.row_str(row, "full_name") or "",
                    position=self.row_str(row, "position"),
                    team=self.row_str(row, "team"),
                ),
                status=self._players.to_injury_status(row),
            )
            for row in frame.iter_rows(named=True)
        ]
        distinct_players = {entry.player.player_id for entry in entries}
        return TeamInjuryReport(
            team=normalized,
            window=self.build_season_window(seasons, SeasonType.ALL, week),
            entries=entries,
            player_count=len(distinct_players),
        )
