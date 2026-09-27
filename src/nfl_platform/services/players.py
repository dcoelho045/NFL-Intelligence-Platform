"""Player analytics service.

Wraps the Milestone 1 player functions and converts their frames into DTOs. The
arithmetic stays in ``nfl_platform.analytics.players``; this class composes,
validates, and translates.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl

from nfl_platform.analytics.injuries import get_injury_information
from nfl_platform.analytics.players import (
    get_player_recent_stats,
    get_player_usage,
    summarize_player_stats,
)
from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.domain import SeasonInput
from nfl_platform.schemas.common import PlayerRef, SeasonType
from nfl_platform.schemas.player import (
    PlayerComparison,
    PlayerGameLog,
    PlayerGameStats,
    PlayerInjuryReport,
    PlayerInjuryStatus,
    PlayerSeasonStats,
    PlayerStatAverages,
    PlayerStatLine,
    PlayerUsage,
    PlayerUsageGame,
)
from nfl_platform.services.base import AnalyticsService
from nfl_platform.services.directory import PlayerDirectoryService

# Box-score columns surfaced on PlayerStatLine, split by how they aggregate.
_COUNTING_STATS: tuple[str, ...] = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
)
_RATE_STATS: tuple[str, ...] = ("passing_epa", "rushing_epa", "receiving_epa")
_FANTASY_STATS: tuple[str, ...] = ("fantasy_points", "fantasy_points_ppr")


class PlayerAnalyticsService(AnalyticsService):
    """Recent form, season production, usage, game logs, and comparisons."""

    def __init__(
        self, loader: NflDataLoader, directory: PlayerDirectoryService | None = None
    ) -> None:
        super().__init__(loader)
        self._directory = directory or PlayerDirectoryService(loader)

    @property
    def directory(self) -> PlayerDirectoryService:
        return self._directory

    def _player_ref(self, player_id: str, frame: pl.DataFrame) -> PlayerRef:
        """Build a PlayerRef, preferring the stats frame's own name and team."""
        name = self.cell(frame, "player_display_name")
        position = self.cell(frame, "position")
        team = self.cell(frame, "team")
        if name is None:
            name, position, team = self._directory.reference(player_id)
        return PlayerRef(
            player_id=player_id,
            name=self.optional_str(name) or player_id,
            position=self.optional_str(position),
            team=self.optional_str(team),
        )

    # --- game log -------------------------------------------------------------

    def game_log(
        self,
        player_id: str,
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = 4,
        weeks: int | Iterable[int] | None = None,
        season_type: SeasonType = SeasonType.ALL,
    ) -> PlayerGameLog:
        """Game-by-game production, newest first."""
        resolved_id = self._directory.resolve(player_id)
        frame = get_player_recent_stats(
            resolved_id,
            loader=self._loader,
            seasons=seasons,
            last_n_games=last_n_games,
            weeks=weeks,
            season_type=self.season_type_filter(season_type),
        )
        return PlayerGameLog(
            player=self._player_ref(resolved_id, frame),
            window=self.build_stat_window(
                seasons, season_type, frame.height, last_n_games=last_n_games, weeks=weeks
            ),
            games=[self._to_game_stats(row) for row in frame.iter_rows(named=True)],
        )

    def _to_game_stats(self, row: dict[str, object]) -> PlayerGameStats:
        return PlayerGameStats(
            season=self.row_int(row, "season") or 0,
            week=self.row_int(row, "week") or 0,
            season_type=self.row_str(row, "season_type"),
            game_id=self.row_str(row, "game_id"),
            team=self.row_str(row, "team"),
            opponent=self.row_str(row, "opponent_team"),
            stats=self._to_stat_line(row),
        )

    def _to_stat_line(self, row: dict[str, object]) -> PlayerStatLine:
        values: dict[str, object] = {}
        for column in _COUNTING_STATS:
            values[column] = self.row_int(row, column)
        for column in (*_RATE_STATS, *_FANTASY_STATS):
            values[column] = self.row_float(row, column)
        return PlayerStatLine(**values)  # type: ignore[arg-type]

    # --- aggregated production ------------------------------------------------

    def recent_performance(
        self,
        player_id: str,
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = 4,
        season_type: SeasonType = SeasonType.ALL,
    ) -> PlayerSeasonStats:
        """Totals and per-game averages over the most recent games."""
        return self._aggregate(
            player_id, seasons=seasons, last_n_games=last_n_games, season_type=season_type
        )

    def season_performance(
        self,
        player_id: str,
        *,
        seasons: SeasonInput = None,
        season_type: SeasonType = SeasonType.REGULAR,
    ) -> PlayerSeasonStats:
        """Totals and per-game averages over a full season.

        ``last_n_games=None`` is what distinguishes this from recent form.
        """
        return self._aggregate(
            player_id, seasons=seasons, last_n_games=None, season_type=season_type
        )

    def _aggregate(
        self,
        player_id: str,
        *,
        seasons: SeasonInput,
        last_n_games: int | None,
        season_type: SeasonType,
    ) -> PlayerSeasonStats:
        resolved_id = self._directory.resolve(player_id)
        frame = get_player_recent_stats(
            resolved_id,
            loader=self._loader,
            seasons=seasons,
            last_n_games=last_n_games,
            season_type=self.season_type_filter(season_type),
        )
        summary = summarize_player_stats(frame)

        totals: dict[str, object] = {}
        per_game: dict[str, object] = {}
        for column in _COUNTING_STATS:
            totals[column] = self.optional_int(self.cell(summary, f"total_{column}"))
            per_game[column] = self.optional_float(self.cell(summary, f"avg_{column}"))
        for column in _FANTASY_STATS:
            totals[column] = self.optional_float(self.cell(summary, f"total_{column}"))
            per_game[column] = self.optional_float(self.cell(summary, f"avg_{column}"))
        for column in _RATE_STATS:
            # EPA is a rate; summing it across games is not meaningful.
            totals[column] = None
            per_game[column] = self.column_mean(frame, column)

        return PlayerSeasonStats(
            player=self._player_ref(resolved_id, frame),
            window=self.build_stat_window(
                seasons, season_type, frame.height, last_n_games=last_n_games
            ),
            totals=PlayerStatLine(**totals),  # type: ignore[arg-type]
            per_game=PlayerStatAverages(**per_game),  # type: ignore[arg-type]
        )

    # --- usage ----------------------------------------------------------------

    def usage(
        self,
        player_id: str,
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = 4,
        season_type: SeasonType = SeasonType.ALL,
        include_snap_share: bool = True,
    ) -> PlayerUsage:
        """Opportunity metrics: snaps, targets, carries, and shares."""
        resolved_id = self._directory.resolve(player_id)
        frame = get_player_usage(
            resolved_id,
            loader=self._loader,
            seasons=seasons,
            last_n_games=last_n_games,
            season_type=self.season_type_filter(season_type),
            include_snap_share=include_snap_share,
        )
        snap_available = "offense_snap_pct" in frame.columns and (
            frame.get_column("offense_snap_pct").drop_nulls().len() > 0
            if not frame.is_empty()
            else False
        )
        return PlayerUsage(
            player=self._player_ref(resolved_id, frame),
            window=self.build_stat_window(
                seasons, season_type, frame.height, last_n_games=last_n_games
            ),
            games=[self._to_usage_game(row) for row in frame.iter_rows(named=True)],
            avg_offense_snap_pct=self.column_mean(frame, "offense_snap_pct"),
            avg_target_share=self.column_mean(frame, "target_share"),
            avg_wopr=self.column_mean(frame, "wopr"),
            total_targets=self.optional_int(self.column_sum(frame, "targets")),
            total_carries=self.optional_int(self.column_sum(frame, "carries")),
            snap_data_available=snap_available,
        )

    def _to_usage_game(self, row: dict[str, object]) -> PlayerUsageGame:
        return PlayerUsageGame(
            season=self.row_int(row, "season") or 0,
            week=self.row_int(row, "week") or 0,
            game_id=self.row_str(row, "game_id"),
            team=self.row_str(row, "team"),
            opponent=self.row_str(row, "opponent_team"),
            attempts=self.row_int(row, "attempts"),
            carries=self.row_int(row, "carries"),
            targets=self.row_int(row, "targets"),
            receptions=self.row_int(row, "receptions"),
            target_share=self.row_float(row, "target_share"),
            air_yards_share=self.row_float(row, "air_yards_share"),
            wopr=self.row_float(row, "wopr"),
            offense_snaps=self.row_int(row, "offense_snaps"),
            offense_snap_pct=self.row_float(row, "offense_snap_pct"),
        )

    # --- injuries -------------------------------------------------------------

    def injury_report(
        self,
        player_id: str,
        *,
        seasons: SeasonInput = None,
        weeks: int | Iterable[int] | None = None,
        latest_only: bool = False,
    ) -> PlayerInjuryReport:
        """Injury-report history. An empty result means no injury was reported."""
        resolved_id = self._directory.resolve(player_id)
        frame = get_injury_information(
            resolved_id,
            loader=self._loader,
            seasons=seasons,
            weeks=weeks,
            latest_only=latest_only,
        )
        name, position, team = self._directory.reference(resolved_id)
        return PlayerInjuryReport(
            player=PlayerRef(player_id=resolved_id, name=name, position=position, team=team),
            window=self.build_season_window(seasons, SeasonType.ALL, weeks),
            entries=[self.to_injury_status(row) for row in frame.iter_rows(named=True)],
            has_injury_data=not frame.is_empty(),
        )

    def to_injury_status(self, row: dict[str, object]) -> PlayerInjuryStatus:
        """Shared with the team service, which reports the same rows team-wide."""
        return PlayerInjuryStatus(
            season=self.row_int(row, "season") or 0,
            week=self.row_int(row, "week") or 0,
            team=self.row_str(row, "team"),
            report_status=self.row_str(row, "report_status"),
            report_primary_injury=self.row_str(row, "report_primary_injury"),
            report_secondary_injury=self.row_str(row, "report_secondary_injury"),
            practice_status=self.row_str(row, "practice_status"),
            practice_primary_injury=self.row_str(row, "practice_primary_injury"),
            date_modified=self.row_str(row, "date_modified"),
        )

    # --- comparison -----------------------------------------------------------

    def compare(
        self,
        player_ids: Sequence[str],
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = 4,
        season_type: SeasonType = SeasonType.ALL,
    ) -> PlayerComparison:
        """Compare players over an identical window.

        The shared window is the point: comparing players across different game
        counts is the most common way a comparison misleads.
        """
        if len(player_ids) < 2:
            raise ValueError("Comparing players requires at least two player identifiers.")

        # Resolved up front so every unidentifiable name is reported together.
        resolved_ids = self.directory.resolve_all(player_ids)

        results = [
            self.recent_performance(
                player_id, seasons=seasons, last_n_games=last_n_games, season_type=season_type
            )
            for player_id in resolved_ids
        ]

        game_counts = {result.window.games for result in results}
        note = None
        if len(game_counts) > 1:
            note = (
                "Players have different game counts in this window "
                f"({sorted(game_counts)}); totals are not directly comparable."
            )

        return PlayerComparison(
            window=self.build_stat_window(
                seasons,
                season_type,
                max(game_counts) if game_counts else 0,
                last_n_games=last_n_games,
            ),
            players=results,
            note=note,
        )
