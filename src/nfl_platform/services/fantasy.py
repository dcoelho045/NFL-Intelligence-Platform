"""Fantasy analytics service.

Re-frames the same deterministic data as the player service around fantasy
scoring, opportunity, and matchup context. Nothing here projects or predicts.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

import polars as pl

from nfl_platform.analytics.players import get_player_recent_stats, get_player_usage
from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.domain import SeasonInput, normalize_team
from nfl_platform.schemas.common import PlayerRef, SeasonType
from nfl_platform.schemas.fantasy import (
    FantasyComparison,
    FantasyMatchupContext,
    FantasyProduction,
    FantasyScoring,
    FantasyUsageTrend,
    FantasyWeek,
)
from nfl_platform.services.base import AnalyticsService
from nfl_platform.services.directory import PlayerDirectoryService
from nfl_platform.services.teams import TeamAnalyticsService

_SCORING_COLUMN: dict[FantasyScoring, str] = {
    FantasyScoring.STANDARD: "fantasy_points",
    FantasyScoring.PPR: "fantasy_points_ppr",
}

# Below this many games a trend comparison is noise rather than signal.
_MIN_GAMES_FOR_TREND = 4

# Relative change required before a trend is called rising or falling.
_TREND_THRESHOLD = 0.10


class FantasyAnalyticsService(AnalyticsService):
    """Fantasy production, usage trends, matchup context, and comparisons."""

    def __init__(
        self,
        loader: NflDataLoader,
        directory: PlayerDirectoryService | None = None,
        teams: TeamAnalyticsService | None = None,
    ) -> None:
        super().__init__(loader)
        self._directory = directory or PlayerDirectoryService(loader)
        self._teams = teams or TeamAnalyticsService(loader)

    def _reference(self, player_id: str) -> PlayerRef:
        name, position, team = self._directory.reference(player_id)
        return PlayerRef(player_id=player_id, name=name, position=position, team=team)

    # --- production -----------------------------------------------------------

    def production(
        self,
        player_id: str,
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = 4,
        scoring: FantasyScoring = FantasyScoring.PPR,
        season_type: SeasonType = SeasonType.ALL,
    ) -> FantasyProduction:
        """Recent fantasy scoring, with a consistency measure."""
        resolved_id = self._directory.resolve(player_id)
        frame = get_player_usage(
            resolved_id,
            loader=self._loader,
            seasons=seasons,
            last_n_games=last_n_games,
            season_type=self.season_type_filter(season_type),
        )
        stats = get_player_recent_stats(
            resolved_id,
            loader=self._loader,
            seasons=seasons,
            last_n_games=last_n_games,
            season_type=self.season_type_filter(season_type),
        )
        weeks = self._to_weeks(stats, frame, scoring)
        points = [week.points for week in weeks if week.points is not None]

        return FantasyProduction(
            player=self._reference(resolved_id),
            window=self.build_stat_window(
                seasons, season_type, stats.height, last_n_games=last_n_games
            ),
            scoring=scoring,
            weeks=weeks,
            total_points=sum(points) if points else None,
            avg_points=sum(points) / len(points) if points else None,
            best_week=max(points) if points else None,
            worst_week=min(points) if points else None,
            std_dev=statistics.pstdev(points) if len(points) > 1 else None,
        )

    def _to_weeks(
        self, stats: pl.DataFrame, usage: pl.DataFrame, scoring: FantasyScoring
    ) -> list[FantasyWeek]:
        """Join production to opportunity on (season, week)."""
        column = _SCORING_COLUMN[scoring]
        usage_by_week: dict[tuple[int, int], dict[str, object]] = {}
        for row in usage.iter_rows(named=True):
            season = self.row_int(row, "season")
            week = self.row_int(row, "week")
            if season is not None and week is not None:
                usage_by_week[(season, week)] = row

        weeks: list[FantasyWeek] = []
        for row in stats.iter_rows(named=True):
            season = self.row_int(row, "season") or 0
            week_number = self.row_int(row, "week") or 0
            usage_row = usage_by_week.get((season, week_number), {})
            weeks.append(
                FantasyWeek(
                    season=season,
                    week=week_number,
                    opponent=self.row_str(row, "opponent_team"),
                    points=self.row_float(row, column),
                    targets=self.row_int(row, "targets"),
                    carries=self.row_int(row, "carries"),
                    receptions=self.row_int(row, "receptions"),
                    offense_snap_pct=self.row_float(usage_row, "offense_snap_pct"),
                )
            )
        return weeks

    # --- usage trend ----------------------------------------------------------

    def usage_trend(
        self,
        player_id: str,
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = 8,
        season_type: SeasonType = SeasonType.ALL,
    ) -> FantasyUsageTrend:
        """Compare the earlier and more recent halves of the window.

        This describes what already happened. It is not a forecast, and the
        direction is left null when the sample is too small to mean anything.
        """
        resolved_id = self._directory.resolve(player_id)
        production = self.production(
            player_id,
            seasons=seasons,
            last_n_games=last_n_games,
            season_type=season_type,
        )
        usage = get_player_usage(
            resolved_id,
            loader=self._loader,
            seasons=seasons,
            last_n_games=last_n_games,
            season_type=self.season_type_filter(season_type),
        )

        # Frames are newest-first, so the first half is the recent half.
        midpoint = usage.height // 2
        recent = usage.head(midpoint)
        early = usage.tail(usage.height - midpoint)

        target_recent = self.column_mean(recent, "target_share")
        target_early = self.column_mean(early, "target_share")
        snap_recent = self.column_mean(recent, "offense_snap_pct")
        snap_early = self.column_mean(early, "offense_snap_pct")
        sufficient = usage.height >= _MIN_GAMES_FOR_TREND

        return FantasyUsageTrend(
            player=self._reference(resolved_id),
            window=self.build_stat_window(
                seasons, season_type, usage.height, last_n_games=last_n_games
            ),
            weeks=production.weeks,
            avg_target_share_early=target_early,
            avg_target_share_recent=target_recent,
            avg_snap_pct_early=snap_early,
            avg_snap_pct_recent=snap_recent,
            target_share_direction=(
                self._direction(target_early, target_recent) if sufficient else None
            ),
            snap_share_direction=(self._direction(snap_early, snap_recent) if sufficient else None),
        )

    @staticmethod
    def _direction(early: float | None, recent: float | None) -> str | None:
        """Classify a change as rising, falling, or stable."""
        if early is None or recent is None:
            return None
        if early == 0:
            return "rising" if recent > 0 else "stable"
        change = (recent - early) / abs(early)
        if change > _TREND_THRESHOLD:
            return "rising"
        if change < -_TREND_THRESHOLD:
            return "falling"
        return "stable"

    # --- matchup context ------------------------------------------------------

    def matchup_context(
        self,
        player_id: str,
        *,
        opponent: str,
        seasons: SeasonInput = None,
        last_n_games: int | None = 4,
        season_type: SeasonType = SeasonType.ALL,
    ) -> FantasyMatchupContext:
        """What the opponent's defence has allowed recently."""
        resolved_id = self._directory.resolve(player_id)
        normalized_opponent = normalize_team(opponent)
        defense = self._teams.defense(
            normalized_opponent,
            seasons=seasons,
            last_n_games=last_n_games,
            season_type=season_type,
        )
        note = None
        if defense.window.games == 0:
            note = f"No completed games found for {normalized_opponent} in this window."

        return FantasyMatchupContext(
            player=self._reference(resolved_id),
            opponent=normalized_opponent,
            window=defense.window,
            opponent_defense=defense,
            note=note,
        )

    # --- comparison -----------------------------------------------------------

    def compare(
        self,
        player_ids: Sequence[str],
        *,
        seasons: SeasonInput = None,
        last_n_games: int | None = 4,
        scoring: FantasyScoring = FantasyScoring.PPR,
        season_type: SeasonType = SeasonType.ALL,
    ) -> FantasyComparison:
        """Compare fantasy production across players over one shared window."""
        if len(player_ids) < 2:
            raise ValueError("Comparing players requires at least two player identifiers.")

        results = [
            self.production(
                player_id,
                seasons=seasons,
                last_n_games=last_n_games,
                scoring=scoring,
                season_type=season_type,
            )
            for player_id in player_ids
        ]
        game_counts = {result.window.games for result in results}
        note = None
        if len(game_counts) > 1:
            note = (
                "Players have different game counts in this window "
                f"({sorted(game_counts)}); compare averages rather than totals."
            )

        return FantasyComparison(
            window=self.build_stat_window(
                seasons,
                season_type,
                max(game_counts) if game_counts else 0,
                last_n_games=last_n_games,
            ),
            scoring=scoring,
            players=results,
            note=note,
        )
