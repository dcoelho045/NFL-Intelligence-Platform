"""Tests for FantasyAnalyticsService."""

from __future__ import annotations

import pytest

from nfl_platform.schemas.common import SeasonType
from nfl_platform.schemas.fantasy import FantasyScoring
from nfl_platform.services import ServiceContainer
from support import ALLEN_ID, MAHOMES_ID


class TestProduction:
    def test_summarises_scoring(self, services: ServiceContainer):
        production = services.fantasy.production(MAHOMES_ID, seasons=2024, last_n_games=4)
        assert production.scoring is FantasyScoring.PPR
        assert len(production.weeks) == 4
        assert production.avg_points is not None

    def test_total_matches_the_sum_of_weeks(self, services: ServiceContainer):
        production = services.fantasy.production(MAHOMES_ID, seasons=2024, last_n_games=4)
        expected = sum(week.points for week in production.weeks if week.points is not None)
        assert production.total_points == pytest.approx(expected)

    def test_best_and_worst_bound_the_weeks(self, services: ServiceContainer):
        production = services.fantasy.production(MAHOMES_ID, seasons=2024, last_n_games=5)
        points = [week.points for week in production.weeks if week.points is not None]
        assert production.best_week == max(points)
        assert production.worst_week == min(points)

    def test_scoring_choice_changes_the_column(self, services: ServiceContainer):
        ppr = services.fantasy.production(
            MAHOMES_ID, seasons=2024, last_n_games=4, scoring=FantasyScoring.PPR
        )
        standard = services.fantasy.production(
            MAHOMES_ID, seasons=2024, last_n_games=4, scoring=FantasyScoring.STANDARD
        )
        assert ppr.scoring is FantasyScoring.PPR
        assert standard.scoring is FantasyScoring.STANDARD

    def test_postseason_window_is_empty_in_the_fixture(self, services: ServiceContainer):
        production = services.fantasy.production(
            MAHOMES_ID, seasons=2024, season_type=SeasonType.POST
        )
        assert production.weeks == []
        assert production.avg_points is None


class TestUsageTrend:
    def test_reports_a_direction(self, services: ServiceContainer):
        trend = services.fantasy.usage_trend(MAHOMES_ID, seasons=2024, last_n_games=6)
        assert trend.snap_share_direction in {"rising", "falling", "stable", None}

    def test_trend_is_suppressed_on_a_tiny_sample(self, services: ServiceContainer):
        """Below four games a trend is noise, so no direction is claimed."""
        trend = services.fantasy.usage_trend(MAHOMES_ID, seasons=2024, last_n_games=2)
        assert trend.target_share_direction is None
        assert trend.snap_share_direction is None


class TestMatchupContext:
    def test_returns_opponent_defence(self, services: ServiceContainer):
        context = services.fantasy.matchup_context(
            MAHOMES_ID, opponent="BUF", seasons=2024, last_n_games=4
        )
        assert context.opponent == "BUF"
        assert context.opponent_defense.team == "BUF"


class TestComparison:
    def test_requires_two_players(self, services: ServiceContainer):
        with pytest.raises(ValueError):
            services.fantasy.compare([MAHOMES_ID], seasons=2024)

    def test_shares_one_window(self, services: ServiceContainer):
        comparison = services.fantasy.compare([MAHOMES_ID, ALLEN_ID], seasons=2024, last_n_games=4)
        assert len(comparison.players) == 2
        assert comparison.window.requested_last_n_games == 4
