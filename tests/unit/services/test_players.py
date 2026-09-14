"""Tests for PlayerAnalyticsService."""

from __future__ import annotations

import pytest

from nfl_platform.exceptions import AmbiguousPlayerError, InvalidWeekError, PlayerNotFoundError
from nfl_platform.schemas.common import SeasonType
from nfl_platform.services import ServiceContainer
from support import ALLEN_ID, MAHOMES_ID, UNKNOWN_PLAYER_ID


class TestGameLog:
    def test_returns_requested_game_count(self, services: ServiceContainer):
        log = services.players.game_log(MAHOMES_ID, seasons=2024, last_n_games=3)
        assert len(log.games) == 3
        assert log.window.games == 3
        assert log.window.requested_last_n_games == 3

    def test_newest_first(self, services: ServiceContainer):
        log = services.players.game_log(MAHOMES_ID, seasons=2024, last_n_games=None)
        weeks = [game.week for game in log.games]
        assert weeks == sorted(weeks, reverse=True)

    def test_player_reference_is_populated(self, services: ServiceContainer):
        log = services.players.game_log(MAHOMES_ID, seasons=2024, last_n_games=1)
        assert log.player.player_id == MAHOMES_ID
        assert log.player.name == "Patrick Mahomes"
        assert log.player.position == "QB"

    def test_accepts_a_player_name(self, services: ServiceContainer):
        log = services.players.game_log("Patrick Mahomes", seasons=2024, last_n_games=1)
        assert log.player.player_id == MAHOMES_ID

    def test_ambiguous_name_raises(self, services: ServiceContainer):
        with pytest.raises(AmbiguousPlayerError):
            services.players.game_log("Mike Williams", seasons=2024)

    def test_unknown_player_raises(self, services: ServiceContainer):
        with pytest.raises(PlayerNotFoundError):
            services.players.game_log(UNKNOWN_PLAYER_ID, seasons=2024)

    def test_invalid_week_raises(self, services: ServiceContainer):
        with pytest.raises(InvalidWeekError):
            services.players.game_log(MAHOMES_ID, seasons=2024, weeks=99)

    def test_week_filter_narrows_results(self, services: ServiceContainer):
        log = services.players.game_log(MAHOMES_ID, seasons=2024, weeks=[1, 2], last_n_games=None)
        assert sorted(game.week for game in log.games) == [1, 2]

    def test_window_records_zero_games_when_empty(self, services: ServiceContainer):
        log = services.players.game_log(MAHOMES_ID, seasons=2024, season_type=SeasonType.POST)
        assert log.games == []
        assert log.window.games == 0


class TestAggregation:
    """Aggregation correctness against values computed straight from the frames."""

    def test_totals_match_the_sum_of_the_game_log(self, services: ServiceContainer):
        log = services.players.game_log(MAHOMES_ID, seasons=2024, last_n_games=4)
        summary = services.players.recent_performance(MAHOMES_ID, seasons=2024, last_n_games=4)
        expected = sum(game.stats.passing_yards for game in log.games if game.stats.passing_yards)
        assert summary.totals.passing_yards == expected

    def test_per_game_is_the_mean_of_the_totals(self, services: ServiceContainer):
        summary = services.players.recent_performance(MAHOMES_ID, seasons=2024, last_n_games=4)
        assert summary.totals.passing_yards is not None
        assert summary.per_game.passing_yards == pytest.approx(
            summary.totals.passing_yards / summary.window.games
        )

    def test_per_game_averages_may_be_fractional(self, services: ServiceContainer):
        """Totals are integers; averages are not. They are separate types for this reason."""
        summary = services.players.recent_performance(MAHOMES_ID, seasons=2024, last_n_games=3)
        assert isinstance(summary.totals.passing_yards, int)
        assert isinstance(summary.per_game.passing_yards, float)

    def test_epa_is_averaged_never_summed(self, services: ServiceContainer):
        """EPA is a rate statistic; a season total would be meaningless."""
        summary = services.players.recent_performance(MAHOMES_ID, seasons=2024, last_n_games=4)
        assert summary.totals.passing_epa is None
        assert summary.per_game.passing_epa is not None

    def test_season_performance_covers_every_game(self, services: ServiceContainer):
        season = services.players.season_performance(
            MAHOMES_ID, seasons=2024, season_type=SeasonType.ALL
        )
        full_log = services.players.game_log(MAHOMES_ID, seasons=2024, last_n_games=None)
        assert season.window.games == len(full_log.games)
        assert season.window.requested_last_n_games is None


class TestUsage:
    def test_reports_opportunity_not_production(self, services: ServiceContainer):
        usage = services.players.usage(MAHOMES_ID, seasons=2024, last_n_games=3)
        assert len(usage.games) == 3
        assert usage.games[0].attempts is not None

    def test_snap_share_is_joined(self, services: ServiceContainer):
        usage = services.players.usage(MAHOMES_ID, seasons=2024, last_n_games=3)
        assert usage.snap_data_available is True
        assert usage.avg_offense_snap_pct is not None
        assert 0.0 <= usage.avg_offense_snap_pct <= 1.0

    def test_snap_share_can_be_disabled(self, services: ServiceContainer):
        usage = services.players.usage(
            MAHOMES_ID, seasons=2024, last_n_games=3, include_snap_share=False
        )
        assert usage.snap_data_available is False
        assert all(game.offense_snap_pct is None for game in usage.games)

    def test_totals_match_the_per_game_values(self, services: ServiceContainer):
        usage = services.players.usage(ALLEN_ID, seasons=2024, last_n_games=4)
        expected = sum(game.carries for game in usage.games if game.carries)
        assert usage.total_carries == expected


class TestInjuryReport:
    def test_player_without_injuries_is_not_an_error(self, services: ServiceContainer):
        report = services.players.injury_report(MAHOMES_ID, seasons=2024)
        assert report.has_injury_data in {True, False}
        assert report.player.player_id == MAHOMES_ID

    def test_entries_are_newest_first(self, services: ServiceContainer):
        injuries = services.loader.load_injuries(2024)
        busiest = str(
            injuries.group_by("gsis_id").len().sort("len", descending=True).item(0, "gsis_id")
        )
        report = services.players.injury_report(busiest, seasons=2024)
        weeks = [entry.week for entry in report.entries]
        assert weeks == sorted(weeks, reverse=True)

    def test_latest_only_returns_one_entry(self, services: ServiceContainer):
        injuries = services.loader.load_injuries(2024)
        busiest = str(
            injuries.group_by("gsis_id").len().sort("len", descending=True).item(0, "gsis_id")
        )
        report = services.players.injury_report(busiest, seasons=2024, latest_only=True)
        assert len(report.entries) == 1


class TestComparison:
    def test_compares_two_players_over_one_window(self, services: ServiceContainer):
        comparison = services.players.compare([MAHOMES_ID, ALLEN_ID], seasons=2024, last_n_games=4)
        assert len(comparison.players) == 2
        assert {player.player.player_id for player in comparison.players} == {
            MAHOMES_ID,
            ALLEN_ID,
        }

    def test_requires_at_least_two_players(self, services: ServiceContainer):
        with pytest.raises(ValueError):
            services.players.compare([MAHOMES_ID], seasons=2024)

    def test_names_are_accepted(self, services: ServiceContainer):
        """Names and ids are interchangeable, and may be mixed."""
        comparison = services.players.compare(
            ["Patrick Mahomes", ALLEN_ID], seasons=2024, last_n_games=2
        )
        assert len(comparison.players) == 2

    def test_ambiguous_name_in_a_comparison_raises(self, services: ServiceContainer):
        with pytest.raises(AmbiguousPlayerError):
            services.players.compare(["Mike Williams", MAHOMES_ID], seasons=2024, last_n_games=2)

    def test_note_warns_when_game_counts_differ(self, services: ServiceContainer):
        """Unequal samples are the most common way a comparison misleads."""
        comparison = services.players.compare([MAHOMES_ID, ALLEN_ID], seasons=2024, last_n_games=6)
        counts = {player.window.games for player in comparison.players}
        if len(counts) > 1:
            assert comparison.note is not None
            assert "different game counts" in comparison.note
        else:
            assert comparison.note is None
