"""Tests for TeamAnalyticsService."""

from __future__ import annotations

import pytest

from nfl_platform.exceptions import TeamNotFoundError
from nfl_platform.schemas.team import TeamDefenseSummary, TeamOffenseSummary, TeamSplit
from nfl_platform.services import ServiceContainer


class TestTeamOffenseAndDefense:
    def test_offense_reports_the_team_own_production(self, services: ServiceContainer):
        offense = services.teams.offense("KC", seasons=2024, last_n_games=4)
        assert offense.team == "KC"
        assert offense.window.games == 4
        assert offense.avg_passing_yards is not None

    def test_defense_reports_what_was_allowed(self, services: ServiceContainer):
        defense = services.teams.defense("KC", seasons=2024, last_n_games=4)
        assert defense.team == "KC"
        assert defense.avg_allowed_passing_yards is not None

    def test_offense_and_defense_are_different_numbers(self, services: ServiceContainer):
        """Guards the perspective inversion that would silently produce wrong answers."""
        offense = services.teams.offense("KC", seasons=2024, last_n_games=4)
        defense = services.teams.defense("KC", seasons=2024, last_n_games=4)
        assert offense.avg_passing_yards != defense.avg_allowed_passing_yards

    def test_defense_equals_the_opponents_offense(self, services: ServiceContainer):
        """What KC allowed in week 1 must equal what BAL produced in week 1."""
        kc_allowed = services.teams.defense("KC", seasons=2024, weeks=1, last_n_games=None)
        bal_produced = services.teams.offense("BAL", seasons=2024, weeks=1, last_n_games=None)
        assert kc_allowed.total_allowed_passing_yards == bal_produced.total_passing_yards
        assert kc_allowed.total_allowed_rushing_yards == bal_produced.total_rushing_yards

    def test_total_yards_sums_passing_and_rushing(self, services: ServiceContainer):
        offense = services.teams.offense("KC", seasons=2024, last_n_games=4)
        assert offense.avg_total_yards == pytest.approx(
            (offense.avg_passing_yards or 0) + (offense.avg_rushing_yards or 0)
        )

    def test_total_touchdowns_are_supplied_not_left_to_the_caller(self, services: ServiceContainer):
        """Mirrors avg_total_yards: a combination a caller would otherwise compute."""
        offense = services.teams.offense("KC", seasons=2024, last_n_games=4)
        assert offense.avg_total_tds == pytest.approx(
            (offense.avg_passing_tds or 0) + (offense.avg_rushing_tds or 0)
        )
        defense = services.teams.defense("KC", seasons=2024, last_n_games=4)
        assert defense.avg_allowed_total_tds == pytest.approx(
            (defense.avg_allowed_passing_tds or 0) + (defense.avg_allowed_rushing_tds or 0)
        )

    def test_defence_reports_touchdowns_allowed_per_game(self, services: ServiceContainer):
        """Every total needs its average, or the caller divides and calls it data."""
        defense = services.teams.defense("KC", seasons=2024, last_n_games=4)
        assert defense.avg_allowed_passing_tds == pytest.approx(
            defense.total_allowed_passing_tds / defense.window.games
        )
        assert defense.avg_allowed_rushing_tds == pytest.approx(
            defense.total_allowed_rushing_tds / defense.window.games
        )

    def test_offence_reports_touchdowns_and_turnovers_per_game(self, services: ServiceContainer):
        offense = services.teams.offense("KC", seasons=2024, last_n_games=4)
        assert offense.avg_passing_tds == pytest.approx(
            offense.total_passing_tds / offense.window.games
        )
        assert offense.avg_rushing_tds == pytest.approx(
            offense.total_rushing_tds / offense.window.games
        )
        assert offense.avg_turnovers == pytest.approx(
            offense.total_turnovers / offense.window.games
        )

    @pytest.mark.parametrize(
        ("dto", "total_prefix", "avg_prefix"),
        [
            (TeamOffenseSummary, "total_", "avg_"),
            (TeamDefenseSummary, "total_allowed_", "avg_allowed_"),
        ],
        ids=["offense", "defense"],
    )
    def test_every_total_has_a_per_game_counterpart(
        self, dto: type, total_prefix: str, avg_prefix: str
    ):
        """Stops the gap reopening when a new total is added later."""
        totals = {
            field.removeprefix(total_prefix)
            for field in dto.model_fields
            if field.startswith(total_prefix)
        }
        averages = {
            field.removeprefix(avg_prefix)
            for field in dto.model_fields
            if field.startswith(avg_prefix)
        }
        assert not totals - averages

    def test_unknown_team_raises(self, services: ServiceContainer):
        with pytest.raises(TeamNotFoundError):
            services.teams.offense("ZZZ", seasons=2024)

    def test_team_abbreviation_is_normalised(self, services: ServiceContainer):
        assert services.teams.offense(" kc ", seasons=2024).team == "KC"


class TestTeamFormAndSplits:
    def test_recent_form_counts_results(self, services: ServiceContainer):
        form = services.teams.recent_form("KC", seasons=2024, last_n_games=5)
        assert form.wins + form.losses + form.ties == form.window.games
        assert len(form.results) == form.window.games

    def test_results_are_newest_first(self, services: ServiceContainer):
        log = services.teams.game_log("KC", seasons=2024, last_n_games=None)
        weeks = [game.week for game in log.games]
        assert weeks == sorted(weeks, reverse=True)

    def test_game_log_is_from_the_team_perspective(self, services: ServiceContainer):
        """Week 1 2024 was BAL at KC, 27-20."""
        log = services.teams.game_log("KC", seasons=2024, last_n_games=None)
        week_one = next(game for game in log.games if game.week == 1)
        assert week_one.opponent == "BAL"
        assert week_one.is_home is True
        assert week_one.points_for == 27
        assert week_one.points_against == 20
        assert week_one.outcome == "W"
        assert week_one.point_differential == 7

    def test_home_and_away_split_partitions_the_games(self, services: ServiceContainer):
        splits = services.teams.home_away_splits("KC", seasons=2024)
        total = services.teams.recent_form("KC", seasons=2024, last_n_games=None)
        assert splits.home.window.games + splits.away.window.games == total.window.games

    def test_splits_are_labelled(self, services: ServiceContainer):
        splits = services.teams.home_away_splits("KC", seasons=2024)
        assert splits.home.split is TeamSplit.HOME
        assert splits.away.split is TeamSplit.AWAY

    def test_home_split_contains_only_home_games(self, services: ServiceContainer):
        log = services.teams.game_log("KC", seasons=2024, last_n_games=None)
        home_wins = sum(1 for game in log.games if game.is_home and game.outcome == "W")
        assert services.teams.home_away_splits("KC", seasons=2024).home.wins == home_wins

    def test_last_n_games_applies_per_split(self, services: ServiceContainer):
        """'Last 2 home games' should mean 2 home games, not 2 games split unevenly."""
        splits = services.teams.home_away_splits("KC", seasons=2024, last_n_games=2)
        assert splits.home.window.games <= 2
        assert splits.away.window.games <= 2


class TestTeamInjuries:
    def test_entries_carry_player_identity(self, services: ServiceContainer):
        report = services.teams.injury_report("KC", seasons=2024)
        assert report.team == "KC"
        if report.entries:
            assert report.entries[0].player.player_id
            assert report.entries[0].player.name

    def test_player_count_is_distinct_not_row_count(self, services: ServiceContainer):
        report = services.teams.injury_report("KC", seasons=2024)
        assert report.player_count <= len(report.entries)

    def test_week_filter(self, services: ServiceContainer):
        report = services.teams.injury_report("KC", seasons=2024, week=2)
        assert all(entry.status.week == 2 for entry in report.entries)
