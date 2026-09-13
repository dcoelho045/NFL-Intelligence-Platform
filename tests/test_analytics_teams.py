"""Tests for team and game analytics."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_platform.analytics import (
    get_recent_games,
    get_team_defense,
    get_team_stats,
    summarize_recent_games,
    summarize_team_stats,
)
from nfl_platform.data.ingestion import NflDataLoader
from nfl_platform.exceptions import InvalidGameCountError, TeamNotFoundError


class TestTeamStats:
    def test_returns_only_the_requested_team(self, loader: NflDataLoader):
        frame = get_team_stats("KC", loader=loader, last_n_games=None)
        assert frame.get_column("team").unique().to_list() == ["KC"]

    def test_limits_to_last_n_games(self, loader: NflDataLoader):
        assert get_team_stats("KC", loader=loader, last_n_games=2).height == 2

    def test_ordered_most_recent_first(self, loader: NflDataLoader):
        weeks = get_team_stats("KC", loader=loader, last_n_games=None).get_column("week").to_list()
        assert weeks == sorted(weeks, reverse=True)

    def test_team_abbreviation_is_case_insensitive(self, loader: NflDataLoader):
        assert get_team_stats("kc", loader=loader, last_n_games=1).height == 1

    def test_week_filter(self, loader: NflDataLoader):
        frame = get_team_stats("KC", loader=loader, weeks=[1, 2], last_n_games=None)
        assert sorted(frame.get_column("week").to_list()) == [1, 2]

    def test_unknown_team_raises(self, loader: NflDataLoader):
        with pytest.raises(TeamNotFoundError) as excinfo:
            get_team_stats("ZZZ", loader=loader)
        assert "ZZZ" in str(excinfo.value)

    def test_invalid_game_count_raises(self, loader: NflDataLoader):
        with pytest.raises(InvalidGameCountError):
            get_team_stats("KC", loader=loader, last_n_games=-1)


class TestTeamDefense:
    def test_columns_are_prefixed_to_make_perspective_explicit(self, loader: NflDataLoader):
        frame = get_team_defense("KC", loader=loader, last_n_games=None)
        assert "allowed_passing_yards" in frame.columns
        assert "passing_yards" not in frame.columns

    def test_team_column_identifies_the_defense(self, loader: NflDataLoader):
        frame = get_team_defense("KC", loader=loader, last_n_games=None)
        assert frame.get_column("team").unique().to_list() == ["KC"]
        assert "KC" not in frame.get_column("offense_team").to_list()

    def test_defense_reflects_opponent_offense(self, loader: NflDataLoader):
        """What KC allowed in week 1 must equal what BAL produced in week 1."""
        kc_allowed = get_team_defense("KC", loader=loader, weeks=1, last_n_games=None)
        bal_offense = get_team_stats("BAL", loader=loader, weeks=1, last_n_games=None)
        assert kc_allowed.item(0, "allowed_passing_yards") == bal_offense.item(0, "passing_yards")
        assert kc_allowed.item(0, "allowed_rushing_yards") == bal_offense.item(0, "rushing_yards")

    def test_offense_and_defense_have_the_same_game_count(self, loader: NflDataLoader):
        offense = get_team_stats("KC", loader=loader, last_n_games=None)
        defense = get_team_defense("KC", loader=loader, last_n_games=None)
        assert offense.height == defense.height

    def test_unknown_team_raises(self, loader: NflDataLoader):
        with pytest.raises(TeamNotFoundError):
            get_team_defense("ZZZ", loader=loader)


class TestSummarizeTeamStats:
    @pytest.fixture
    def synthetic(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "passing_yards": [250, 150, 200],
                "rushing_yards": [100, 120, 80],
                "passing_epa": [1.0, -1.0, 3.0],
            }
        )

    def test_totals_and_averages(self, synthetic: pl.DataFrame):
        summary = summarize_team_stats(synthetic)
        assert summary.item(0, "games") == 3
        assert summary.item(0, "total_passing_yards") == 600
        assert summary.item(0, "avg_passing_yards") == pytest.approx(200.0)
        assert summary.item(0, "avg_rushing_yards") == pytest.approx(100.0)

    def test_epa_is_averaged_not_summed(self, synthetic: pl.DataFrame):
        summary = summarize_team_stats(synthetic)
        assert summary.item(0, "avg_passing_epa") == pytest.approx(1.0)
        assert "total_passing_epa" not in summary.columns

    def test_prefix_handles_defense_frames(self, loader: NflDataLoader):
        defense = get_team_defense("KC", loader=loader, last_n_games=None)
        summary = summarize_team_stats(defense, prefix="allowed_")
        assert summary.item(0, "games") == defense.height
        expected = sum(defense.get_column("allowed_passing_yards").drop_nulls().to_list())
        assert summary.item(0, "total_allowed_passing_yards") == expected

    def test_empty_input_yields_zero_games(self, synthetic: pl.DataFrame):
        assert summarize_team_stats(synthetic.clear()).item(0, "games") == 0


class TestRecentGames:
    def test_home_game_perspective(self, loader: NflDataLoader):
        """Fixture: 2024 week 1, KC beat BAL 27-20 at home."""
        games = get_recent_games("KC", loader=loader, last_n_games=None)
        week_one = games.filter(pl.col("week") == 1)
        assert week_one.item(0, "is_home") is True
        assert week_one.item(0, "opponent") == "BAL"
        assert week_one.item(0, "points_for") == 27
        assert week_one.item(0, "points_against") == 20
        assert week_one.item(0, "point_differential") == 7
        assert week_one.item(0, "outcome") == "W"

    def test_away_game_perspective(self, loader: NflDataLoader):
        """Fixture: 2024 week 2, BUF won 31-10 at MIA."""
        games = get_recent_games("BUF", loader=loader, last_n_games=None)
        week_two = games.filter(pl.col("week") == 2)
        assert week_two.item(0, "is_home") is False
        assert week_two.item(0, "opponent") == "MIA"
        assert week_two.item(0, "points_for") == 31
        assert week_two.item(0, "points_against") == 10
        assert week_two.item(0, "outcome") == "W"

    def test_loss_is_recorded(self, loader: NflDataLoader):
        """Fixture: 2024 week 4, BUF lost 10-35 at BAL."""
        games = get_recent_games("BUF", loader=loader, last_n_games=None)
        week_four = games.filter(pl.col("week") == 4)
        assert week_four.item(0, "outcome") == "L"
        assert week_four.item(0, "point_differential") == -25

    def test_ordered_most_recent_first(self, loader: NflDataLoader):
        games = get_recent_games("KC", loader=loader, last_n_games=None)
        weeks = games.get_column("week").to_list()
        assert weeks == sorted(weeks, reverse=True)

    def test_limits_to_last_n_games(self, loader: NflDataLoader):
        assert get_recent_games("KC", loader=loader, last_n_games=2).height == 2

    def test_unknown_team_raises(self, loader: NflDataLoader):
        with pytest.raises(TeamNotFoundError):
            get_recent_games("ZZZ", loader=loader)

    def test_summary_matches_manual_count(self, loader: NflDataLoader):
        games = get_recent_games("KC", loader=loader, last_n_games=None)
        summary = summarize_recent_games(games)
        outcomes = games.get_column("outcome").to_list()
        assert summary.item(0, "games") == len(outcomes)
        assert summary.item(0, "wins") == outcomes.count("W")
        assert summary.item(0, "losses") == outcomes.count("L")
        expected_avg = sum(games.get_column("points_for").to_list()) / games.height
        assert summary.item(0, "avg_points_for") == pytest.approx(expected_avg)
