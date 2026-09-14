"""Tests for game-level analytics derived from the schedules dataset."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_platform.analytics import get_recent_games, summarize_recent_games
from nfl_platform.data.ingestion import NflDataLoader
from nfl_platform.exceptions import TeamNotFoundError


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


class TestSummarizeRecentGames:
    def test_summary_matches_manual_count(self, loader: NflDataLoader):
        games = get_recent_games("KC", loader=loader, last_n_games=None)
        summary = summarize_recent_games(games)
        outcomes = games.get_column("outcome").to_list()
        assert summary.item(0, "games") == len(outcomes)
        assert summary.item(0, "wins") == outcomes.count("W")
        assert summary.item(0, "losses") == outcomes.count("L")
        expected_avg = sum(games.get_column("points_for").to_list()) / games.height
        assert summary.item(0, "avg_points_for") == pytest.approx(expected_avg)
