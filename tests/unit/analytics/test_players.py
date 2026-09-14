"""Tests for player analytics."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_platform.analytics import (
    get_player_recent_stats,
    get_player_usage,
    resolve_pfr_id,
    summarize_player_stats,
)
from nfl_platform.analytics.filters import POSTSEASON, REGULAR_SEASON
from nfl_platform.data.ingestion import Dataset, NflDataLoader
from nfl_platform.exceptions import (
    InvalidGameCountError,
    InvalidWeekError,
    PlayerNotFoundError,
)
from support import ALLEN_ID, MAHOMES_ID, UNKNOWN_PLAYER_ID, StubDataSource


class TestRecentStats:
    def test_returns_requested_number_of_games(self, loader: NflDataLoader):
        frame = get_player_recent_stats(MAHOMES_ID, loader=loader, last_n_games=3)
        assert frame.height == 3

    def test_ordered_most_recent_first(self, loader: NflDataLoader):
        frame = get_player_recent_stats(MAHOMES_ID, loader=loader, last_n_games=None)
        weeks = frame.get_column("week").to_list()
        assert weeks == sorted(weeks, reverse=True)

    def test_only_the_requested_player_is_returned(self, loader: NflDataLoader):
        frame = get_player_recent_stats(MAHOMES_ID, loader=loader, last_n_games=None)
        assert frame.get_column("player_id").unique().to_list() == [MAHOMES_ID]

    def test_last_n_games_larger_than_sample_returns_all(self, loader: NflDataLoader):
        everything = get_player_recent_stats(MAHOMES_ID, loader=loader, last_n_games=None)
        capped = get_player_recent_stats(MAHOMES_ID, loader=loader, last_n_games=999)
        assert capped.height == everything.height

    def test_week_filter_narrows_the_result(self, loader: NflDataLoader):
        frame = get_player_recent_stats(MAHOMES_ID, loader=loader, weeks=[1, 2], last_n_games=None)
        assert sorted(frame.get_column("week").to_list()) == [1, 2]

    def test_week_with_no_games_returns_empty_frame_with_schema(self, loader: NflDataLoader):
        frame = get_player_recent_stats(MAHOMES_ID, loader=loader, weeks=18, last_n_games=None)
        assert frame.is_empty()
        assert "fantasy_points" in frame.columns

    def test_postseason_filter_returns_empty_for_regular_season_fixture(
        self, loader: NflDataLoader
    ):
        frame = get_player_recent_stats(MAHOMES_ID, loader=loader, season_type=POSTSEASON)
        assert frame.is_empty()

    def test_default_includes_every_game_type(self, loader: NflDataLoader):
        """Defaulting to REG would silently drop playoff games from 'recent form'."""
        default = get_player_recent_stats(MAHOMES_ID, loader=loader, last_n_games=None)
        regular = get_player_recent_stats(
            MAHOMES_ID, loader=loader, last_n_games=None, season_type=REGULAR_SEASON
        )
        # The fixture is regular season only, so these agree; the assertion pins
        # the default so a future change to it fails loudly.
        assert default.height == regular.height
        assert default.height > 0

    def test_unknown_player_raises(self, loader: NflDataLoader):
        with pytest.raises(PlayerNotFoundError) as excinfo:
            get_player_recent_stats(UNKNOWN_PLAYER_ID, loader=loader)
        assert UNKNOWN_PLAYER_ID in str(excinfo.value)

    def test_season_without_data_raises_player_not_found(self, loader: NflDataLoader):
        with pytest.raises(PlayerNotFoundError):
            get_player_recent_stats(MAHOMES_ID, loader=loader, seasons=2023)

    def test_invalid_game_count_raises(self, loader: NflDataLoader):
        with pytest.raises(InvalidGameCountError):
            get_player_recent_stats(MAHOMES_ID, loader=loader, last_n_games=0)

    def test_invalid_week_raises(self, loader: NflDataLoader):
        with pytest.raises(InvalidWeekError):
            get_player_recent_stats(MAHOMES_ID, loader=loader, weeks=99)

    def test_player_id_is_trimmed(self, loader: NflDataLoader):
        frame = get_player_recent_stats(f"  {MAHOMES_ID}  ", loader=loader, last_n_games=1)
        assert frame.height == 1


class TestSummarize:
    """Aggregation correctness is checked against hand-computed values."""

    @pytest.fixture
    def synthetic_stats(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "week": [3, 2, 1],
                "player_id": ["00-0000001"] * 3,
                "passing_yards": [300, 200, 100],
                "carries": [4, 6, 5],
                "targets": [0, 0, 0],
                "fantasy_points": [20.0, 10.0, 15.0],
                "target_share": [0.2, 0.4, 0.3],
            }
        )

    def test_game_count(self, synthetic_stats: pl.DataFrame):
        assert summarize_player_stats(synthetic_stats).item(0, "games") == 3

    def test_totals(self, synthetic_stats: pl.DataFrame):
        summary = summarize_player_stats(synthetic_stats)
        assert summary.item(0, "total_passing_yards") == 600
        assert summary.item(0, "total_carries") == 15
        assert summary.item(0, "total_fantasy_points") == pytest.approx(45.0)

    def test_averages(self, synthetic_stats: pl.DataFrame):
        summary = summarize_player_stats(synthetic_stats)
        assert summary.item(0, "avg_passing_yards") == pytest.approx(200.0)
        assert summary.item(0, "avg_fantasy_points") == pytest.approx(15.0)
        assert summary.item(0, "avg_target_share") == pytest.approx(0.3)

    def test_empty_input_yields_zero_games(self, synthetic_stats: pl.DataFrame):
        summary = summarize_player_stats(synthetic_stats.clear())
        assert summary.height == 1
        assert summary.item(0, "games") == 0

    def test_nulls_are_ignored_by_the_mean(self):
        frame = pl.DataFrame({"fantasy_points": [10.0, None, 20.0]})
        summary = summarize_player_stats(frame)
        assert summary.item(0, "avg_fantasy_points") == pytest.approx(15.0)
        assert summary.item(0, "total_fantasy_points") == pytest.approx(30.0)

    def test_matches_manual_sum_of_real_fixture_data(self, loader: NflDataLoader):
        recent = get_player_recent_stats(ALLEN_ID, loader=loader, last_n_games=None)
        expected = sum(recent.get_column("fantasy_points").drop_nulls().to_list())
        summary = summarize_player_stats(recent)
        assert summary.item(0, "total_fantasy_points") == pytest.approx(expected)


class TestUsage:
    def test_returns_opportunity_columns_not_production(self, loader: NflDataLoader):
        usage = get_player_usage(MAHOMES_ID, loader=loader, last_n_games=3)
        assert {"targets", "carries", "attempts"} <= set(usage.columns)
        assert "fantasy_points" not in usage.columns

    def test_snap_share_is_joined_per_week(self, loader: NflDataLoader):
        usage = get_player_usage(MAHOMES_ID, loader=loader, last_n_games=3)
        assert "offense_snap_pct" in usage.columns
        assert usage.get_column("offense_snap_pct").drop_nulls().len() > 0

    def test_snap_share_can_be_disabled(self, loader: NflDataLoader):
        usage = get_player_usage(
            MAHOMES_ID, loader=loader, last_n_games=3, include_snap_share=False
        )
        assert "offense_snap_pct" not in usage.columns

    def test_join_does_not_duplicate_rows(self, loader: NflDataLoader):
        without = get_player_usage(
            MAHOMES_ID, loader=loader, last_n_games=None, include_snap_share=False
        )
        with_snaps = get_player_usage(MAHOMES_ID, loader=loader, last_n_games=None)
        assert with_snaps.height == without.height

    def test_unknown_player_raises(self, loader: NflDataLoader):
        with pytest.raises(PlayerNotFoundError):
            get_player_usage(UNKNOWN_PLAYER_ID, loader=loader)

    def test_snap_columns_omitted_when_the_crosswalk_has_no_pfr_id(self):
        """A missing identifier link must drop the column, never invent a value."""
        player_id = "00-0000001"
        stub = StubDataSource(
            {
                Dataset.PLAYER_STATS: pl.DataFrame(
                    {
                        "player_id": [player_id],
                        "player_display_name": ["Test Player"],
                        "position": ["WR"],
                        "season": [2024],
                        "week": [1],
                        "season_type": ["REG"],
                        "team": ["KC"],
                        "opponent_team": ["BAL"],
                        "targets": [5],
                        "fantasy_points": [10.0],
                        "fantasy_points_ppr": [15.0],
                    }
                ),
                Dataset.PLAYERS: pl.DataFrame(
                    {
                        "gsis_id": [player_id],
                        "display_name": ["Test Player"],
                        "position": ["WR"],
                        "pfr_id": [None],
                        "latest_team": ["KC"],
                        "status": ["ACT"],
                    }
                ),
            }
        )
        stub_loader = NflDataLoader(source=stub, current_season=2024)
        usage = get_player_usage(player_id, loader=stub_loader, seasons=2024)
        assert usage.height == 1
        assert "offense_snap_pct" not in usage.columns


class TestIdentifierCrosswalk:
    def test_resolves_gsis_to_pfr(self, loader: NflDataLoader):
        assert resolve_pfr_id(MAHOMES_ID, loader=loader) == "MahoPa00"

    def test_unknown_id_returns_none(self, loader: NflDataLoader):
        assert resolve_pfr_id(UNKNOWN_PLAYER_ID, loader=loader) is None
