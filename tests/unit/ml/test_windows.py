"""As-of windowing, and the leakage guarantee that rests on it.

The truncation-invariance tests are the most important ones in the milestone.
Every other check here could pass while the model is quietly trained on the
future; these cannot. The argument is simple: if deleting every row after week
*W* leaves the week-*W* features unchanged, then nothing after week *W* was ever
used to compute them.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_platform.ml.features.windows import (
    TIME_INDEX,
    join_as_of,
    safe_ratio,
    time_index,
    trailing_count,
    trailing_std,
    trailing_sums,
)


def _history(values: list[tuple[str, int, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": [row[0] for row in values],
            "season": [row[1] for row in values],
            "week": [row[2] for row in values],
            "points": [row[3] for row in values],
        }
    ).with_columns(time_index())


class TestTimeIndex:
    def test_orders_within_a_season(self):
        frame = _history([("A", 2023, 1, 1.0), ("A", 2023, 12, 2.0)])
        assert frame.get_column(TIME_INDEX).to_list() == [202301, 202312]

    def test_orders_across_a_season_boundary(self):
        """Postseason week 22 must sort before the next season's week 1."""
        frame = _history([("A", 2023, 22, 1.0), ("A", 2024, 1, 2.0)])
        values = frame.get_column(TIME_INDEX).to_list()
        assert values[0] < values[1]

    def test_consecutive_games_differ_by_at_least_one(self):
        """The as-of join decrements by one, so a smaller gap would self-match."""
        frame = _history([("A", 2023, 1, 1.0), ("A", 2023, 2, 2.0)])
        values = frame.get_column(TIME_INDEX).to_list()
        assert values[1] - values[0] >= 1


class TestTrailingSums:
    def test_sums_the_window_inclusive_of_the_current_row(self):
        frame = trailing_sums(
            _history([("A", 2023, w, float(w)) for w in (1, 2, 3, 4)]),
            group_by=["team"],
            order_by=[TIME_INDEX],
            columns=("points",),
            window=2,
            suffix="_w2",
        )
        assert frame.get_column("points_w2").to_list() == [1.0, 3.0, 5.0, 7.0]

    def test_groups_do_not_bleed_into_each_other(self):
        frame = trailing_sums(
            _history([("A", 2023, 1, 10.0), ("B", 2023, 1, 5.0), ("B", 2023, 2, 5.0)]),
            group_by=["team"],
            order_by=[TIME_INDEX],
            columns=("points",),
            window=5,
            suffix="_w",
        )
        rows = {(r["team"], r["week"]): r["points_w"] for r in frame.to_dicts()}
        assert rows[("B", 2)] == 10.0
        assert rows[("A", 1)] == 10.0

    def test_nulls_count_as_zero_rather_than_voiding_the_window(self):
        frame = trailing_sums(
            _history([("A", 2023, 1, 4.0)]).with_columns(
                pl.lit(None, dtype=pl.Float64).alias("points")
            ),
            group_by=["team"],
            order_by=[TIME_INDEX],
            columns=("points",),
            window=3,
            suffix="_w",
        )
        assert frame.get_column("points_w").to_list() == [0.0]

    def test_missing_ordering_column_is_rejected(self):
        with pytest.raises(ValueError, match="ordering column"):
            trailing_sums(
                pl.DataFrame({"team": ["A"]}),
                group_by=["team"],
                order_by=["nope"],
                columns=(),
                window=2,
                suffix="_w",
            )


class TestTrailingCountAndStd:
    def test_count_is_capped_by_the_window(self):
        frame = trailing_count(
            _history([("A", 2023, w, 1.0) for w in range(1, 6)]),
            group_by=["team"],
            order_by=[TIME_INDEX],
            alias="n",
            window=3,
        )
        assert frame.get_column("n").to_list() == [1, 2, 3, 3, 3]

    def test_count_is_cumulative_without_a_window(self):
        frame = trailing_count(
            _history([("A", 2023, w, 1.0) for w in range(1, 4)]),
            group_by=["team"],
            order_by=[TIME_INDEX],
            alias="n",
        )
        assert frame.get_column("n").to_list() == [1, 2, 3]

    def test_std_needs_two_observations(self):
        frame = trailing_std(
            _history([("A", 2023, 1, 1.0), ("A", 2023, 2, 3.0)]),
            group_by=["team"],
            order_by=[TIME_INDEX],
            column="points",
            window=4,
            alias="sd",
        )
        values = frame.get_column("sd").to_list()
        assert values[0] is None
        assert values[1] == pytest.approx(1.4142, abs=1e-3)


class TestSafeRatio:
    @pytest.mark.parametrize("denominator", [0.0, None])
    def test_unknown_rather_than_zero_when_the_denominator_is_unusable(self, denominator):
        """A team with no plays has an unknown EPA per play, not an EPA of zero."""
        frame = pl.DataFrame({"num": [5.0], "den": [denominator]}).with_columns(
            safe_ratio("num", "den", "rate")
        )
        assert frame.item(0, "rate") is None

    def test_divides_when_the_denominator_is_positive(self):
        frame = pl.DataFrame({"num": [5.0], "den": [2.0]}).with_columns(
            safe_ratio("num", "den", "rate")
        )
        assert frame.item(0, "rate") == pytest.approx(2.5)


class TestJoinAsOf:
    @pytest.fixture
    def history(self) -> pl.DataFrame:
        return trailing_sums(
            _history([("A", 2023, w, float(w)) for w in (1, 2, 3)]),
            group_by=["team"],
            order_by=[TIME_INDEX],
            columns=("points",),
            window=5,
            suffix="_total",
        )

    def test_a_row_never_sees_its_own_game(self, history: pl.DataFrame):
        """The guarantee the whole leakage story rests on."""
        targets = _history([("A", 2023, 3, 0.0)])
        joined = join_as_of(targets, history, by=["team"], columns=("points_total",))
        # Weeks 1 and 2 total 3; week 3's own 3 points must be excluded.
        assert joined.item(0, "points_total") == 3.0

    def test_the_first_ever_game_gets_null_not_zero(self, history: pl.DataFrame):
        targets = _history([("A", 2023, 1, 0.0)])
        joined = join_as_of(targets, history, by=["team"], columns=("points_total",))
        assert joined.item(0, "points_total") is None

    def test_a_distant_future_target_still_sees_all_prior_history(self, history: pl.DataFrame):
        """Predicting week 12 from a week-3 history must not truncate the window.

        This is the case a shifted rolling window gets wrong: it counts the
        intervening unplayed rows and silently shortens the lookback.
        """
        targets = _history([("A", 2023, 12, 0.0)])
        joined = join_as_of(targets, history, by=["team"], columns=("points_total",))
        assert joined.item(0, "points_total") == 6.0

    def test_unknown_entities_get_nulls_rather_than_another_entity_s_history(
        self, history: pl.DataFrame
    ):
        targets = _history([("Z", 2023, 5, 0.0)])
        joined = join_as_of(targets, history, by=["team"], columns=("points_total",))
        assert joined.item(0, "points_total") is None

    def test_a_frame_without_a_time_index_is_rejected(self, history: pl.DataFrame):
        with pytest.raises(ValueError, match=TIME_INDEX):
            join_as_of(
                pl.DataFrame({"team": ["A"]}), history, by=["team"], columns=("points_total",)
            )

    def test_future_history_rows_cannot_reach_an_earlier_target(self):
        """Truncation invariance, at the primitive level."""
        full = trailing_sums(
            _history([("A", 2023, w, float(w)) for w in range(1, 11)]),
            group_by=["team"],
            order_by=[TIME_INDEX],
            columns=("points",),
            window=20,
            suffix="_total",
        )
        truncated = full.filter(pl.col("week") <= 4)
        targets = _history([("A", 2023, 4, 0.0)])

        from_full = join_as_of(targets, full, by=["team"], columns=("points_total",))
        from_truncated = join_as_of(targets, truncated, by=["team"], columns=("points_total",))
        assert_frame_equal(from_full, from_truncated)
