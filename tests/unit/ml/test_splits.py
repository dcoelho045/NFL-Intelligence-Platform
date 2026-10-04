"""Time-aware splitting and the frozen test period."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_platform.ml.datasets.splits import (
    MIN_COMPLETE_SEASON_GAMES,
    SplitPlan,
    TimeFold,
    complete_seasons,
    plan_splits,
)


class TestTimeFold:
    def test_a_fold_that_trains_on_its_own_evaluation_season_cannot_be_built(self):
        """The leakage guard lives in the type, so no harness can construct one."""
        with pytest.raises(ValueError, match="strictly before"):
            TimeFold("bad", train_seasons=(2020, 2021), eval_seasons=(2021,))

    def test_a_fold_that_trains_on_a_later_season_cannot_be_built(self):
        with pytest.raises(ValueError, match="strictly before"):
            TimeFold("bad", train_seasons=(2023,), eval_seasons=(2020,))

    def test_a_valid_fold_is_accepted(self):
        fold = TimeFold("ok", train_seasons=(2018, 2019), eval_seasons=(2020,))
        assert fold.eval_seasons == (2020,)


class TestCompleteSeasons:
    def _games(
        self, season: int, count: int, *, game_type: str, completed: bool = True
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "season": [season] * count,
                "game_id": [f"{season}_{i}" for i in range(count)],
                "game_type": [game_type] * count,
                "completed": [completed] * count,
            }
        )

    def test_a_season_with_a_championship_game_is_complete(self):
        frame = pl.concat(
            [self._games(2022, 5, game_type="REG"), self._games(2022, 1, game_type="SB")]
        )
        assert complete_seasons(frame) == (2022,)

    def test_an_in_progress_season_is_not_complete(self):
        frame = pl.concat(
            [
                self._games(2022, 5, game_type="REG"),
                self._games(2022, 1, game_type="SB"),
                self._games(2023, 40, game_type="REG"),
            ]
        )
        assert complete_seasons(frame) == (2022,)

    def test_unplayed_games_do_not_make_a_season_complete(self):
        frame = self._games(2023, 300, game_type="SB", completed=False)
        assert complete_seasons(frame) == ()

    def test_the_fallback_counts_distinct_games_not_rows(self):
        """A player-week frame reaches 200 rows in one weekend.

        Counting rows would let an in-progress season become the frozen test
        period, which is the specific failure this guards.
        """
        rows = pl.DataFrame(
            {
                "season": [2023] * 2000,
                "game_id": [f"2023_{i % 10}" for i in range(2000)],
                "completed": [True] * 2000,
            }
        )
        assert complete_seasons(rows) == ()

    def test_the_fallback_accepts_a_full_slate_of_distinct_games(self):
        count = MIN_COMPLETE_SEASON_GAMES
        rows = pl.DataFrame(
            {
                "season": [2023] * count,
                "game_id": [f"2023_{i}" for i in range(count)],
                "completed": [True] * count,
            }
        )
        assert complete_seasons(rows) == (2023,)

    def test_an_empty_frame_yields_nothing(self):
        assert complete_seasons(pl.DataFrame()) == ()


class TestPlanSplits:
    @pytest.fixture
    def plan(self) -> SplitPlan:
        return plan_splits(range(2010, 2024))

    def test_the_test_period_is_the_latest_complete_seasons(self, plan: SplitPlan):
        assert plan.test_seasons == (2022, 2023)

    def test_validation_sits_immediately_before_the_test_period(self, plan: SplitPlan):
        assert plan.validation_seasons == (2018, 2019, 2020, 2021)

    def test_the_three_blocks_do_not_overlap(self, plan: SplitPlan):
        blocks = [set(plan.train_seasons), set(plan.validation_seasons), set(plan.test_seasons)]
        assert not blocks[0] & blocks[1]
        assert not blocks[1] & blocks[2]
        assert not blocks[0] & blocks[2]

    def test_the_test_period_is_never_fitted_on(self, plan: SplitPlan):
        assert not set(plan.fit_seasons) & set(plan.test_seasons)

    def test_the_final_fit_uses_train_plus_validation(self, plan: SplitPlan):
        assert set(plan.fit_seasons) == set(plan.train_seasons) | set(plan.validation_seasons)

    def test_folds_expand(self, plan: SplitPlan):
        sizes = [len(fold.train_seasons) for fold in plan.folds]
        assert sizes == sorted(sizes)
        assert len(set(sizes)) == len(sizes)

    def test_no_fold_trains_on_the_test_period(self, plan: SplitPlan):
        for fold in plan.folds:
            assert not set(fold.train_seasons) & set(plan.test_seasons)

    def test_every_fold_evaluates_exactly_one_validation_season(self, plan: SplitPlan):
        assert [fold.eval_seasons for fold in plan.folds] == [
            (season,) for season in plan.validation_seasons
        ]

    def test_the_test_period_is_derived_not_hard_coded(self):
        """Running against a newer snapshot must move the test period forward."""
        later = plan_splits(range(2012, 2026))
        assert later.test_seasons == (2024, 2025)

    def test_too_few_seasons_is_an_error_rather_than_a_silent_shrink(self):
        with pytest.raises(ValueError, match="at least"):
            plan_splits(range(2020, 2025))

    def test_the_rationale_names_the_test_seasons(self, plan: SplitPlan):
        assert "2022" in plan.rationale and "2023" in plan.rationale

    def test_the_plan_serialises_for_metadata(self, plan: SplitPlan):
        payload = plan.as_dict()
        assert payload["test_seasons"] == [2022, 2023]
        assert len(payload["folds"]) == len(plan.validation_seasons)
