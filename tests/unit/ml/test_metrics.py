"""Evaluation metrics.

Asserted against hand-worked values rather than against another library. A test
that only proves two implementations agree proves nothing about whether either
is right, and these numbers are what the milestone's conclusions rest on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nfl_platform.ml.evaluation.metrics import (
    accuracy,
    brier_score,
    brier_skill_score,
    classification_metrics,
    expected_calibration_error,
    grouped_spearman,
    log_loss,
    mean_absolute_error,
    r_squared,
    regression_metrics,
    reliability_table,
    roc_auc,
    root_mean_squared_error,
    spearman,
)


class TestClassificationMetrics:
    def test_accuracy_thresholds_at_one_half(self):
        assert accuracy([1, 0, 1, 0], [0.9, 0.4, 0.6, 0.51]) == pytest.approx(0.75)

    def test_brier_is_mean_squared_probability_error(self):
        # (0.2^2 + 0.3^2) / 2
        assert brier_score([1, 0], [0.8, 0.3]) == pytest.approx(0.065)

    def test_a_coin_flip_scores_one_quarter(self):
        assert brier_score([1, 0, 1, 0], [0.5] * 4) == pytest.approx(0.25)

    def test_log_loss_matches_the_hand_computed_value(self):
        expected = -(math.log(0.8) + math.log(0.7)) / 2
        assert log_loss([1, 0], [0.8, 0.3]) == pytest.approx(expected)

    def test_log_loss_is_finite_for_a_confidently_wrong_prediction(self):
        """Unclipped, one miss sends the mean to infinity and hides everything else."""
        assert math.isfinite(log_loss([1], [0.0]))

    def test_brier_skill_is_positive_when_the_model_beats_the_reference(self):
        skill = brier_skill_score([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], [0.5] * 4)
        assert skill > 0

    def test_brier_skill_is_zero_against_itself(self):
        probabilities = [0.7, 0.2, 0.6, 0.4]
        assert brier_skill_score([1, 0, 1, 0], probabilities, probabilities) == pytest.approx(0.0)

    def test_brier_skill_is_negative_when_the_model_is_worse(self):
        assert brier_skill_score([1, 0], [0.1, 0.9], [0.5, 0.5]) < 0

    def test_perfect_ranking_scores_one(self):
        assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero(self):
        assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)

    def test_all_ties_score_one_half(self):
        """Tie handling is the part of the rank identity that is easy to get wrong."""
        assert roc_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)

    def test_auc_is_undefined_with_only_one_class(self):
        assert math.isnan(roc_auc([1, 1, 1], [0.9, 0.5, 0.2]))

    def test_a_perfectly_calibrated_forecast_has_no_calibration_error(self):
        rng = np.random.default_rng(3)
        probabilities = rng.uniform(0.05, 0.95, 20_000)
        outcomes = (rng.uniform(size=probabilities.size) < probabilities).astype(float)
        assert expected_calibration_error(outcomes, probabilities) < 0.02

    def test_a_systematically_overconfident_forecast_is_detected(self):
        outcomes = [1] * 50 + [0] * 50
        probabilities = [0.99] * 50 + [0.9] * 50
        assert expected_calibration_error(outcomes, probabilities) > 0.3

    def test_the_reliability_table_bins_cover_the_unit_interval(self):
        table = reliability_table([1, 0], [0.9, 0.1], bins=5)
        assert len(table) == 5
        assert table[0]["lower"] == 0.0
        assert table[-1]["upper"] == 1.0

    def test_a_probability_of_one_lands_in_the_final_bin(self):
        table = reliability_table([1], [1.0], bins=4)
        assert table[-1]["count"] == 1

    def test_the_bundle_carries_the_base_rate_and_the_skill_score(self):
        metrics = classification_metrics(
            [1, 1, 0, 0], [0.8, 0.7, 0.3, 0.2], baseline_prob=[0.5] * 4
        )
        assert metrics.rows == 4
        assert metrics.base_rate == pytest.approx(0.5)
        assert metrics.brier_skill_vs_baseline is not None
        assert metrics.reliability


class TestRegressionMetrics:
    def test_mae_is_the_mean_absolute_difference(self):
        assert mean_absolute_error([10, 20], [12, 17]) == pytest.approx(2.5)

    def test_rmse_penalises_the_larger_miss_more(self):
        assert root_mean_squared_error([10, 20], [12, 17]) > mean_absolute_error([10, 20], [12, 17])

    def test_r_squared_is_one_for_a_perfect_fit(self):
        assert r_squared([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_r_squared_is_zero_for_predicting_the_mean(self):
        assert r_squared([1, 2, 3, 4], [2.5] * 4) == pytest.approx(0.0)

    def test_spearman_is_one_for_a_monotone_but_nonlinear_relationship(self):
        assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)

    def test_spearman_is_minus_one_when_reversed(self):
        assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_grouped_spearman_averages_within_groups(self):
        """Pooled correlation would be flattered by the gap between groups."""
        truth = [1, 2, 3, 10, 20, 30]
        predicted = [1, 2, 3, 10, 20, 30]
        groups = ["w1", "w1", "w1", "w2", "w2", "w2"]
        assert grouped_spearman(truth, predicted, groups) == pytest.approx(1.0)

    def test_grouped_spearman_detects_a_wrong_order_inside_a_week(self):
        truth = [1, 2, 3, 10, 20, 30]
        predicted = [3, 2, 1, 30, 20, 10]
        groups = ["w1", "w1", "w1", "w2", "w2", "w2"]
        assert grouped_spearman(truth, predicted, groups) == pytest.approx(-1.0)

    def test_groups_too_small_to_rank_are_skipped(self):
        assert math.isnan(grouped_spearman([1, 2], [1, 2], ["a", "b"]))

    def test_the_bundle_reports_improvement_over_a_baseline(self):
        metrics = regression_metrics(
            [10, 20, 30], [11, 19, 31], baseline_pred=[15, 15, 15], groups=["w", "w", "w"]
        )
        assert metrics.mae_improvement_vs_baseline > 0.5
        assert metrics.within_week_spearman == pytest.approx(1.0)


class TestEdgeCases:
    @pytest.mark.parametrize(
        "metric", [accuracy, brier_score, log_loss, mean_absolute_error, root_mean_squared_error]
    )
    def test_empty_input_is_not_an_error(self, metric):
        assert math.isnan(metric([], []))

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            brier_score([1, 0], [0.5])

    def test_non_finite_pairs_are_dropped_rather_than_poisoning_the_mean(self):
        assert mean_absolute_error([1.0, np.nan], [2.0, 5.0]) == pytest.approx(1.0)
