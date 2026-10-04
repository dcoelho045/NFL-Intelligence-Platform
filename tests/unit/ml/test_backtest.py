"""Walk-forward backtesting and the promotion rule.

The promotion tests encode a decision made before any model was trained: a
candidate earns the default slot only by beating the simpler model by more than
the fold-to-fold variation of that advantage. Writing it down as a test is what
stops it being relaxed later to let a favoured model through.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from nfl_platform.ml.datasets.builder import TrainingDataset
from nfl_platform.ml.datasets.splits import plan_splits
from nfl_platform.ml.evaluation.backtest import (
    BacktestResult,
    FoldResult,
    decide_promotion,
    walk_forward,
)
from nfl_platform.ml.evaluation.metrics import classification_metrics
from nfl_platform.ml.features.schema import Availability, FeatureSet, FeatureSpec
from nfl_platform.ml.models.base import FittedModel
from nfl_platform.ml.models.baselines import HomeFieldBaseline
from nfl_platform.ml.models.linear import LogisticRegressionModel

FEATURE_SET = FeatureSet(
    "test-v1",
    (
        FeatureSpec("x", "A signal.", "synthetic", Availability.PRIOR_GAMES, "n/a"),
        FeatureSpec("y", "Noise.", "synthetic", Availability.PRIOR_GAMES, "n/a"),
    ),
)


@pytest.fixture
def dataset() -> TrainingDataset:
    rng = np.random.default_rng(21)
    rows = []
    for season in range(2012, 2024):
        for _ in range(120):
            x = float(rng.normal())
            rows.append(
                {
                    "season": season,
                    "week": int(rng.integers(1, 15)),
                    "x": x,
                    "y": float(rng.normal()),
                    "label": float(rng.uniform() < 1 / (1 + np.exp(-1.5 * x))),
                }
            )
    return TrainingDataset(pl.DataFrame(rows), FEATURE_SET, "label")


def _score(actual: np.ndarray, predicted: np.ndarray, _: TrainingDataset) -> dict[str, Any]:
    return classification_metrics(actual, predicted).as_dict()


def _fit_logistic(train: TrainingDataset) -> FittedModel:
    return LogisticRegressionModel.fit(train.feature_names, train.matrix(), train.targets())


def _fit_baseline(train: TrainingDataset) -> FittedModel:
    return HomeFieldBaseline.fit(train.feature_names, train.matrix(), train.targets())


class TestTrainingDataset:
    def test_a_missing_declared_feature_is_rejected(self):
        with pytest.raises(ValueError, match="missing declared feature"):
            TrainingDataset(pl.DataFrame({"season": [1], "label": [1.0]}), FEATURE_SET, "label")

    def test_a_missing_label_is_rejected(self):
        frame = pl.DataFrame({"season": [1], "x": [1.0], "y": [1.0]})
        with pytest.raises(ValueError, match="missing the label"):
            TrainingDataset(frame, FEATURE_SET, "label")

    def test_the_matrix_follows_the_declared_feature_order(self, dataset: TrainingDataset):
        shuffled = dataset.frame.select(["y", "x", "season", "week", "label"])
        rebuilt = TrainingDataset(shuffled, FEATURE_SET, "label")
        assert rebuilt.matrix()[:, 0].tolist() == dataset.matrix()[:, 0].tolist()

    def test_infinities_become_missing_values(self):
        frame = pl.DataFrame({"season": [1], "x": [float("inf")], "y": [1.0], "label": [1.0]})
        matrix = TrainingDataset(frame, FEATURE_SET, "label").matrix()
        assert np.isnan(matrix[0, 0])

    def test_filtering_by_season_preserves_the_binding(self, dataset: TrainingDataset):
        subset = dataset.filter_seasons([2015, 2016])
        assert subset.seasons == (2015, 2016)
        assert subset.feature_names == dataset.feature_names


class TestWalkForward:
    def test_every_validation_season_produces_a_fold(self, dataset: TrainingDataset):
        plan = plan_splits(dataset.seasons)
        result = walk_forward(dataset, plan, _fit_logistic, _score)
        assert len(result.folds) == len(plan.validation_seasons)

    def test_out_of_fold_predictions_cover_every_validation_row(self, dataset: TrainingDataset):
        plan = plan_splits(dataset.seasons)
        result = walk_forward(dataset, plan, _fit_logistic, _score)
        expected = len(dataset.filter_seasons(plan.validation_seasons))
        assert result.predictions.size == expected

    def test_a_fold_never_trains_on_the_season_it_scores(self, dataset: TrainingDataset):
        plan = plan_splits(dataset.seasons)
        for fold in plan.folds:
            assert max(fold.train_seasons) < min(fold.eval_seasons)

    def test_the_index_is_aligned_with_the_predictions(self, dataset: TrainingDataset):
        plan = plan_splits(dataset.seasons)
        result = walk_forward(dataset, plan, _fit_logistic, _score)
        assert result.index.height == result.predictions.size

    def test_a_model_with_signal_beats_the_constant_baseline(self, dataset: TrainingDataset):
        """Guards against the harness quietly scoring the wrong rows."""
        plan = plan_splits(dataset.seasons)
        model = walk_forward(dataset, plan, _fit_logistic, _score)
        baseline = walk_forward(dataset, plan, _fit_baseline, _score)
        assert np.nanmean(model.fold_metric("brier_score")) < np.nanmean(
            baseline.fold_metric("brier_score")
        )

    def test_empty_folds_are_recorded_rather_than_crashing(self, dataset: TrainingDataset):
        sparse = dataset.filter_seasons([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
        plan = plan_splits(dataset.seasons)
        result = walk_forward(sparse, plan, _fit_logistic, _score)
        assert any("skipped" in note for note in result.notes)


def _backtest(values: list[float], metric: str = "brier_score") -> BacktestResult:
    return BacktestResult(
        folds=tuple(
            FoldResult(f"f{i}", 100, 50, {metric: value}) for i, value in enumerate(values)
        ),
        predictions=np.zeros(1),
        targets=np.zeros(1),
        index=pl.DataFrame(),
    )


class TestPromotionRule:
    def test_a_consistent_improvement_is_promoted(self):
        simple = _backtest([0.25, 0.25, 0.25, 0.25])
        candidate = _backtest([0.22, 0.22, 0.22, 0.22])
        decision = decide_promotion(simple, candidate, metric="brier_score", lower_is_better=True)
        assert decision.promoted
        assert decision.improvement == pytest.approx(0.03)

    def test_an_improvement_smaller_than_fold_noise_is_not_promoted(self):
        """The decision this rule exists to make."""
        simple = _backtest([0.25, 0.20, 0.30, 0.22])
        candidate = _backtest([0.24, 0.23, 0.27, 0.23])
        decision = decide_promotion(simple, candidate, metric="brier_score", lower_is_better=True)
        assert not decision.promoted
        assert "not distinguishable from noise" in decision.reason

    def test_a_worse_candidate_is_not_promoted(self):
        simple = _backtest([0.22, 0.22, 0.22, 0.22])
        candidate = _backtest([0.25, 0.25, 0.25, 0.25])
        decision = decide_promotion(simple, candidate, metric="brier_score", lower_is_better=True)
        assert not decision.promoted
        assert decision.improvement < 0

    def test_higher_is_better_metrics_invert_the_comparison(self):
        simple = _backtest([0.60] * 4, metric="accuracy")
        candidate = _backtest([0.66] * 4, metric="accuracy")
        decision = decide_promotion(simple, candidate, metric="accuracy", lower_is_better=False)
        assert decision.promoted

    def test_too_few_comparable_folds_keeps_the_simpler_model(self):
        decision = decide_promotion(
            _backtest([0.25]), _backtest([0.10]), metric="brier_score", lower_is_better=True
        )
        assert not decision.promoted
        assert "Fewer than two comparable folds" in decision.reason

    def test_the_decision_records_its_evidence(self):
        decision = decide_promotion(
            _backtest([0.25] * 4),
            _backtest([0.22] * 4),
            metric="brier_score",
            lower_is_better=True,
        )
        payload = decision.as_dict()
        assert payload["metric"] == "brier_score"
        assert len(payload["fold_differences"]) == 4
        assert payload["reason"]

    def test_the_reason_names_both_models(self):
        decision = decide_promotion(
            _backtest([0.25] * 4),
            _backtest([0.22] * 4),
            metric="brier_score",
            lower_is_better=True,
            simple_name="logistic regression",
            candidate_name="LightGBM",
        )
        assert "logistic regression" in decision.reason
        assert "LightGBM" in decision.reason
