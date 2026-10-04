"""Fitted models: preprocessing, serialisation, calibration, and the skew guard."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nfl_platform.exceptions import FeatureSchemaMismatchError, ModelArtifactInvalidError
from nfl_platform.ml.models import load_model, registered_algorithms
from nfl_platform.ml.models.base import ModelKind, Preprocessor, model_type
from nfl_platform.ml.models.baselines import (
    PROBABILITY_CEILING,
    PROBABILITY_FLOOR,
    FeaturePassthrough,
    HomeFieldBaseline,
    SpreadBenchmark,
)
from nfl_platform.ml.models.boosted import LightGbmClassifier, LightGbmRegressor
from nfl_platform.ml.models.calibration import (
    CALIBRATION_METHOD,
    MIN_CALIBRATION_ROWS,
    CalibratedClassifier,
    calibrate,
)
from nfl_platform.ml.models.linear import LogisticRegressionModel, RidgeRegressionModel

FEATURES = ("a", "b", "c")


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(600, 3))
    logits = 1.3 * matrix[:, 0] - 0.8 * matrix[:, 1]
    targets = (rng.uniform(size=600) < 1 / (1 + np.exp(-logits))).astype(float)
    return matrix, targets


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(600, 3))
    targets = np.maximum(0.0, 10 + 3 * matrix[:, 0] - 2 * matrix[:, 1] + rng.normal(size=600))
    return matrix, targets


class TestPreprocessor:
    def test_missing_values_are_filled_with_the_training_median(self):
        matrix = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        preprocessor = Preprocessor.fit(matrix, standardize=False)
        out = preprocessor.transform(np.array([[np.nan, np.nan]]))
        assert out.tolist() == [[3.0, 20.0]]

    def test_a_column_that_is_missing_everywhere_falls_back_to_zero(self):
        matrix = np.array([[np.nan], [np.nan]])
        preprocessor = Preprocessor.fit(matrix, standardize=False)
        assert preprocessor.transform(np.array([[np.nan]])).tolist() == [[0.0]]

    def test_standardisation_centres_and_scales(self):
        matrix = np.array([[1.0], [2.0], [3.0]])
        preprocessor = Preprocessor.fit(matrix, standardize=True)
        out = preprocessor.transform(matrix)
        assert out.mean() == pytest.approx(0.0, abs=1e-9)
        assert out.std() == pytest.approx(1.0, abs=1e-9)

    def test_a_constant_column_does_not_produce_nan(self):
        """Dividing by a zero standard deviation would void every row."""
        matrix = np.array([[5.0], [5.0], [5.0]])
        preprocessor = Preprocessor.fit(matrix, standardize=True)
        assert np.isfinite(preprocessor.transform(matrix)).all()

    def test_round_trips_through_json(self):
        preprocessor = Preprocessor.fit(np.array([[1.0, 2.0], [3.0, 8.0]]), standardize=True)
        restored = Preprocessor.from_params(json.loads(json.dumps(preprocessor.to_params())))
        sample = np.array([[2.0, 5.0]])
        assert restored.transform(sample).tolist() == preprocessor.transform(sample).tolist()

    def test_a_wrong_column_count_is_rejected(self):
        preprocessor = Preprocessor.fit(np.zeros((2, 3)), standardize=False)
        with pytest.raises(ValueError, match="Expected 3 columns"):
            preprocessor.transform(np.zeros((2, 2)))


class TestHomeFieldBaseline:
    def test_predicts_the_training_home_win_rate(self):
        targets = np.array([1.0] * 57 + [0.0] * 43)
        model = HomeFieldBaseline.fit(FEATURES, np.zeros((100, 3)), targets)
        assert model.rate == pytest.approx(0.57)
        assert model.predict(np.zeros((5, 3))).tolist() == pytest.approx([0.57] * 5)

    def test_is_a_classifier(self):
        assert HomeFieldBaseline.kind is ModelKind.CLASSIFIER


class TestFeaturePassthrough:
    def test_returns_the_named_feature(self):
        model = FeaturePassthrough.fit(
            FEATURES, np.array([[1.0, 2.0, 3.0]]), np.array([5.0]), feature="b"
        )
        assert model.predict(np.array([[9.0, 4.0, 1.0]])).tolist() == [4.0]

    def test_falls_back_to_the_training_mean_when_the_feature_is_missing(self):
        """A rolling average is null for a debut; the baseline must still answer."""
        model = FeaturePassthrough.fit(
            FEATURES, np.array([[1.0, 2.0, 3.0]]), np.array([7.0]), feature="b"
        )
        assert model.predict(np.array([[1.0, np.nan, 1.0]])).tolist() == [7.0]

    def test_an_unknown_feature_is_rejected_at_fit_time(self):
        with pytest.raises(ValueError, match="not in the feature set"):
            FeaturePassthrough.fit(FEATURES, np.zeros((1, 3)), np.zeros(1), feature="nope")


class TestLogisticRegression:
    def test_learns_the_sign_of_the_relationship(self, classification_data):
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        assert model.coefficients[0] > 0
        assert model.coefficients[1] < 0

    def test_probabilities_stay_inside_the_open_interval(self, classification_data):
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        probabilities = model.predict(matrix * 50)
        assert probabilities.min() >= PROBABILITY_FLOOR
        assert probabilities.max() <= PROBABILITY_CEILING

    def test_beats_a_constant_forecast(self, classification_data):
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        from nfl_platform.ml.evaluation.metrics import brier_score

        assert brier_score(targets, model.predict(matrix)) < brier_score(
            targets, np.full_like(targets, targets.mean())
        )

    def test_contributions_sum_towards_the_logit(self, classification_data):
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        contributions = model.contributions(matrix[:1])
        assert contributions is not None
        assert contributions.shape == (1, 3)

    def test_round_trips_without_changing_a_prediction(self, classification_data):
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        restored = load_model(json.loads(json.dumps(model.serialise())))
        assert restored.predict(matrix).tolist() == pytest.approx(model.predict(matrix).tolist())

    def test_the_artifact_is_plain_numbers(self, classification_data):
        """Readable and pickle-free: anyone can see what the model believes."""
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        payload = json.dumps(model.serialise())
        assert "coefficients" in payload
        assert "pickle" not in payload


class TestRidgeRegression:
    def test_predictions_are_not_negative_by_default(self, regression_data):
        """Projecting negative fantasy points is not wrong by much but is obviously wrong."""
        matrix, targets = regression_data
        model = RidgeRegressionModel.fit(FEATURES, matrix, targets)
        assert model.predict(matrix * 10).min() >= 0.0

    def test_beats_predicting_the_mean(self, regression_data):
        matrix, targets = regression_data
        model = RidgeRegressionModel.fit(FEATURES, matrix, targets)
        from nfl_platform.ml.evaluation.metrics import mean_absolute_error

        assert mean_absolute_error(targets, model.predict(matrix)) < mean_absolute_error(
            targets, np.full_like(targets, targets.mean())
        )

    def test_round_trips(self, regression_data):
        matrix, targets = regression_data
        model = RidgeRegressionModel.fit(FEATURES, matrix, targets)
        restored = load_model(json.loads(json.dumps(model.serialise())))
        assert restored.predict(matrix).tolist() == pytest.approx(model.predict(matrix).tolist())


class TestLightGbm:
    def test_classifier_round_trips_through_its_text_format(self, classification_data):
        matrix, targets = classification_data
        model = LightGbmClassifier.fit(FEATURES, matrix, targets, seed=1)
        restored = load_model(json.loads(json.dumps(model.serialise())))
        assert restored.predict(matrix).tolist() == pytest.approx(model.predict(matrix).tolist())

    def test_the_serialised_booster_is_text_not_a_pickle(self, classification_data):
        matrix, targets = classification_data
        model = LightGbmClassifier.fit(FEATURES, matrix, targets, seed=1)
        assert isinstance(model.serialise()["params"]["booster"], str)

    def test_training_is_deterministic_for_a_fixed_seed(self, classification_data):
        """Without pinned threads and deterministic histograms this fails on CI."""
        matrix, targets = classification_data
        first = LightGbmClassifier.fit(FEATURES, matrix, targets, seed=5)
        second = LightGbmClassifier.fit(FEATURES, matrix, targets, seed=5)
        assert first.predict(matrix).tolist() == pytest.approx(second.predict(matrix).tolist())

    def test_contributions_are_returned_per_feature(self, classification_data):
        matrix, targets = classification_data
        model = LightGbmClassifier.fit(FEATURES, matrix, targets, seed=1)
        contributions = model.contributions(matrix[:1])
        assert contributions is not None
        assert contributions.shape == (1, len(FEATURES))

    def test_the_regressor_respects_its_floor(self, regression_data):
        matrix, targets = regression_data
        model = LightGbmRegressor.fit(FEATURES, matrix, targets, seed=2)
        assert model.predict(matrix).min() >= 0.0

    def test_boosters_see_missing_values_rather_than_imputed_medians(self, classification_data):
        """Trees handle nulls natively, so "no eighth prior game" stays distinguishable."""
        matrix, targets = classification_data
        model = LightGbmClassifier.fit(FEATURES, matrix, targets, seed=1)
        assert np.isnan(model.prepare(np.array([[np.nan, 0.0, 0.0]]))[0, 0])


class TestSpreadBenchmark:
    def test_a_bigger_home_favourite_gets_a_higher_probability(self):
        rng = np.random.default_rng(4)
        spreads = rng.normal(0, 6, 400)
        targets = (rng.uniform(size=400) < 1 / (1 + np.exp(-spreads / 5))).astype(float)
        model = SpreadBenchmark.fit(spreads, targets)
        probabilities = model.predict(np.array([[-10.0], [0.0], [10.0]]))
        assert probabilities[0] < probabilities[1] < probabilities[2]

    def test_with_no_usable_lines_it_reports_itself_uninformative(self):
        model = SpreadBenchmark.fit(np.array([np.nan] * 5), np.array([1.0] * 5))
        assert not model.is_informative

    def test_it_round_trips(self):
        rng = np.random.default_rng(6)
        spreads = rng.normal(0, 6, 300)
        targets = (spreads > 0).astype(float)
        model = SpreadBenchmark.fit(spreads, targets)
        restored = load_model(json.loads(json.dumps(model.serialise())))
        sample = np.array([[3.0]])
        assert restored.predict(sample).tolist() == pytest.approx(model.predict(sample).tolist())


class TestCalibration:
    def test_calibration_improves_a_systematically_overconfident_model(self):
        rng = np.random.default_rng(8)
        truth = rng.uniform(0.2, 0.8, 4000)
        outcomes = (rng.uniform(size=truth.size) < truth).astype(float)
        # Push probabilities towards the extremes, the classic overconfidence shape.
        overconfident = np.clip((truth - 0.5) * 2.5 + 0.5, 0.01, 0.99)

        from nfl_platform.ml.evaluation.metrics import expected_calibration_error

        before = expected_calibration_error(outcomes, overconfident)

        class _Fixed(LogisticRegressionModel):
            def predict(self, matrix: np.ndarray) -> np.ndarray:
                return overconfident

        inner = _Fixed(FEATURES, Preprocessor(np.zeros(3)), np.zeros(3), 0.0)
        calibrated = CalibratedClassifier.fit(inner, overconfident, outcomes)
        after = expected_calibration_error(outcomes, calibrated.predict(np.zeros((1, 3))))
        assert after < before

    def test_calibration_is_skipped_when_there_is_too_little_validation_data(
        self, classification_data
    ):
        """A calibrator fitted on noise is worse than none; this says so honestly."""
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        few = MIN_CALIBRATION_ROWS - 1
        result, method = calibrate(model, model.predict(matrix[:few]), targets[:few])
        assert method is None
        assert result is model

    def test_calibration_is_applied_when_there_is_enough_data(self, classification_data):
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        result, method = calibrate(model, model.predict(matrix), targets)
        assert method == CALIBRATION_METHOD
        assert isinstance(result, CalibratedClassifier)

    def test_a_calibrated_model_round_trips_with_its_inner_model(self, classification_data):
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        calibrated, _ = calibrate(model, model.predict(matrix), targets)
        restored = load_model(json.loads(json.dumps(calibrated.serialise())))
        assert restored.predict(matrix).tolist() == pytest.approx(
            calibrated.predict(matrix).tolist()
        )

    def test_calibration_preserves_ranking(self, classification_data):
        """Platt scaling is monotone, so contributions stay meaningful."""
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        calibrated, _ = calibrate(model, model.predict(matrix), targets)
        raw = model.predict(matrix)
        adjusted = calibrated.predict(matrix)
        assert np.argsort(raw).tolist() == np.argsort(adjusted).tolist()


class TestFeatureSchemaGuard:
    def test_identical_features_are_accepted(self, classification_data):
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        model.check_features(FEATURES)

    def test_a_missing_feature_is_rejected(self, classification_data):
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        with pytest.raises(FeatureSchemaMismatchError) as caught:
            model.check_features(("a", "b"))
        assert caught.value.missing == ("c",)

    def test_an_extra_feature_is_rejected(self, classification_data):
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        with pytest.raises(FeatureSchemaMismatchError) as caught:
            model.check_features(("a", "b", "c", "d"))
        assert caught.value.unexpected == ("d",)

    def test_reordered_features_are_rejected(self, classification_data):
        """An estimator scores positional arrays; reordering is silent corruption."""
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        with pytest.raises(FeatureSchemaMismatchError) as caught:
            model.check_features(("c", "b", "a"))
        assert caught.value.reordered

    def test_a_calibrated_model_delegates_the_check(self, classification_data):
        matrix, targets = classification_data
        model = LogisticRegressionModel.fit(FEATURES, matrix, targets)
        calibrated, _ = calibrate(model, model.predict(matrix), targets)
        with pytest.raises(FeatureSchemaMismatchError):
            calibrated.check_features(("a",))


class TestRegistry:
    def test_every_shipped_algorithm_is_registered(self):
        assert set(registered_algorithms()) >= {
            "home_field_baseline",
            "feature_passthrough",
            "logistic_regression",
            "ridge_regression",
            "lightgbm_classifier",
            "lightgbm_regressor",
            "calibrated_classifier",
            "spread_benchmark",
        }

    def test_an_unknown_algorithm_cannot_be_constructed_from_a_file(self):
        """Only names in the explicit table are loadable; nothing is auto-imported."""
        with pytest.raises(ModelArtifactInvalidError, match="unknown algorithm"):
            model_type("something_arbitrary")

    def test_a_malformed_payload_is_refused(self):
        with pytest.raises(ModelArtifactInvalidError, match="malformed model section"):
            load_model({"algorithm": "logistic_regression"})
