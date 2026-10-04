"""Probability calibration.

A classifier that ranks well can still be systematically over- or
under-confident, and for a win-probability product that is the failure that
matters. "The model says 80%" has to mean the home team wins about four times in
five, or the number is decoration.

Platt scaling (a one-parameter logistic on the model's own log-odds) rather than
isotonic regression, for a reason specific to this dataset: a validation fold
here is a few hundred games. Isotonic is non-parametric and will happily carve
that into a step function that fits the fold's noise, and a calibrator that
overfits is worse than none -- it looks principled while moving probabilities in
the wrong direction.

The calibrator is fitted on **validation-fold predictions only**. Fitting it on
training predictions calibrates against numbers the model has already seen and
reliably makes real-world calibration worse; fitting it on the test period would
use the frozen block for a modelling decision.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np

from nfl_platform.ml.models.base import (
    FittedModel,
    ModelKind,
    Preprocessor,
    load_model,
    register_model,
)
from nfl_platform.ml.models.baselines import (
    PROBABILITY_CEILING,
    PROBABILITY_FLOOR,
    clip_probability,
)

CALIBRATION_METHOD = "platt_sigmoid"
#: Below this many validation rows, calibration is skipped rather than fitted on
#: noise. The uncalibrated model is the more honest artifact at that point.
MIN_CALIBRATION_ROWS = 200


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, PROBABILITY_FLOOR, PROBABILITY_CEILING)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


@register_model
class CalibratedClassifier(FittedModel):
    """A classifier whose probabilities have been Platt-scaled.

    Wraps rather than replaces the inner model, so the uncalibrated predictions
    stay reachable and the artifact records exactly what was calibrated.
    """

    algorithm: ClassVar[str] = "calibrated_classifier"
    kind: ClassVar[ModelKind] = ModelKind.CLASSIFIER

    def __init__(
        self,
        feature_names: tuple[str, ...],
        preprocessor: Preprocessor,
        inner: FittedModel,
        slope: float,
        intercept: float,
    ) -> None:
        super().__init__(feature_names, preprocessor)
        self._inner = inner
        self._slope = float(slope)
        self._intercept = float(intercept)

    @classmethod
    def fit(
        cls,
        inner: FittedModel,
        validation_probabilities: np.ndarray,
        validation_targets: np.ndarray,
        *,
        seed: int = 0,
    ) -> Self:
        from sklearn.linear_model import LogisticRegression

        probabilities = np.asarray(validation_probabilities, dtype=np.float64).ravel()
        targets = np.asarray(validation_targets, dtype=np.float64).ravel()
        usable = np.isfinite(probabilities) & np.isfinite(targets)
        estimator = LogisticRegression(random_state=seed)
        estimator.fit(_logit(probabilities[usable]).reshape(-1, 1), targets[usable].astype(int))
        return cls(
            inner.feature_names,
            inner.preprocessor,
            inner,
            float(estimator.coef_[0][0]),
            float(estimator.intercept_[0]),
        )

    @property
    def inner(self) -> FittedModel:
        return self._inner

    @property
    def method(self) -> str:
        return CALIBRATION_METHOD

    def check_features(self, names: tuple[str, ...]) -> None:
        self._inner.check_features(names)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        # The inner model applies its own preprocessing, so this must not.
        raw = _logit(self._inner.predict(matrix))
        return clip_probability(1.0 / (1.0 + np.exp(-(self._intercept + self._slope * raw))))

    def predict_uncalibrated(self, matrix: np.ndarray) -> np.ndarray:
        return self._inner.predict(matrix)

    def contributions(self, matrix: np.ndarray) -> np.ndarray | None:
        """Delegated to the inner model.

        Platt scaling is monotone, so it cannot change which features pushed a
        prediction up or down, nor their order. It does change the units, which
        is why ``FeatureContribution`` describes direction and relative size
        rather than claiming a probability delta.
        """
        return self._inner.contributions(matrix)

    def to_params(self) -> dict[str, Any]:
        return {
            "slope": self._slope,
            "intercept": self._intercept,
            "method": CALIBRATION_METHOD,
            "inner": self._inner.serialise(),
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        inner = load_model(dict(params["inner"]))
        return cls(
            inner.feature_names,
            inner.preprocessor,
            inner,
            float(params["slope"]),
            float(params["intercept"]),
        )


def calibrate(
    inner: FittedModel,
    validation_probabilities: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[FittedModel, str | None]:
    """Calibrate when there is enough validation data, otherwise leave it alone.

    Returns the model to ship and the method applied, so the artifact can record
    ``calibration_method: null`` honestly rather than implying a calibration
    that was skipped.
    """
    probabilities = np.asarray(validation_probabilities, dtype=np.float64).ravel()
    targets = np.asarray(validation_targets, dtype=np.float64).ravel()
    usable = int((np.isfinite(probabilities) & np.isfinite(targets)).sum())
    if usable < MIN_CALIBRATION_ROWS or len(np.unique(targets[np.isfinite(targets)])) < 2:
        return inner, None
    return (
        CalibratedClassifier.fit(inner, probabilities, targets, seed=seed),
        CALIBRATION_METHOD,
    )
