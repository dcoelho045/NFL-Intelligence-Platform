"""Baselines and the market benchmark.

A baseline here is a real, shippable model, not a number quoted once in a
report. The promotion rule in :mod:`nfl_platform.ml.training.pipeline` can only
mean something if "the baseline wins" is an outcome the system can actually
deliver, and a baseline that cannot be serialised, loaded and served is not an
outcome -- it is a footnote.

Three things live here:

:class:`HomeFieldBaseline`
    The honest floor for game prediction. Home teams win about 57% of the time,
    so a model beating a coin flip has demonstrated nothing.

:class:`FeaturePassthrough`
    Predicts one feature's value directly: the rolling-average projection that
    every fantasy model has to beat before it has earned anything. Beating a
    three-game average at weekly fantasy is harder than it sounds.

:class:`SpreadBenchmark`
    The closing betting line converted to a probability. A *benchmark*, never a
    default and never an input: it tells us how much of the available signal the
    football-only model recovered, which is the context that stops a 63%
    accuracy from sounding either good or bad in a vacuum.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np

from nfl_platform.ml.models.base import (
    FittedModel,
    ModelKind,
    Preprocessor,
    register_model,
)

#: Probabilities are kept away from the ends, since no NFL game is a certainty
#: and a 0 or 1 makes log loss infinite the first time it is wrong.
PROBABILITY_FLOOR = 0.01
PROBABILITY_CEILING = 0.99


def clip_probability(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(values, PROBABILITY_FLOOR, PROBABILITY_CEILING), dtype=np.float64)


@register_model
class HomeFieldBaseline(FittedModel):
    """Predicts the training-set home win rate for every game."""

    algorithm: ClassVar[str] = "home_field_baseline"
    kind: ClassVar[ModelKind] = ModelKind.CLASSIFIER

    def __init__(
        self, feature_names: tuple[str, ...], preprocessor: Preprocessor, rate: float
    ) -> None:
        super().__init__(feature_names, preprocessor)
        self._rate = float(rate)

    @classmethod
    def fit(cls, feature_names: tuple[str, ...], matrix: np.ndarray, targets: np.ndarray) -> Self:
        finite = np.isfinite(targets)
        rate = float(targets[finite].mean()) if finite.any() else 0.5
        return cls(feature_names, Preprocessor.fit(matrix, standardize=False), rate)

    @property
    def rate(self) -> float:
        return self._rate

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return clip_probability(np.full(np.asarray(matrix).shape[0], self._rate, dtype=np.float64))

    def to_params(self) -> dict[str, Any]:
        return {"rate": self._rate}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        return cls((), Preprocessor(np.zeros(0)), float(params["rate"]))


@register_model
class FeaturePassthrough(FittedModel):
    """Predicts one feature's value, with the training mean as a fallback.

    The fallback matters: a rolling average is null for a player's first game,
    and a baseline that returns NaN there would be excluded from exactly the
    rows where baselines are most informative about a model's value.
    """

    algorithm: ClassVar[str] = "feature_passthrough"
    kind: ClassVar[ModelKind] = ModelKind.REGRESSOR

    def __init__(
        self,
        feature_names: tuple[str, ...],
        preprocessor: Preprocessor,
        feature: str,
        fallback: float,
    ) -> None:
        super().__init__(feature_names, preprocessor)
        self._feature = feature
        self._fallback = float(fallback)

    @classmethod
    def fit(
        cls,
        feature_names: tuple[str, ...],
        matrix: np.ndarray,
        targets: np.ndarray,
        *,
        feature: str,
    ) -> Self:
        if feature not in feature_names:
            raise ValueError(f"Baseline feature {feature!r} is not in the feature set.")
        finite = np.isfinite(targets)
        fallback = float(targets[finite].mean()) if finite.any() else 0.0
        return cls(feature_names, Preprocessor.fit(matrix, standardize=False), feature, fallback)

    @property
    def feature(self) -> str:
        return self._feature

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        data = np.asarray(matrix, dtype=np.float64)
        column = data[:, self.feature_names.index(self._feature)]
        return np.where(np.isfinite(column), column, self._fallback)

    def to_params(self) -> dict[str, Any]:
        return {"feature": self._feature, "fallback": self._fallback}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        return cls((), Preprocessor(np.zeros(0)), str(params["feature"]), float(params["fallback"]))


@register_model
class SpreadBenchmark(FittedModel):
    """Closing spread converted to a home win probability by a one-parameter logistic.

    Fitted on training rows only, exactly like every other model, so the
    comparison is like for like rather than a published conversion table
    borrowed from elsewhere.

    Not a candidate for promotion under any circumstance. Shipping it would mean
    the platform's "prediction" is a restatement of a betting line, which is a
    data-licensing position rather than a modelling one.
    """

    algorithm: ClassVar[str] = "spread_benchmark"
    kind: ClassVar[ModelKind] = ModelKind.CLASSIFIER

    def __init__(
        self,
        feature_names: tuple[str, ...],
        preprocessor: Preprocessor,
        slope: float,
        intercept: float,
    ) -> None:
        super().__init__(feature_names, preprocessor)
        self._slope = float(slope)
        self._intercept = float(intercept)

    @classmethod
    def fit(cls, spreads: np.ndarray, targets: np.ndarray) -> Self:
        from sklearn.linear_model import LogisticRegression

        values = np.asarray(spreads, dtype=np.float64).reshape(-1, 1)
        usable = np.isfinite(values).ravel() & np.isfinite(targets)
        names = ("spread_line",)
        if usable.sum() < 10 or len(np.unique(targets[usable])) < 2:
            # Too few lines to fit: degrade to "no information", not to a guess
            # that looks like a market opinion.
            return cls(names, Preprocessor(np.zeros(1)), 0.0, 0.0)
        estimator = LogisticRegression()
        estimator.fit(values[usable], targets[usable].astype(int))
        return cls(
            names,
            Preprocessor.fit(values[usable], standardize=False),
            float(estimator.coef_[0][0]),
            float(estimator.intercept_[0]),
        )

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64).reshape(-1, 1)
        prepared = self.prepare(values)[:, 0]
        return clip_probability(1.0 / (1.0 + np.exp(-(self._intercept + self._slope * prepared))))

    @property
    def is_informative(self) -> bool:
        """False when no line could be fitted, so callers can omit the benchmark."""
        return self._slope != 0.0

    def to_params(self) -> dict[str, Any]:
        return {"slope": self._slope, "intercept": self._intercept}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        return cls(
            ("spread_line",),
            Preprocessor(np.zeros(1)),
            float(params["slope"]),
            float(params["intercept"]),
        )
