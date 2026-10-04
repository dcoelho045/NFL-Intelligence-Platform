"""Linear models.

At roughly 250 usable games per season, a regularised linear model is not the
warm-up act for gradient boosting -- it is a serious candidate. Twenty-two
differenced features over a few thousand rows is close to the regime where L2
logistic regression is the right answer, and the promotion rule is written so
that it can win.

Coefficients are stored as plain numbers, so the artifact is readable: anyone
can see what the model believes without running it, and a sign that flips
between retrainings is visible in a diff.
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
from nfl_platform.ml.models.baselines import clip_probability


@register_model
class LogisticRegressionModel(FittedModel):
    """L2-regularised logistic regression on standardised features."""

    algorithm: ClassVar[str] = "logistic_regression"
    kind: ClassVar[ModelKind] = ModelKind.CLASSIFIER

    def __init__(
        self,
        feature_names: tuple[str, ...],
        preprocessor: Preprocessor,
        coefficients: np.ndarray,
        intercept: float,
    ) -> None:
        super().__init__(feature_names, preprocessor)
        self._coefficients = np.asarray(coefficients, dtype=np.float64)
        self._intercept = float(intercept)

    @classmethod
    def fit(
        cls,
        feature_names: tuple[str, ...],
        matrix: np.ndarray,
        targets: np.ndarray,
        *,
        c: float = 1.0,
        seed: int = 0,
    ) -> Self:
        from sklearn.linear_model import LogisticRegression

        # Standardisation is not cosmetic for a penalised model: without it the
        # L2 term penalises a feature measured in yards far more than one
        # measured in EPA, which is a modelling decision nobody made.
        preprocessor = Preprocessor.fit(matrix, standardize=True)
        prepared = preprocessor.transform(matrix)
        estimator = LogisticRegression(C=c, max_iter=2000, random_state=seed)
        estimator.fit(prepared, targets.astype(int))
        return cls(feature_names, preprocessor, estimator.coef_[0], float(estimator.intercept_[0]))

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients.copy()

    def _logits(self, matrix: np.ndarray) -> np.ndarray:
        return self.prepare(matrix) @ self._coefficients + self._intercept

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return clip_probability(1.0 / (1.0 + np.exp(-self._logits(matrix))))

    def contributions(self, matrix: np.ndarray) -> np.ndarray | None:
        """Each feature's share of the log-odds: coefficient times standardised value."""
        return self.prepare(matrix) * self._coefficients

    def to_params(self) -> dict[str, Any]:
        return {
            "coefficients": self._coefficients.tolist(),
            "intercept": self._intercept,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        return cls(
            (),
            Preprocessor(np.zeros(0)),
            np.asarray(params["coefficients"], dtype=np.float64),
            float(params["intercept"]),
        )


@register_model
class RidgeRegressionModel(FittedModel):
    """L2-penalised linear regression on standardised features."""

    algorithm: ClassVar[str] = "ridge_regression"
    kind: ClassVar[ModelKind] = ModelKind.REGRESSOR

    def __init__(
        self,
        feature_names: tuple[str, ...],
        preprocessor: Preprocessor,
        coefficients: np.ndarray,
        intercept: float,
        floor: float | None = 0.0,
    ) -> None:
        super().__init__(feature_names, preprocessor)
        self._coefficients = np.asarray(coefficients, dtype=np.float64)
        self._intercept = float(intercept)
        self._floor = floor

    @classmethod
    def fit(
        cls,
        feature_names: tuple[str, ...],
        matrix: np.ndarray,
        targets: np.ndarray,
        *,
        alpha: float = 1.0,
        floor: float | None = 0.0,
        seed: int = 0,
    ) -> Self:
        from sklearn.linear_model import Ridge

        preprocessor = Preprocessor.fit(matrix, standardize=True)
        prepared = preprocessor.transform(matrix)
        estimator = Ridge(alpha=alpha, random_state=seed)
        estimator.fit(prepared, targets)
        return cls(
            feature_names,
            preprocessor,
            np.asarray(estimator.coef_, dtype=np.float64).ravel(),
            float(estimator.intercept_),
            floor,
        )

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = self.prepare(matrix) @ self._coefficients + self._intercept
        # Fantasy scoring is bounded below in practice; an unclipped linear model
        # will happily project negative points for a deep bench player, which is
        # not wrong by much but is obviously wrong to any reader.
        return values if self._floor is None else np.maximum(values, self._floor)

    def contributions(self, matrix: np.ndarray) -> np.ndarray | None:
        return self.prepare(matrix) * self._coefficients

    def to_params(self) -> dict[str, Any]:
        return {
            "coefficients": self._coefficients.tolist(),
            "intercept": self._intercept,
            "floor": self._floor,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        return cls(
            (),
            Preprocessor(np.zeros(0)),
            np.asarray(params["coefficients"], dtype=np.float64),
            float(params["intercept"]),
            params.get("floor", 0.0),
        )
