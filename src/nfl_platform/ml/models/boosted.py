"""Gradient-boosted models.

LightGBM rather than XGBoost for three reasons that are specific rather than
preferential: its native artifact is plain text, so the project needs no pickle
anywhere; it returns per-prediction SHAP contributions from ``predict`` itself,
so ``top_factors`` costs no extra dependency; and it is CPU-first, so nothing
here implies a GPU.

The hyperparameters are deliberately small -- shallow trees, a high minimum leaf
size, heavy feature and row subsampling. With a few thousand rows the realistic
failure is not underfitting, and a booster allowed to grow freely will memorise
the training seasons and then lose to logistic regression on the validation
folds. Constraining it up front is cheaper than discovering that repeatedly.

Determinism is explicit. LightGBM's default histogram construction varies with
thread count, so an unpinned run on a different machine produces different
numbers from the same data and seed, which would make the recorded metrics
unreproducible.
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

#: Shared across both objectives. ``deterministic`` plus a fixed thread count is
#: what makes a re-run reproduce the recorded metric rather than approximately
#: reproduce it.
BASE_PARAMS: dict[str, Any] = {
    "num_leaves": 15,
    "max_depth": 4,
    "learning_rate": 0.03,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
}

DEFAULT_ROUNDS = 400
EARLY_STOPPING_ROUNDS = 40


def _train_booster(
    matrix: np.ndarray,
    targets: np.ndarray,
    *,
    objective: str,
    metric: str,
    seed: int,
    rounds: int,
    validation: tuple[np.ndarray, np.ndarray] | None,
) -> Any:
    import lightgbm as lgb

    params = {**BASE_PARAMS, "objective": objective, "metric": metric, "seed": seed}
    train_set = lgb.Dataset(matrix, label=targets, free_raw_data=False)
    callbacks: list[Any] = [lgb.log_evaluation(period=0)]
    valid_sets = None
    if validation is not None and validation[0].shape[0] > 0:
        valid_sets = [lgb.Dataset(validation[0], label=validation[1], reference=train_set)]
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))
    return lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


class _BoosterModel(FittedModel):
    """Shared serialisation for the LightGBM-backed models."""

    def __init__(
        self, feature_names: tuple[str, ...], preprocessor: Preprocessor, booster: Any
    ) -> None:
        super().__init__(feature_names, preprocessor)
        self._booster = booster

    @property
    def booster(self) -> Any:
        return self._booster

    @property
    def best_iteration(self) -> int | None:
        value = getattr(self._booster, "best_iteration", None)
        return int(value) if value else None

    def _raw(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self._booster.predict(self.prepare(matrix)), dtype=np.float64).ravel()

    def contributions(self, matrix: np.ndarray) -> np.ndarray | None:
        """SHAP values from LightGBM itself; the trailing column is the base value."""
        raw = np.asarray(
            self._booster.predict(self.prepare(matrix), pred_contrib=True), dtype=np.float64
        )
        return raw[:, :-1]

    def to_params(self) -> dict[str, Any]:
        return {"booster": self._booster.model_to_string()}

    @classmethod
    def _booster_from(cls, params: dict[str, Any]) -> Any:
        import lightgbm as lgb

        return lgb.Booster(model_str=str(params["booster"]))


@register_model
class LightGbmClassifier(_BoosterModel):
    """Binary gradient boosting returning a calibrated-ready probability."""

    algorithm: ClassVar[str] = "lightgbm_classifier"
    kind: ClassVar[ModelKind] = ModelKind.CLASSIFIER

    @classmethod
    def fit(
        cls,
        feature_names: tuple[str, ...],
        matrix: np.ndarray,
        targets: np.ndarray,
        *,
        seed: int = 0,
        rounds: int = DEFAULT_ROUNDS,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Self:
        # Boosters handle missing values natively, so imputation is skipped and
        # "this team has no eighth prior game" stays distinguishable from
        # "this team's eighth prior game was average".
        preprocessor = Preprocessor(np.full(matrix.shape[1], np.nan))
        prepared_validation = (
            None
            if validation is None
            else (preprocessor.transform(validation[0]), validation[1].astype(int))
        )
        booster = _train_booster(
            preprocessor.transform(matrix),
            targets.astype(int),
            objective="binary",
            metric="binary_logloss",
            seed=seed,
            rounds=rounds,
            validation=prepared_validation,
        )
        return cls(feature_names, preprocessor, booster)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return clip_probability(self._raw(matrix))

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        return cls((), Preprocessor(np.zeros(0)), cls._booster_from(params))


@register_model
class LightGbmRegressor(_BoosterModel):
    """Gradient boosting for a non-negative, right-skewed target."""

    algorithm: ClassVar[str] = "lightgbm_regressor"
    kind: ClassVar[ModelKind] = ModelKind.REGRESSOR

    def __init__(
        self,
        feature_names: tuple[str, ...],
        preprocessor: Preprocessor,
        booster: Any,
        floor: float | None = 0.0,
    ) -> None:
        super().__init__(feature_names, preprocessor, booster)
        self._floor = floor

    @classmethod
    def fit(
        cls,
        feature_names: tuple[str, ...],
        matrix: np.ndarray,
        targets: np.ndarray,
        *,
        objective: str = "regression",
        seed: int = 0,
        rounds: int = DEFAULT_ROUNDS,
        floor: float | None = 0.0,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Self:
        preprocessor = Preprocessor(np.full(matrix.shape[1], np.nan))
        prepared_validation = (
            None if validation is None else (preprocessor.transform(validation[0]), validation[1])
        )
        booster = _train_booster(
            preprocessor.transform(matrix),
            targets,
            objective=objective,
            metric="l1",
            seed=seed,
            rounds=rounds,
            validation=prepared_validation,
        )
        return cls(feature_names, preprocessor, booster, floor)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = self._raw(matrix)
        return values if self._floor is None else np.maximum(values, self._floor)

    def to_params(self) -> dict[str, Any]:
        return {**super().to_params(), "floor": self._floor}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        return cls(
            (), Preprocessor(np.zeros(0)), cls._booster_from(params), params.get("floor", 0.0)
        )
