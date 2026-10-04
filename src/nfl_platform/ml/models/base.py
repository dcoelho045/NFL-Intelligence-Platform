"""The fitted-model contract and its preprocessing.

Every estimator in this project is one object that owns its own preprocessing,
answers :meth:`FittedModel.predict`, and can describe itself as plain JSON. That
is a deliberately small contract, and the smallness is what makes the artifact
format safe: there is nothing to serialise except numbers, so nothing has to be
executed to deserialise them.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, ClassVar, Self

import numpy as np

from nfl_platform.exceptions import FeatureSchemaMismatchError, ModelArtifactInvalidError


class ModelKind(StrEnum):
    """What a model's output means."""

    CLASSIFIER = "classifier"
    """``predict`` returns P(label = 1)."""

    REGRESSOR = "regressor"
    """``predict`` returns the expected value of the target."""


class Preprocessor:
    """Median imputation, then optional standardisation.

    Fitted inside a training fold and never outside one. Computing medians over
    the whole dataset is the quietest form of leakage there is: it moves no rows
    across the split, produces no obvious error, and lets every fold's imputed
    values be informed by seasons that fold is supposed to be blind to.

    Median rather than mean because the inputs include ratios whose tails are
    long, and a single blow-out game should not move the value a missing week
    gets filled with.
    """

    def __init__(
        self,
        medians: np.ndarray,
        means: np.ndarray | None = None,
        scales: np.ndarray | None = None,
    ) -> None:
        self._medians = np.asarray(medians, dtype=np.float64)
        self._means = None if means is None else np.asarray(means, dtype=np.float64)
        self._scales = None if scales is None else np.asarray(scales, dtype=np.float64)

    @classmethod
    def fit(cls, matrix: np.ndarray, *, standardize: bool) -> Preprocessor:
        data = np.asarray(matrix, dtype=np.float64)
        if data.size == 0:
            medians = np.zeros(data.shape[1], dtype=np.float64)
        else:
            # An all-missing column warns rather than failing; the fallback below
            # is the intended handling, so the warning is noise.
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                medians = np.nanmedian(data, axis=0)
            # A column that is missing everywhere has no median. Zero is the
            # only defensible fill, and it is recorded rather than inferred.
            medians = np.where(np.isfinite(medians), medians, 0.0)
        if not standardize:
            return cls(medians)
        imputed = cls(medians).transform(data)
        means = imputed.mean(axis=0) if imputed.size else np.zeros_like(medians)
        scales = imputed.std(axis=0) if imputed.size else np.ones_like(medians)
        # A constant column has zero spread; dividing by it would produce NaN for
        # every row rather than the zero-information column it actually is.
        scales = np.where(scales > 1e-12, scales, 1.0)
        return cls(medians, means, scales)

    @property
    def standardizes(self) -> bool:
        return self._means is not None and self._scales is not None

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        data = np.asarray(matrix, dtype=np.float64).copy()
        if data.ndim != 2 or data.shape[1] != self._medians.size:
            raise ValueError(
                f"Expected {self._medians.size} columns, got "
                f"{data.shape[1] if data.ndim == 2 else data.shape}"
            )
        filled = np.where(np.isfinite(data), data, np.broadcast_to(self._medians, data.shape))
        if self._means is None or self._scales is None:
            return filled
        return (filled - self._means) / self._scales

    def to_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"medians": self._medians.tolist()}
        if self._means is not None and self._scales is not None:
            params["means"] = self._means.tolist()
            params["scales"] = self._scales.tolist()
        return params

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Preprocessor:
        means = params.get("means")
        scales = params.get("scales")
        return cls(
            np.asarray(params["medians"], dtype=np.float64),
            None if means is None else np.asarray(means, dtype=np.float64),
            None if scales is None else np.asarray(scales, dtype=np.float64),
        )


class FittedModel(ABC):
    """One trained estimator, ready to score and to serialise."""

    algorithm: ClassVar[str]
    kind: ClassVar[ModelKind]

    def __init__(self, feature_names: tuple[str, ...], preprocessor: Preprocessor) -> None:
        self._feature_names = tuple(feature_names)
        self._preprocessor = preprocessor

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def preprocessor(self) -> Preprocessor:
        return self._preprocessor

    def check_features(self, names: tuple[str, ...]) -> None:
        """Refuse to score a matrix whose columns are not the fitted ones.

        The train/serve skew guard. Order is checked as strictly as membership,
        because an estimator fitted on positional arrays will happily score
        reordered columns and return a confident, meaningless number.
        """
        if names == self._feature_names:
            return
        expected, received = set(self._feature_names), set(names)
        raise FeatureSchemaMismatchError(
            self.algorithm,
            missing=tuple(sorted(expected - received)),
            unexpected=tuple(sorted(received - expected)),
            reordered=expected == received,
        )

    def prepare(self, matrix: np.ndarray) -> np.ndarray:
        return self._preprocessor.transform(matrix)

    @abstractmethod
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """Probabilities for a classifier, expected values for a regressor."""

    def contributions(self, matrix: np.ndarray) -> np.ndarray | None:
        """Per-feature push on each prediction, or None when unsupported.

        Returned in model output units, which differ by family: log-odds for the
        linear classifier, points for the regressors. Comparable within one
        prediction, which is all the ``top_factors`` field claims.
        """
        return None

    @abstractmethod
    def to_params(self) -> dict[str, Any]:
        """JSON-serialisable state, excluding the shared header fields."""

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        """Rebuild from :meth:`to_params` output."""

    def serialise(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "kind": str(self.kind),
            "feature_names": list(self._feature_names),
            "preprocessor": self._preprocessor.to_params(),
            "params": self.to_params(),
        }


_MODEL_TYPES: dict[str, type[FittedModel]] = {}


def register_model(model_type_: type[FittedModel]) -> type[FittedModel]:
    """Record a model class so artifacts naming it can be rebuilt.

    Registration is explicit rather than discovered by scanning, so loading an
    artifact can never import something unexpected. Only names in this table are
    constructible from a file on disk.
    """
    existing = _MODEL_TYPES.get(model_type_.algorithm)
    if existing is not None and existing is not model_type_:
        raise ValueError(f"Duplicate model algorithm {model_type_.algorithm!r}")
    _MODEL_TYPES[model_type_.algorithm] = model_type_
    return model_type_


def model_type(algorithm: str) -> type[FittedModel]:
    try:
        return _MODEL_TYPES[algorithm]
    except KeyError:
        known = ", ".join(sorted(_MODEL_TYPES)) or "none"
        raise ModelArtifactInvalidError(
            algorithm, f"unknown algorithm; registered algorithms are: {known}"
        ) from None


def load_model(payload: dict[str, Any]) -> FittedModel:
    """Rebuild a model from its serialised form."""
    try:
        algorithm = str(payload["algorithm"])
        names = tuple(str(name) for name in payload["feature_names"])
        preprocessor = Preprocessor.from_params(payload["preprocessor"])
        params = dict(payload["params"])
    except (KeyError, TypeError) as exc:
        raise ModelArtifactInvalidError("<payload>", f"malformed model section: {exc}") from exc

    built = model_type(algorithm).from_params(params)
    built._feature_names = names
    built._preprocessor = preprocessor
    return built


def registered_algorithms() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_TYPES))
