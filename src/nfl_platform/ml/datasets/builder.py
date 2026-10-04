"""The bridge from a Polars feature frame to the arrays estimators accept.

Three things happen here and nowhere else, so that each is testable on its own:

*Column order is fixed* by the :class:`~nfl_platform.ml.features.schema.FeatureSet`,
not by whatever order the frame happens to be in. Estimators are fitted on
positional arrays; a frame that reorders itself between training and inference
would score the right model against the wrong columns and report it as success.

*Non-finite values become null.* A division that produced an infinity is a
missing value, not a very large one, and only one of those two readings survives
imputation intact.

*Rows are selected by season*, which is the only selection a time-aware design
needs and the only one that cannot accidentally mix eras.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from nfl_platform.ml.features.schema import FeatureSet


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    """A feature frame bound to the feature set that defines its columns."""

    frame: pl.DataFrame
    feature_set: FeatureSet
    label: str
    season_column: str = "season"

    def __post_init__(self) -> None:
        missing = [name for name in self.feature_set.names if name not in self.frame.columns]
        if missing:
            raise ValueError(f"Frame is missing declared feature(s): {', '.join(missing)}")
        if self.label not in self.frame.columns:
            raise ValueError(f"Frame is missing the label column {self.label!r}")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.feature_set.names

    @property
    def seasons(self) -> tuple[int, ...]:
        values = self.frame.get_column(self.season_column).unique().to_list()
        return tuple(sorted(int(value) for value in values if value is not None))

    def filter_seasons(self, seasons: Sequence[int]) -> TrainingDataset:
        """The subset falling in ``seasons``, preserving every other binding."""
        subset = self.frame.filter(pl.col(self.season_column).is_in(list(seasons)))
        return TrainingDataset(subset, self.feature_set, self.label, self.season_column)

    def select_features(self, feature_set: FeatureSet) -> TrainingDataset:
        """Rebind to a narrower feature set, e.g. the market-free variant."""
        return TrainingDataset(self.frame, feature_set, self.label, self.season_column)

    def matrix(self) -> np.ndarray:
        """Features as a float array, in declared order, with NaN for missing."""
        return to_matrix(self.frame, self.feature_names)

    def targets(self) -> np.ndarray:
        return self.frame.get_column(self.label).cast(pl.Float64).to_numpy().astype(np.float64)

    def column(self, name: str) -> np.ndarray:
        """Any non-feature column as an array, for benchmarks and grouped metrics."""
        return self.frame.get_column(name).to_numpy()

    def __len__(self) -> int:
        return self.frame.height


def to_matrix(frame: pl.DataFrame, names: Sequence[str]) -> np.ndarray:
    """Project ``names`` out of ``frame`` as a float64 matrix with NaN for missing."""
    if not names:
        return np.empty((frame.height, 0), dtype=np.float64)
    projected = frame.select(
        [pl.col(name).cast(pl.Float64, strict=False).alias(name) for name in names]
    )
    matrix = projected.to_numpy().astype(np.float64)
    # An infinity is a failed division, which is missing information rather than
    # an extreme observation. Imputation can handle the first honestly.
    return np.where(np.isfinite(matrix), matrix, np.nan)


def content_hash(matrix: np.ndarray, targets: np.ndarray) -> str:
    """A stable digest of the fitted data, recorded in artifact metadata.

    Lets a later run prove it trained on the same rows rather than asserting it.
    NaNs are canonicalised first, since their bit patterns are not stable across
    platforms and would make an identical dataset hash differently.
    """
    import hashlib

    digest = hashlib.sha256()
    for array in (matrix, targets):
        clean = np.nan_to_num(np.ascontiguousarray(array, dtype=np.float64), nan=-9.87654321e30)
        digest.update(str(clean.shape).encode())
        digest.update(np.round(clean, 8).tobytes())
    return digest.hexdigest()[:16]
