"""What a predictor hands back to the service layer.

A plain dataclass rather than a DTO. The service owns DTO construction, because
it is the service that knows how to resolve a player reference, attach a
schedule row and phrase a caveat. Returning a DTO from here would put half the
presentation logic in the ML layer and make it untestable without the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from nfl_platform.ml.models.base import FittedModel
from nfl_platform.ml.registry.artifacts import ModelMetadata

#: How many feature contributions to surface. Enough to explain, few enough that
#: a reader does not have to triage twenty near-zero values to find the signal.
TOP_FACTORS = 5


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """One scored row, with the context needed to explain and qualify it."""

    value: float
    metadata: ModelMetadata
    feature_values: dict[str, float | None] = field(default_factory=dict)
    contributions: list[tuple[str, float | None, float]] = field(default_factory=list)
    baseline_value: float | None = None
    history: dict[str, int] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def top_contributions(
    model: FittedModel,
    matrix: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    limit: int = TOP_FACTORS,
) -> list[tuple[str, float | None, float]]:
    """The largest-magnitude feature pushes for a single row.

    Returns an empty list when the model family cannot attribute its output,
    which is honest: an empty ``top_factors`` says "this model does not explain
    itself", where a fabricated ranking would say something false.
    """
    raw = model.contributions(matrix)
    if raw is None or raw.size == 0:
        return []
    values = np.asarray(raw, dtype=np.float64)[0]
    row = np.asarray(matrix, dtype=np.float64)[0]
    order = np.argsort(np.abs(values))[::-1][:limit]
    ranked: list[tuple[str, float | None, float]] = []
    for position in order:
        contribution = float(values[position])
        if abs(contribution) < 1e-9:
            continue
        observed = float(row[position]) if np.isfinite(row[position]) else None
        ranked.append((feature_names[position], observed, contribution))
    return ranked


def feature_matrix(row: pl.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    """Project a single feature row into the array the artifact expects.

    Column order comes from the artifact's recorded feature list, never from the
    frame, which is what makes the shared builder and the fitted model agree.
    """
    projected = row.select(
        [pl.col(name).cast(pl.Float64, strict=False).alias(name) for name in names]
    )
    matrix = projected.to_numpy().astype(np.float64)
    return np.where(np.isfinite(matrix), matrix, np.nan)


def optional_float(value: object) -> float | None:
    """Coerce a frame cell to a finite float, or None."""
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
