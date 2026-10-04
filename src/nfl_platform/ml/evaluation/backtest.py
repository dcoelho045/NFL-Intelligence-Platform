"""Walk-forward backtesting and the promotion rule.

Two things live here, and they are related: the harness that produces
out-of-fold predictions without ever letting a fold see its own future, and the
rule that decides whether a more complex model has earned the right to be the
default.

## Why the promotion rule needs a threshold

Comparing two models on a single pooled validation score and shipping whichever
number is lower is how projects end up with a gradient-boosted model that is
worse in production than the logistic regression it replaced. A difference of
0.003 Brier across four seasons is indistinguishable from which seasons happened
to land in the validation window.

So the comparison is made fold by fold, and the margin a candidate has to clear
is the standard deviation of the per-fold differences. If the candidate's
advantage is smaller than the variation in that advantage, it has not been
demonstrated, and the simpler model stays the default. That rule is written
before any model is trained, which is the only time it can be written honestly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from nfl_platform.config.logging import get_logger
from nfl_platform.ml.datasets.builder import TrainingDataset
from nfl_platform.ml.datasets.splits import SplitPlan
from nfl_platform.ml.models.base import FittedModel

logger = get_logger(__name__)

FitFunction = Callable[[TrainingDataset], FittedModel]
ScoreFunction = Callable[[np.ndarray, np.ndarray, TrainingDataset], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class FoldResult:
    """One expanding-window fold's outcome."""

    name: str
    train_rows: int
    eval_rows: int
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.name,
            "train_rows": self.train_rows,
            "eval_rows": self.eval_rows,
            "metrics": self.metrics,
        }


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Out-of-fold predictions plus the per-fold record that produced them."""

    folds: tuple[FoldResult, ...]
    predictions: np.ndarray
    targets: np.ndarray
    index: pl.DataFrame
    """Identity columns of the evaluated rows, aligned with ``predictions``."""
    notes: list[str] = field(default_factory=list)

    def fold_metric(self, metric: str) -> list[float]:
        values: list[float] = []
        for fold in self.folds:
            value = fold.metrics.get(metric)
            values.append(float(value) if isinstance(value, int | float) else math.nan)
        return values

    def as_dict(self) -> dict[str, Any]:
        return {"folds": [fold.as_dict() for fold in self.folds], "notes": self.notes}


def walk_forward(
    dataset: TrainingDataset,
    plan: SplitPlan,
    fit: FitFunction,
    score: ScoreFunction,
    *,
    index_columns: Sequence[str] = ("season", "week"),
) -> BacktestResult:
    """Fit on everything before each validation season, then score that season.

    The fold boundary is enforced by :class:`~nfl_platform.ml.datasets.splits.TimeFold`
    itself, which refuses to exist if its training seasons reach the season it
    is supposed to evaluate. The harness therefore cannot construct an invalid
    fold even if asked to.
    """
    results: list[FoldResult] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    index_frames: list[pl.DataFrame] = []
    notes: list[str] = []

    for fold in plan.folds:
        train = dataset.filter_seasons(fold.train_seasons)
        evaluation = dataset.filter_seasons(fold.eval_seasons)
        if len(train) == 0 or len(evaluation) == 0:
            notes.append(
                f"Fold {fold.name} skipped: {len(train)} train / {len(evaluation)} eval rows."
            )
            continue

        model = fit(train)
        predicted = model.predict(evaluation.matrix())
        actual = evaluation.targets()

        results.append(
            FoldResult(
                name=fold.name,
                train_rows=len(train),
                eval_rows=len(evaluation),
                metrics=score(actual, predicted, evaluation),
            )
        )
        predictions.append(predicted)
        targets.append(actual)
        kept = [c for c in index_columns if c in evaluation.frame.columns]
        index_frames.append(evaluation.frame.select(kept))

    if not predictions:
        return BacktestResult(
            folds=(),
            predictions=np.empty(0),
            targets=np.empty(0),
            index=pl.DataFrame(),
            notes=[*notes, "No fold produced predictions."],
        )

    return BacktestResult(
        folds=tuple(results),
        predictions=np.concatenate(predictions),
        targets=np.concatenate(targets),
        index=pl.concat(index_frames, how="vertical"),
        notes=notes,
    )


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Whether the stronger model replaces the simpler one, and on what evidence."""

    promoted: bool
    metric: str
    simple_score: float
    candidate_score: float
    improvement: float
    required_margin: float
    fold_differences: list[float]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "metric": self.metric,
            "simple_score": self.simple_score,
            "candidate_score": self.candidate_score,
            "improvement": self.improvement,
            "required_margin": self.required_margin,
            "fold_differences": self.fold_differences,
            "reason": self.reason,
        }


def decide_promotion(
    simple: BacktestResult,
    candidate: BacktestResult,
    *,
    metric: str,
    lower_is_better: bool,
    simple_name: str = "baseline",
    candidate_name: str = "candidate",
) -> PromotionDecision:
    """Promote the candidate only if it beats the simpler model by more than fold noise.

    Args:
        simple: Backtest of the model that is currently the default.
        candidate: Backtest of the more complex alternative.
        metric: Key in the fold metrics to compare on.
        lower_is_better: True for Brier and MAE, false for accuracy or rank
            correlation.
        simple_name: Label for the reason string.
        candidate_name: Label for the reason string.

    Returns:
        The decision and the numbers behind it, recorded in artifact metadata so
        the choice can be re-examined without retraining.
    """
    simple_folds = np.asarray(simple.fold_metric(metric), dtype=np.float64)
    candidate_folds = np.asarray(candidate.fold_metric(metric), dtype=np.float64)
    usable = np.isfinite(simple_folds) & np.isfinite(candidate_folds)

    sign = 1.0 if lower_is_better else -1.0
    simple_score = float(np.nanmean(simple_folds)) if usable.any() else math.nan
    candidate_score = float(np.nanmean(candidate_folds)) if usable.any() else math.nan

    if usable.sum() < 2:
        return PromotionDecision(
            promoted=False,
            metric=metric,
            simple_score=simple_score,
            candidate_score=candidate_score,
            improvement=math.nan,
            required_margin=math.inf,
            fold_differences=[],
            reason=(
                f"Fewer than two comparable folds, so no difference between {candidate_name} "
                f"and {simple_name} can be distinguished from noise. Keeping {simple_name}."
            ),
        )

    differences = (simple_folds[usable] - candidate_folds[usable]) * sign
    improvement = float(differences.mean())
    # Population standard deviation of the per-fold advantage. With four folds a
    # sample standard deviation would be jumpy enough to be its own source of
    # randomness in the decision.
    margin = float(differences.std())
    promoted = bool(improvement > margin and improvement > 0)

    if promoted:
        reason = (
            f"{candidate_name} improves {metric} by {improvement:.4f} on average across "
            f"{int(usable.sum())} validation folds, which exceeds the fold-to-fold "
            f"variation of {margin:.4f}. Promoted over {simple_name}."
        )
    else:
        reason = (
            f"{candidate_name} changes {metric} by {improvement:.4f} across "
            f"{int(usable.sum())} validation folds, which does not exceed the fold-to-fold "
            f"variation of {margin:.4f}. The improvement is not distinguishable from "
            f"noise, so {simple_name} stays the default."
        )

    logger.info("Promotion decision", extra={"metric": metric, "promoted": promoted})
    return PromotionDecision(
        promoted=promoted,
        metric=metric,
        simple_score=simple_score,
        candidate_score=candidate_score,
        improvement=improvement,
        required_margin=margin,
        fold_differences=[float(value) for value in differences],
        reason=reason,
    )
