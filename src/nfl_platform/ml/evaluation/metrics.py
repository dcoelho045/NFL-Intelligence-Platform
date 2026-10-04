"""Evaluation metrics, implemented directly on NumPy.

scikit-learn has all of these. They are written out anyway for two reasons.

*The quarantine.* ``sklearn`` is confined to :mod:`nfl_platform.ml.models` so
that "which code can fit a model" stays answerable by naming one package. Having
the evaluation layer import it to compute an average would dissolve that line
for no benefit.

*The tests mean something.* A test asserting ``roc_auc`` matches
``sklearn.roc_auc_score`` only proves two libraries agree. A test asserting it
matches a hand-worked value proves the number is right, which is the claim the
milestone actually rests on.

Scoring-rule choices worth stating:

``brier_score`` is primary for the game model, not accuracy. Accuracy throws
away the whole probability and cannot tell a model that says 51% from one that
says 99%, which for a win-probability product is most of the information.

``brier_skill_score`` is reported against the simple baseline rather than
against a constant 0.5, because beating a coin flip at predicting NFL games is
not an achievement -- home teams win about 57% of the time, so the honest bar is
the baseline that already knows that.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

#: Probabilities are clipped before taking a logarithm. Without this one
#: confident miss sends log loss to infinity and the mean becomes meaningless.
_EPSILON = 1e-15


def _as_float(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).ravel()


def _paired(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    truth, prediction = _as_float(y_true), _as_float(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: {truth.shape} vs {prediction.shape}")
    finite = np.isfinite(truth) & np.isfinite(prediction)
    return truth[finite], prediction[finite]


# --- classification -----------------------------------------------------------


def accuracy(y_true: Any, y_prob: Any, *, threshold: float = 0.5) -> float:
    truth, prob = _paired(y_true, y_prob)
    if truth.size == 0:
        return float("nan")
    return float(np.mean((prob >= threshold).astype(np.float64) == truth))


def log_loss(y_true: Any, y_prob: Any) -> float:
    truth, prob = _paired(y_true, y_prob)
    if truth.size == 0:
        return float("nan")
    clipped = np.clip(prob, _EPSILON, 1 - _EPSILON)
    return float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)))


def brier_score(y_true: Any, y_prob: Any) -> float:
    """Mean squared error of the probability. Lower is better; 0.25 is a coin flip."""
    truth, prob = _paired(y_true, y_prob)
    if truth.size == 0:
        return float("nan")
    return float(np.mean((prob - truth) ** 2))


def brier_skill_score(y_true: Any, y_prob: Any, y_reference: Any) -> float:
    """Fractional Brier improvement over a reference forecast.

    Positive means better than the reference, zero means indistinguishable,
    negative means worse. This is the number the promotion rule reads.
    """
    model = brier_score(y_true, y_prob)
    reference = brier_score(y_true, y_reference)
    if not np.isfinite(model) or not np.isfinite(reference) or reference <= 0:
        return float("nan")
    return float(1.0 - model / reference)


def roc_auc(y_true: Any, y_prob: Any) -> float:
    """Area under the ROC curve, via the rank-sum identity with tie correction.

    Equivalent to the probability that a randomly chosen positive outranks a
    randomly chosen negative, with ties counting half.
    """
    truth, prob = _paired(y_true, y_prob)
    positives = truth == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(prob)
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged, which is what the AUC identity needs."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = (start + 1 + index) / 2
            start = index
    return ranks


def reliability_table(y_true: Any, y_prob: Any, *, bins: int = 10) -> list[dict[str, float]]:
    """Observed frequency against predicted probability, bin by bin.

    The raw material of a calibration plot, and the only way to see *where* a
    model is miscalibrated. A single ECE number hides whether the error is a
    uniform shift or confident predictions being wrong.
    """
    truth, prob = _paired(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    table: list[dict[str, float]] = []
    for lower, upper in itertools.pairwise(edges):
        in_bin = (prob >= lower) & (prob < upper if upper < 1.0 else prob <= upper)
        count = int(in_bin.sum())
        table.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_predicted": float(prob[in_bin].mean()) if count else float("nan"),
                "observed_rate": float(truth[in_bin].mean()) if count else float("nan"),
            }
        )
    return table


def expected_calibration_error(y_true: Any, y_prob: Any, *, bins: int = 10) -> float:
    """Count-weighted mean gap between predicted probability and observed rate."""
    truth, _ = _paired(y_true, y_prob)
    if truth.size == 0:
        return float("nan")
    total = 0.0
    for row in reliability_table(y_true, y_prob, bins=bins):
        if row["count"]:
            total += row["count"] * abs(row["mean_predicted"] - row["observed_rate"])
    return float(total / truth.size)


# --- regression ---------------------------------------------------------------


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    truth, prediction = _paired(y_true, y_pred)
    if truth.size == 0:
        return float("nan")
    return float(np.mean(np.abs(truth - prediction)))


def root_mean_squared_error(y_true: Any, y_pred: Any) -> float:
    truth, prediction = _paired(y_true, y_pred)
    if truth.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def r_squared(y_true: Any, y_pred: Any) -> float:
    truth, prediction = _paired(y_true, y_pred)
    if truth.size < 2:
        return float("nan")
    total = float(np.sum((truth - truth.mean()) ** 2))
    if total <= 0:
        return float("nan")
    return float(1.0 - np.sum((truth - prediction) ** 2) / total)


def spearman(y_true: Any, y_pred: Any) -> float:
    """Rank correlation: Pearson applied to average ranks."""
    truth, prediction = _paired(y_true, y_pred)
    if truth.size < 2:
        return float("nan")
    rank_truth, rank_pred = _average_ranks(truth), _average_ranks(prediction)
    std_truth, std_pred = rank_truth.std(), rank_pred.std()
    if std_truth == 0 or std_pred == 0:
        return float("nan")
    centred_truth = rank_truth - rank_truth.mean()
    centred_pred = rank_pred - rank_pred.mean()
    return float(np.mean(centred_truth * centred_pred) / (std_truth * std_pred))


def grouped_spearman(
    y_true: Any, y_pred: Any, groups: Sequence[Any], *, min_group_size: int = 3
) -> float:
    """Mean within-group rank correlation, weighted by group size.

    The metric that corresponds to the actual fantasy decision. Pooled
    correlation is dominated by the easy comparison between a starting running
    back and a backup tight end; nobody is choosing between those. Ranking four
    plausible flex options inside one week is the task, so the correlation is
    computed inside each week-position group and then averaged.
    """
    truth, prediction = _as_float(y_true), _as_float(y_pred)
    keys = np.asarray(groups, dtype=object)
    if truth.size == 0 or keys.size != truth.size:
        return float("nan")

    weighted, total = 0.0, 0
    for key in np.unique(keys):
        mask = keys == key
        if int(mask.sum()) < min_group_size:
            continue
        value = spearman(truth[mask], prediction[mask])
        if np.isfinite(value):
            weighted += value * int(mask.sum())
            total += int(mask.sum())
    return float(weighted / total) if total else float("nan")


# --- bundles ------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationMetrics:
    """Everything the game model is judged on, for one set of rows."""

    rows: int
    accuracy: float
    log_loss: float
    brier_score: float
    roc_auc: float
    expected_calibration_error: float
    base_rate: float
    brier_skill_vs_baseline: float | None = None
    reliability: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionMetrics:
    """Everything the fantasy model is judged on, for one set of rows."""

    rows: int
    mae: float
    rmse: float
    r_squared: float
    spearman: float
    within_week_spearman: float | None = None
    mae_improvement_vs_baseline: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def classification_metrics(
    y_true: Any,
    y_prob: Any,
    *,
    baseline_prob: Any = None,
    bins: int = 10,
) -> ClassificationMetrics:
    truth, prob = _paired(y_true, y_prob)
    return ClassificationMetrics(
        rows=int(truth.size),
        accuracy=accuracy(truth, prob),
        log_loss=log_loss(truth, prob),
        brier_score=brier_score(truth, prob),
        roc_auc=roc_auc(truth, prob),
        expected_calibration_error=expected_calibration_error(truth, prob, bins=bins),
        base_rate=float(truth.mean()) if truth.size else float("nan"),
        brier_skill_vs_baseline=(
            None if baseline_prob is None else brier_skill_score(y_true, y_prob, baseline_prob)
        ),
        reliability=reliability_table(truth, prob, bins=bins),
    )


def regression_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    groups: Sequence[Any] | None = None,
    baseline_pred: Any = None,
) -> RegressionMetrics:
    truth, prediction = _paired(y_true, y_pred)
    improvement: float | None = None
    if baseline_pred is not None:
        baseline_mae = mean_absolute_error(y_true, baseline_pred)
        model_mae = mean_absolute_error(y_true, y_pred)
        if np.isfinite(baseline_mae) and baseline_mae > 0:
            improvement = float(1.0 - model_mae / baseline_mae)
    return RegressionMetrics(
        rows=int(truth.size),
        mae=mean_absolute_error(truth, prediction),
        rmse=root_mean_squared_error(truth, prediction),
        r_squared=r_squared(truth, prediction),
        spearman=spearman(truth, prediction),
        within_week_spearman=(None if groups is None else grouped_spearman(y_true, y_pred, groups)),
        mae_improvement_vs_baseline=improvement,
    )
