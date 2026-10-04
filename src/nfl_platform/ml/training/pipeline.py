"""The training pipeline.

One function per prediction problem rather than a configurable framework. Two
problems do not justify an abstraction, and a generic trainer would hide the
decisions that matter -- which metric selects the model, what the baseline is,
when the test period is allowed to be touched -- behind configuration.

The order of operations is the part to read, because it is what keeps the
evaluation honest:

1. Build features and drop rows with too little history.
2. Derive the split from the snapshot's complete seasons.
3. Walk forward across validation folds for every model, including baselines.
4. Decide promotion from those folds alone.
5. Refit the winner on train + validation.
6. Calibrate on out-of-fold validation predictions.
7. Score the frozen test period **once**, after everything above is settled.

Step 7 happens last and is counted. ``test_evaluations`` in the metadata records
how many times the test block was scored; anything other than one means it has
informed a decision and is no longer a held-out estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.ml.datasets.builder import TrainingDataset, content_hash
from nfl_platform.ml.datasets.splits import SplitPlan, complete_seasons, plan_splits
from nfl_platform.ml.evaluation.backtest import (
    BacktestResult,
    PromotionDecision,
    decide_promotion,
    walk_forward,
)
from nfl_platform.ml.evaluation.metrics import classification_metrics, regression_metrics
from nfl_platform.ml.features import fantasy as fantasy_features
from nfl_platform.ml.features import games as game_features
from nfl_platform.ml.models.base import FittedModel
from nfl_platform.ml.models.baselines import FeaturePassthrough, HomeFieldBaseline, SpreadBenchmark
from nfl_platform.ml.models.boosted import BASE_PARAMS, LightGbmClassifier, LightGbmRegressor
from nfl_platform.ml.models.calibration import calibrate
from nfl_platform.ml.models.linear import LogisticRegressionModel, RidgeRegressionModel
from nfl_platform.ml.problems import FANTASY_PROBLEM, GAME_PROBLEM
from nfl_platform.ml.registry.artifacts import ModelArtifact, ModelMetadata, build_metadata
from nfl_platform.schemas.prediction import ModelRole

logger = get_logger(__name__)

GAME_PRIMARY_METRIC = "brier_score"
FANTASY_PRIMARY_METRIC = "mae"

#: Fantasy rank correlation is computed inside (season, week, position) groups,
#: which is the comparison a start/sit decision actually makes.
_FANTASY_GROUP_COLUMNS = ("season", "week", "position")


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Everything one training run produced."""

    problem: str
    artifacts: tuple[ModelArtifact, ...]
    default_version: str
    promotion: PromotionDecision
    plan: SplitPlan
    report: dict[str, Any] = field(default_factory=dict)

    def artifact(self, version: str) -> ModelArtifact:
        for artifact in self.artifacts:
            if artifact.metadata.version == version:
                return artifact
        raise KeyError(version)


def _group_keys(frame: pl.DataFrame, columns: tuple[str, ...]) -> list[str]:
    present = [c for c in columns if c in frame.columns]
    if not present:
        return ["all"]
    return (
        frame.select(pl.concat_str([pl.col(c).cast(pl.Utf8) for c in present], separator="|"))
        .to_series()
        .to_list()
    )


# --- game prediction ----------------------------------------------------------


def train_game_models(
    features: pl.DataFrame,
    *,
    settings: Settings | None = None,
    snapshot: str | None = None,
    command: str | None = None,
    test_seasons: int = 2,
    validation_seasons: int = 4,
) -> TrainingOutcome:
    """Train, evaluate and version the game-prediction models.

    Produces four artifacts: the home-field baseline, logistic regression,
    LightGBM, and the spread benchmark. All four are written, because a baseline
    that cannot be served is not a baseline the promotion rule can fall back to,
    and a benchmark nobody can reproduce is just a number in a document.
    """
    resolved = settings or get_settings()
    seed = resolved.ml_random_seed

    rows = game_features.training_rows(features)
    if rows.is_empty():
        raise ValueError("No usable game rows after filtering; check the snapshot coverage.")

    dataset = TrainingDataset(rows, game_features.GAME_FEATURES, game_features.LABEL)
    plan = plan_splits(
        [s for s in complete_seasons(features) if s in dataset.seasons],
        test_seasons=test_seasons,
        validation_seasons=validation_seasons,
    )

    def score(actual: np.ndarray, predicted: np.ndarray, _: TrainingDataset) -> dict[str, Any]:
        return classification_metrics(actual, predicted).as_dict()

    def fit_baseline(train: TrainingDataset) -> FittedModel:
        return HomeFieldBaseline.fit(train.feature_names, train.matrix(), train.targets())

    def fit_logistic(train: TrainingDataset) -> FittedModel:
        return LogisticRegressionModel.fit(
            train.feature_names, train.matrix(), train.targets(), seed=seed
        )

    def fit_boosted(train: TrainingDataset) -> FittedModel:
        return LightGbmClassifier.fit(
            train.feature_names, train.matrix(), train.targets(), seed=seed
        )

    baseline_back = walk_forward(dataset, plan, fit_baseline, score)
    logistic_back = walk_forward(dataset, plan, fit_logistic, score)
    boosted_back = walk_forward(dataset, plan, fit_boosted, score)

    # Two comparisons, in the order the milestone asks for them: is the modelled
    # answer better than the trivial one, and is the complex model better than
    # the simple one.
    versus_trivial = decide_promotion(
        baseline_back,
        logistic_back,
        metric=GAME_PRIMARY_METRIC,
        lower_is_better=True,
        simple_name="home-field baseline",
        candidate_name="logistic regression",
    )
    promotion = decide_promotion(
        logistic_back,
        boosted_back,
        metric=GAME_PRIMARY_METRIC,
        lower_is_better=True,
        simple_name="logistic regression",
        candidate_name="LightGBM",
    )

    winner = "v1_lightgbm" if promotion.promoted else "v1_logistic"
    fit_rows = dataset.filter_seasons(plan.fit_seasons)
    test_rows = dataset.filter_seasons(plan.test_seasons)

    final_simple = fit_logistic(fit_rows)
    final_boosted = fit_boosted(fit_rows)
    final_baseline = HomeFieldBaseline.fit(
        fit_rows.feature_names, fit_rows.matrix(), fit_rows.targets()
    )

    chosen_raw = final_boosted if promotion.promoted else final_simple
    chosen_backtest = boosted_back if promotion.promoted else logistic_back
    chosen, calibration_method = calibrate(
        chosen_raw, chosen_backtest.predictions, chosen_backtest.targets, seed=seed
    )

    benchmark = SpreadBenchmark.fit(
        fit_rows.column("spread_line").astype(np.float64), fit_rows.targets()
    )

    baseline_validation = classification_metrics(
        baseline_back.targets, baseline_back.predictions
    ).as_dict()
    benchmark_validation: dict[str, Any] = {}
    if benchmark.is_informative and len(test_rows):
        benchmark_validation = classification_metrics(
            test_rows.targets(), benchmark.predict(test_rows.column("spread_line"))
        ).as_dict()

    def test_metrics(model: FittedModel) -> dict[str, Any]:
        if not len(test_rows):
            return {}
        baseline_probabilities = final_baseline.predict(test_rows.matrix())
        return classification_metrics(
            test_rows.targets(),
            model.predict(test_rows.matrix()),
            baseline_prob=baseline_probabilities,
        ).as_dict()

    def train_metrics(model: FittedModel) -> dict[str, Any]:
        return classification_metrics(
            fit_rows.targets(), model.predict(fit_rows.matrix())
        ).as_dict()

    def validation_metrics(result: BacktestResult) -> dict[str, Any]:
        return classification_metrics(
            result.targets, result.predictions, baseline_prob=baseline_back.predictions
        ).as_dict()

    shared = {
        "problem": GAME_PROBLEM,
        "feature_schema_version": game_features.GAME_FEATURE_SCHEMA_VERSION,
        "feature_names": dataset.feature_names,
        "train_seasons": plan.train_seasons,
        "validation_seasons": plan.validation_seasons,
        "test_seasons": plan.test_seasons,
        "fit_seasons": plan.fit_seasons,
        "primary_metric": GAME_PRIMARY_METRIC,
        "random_seed": seed,
        "data_snapshot": snapshot,
        "training_rows": len(fit_rows),
        "feature_matrix_hash": content_hash(fit_rows.matrix(), fit_rows.targets()),
        "split_rationale": plan.rationale,
        "baseline_metrics": baseline_validation,
        "benchmark_metrics": benchmark_validation,
        "command": command,
        "test_evaluations": 1 if len(test_rows) else 0,
    }

    artifacts = (
        ModelArtifact(
            model=final_baseline,
            metadata=build_metadata(
                **shared,
                model_id=f"{GAME_PROBLEM}/v1_baseline",
                version="v1_baseline",
                algorithm=final_baseline.algorithm,
                role=str(ModelRole.BASELINE),
                hyperparameters={"home_win_rate": round(final_baseline.rate, 6)},
                train_metrics=train_metrics(final_baseline),
                validation_metrics=validation_metrics(baseline_back),
                validation_folds=[fold.as_dict() for fold in baseline_back.folds],
                test_metrics=test_metrics(final_baseline),
                notes=[
                    "Predicts the training-set home win rate for every game. The floor "
                    "every other model has to clear."
                ],
            ),
        ),
        ModelArtifact(
            model=final_simple,
            metadata=build_metadata(
                **shared,
                model_id=f"{GAME_PROBLEM}/v1_logistic",
                version="v1_logistic",
                algorithm=final_simple.algorithm,
                role=str(ModelRole.CANDIDATE),
                hyperparameters={"penalty": "l2", "C": 1.0, "standardized": True},
                train_metrics=train_metrics(final_simple),
                validation_metrics=validation_metrics(logistic_back),
                validation_folds=[fold.as_dict() for fold in logistic_back.folds],
                test_metrics=test_metrics(final_simple),
                promotion=versus_trivial.as_dict(),
                notes=["L2 logistic regression on standardised home-minus-away differences."],
            ),
        ),
        ModelArtifact(
            model=final_boosted,
            metadata=build_metadata(
                **shared,
                model_id=f"{GAME_PROBLEM}/v1_lightgbm",
                version="v1_lightgbm",
                algorithm=final_boosted.algorithm,
                role=str(ModelRole.CANDIDATE),
                hyperparameters=dict(BASE_PARAMS),
                train_metrics=train_metrics(final_boosted),
                validation_metrics=validation_metrics(boosted_back),
                validation_folds=[fold.as_dict() for fold in boosted_back.folds],
                test_metrics=test_metrics(final_boosted),
                promotion=promotion.as_dict(),
                notes=["Shallow gradient boosting with heavy regularisation."],
            ),
        ),
        ModelArtifact(
            model=benchmark,
            metadata=build_metadata(
                **{**shared, "feature_names": ("spread_line",)},
                model_id=f"{GAME_PROBLEM}/v1_market_benchmark",
                version="v1_market_benchmark",
                algorithm=benchmark.algorithm,
                role=str(ModelRole.BENCHMARK),
                hyperparameters={"source": "closing spread"},
                test_metrics=benchmark_validation,
                notes=[
                    "Reference only. Converts the published closing spread into a "
                    "probability so the football-only model can be read in context. "
                    "Never promoted: restating a betting line is not a prediction this "
                    "platform made, and no market value is an input to any other model.",
                ],
            ),
        ),
    )

    calibrated_artifact = _calibrated_artifact(
        chosen, calibration_method, artifacts, winner, test_metrics(chosen), train_metrics(chosen)
    )
    report = {
        "problem": GAME_PROBLEM,
        "plan": plan.as_dict(),
        "promotion_vs_trivial": versus_trivial.as_dict(),
        "promotion_vs_simple": promotion.as_dict(),
        "default_version": winner,
        "calibration_method": calibration_method,
        "rows": {"fit": len(fit_rows), "test": len(test_rows), "total": len(dataset)},
    }
    return TrainingOutcome(
        problem=GAME_PROBLEM,
        artifacts=(*(a for a in artifacts if a.metadata.version != winner), calibrated_artifact),
        default_version=winner,
        promotion=promotion,
        plan=plan,
        report=report,
    )


def _calibrated_artifact(
    chosen: FittedModel,
    calibration_method: str | None,
    artifacts: tuple[ModelArtifact, ...],
    winner: str,
    test_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
) -> ModelArtifact:
    """Replace the winning artifact's model with its calibrated form.

    The metadata is reused so the two cannot drift apart; only the model, the
    calibration note and the metrics that change under calibration are updated.
    """
    original = next(a for a in artifacts if a.metadata.version == winner)
    payload = original.metadata.as_dict()
    payload["calibration_method"] = calibration_method
    payload["test_metrics"] = test_metrics
    payload["train_metrics"] = train_metrics
    payload["notes"] = [
        *original.metadata.notes,
        (
            f"Probabilities calibrated with {calibration_method} fitted on out-of-fold "
            "validation predictions."
            if calibration_method
            else "Not calibrated: too few validation rows to fit a calibrator without "
            "overfitting it."
        ),
    ]
    return ModelArtifact(model=chosen, metadata=ModelMetadata.from_dict(payload))


# --- fantasy projection -------------------------------------------------------


def train_fantasy_models(
    features: pl.DataFrame,
    *,
    settings: Settings | None = None,
    snapshot: str | None = None,
    command: str | None = None,
    test_seasons: int = 2,
    validation_seasons: int = 4,
) -> TrainingOutcome:
    """Train, evaluate and version the fantasy-projection models.

    Three artifacts: a rolling-average passthrough (the naive floor), ridge
    regression (the simple baseline), and LightGBM (the candidate). The naive
    floor is included because beating a three-game average at weekly fantasy is
    harder than it sounds, and a model that cannot is worth knowing about.
    """
    resolved = settings or get_settings()
    seed = resolved.ml_random_seed

    rows = fantasy_features.training_rows(features)
    if rows.is_empty():
        raise ValueError("No usable player-week rows after filtering; check snapshot coverage.")

    dataset = TrainingDataset(rows, fantasy_features.FANTASY_FEATURES, fantasy_features.LABEL)
    plan = plan_splits(
        [s for s in complete_seasons(features) if s in dataset.seasons],
        test_seasons=test_seasons,
        validation_seasons=validation_seasons,
    )

    def score(
        actual: np.ndarray, predicted: np.ndarray, evaluation: TrainingDataset
    ) -> dict[str, Any]:
        return regression_metrics(
            actual,
            predicted,
            groups=_group_keys(evaluation.frame, _FANTASY_GROUP_COLUMNS),
        ).as_dict()

    def fit_naive(train: TrainingDataset) -> FittedModel:
        return FeaturePassthrough.fit(
            train.feature_names,
            train.matrix(),
            train.targets(),
            feature="ppr_per_game_recent",
        )

    def fit_ridge(train: TrainingDataset) -> FittedModel:
        return RidgeRegressionModel.fit(
            train.feature_names, train.matrix(), train.targets(), seed=seed
        )

    def fit_boosted(train: TrainingDataset) -> FittedModel:
        return LightGbmRegressor.fit(
            train.feature_names, train.matrix(), train.targets(), seed=seed
        )

    naive_back = walk_forward(dataset, plan, fit_naive, score, index_columns=_FANTASY_GROUP_COLUMNS)
    ridge_back = walk_forward(dataset, plan, fit_ridge, score, index_columns=_FANTASY_GROUP_COLUMNS)
    boosted_back = walk_forward(
        dataset, plan, fit_boosted, score, index_columns=_FANTASY_GROUP_COLUMNS
    )

    versus_naive = decide_promotion(
        naive_back,
        ridge_back,
        metric=FANTASY_PRIMARY_METRIC,
        lower_is_better=True,
        simple_name="rolling-average baseline",
        candidate_name="ridge regression",
    )
    promotion = decide_promotion(
        ridge_back,
        boosted_back,
        metric=FANTASY_PRIMARY_METRIC,
        lower_is_better=True,
        simple_name="ridge regression",
        candidate_name="LightGBM",
    )

    winner = "v1_lightgbm" if promotion.promoted else "v1_ridge"
    fit_rows = dataset.filter_seasons(plan.fit_seasons)
    test_rows = dataset.filter_seasons(plan.test_seasons)

    final_naive = FeaturePassthrough.fit(
        fit_rows.feature_names,
        fit_rows.matrix(),
        fit_rows.targets(),
        feature="ppr_per_game_recent",
    )
    final_ridge = fit_ridge(fit_rows)
    final_boosted = fit_boosted(fit_rows)

    def metrics_for(
        model: FittedModel, subset: TrainingDataset, *, baseline: FittedModel | None = None
    ) -> dict[str, Any]:
        if not len(subset):
            return {}
        return regression_metrics(
            subset.targets(),
            model.predict(subset.matrix()),
            groups=_group_keys(subset.frame, _FANTASY_GROUP_COLUMNS),
            baseline_pred=None if baseline is None else baseline.predict(subset.matrix()),
        ).as_dict()

    def validation_metrics(result: BacktestResult) -> dict[str, Any]:
        return regression_metrics(
            result.targets,
            result.predictions,
            groups=_group_keys(result.index, _FANTASY_GROUP_COLUMNS),
            baseline_pred=naive_back.predictions,
        ).as_dict()

    shared = {
        "problem": FANTASY_PROBLEM,
        "feature_schema_version": fantasy_features.FANTASY_FEATURE_SCHEMA_VERSION,
        "feature_names": dataset.feature_names,
        "train_seasons": plan.train_seasons,
        "validation_seasons": plan.validation_seasons,
        "test_seasons": plan.test_seasons,
        "fit_seasons": plan.fit_seasons,
        "primary_metric": FANTASY_PRIMARY_METRIC,
        "random_seed": seed,
        "data_snapshot": snapshot,
        "training_rows": len(fit_rows),
        "feature_matrix_hash": content_hash(fit_rows.matrix(), fit_rows.targets()),
        "split_rationale": plan.rationale,
        "baseline_metrics": validation_metrics(naive_back),
        "command": command,
        "test_evaluations": 1 if len(test_rows) else 0,
        "notes": [
            "Target is PPR points in games the player took an offensive snap in. The "
            "projection is conditional on playing and contains no availability estimate."
        ],
    }

    artifacts = (
        ModelArtifact(
            model=final_naive,
            metadata=build_metadata(
                **shared,
                model_id=f"{FANTASY_PROBLEM}/v1_rolling_average",
                version="v1_rolling_average",
                algorithm=final_naive.algorithm,
                role=str(ModelRole.BASELINE),
                hyperparameters={"feature": final_naive.feature},
                train_metrics=metrics_for(final_naive, fit_rows),
                validation_metrics=validation_metrics(naive_back),
                validation_folds=[fold.as_dict() for fold in naive_back.folds],
                test_metrics=metrics_for(final_naive, test_rows),
            ),
        ),
        ModelArtifact(
            model=final_ridge,
            metadata=build_metadata(
                **shared,
                model_id=f"{FANTASY_PROBLEM}/v1_ridge",
                version="v1_ridge",
                algorithm=final_ridge.algorithm,
                role=str(ModelRole.CANDIDATE),
                hyperparameters={"alpha": 1.0, "standardized": True, "floor": 0.0},
                train_metrics=metrics_for(final_ridge, fit_rows),
                validation_metrics=validation_metrics(ridge_back),
                validation_folds=[fold.as_dict() for fold in ridge_back.folds],
                test_metrics=metrics_for(final_ridge, test_rows, baseline=final_naive),
                promotion=versus_naive.as_dict(),
            ),
        ),
        ModelArtifact(
            model=final_boosted,
            metadata=build_metadata(
                **shared,
                model_id=f"{FANTASY_PROBLEM}/v1_lightgbm",
                version="v1_lightgbm",
                algorithm=final_boosted.algorithm,
                role=str(ModelRole.CANDIDATE),
                hyperparameters=dict(BASE_PARAMS),
                train_metrics=metrics_for(final_boosted, fit_rows),
                validation_metrics=validation_metrics(boosted_back),
                validation_folds=[fold.as_dict() for fold in boosted_back.folds],
                test_metrics=metrics_for(final_boosted, test_rows, baseline=final_naive),
                promotion=promotion.as_dict(),
            ),
        ),
    )

    report = {
        "problem": FANTASY_PROBLEM,
        "plan": plan.as_dict(),
        "promotion_vs_naive": versus_naive.as_dict(),
        "promotion_vs_simple": promotion.as_dict(),
        "default_version": winner,
        "rows": {"fit": len(fit_rows), "test": len(test_rows), "total": len(dataset)},
    }
    return TrainingOutcome(
        problem=FANTASY_PROBLEM,
        artifacts=artifacts,
        default_version=winner,
        promotion=promotion,
        plan=plan,
        report=report,
    )


def summarise(outcome: TrainingOutcome) -> str:
    """A short human-readable summary, printed by the training script."""
    fit = outcome.plan.fit_seasons
    lines = [
        f"{outcome.problem}: default = {outcome.default_version}",
        f"  split: {fit[0]}-{fit[-1]} fit, test {list(outcome.plan.test_seasons)}",
        f"  {outcome.promotion.reason}",
    ]
    for artifact in outcome.artifacts:
        metadata = artifact.metadata
        score = metadata.validation_score
        rendered = "n/a" if score is None or math.isnan(score) else f"{score:.4f}"
        lines.append(
            f"  {metadata.version:<22} {metadata.role:<9} "
            f"validation {metadata.primary_metric}={rendered}"
        )
    return "\n".join(lines)
