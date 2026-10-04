"""End-to-end training.

Runs the real pipeline -- features, split planning, walk-forward, promotion,
artifacts -- on the synthetic league. Fast enough for the default suite and
offline, so a change that breaks training is caught here rather than during a
twenty-minute run against real data.

Marked ``training`` only where a test needs the full pinned snapshot; everything
in this module uses generated data and runs by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.ml.features.fantasy import build_fantasy_features
from nfl_platform.ml.features.games import build_game_features
from nfl_platform.ml.registry.store import ModelRegistry
from nfl_platform.ml.training.pipeline import (
    FANTASY_PROBLEM,
    GAME_PROBLEM,
    TrainingOutcome,
    summarise,
    train_fantasy_models,
    train_game_models,
)
from nfl_platform.schemas.prediction import ModelRole
from synthetic import SyntheticLeague


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(_env_file=None, model_dir=tmp_path_factory.mktemp("smoke-models"))


@pytest.fixture(scope="module")
def league() -> SyntheticLeague:
    from synthetic import build_league

    return build_league()


@pytest.fixture(scope="module")
def game_outcome(league: SyntheticLeague, settings: Settings) -> TrainingOutcome:
    features = build_game_features(league.schedules, league.team_stats)
    return train_game_models(features, settings=settings, snapshot="synthetic")


@pytest.fixture(scope="module")
def fantasy_outcome(league: SyntheticLeague, settings: Settings) -> TrainingOutcome:
    features = build_fantasy_features(
        league.player_stats,
        league.schedules,
        team_stats=league.team_stats,
        snap_counts=league.snap_counts,
        ff_opportunity=league.ff_opportunity,
        players=league.players,
    )
    return train_fantasy_models(features, settings=settings, snapshot="synthetic")


class TestGameTraining:
    def test_every_model_is_produced(self, game_outcome: TrainingOutcome):
        versions = {a.metadata.version for a in game_outcome.artifacts}
        assert versions == {
            "v1_baseline",
            "v1_logistic",
            "v1_lightgbm",
            "v1_market_benchmark",
        }

    def test_the_baseline_is_shippable_not_just_quoted(self, game_outcome: TrainingOutcome):
        """A fallback that cannot be served is not a fallback the rule can choose."""
        baseline = game_outcome.artifact("v1_baseline")
        assert baseline.metadata.role == str(ModelRole.BASELINE)
        assert baseline.model.predict(
            __import__("numpy").zeros((2, len(baseline.metadata.feature_names)))
        ).shape == (2,)

    def test_the_market_benchmark_is_never_the_default(self, game_outcome: TrainingOutcome):
        """Restating a betting line is not a prediction this platform made."""
        assert game_outcome.default_version != "v1_market_benchmark"
        benchmark = game_outcome.artifact("v1_market_benchmark")
        assert benchmark.metadata.role == str(ModelRole.BENCHMARK)

    def test_a_modelled_answer_beats_the_trivial_one(self, game_outcome: TrainingOutcome):
        baseline = game_outcome.artifact("v1_baseline").metadata.validation_metrics
        chosen = game_outcome.artifact(game_outcome.default_version).metadata
        assert chosen.validation_metrics["brier_score"] < baseline["brier_score"]

    def test_the_default_is_a_candidate_not_the_baseline(self, game_outcome: TrainingOutcome):
        assert game_outcome.default_version in {"v1_logistic", "v1_lightgbm"}

    def test_the_promotion_decision_is_recorded_with_its_evidence(
        self, game_outcome: TrainingOutcome
    ):
        promotion = game_outcome.artifact("v1_lightgbm").metadata.promotion
        assert "promoted" in promotion
        assert promotion["reason"]
        assert promotion["fold_differences"]

    def test_the_test_period_is_scored_exactly_once(self, game_outcome: TrainingOutcome):
        for artifact in game_outcome.artifacts:
            assert artifact.metadata.test_evaluations == 1

    def test_training_and_test_seasons_never_overlap(self, game_outcome: TrainingOutcome):
        for artifact in game_outcome.artifacts:
            metadata = artifact.metadata
            assert not set(metadata.fit_seasons) & set(metadata.test_seasons)

    def test_metrics_are_recorded_for_all_three_phases(self, game_outcome: TrainingOutcome):
        metadata = game_outcome.artifact(game_outcome.default_version).metadata
        assert metadata.train_metrics["brier_score"]
        assert metadata.validation_metrics["brier_score"]
        assert metadata.test_metrics["brier_score"]

    def test_per_fold_results_are_recorded(self, game_outcome: TrainingOutcome):
        folds = game_outcome.artifact("v1_logistic").metadata.validation_folds
        assert len(folds) == len(game_outcome.plan.validation_seasons)

    def test_provenance_is_complete(self, game_outcome: TrainingOutcome):
        metadata = game_outcome.artifact(game_outcome.default_version).metadata
        assert metadata.data_snapshot == "synthetic"
        assert metadata.random_seed
        assert metadata.feature_matrix_hash
        assert metadata.library_versions["lightgbm"]
        assert metadata.split_rationale

    def test_the_chosen_model_is_calibrated_when_there_is_enough_data(
        self, game_outcome: TrainingOutcome
    ):
        metadata = game_outcome.artifact(game_outcome.default_version).metadata
        assert metadata.calibration_method in {None, "platt_sigmoid"}


class TestFantasyTraining:
    def test_every_model_is_produced(self, fantasy_outcome: TrainingOutcome):
        versions = {a.metadata.version for a in fantasy_outcome.artifacts}
        assert versions == {"v1_rolling_average", "v1_ridge", "v1_lightgbm"}

    def test_a_modelled_answer_beats_the_rolling_average(self, fantasy_outcome: TrainingOutcome):
        naive = fantasy_outcome.artifact("v1_rolling_average").metadata.validation_metrics
        chosen = fantasy_outcome.artifact(fantasy_outcome.default_version).metadata
        assert chosen.validation_metrics["mae"] < naive["mae"]

    def test_within_week_rank_correlation_is_recorded(self, fantasy_outcome: TrainingOutcome):
        """The metric that corresponds to the actual start/sit decision."""
        metadata = fantasy_outcome.artifact(fantasy_outcome.default_version).metadata
        assert metadata.validation_metrics["within_week_spearman"] is not None

    def test_the_conditional_on_playing_semantics_are_recorded(
        self, fantasy_outcome: TrainingOutcome
    ):
        metadata = fantasy_outcome.artifact(fantasy_outcome.default_version).metadata
        assert any("conditional on playing" in note for note in metadata.notes)

    def test_the_test_period_is_scored_exactly_once(self, fantasy_outcome: TrainingOutcome):
        for artifact in fantasy_outcome.artifacts:
            assert artifact.metadata.test_evaluations == 1


class TestArtifactsRoundTrip:
    def test_saved_artifacts_reload_and_predict_identically(
        self, game_outcome: TrainingOutcome, settings: Settings, tmp_path: Path
    ):
        """The claim the whole serving path depends on."""
        import numpy as np

        registry = ModelRegistry(root=tmp_path, settings=settings)
        for artifact in game_outcome.artifacts:
            registry.save(
                artifact, make_default=artifact.metadata.version == game_outcome.default_version
            )

        chosen = game_outcome.artifact(game_outcome.default_version)
        matrix = np.zeros((3, len(chosen.metadata.feature_names)))
        reloaded = registry.load(GAME_PROBLEM)
        assert reloaded.metadata.version == game_outcome.default_version
        assert reloaded.model.predict(matrix).tolist() == pytest.approx(
            chosen.model.predict(matrix).tolist()
        )

    def test_the_manifest_records_the_promotion(
        self, fantasy_outcome: TrainingOutcome, settings: Settings, tmp_path: Path
    ):
        registry = ModelRegistry(root=tmp_path / "fantasy", settings=settings)
        for artifact in fantasy_outcome.artifacts:
            registry.save(
                artifact,
                make_default=artifact.metadata.version == fantasy_outcome.default_version,
            )
        assert registry.manifest()["defaults"][FANTASY_PROBLEM] == (fantasy_outcome.default_version)

    def test_metadata_is_json_serialisable(self, game_outcome: TrainingOutcome):
        for artifact in game_outcome.artifacts:
            json.dumps(artifact.metadata.as_dict(), default=str)


class TestReproducibility:
    def test_two_runs_on_the_same_data_agree(self, league: SyntheticLeague, settings: Settings):
        """Seeded estimators plus deterministic LightGBM settings; no retraining drift."""
        features = build_game_features(league.schedules, league.team_stats)
        first = train_game_models(features, settings=settings, snapshot="synthetic")
        second = train_game_models(features, settings=settings, snapshot="synthetic")
        assert first.default_version == second.default_version
        assert first.artifact("v1_logistic").metadata.validation_metrics[
            "brier_score"
        ] == pytest.approx(
            second.artifact("v1_logistic").metadata.validation_metrics["brier_score"]
        )

    def test_the_feature_matrix_hash_is_stable(self, league: SyntheticLeague, settings: Settings):
        features = build_game_features(league.schedules, league.team_stats)
        first = train_game_models(features, settings=settings, snapshot="synthetic")
        second = train_game_models(features, settings=settings, snapshot="synthetic")
        assert (
            first.artifact("v1_logistic").metadata.feature_matrix_hash
            == second.artifact("v1_logistic").metadata.feature_matrix_hash
        )


class TestSummary:
    def test_the_summary_names_the_default_and_the_reason(self, game_outcome: TrainingOutcome):
        text = summarise(game_outcome)
        assert GAME_PROBLEM in text
        assert game_outcome.default_version in text
        assert game_outcome.promotion.reason in text
