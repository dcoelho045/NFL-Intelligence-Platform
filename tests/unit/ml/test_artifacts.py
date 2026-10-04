"""Artifact integrity and registry resolution.

The checksum and schema tests are the ones that matter. A damaged artifact that
still parses keeps producing confident numbers from the wrong parameters, and
nothing downstream can tell -- so the only safe behaviour is to refuse to load.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.exceptions import (
    ModelArtifactInvalidError,
    ModelNotAvailableError,
    ModelVersionNotFoundError,
)
from nfl_platform.ml.models.baselines import HomeFieldBaseline
from nfl_platform.ml.registry.artifacts import (
    ARTIFACT_SCHEMA,
    METADATA_FILE,
    MODEL_FILE,
    ModelArtifact,
    ModelMetadata,
    build_metadata,
    library_versions,
    read_artifact,
)
from nfl_platform.ml.registry.store import ModelRegistry

FEATURES = ("a", "b")


def make_artifact(version: str = "v1_baseline", problem: str = "game_win") -> ModelArtifact:
    model = HomeFieldBaseline.fit(FEATURES, np.zeros((10, 2)), np.array([1.0] * 6 + [0.0] * 4))
    metadata = build_metadata(
        model_id=f"{problem}/{version}",
        problem=problem,
        version=version,
        algorithm=model.algorithm,
        role="baseline",
        feature_schema_version="game-v1",
        feature_names=FEATURES,
        train_seasons=(2018, 2019),
        validation_seasons=(2020,),
        test_seasons=(2021,),
        fit_seasons=(2018, 2019, 2020),
        primary_metric="brier_score",
        random_seed=1799,
        validation_metrics={"brier_score": 0.24},
        data_snapshot="training",
        test_evaluations=1,
    )
    return ModelArtifact(model=model, metadata=metadata)


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(root=tmp_path, settings=Settings(_env_file=None, model_dir=tmp_path))


class TestMetadata:
    def test_library_versions_record_what_actually_ran(self):
        versions = library_versions()
        assert "python" in versions
        assert "scikit-learn" in versions
        assert "lightgbm" in versions

    def test_metadata_round_trips(self):
        metadata = make_artifact().metadata
        restored = ModelMetadata.from_dict(json.loads(json.dumps(metadata.as_dict())))
        assert restored.model_id == metadata.model_id
        assert restored.feature_names == metadata.feature_names
        assert restored.test_seasons == metadata.test_seasons

    def test_the_validation_score_reads_the_primary_metric(self):
        assert make_artifact().metadata.validation_score == pytest.approx(0.24)

    def test_malformed_metadata_is_refused(self):
        with pytest.raises(ModelArtifactInvalidError, match="malformed metadata"):
            ModelMetadata.from_dict({"model_id": "x"})

    def test_the_test_evaluation_count_is_recorded(self):
        """Anything other than one means the frozen block has informed a decision."""
        assert make_artifact().metadata.test_evaluations == 1


class TestWriteAndRead:
    def test_a_written_artifact_reads_back_identically(self, tmp_path: Path):
        artifact = make_artifact()
        artifact.write(tmp_path / "v1")
        loaded = read_artifact(tmp_path / "v1")
        assert loaded.metadata.model_id == artifact.metadata.model_id
        assert loaded.model.predict(np.zeros((3, 2))).tolist() == pytest.approx(
            artifact.model.predict(np.zeros((3, 2))).tolist()
        )

    def test_both_files_are_written(self, tmp_path: Path):
        make_artifact().write(tmp_path / "v1")
        assert (tmp_path / "v1" / MODEL_FILE).is_file()
        assert (tmp_path / "v1" / METADATA_FILE).is_file()

    def test_the_artifact_is_human_readable_json(self, tmp_path: Path):
        make_artifact().write(tmp_path / "v1")
        payload = json.loads((tmp_path / "v1" / MODEL_FILE).read_text(encoding="utf-8"))
        assert payload["schema"] == ARTIFACT_SCHEMA
        assert payload["model"]["algorithm"] == "home_field_baseline"

    def test_a_checksum_is_recorded(self, tmp_path: Path):
        make_artifact().write(tmp_path / "v1")
        metadata = json.loads((tmp_path / "v1" / METADATA_FILE).read_text(encoding="utf-8"))
        assert len(metadata["artifact_sha256"]) == 64

    def test_a_tampered_model_file_is_refused(self, tmp_path: Path):
        """The core integrity guarantee."""
        make_artifact().write(tmp_path / "v1")
        path = tmp_path / "v1" / MODEL_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["model"]["params"]["rate"] = 0.99
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        with pytest.raises(ModelArtifactInvalidError, match="checksum mismatch"):
            read_artifact(tmp_path / "v1")

    def test_a_truncated_file_is_refused(self, tmp_path: Path):
        make_artifact().write(tmp_path / "v1")
        (tmp_path / "v1" / MODEL_FILE).write_text('{"schema": 1, "mo', encoding="utf-8")
        with pytest.raises(ModelArtifactInvalidError, match="invalid JSON"):
            read_artifact(tmp_path / "v1")

    def test_a_missing_file_is_refused(self, tmp_path: Path):
        make_artifact().write(tmp_path / "v1")
        (tmp_path / "v1" / MODEL_FILE).unlink()
        with pytest.raises(ModelArtifactInvalidError, match=r"missing model\.json"):
            read_artifact(tmp_path / "v1")

    def test_a_future_schema_is_refused(self, tmp_path: Path):
        artifact = make_artifact()
        artifact.write(tmp_path / "v1")
        path = tmp_path / "v1" / MODEL_FILE
        body = path.read_text(encoding="utf-8").replace('"schema": 1', '"schema": 99')
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ModelArtifactInvalidError, match="artifact schema 99"):
            read_artifact(tmp_path / "v1")

    def test_features_disagreeing_with_metadata_are_refused(self, tmp_path: Path):
        """Model and metadata must describe the same thing or neither is trustworthy."""
        artifact = make_artifact()
        artifact.write(tmp_path / "v1")
        metadata_path = tmp_path / "v1" / METADATA_FILE
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["feature_names"] = ["a", "b", "c"]
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ModelArtifactInvalidError, match="disagree with the metadata"):
            read_artifact(tmp_path / "v1")


class TestRegistry:
    def test_nothing_is_available_in_an_empty_directory(self, registry: ModelRegistry):
        assert not registry.available("game_win")
        assert registry.versions("game_win") == ()

    def test_loading_from_an_empty_registry_raises(self, registry: ModelRegistry):
        with pytest.raises(ModelNotAvailableError) as caught:
            registry.load("game_win")
        assert "train_models.py" in str(caught.value)

    def test_a_single_installed_version_becomes_the_default(self, registry: ModelRegistry):
        registry.save(make_artifact())
        assert registry.default_version("game_win") == "v1_baseline"
        assert registry.available("game_win")

    def test_with_several_versions_the_manifest_decides(self, registry: ModelRegistry):
        registry.save(make_artifact("v1_baseline"))
        registry.save(make_artifact("v2_logistic"), make_default=True)
        assert registry.default_version("game_win") == "v2_logistic"

    def test_promotion_is_explicit_not_implied_by_writing(self, registry: ModelRegistry):
        """Training a candidate must not silently change what the API serves."""
        registry.save(make_artifact("v1_baseline"), make_default=True)
        registry.save(make_artifact("v2_logistic"))
        assert registry.default_version("game_win") == "v1_baseline"

    def test_an_unknown_version_is_reported_with_what_is_installed(self, registry: ModelRegistry):
        registry.save(make_artifact("v1_baseline"))
        with pytest.raises(ModelVersionNotFoundError) as caught:
            registry.load("game_win", "v9_nope")
        assert caught.value.known == ("v1_baseline",)

    def test_a_settings_pin_overrides_the_manifest(self, tmp_path: Path):
        settings = Settings(_env_file=None, model_dir=tmp_path, game_model_version="v1_baseline")
        registry = ModelRegistry(root=tmp_path, settings=settings)
        registry.save(make_artifact("v1_baseline"))
        registry.save(make_artifact("v2_logistic"), make_default=True)
        assert registry.load("game_win").metadata.version == "v1_baseline"

    def test_predictions_can_be_switched_off_entirely(self, tmp_path: Path):
        settings = Settings(_env_file=None, model_dir=tmp_path, ml_enable_predictions=False)
        registry = ModelRegistry(root=tmp_path, settings=settings)
        registry.save(make_artifact())
        assert not registry.available("game_win")
        with pytest.raises(ModelNotAvailableError, match="disabled by configuration"):
            registry.load("game_win")

    @pytest.mark.parametrize("name", ["../escape", "a/b", "", "with space", "x" * 100])
    def test_path_traversal_and_odd_names_are_rejected(self, registry: ModelRegistry, name: str):
        """Versions become path segments, so no input may leave the model directory."""
        with pytest.raises(ValueError, match="Invalid"):
            registry.load("game_win", name)

    def test_describe_skips_an_unreadable_artifact_instead_of_failing(
        self, registry: ModelRegistry
    ):
        """A status view must survive the situation it exists to report."""
        registry.save(make_artifact(), make_default=True)
        registry.clear_cache()
        (registry.root / "game_win" / "v1_baseline" / MODEL_FILE).write_text("{", encoding="utf-8")
        assert registry.describe() == []

    def test_describe_lists_installed_defaults(self, registry: ModelRegistry):
        registry.save(make_artifact(), make_default=True)
        registry.save(make_artifact("v1_ridge", problem="fantasy_points"), make_default=True)
        assert {m.problem for m in registry.describe()} == {"game_win", "fantasy_points"}

    def test_loads_are_cached(self, registry: ModelRegistry):
        registry.save(make_artifact())
        assert registry.load("game_win") is registry.load("game_win")

    def test_clearing_the_cache_forces_a_reread(self, registry: ModelRegistry):
        registry.save(make_artifact())
        first = registry.load("game_win")
        registry.clear_cache()
        assert registry.load("game_win") is not first

    def test_a_corrupt_manifest_is_refused(self, registry: ModelRegistry):
        registry.root.mkdir(parents=True, exist_ok=True)
        (registry.root / "manifest.json").write_text("not json", encoding="utf-8")
        with pytest.raises(ModelArtifactInvalidError, match="invalid manifest"):
            registry.manifest()
