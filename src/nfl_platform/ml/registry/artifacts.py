"""Artifact format.

An artifact is two text files in one directory:

``model.json``
    The estimator. JSON for linear models and baselines; for LightGBM, the
    booster's own plain-text dump embedded as a string. **No pickle anywhere.**
    ``joblib.load`` on a path is arbitrary code execution, and a model directory
    is exactly the kind of thing that gets copied between machines, committed by
    accident, or restored from a backup nobody audited. A format that can only
    ever produce numbers removes the question instead of mitigating it.

``metadata.json``
    Everything needed to answer "where did this number come from": the data
    snapshot, the seasons, the seed, the library versions, the feature contract,
    the fold-by-fold validation scores, the single test-period score, and the
    promotion decision with its reasoning.

The metadata records ``artifact_sha256`` over the model file. Verified on every
load, because a truncated or edited artifact that still parses would keep
producing confident numbers from the wrong parameters, and nothing downstream
could tell.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_platform.exceptions import ModelArtifactInvalidError
from nfl_platform.ml.models import load_model
from nfl_platform.ml.models.base import FittedModel

#: Bumped when the layout of these files changes incompatibly.
ARTIFACT_SCHEMA = 1

MODEL_FILE = "model.json"
METADATA_FILE = "metadata.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def library_versions() -> dict[str, str]:
    """Exact resolved versions, recorded so a re-run can be compared like for like."""
    versions = {"python": platform.python_version()}
    for module_name, key in (
        ("numpy", "numpy"),
        ("polars", "polars"),
        ("sklearn", "scikit-learn"),
        ("lightgbm", "lightgbm"),
    ):
        try:
            module = __import__(module_name)
        except ImportError:  # pragma: no cover - all four are hard dependencies
            continue
        versions[key] = str(getattr(module, "__version__", "unknown"))
    return versions


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Provenance and evaluation record for one artifact."""

    model_id: str
    problem: str
    version: str
    algorithm: str
    role: str
    trained_at: str
    feature_schema_version: str
    feature_names: tuple[str, ...]
    train_seasons: tuple[int, ...]
    validation_seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    fit_seasons: tuple[int, ...]
    primary_metric: str
    random_seed: int
    schema: int = ARTIFACT_SCHEMA
    data_snapshot: str | None = None
    data_fingerprint: str | None = None
    training_rows: int = 0
    feature_matrix_hash: str | None = None
    calibration_method: str | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    validation_folds: list[dict[str, Any]] = field(default_factory=list)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    benchmark_metrics: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] = field(default_factory=dict)
    split_rationale: str = ""
    test_evaluations: int = 0
    """How many times the frozen test period was scored. One is correct; more
    means it has been used for selection and is no longer a held-out estimate."""
    library_versions: dict[str, str] = field(default_factory=dict)
    command: str | None = None
    notes: list[str] = field(default_factory=list)
    artifact_sha256: str | None = None

    @property
    def validation_score(self) -> float | None:
        value = self.validation_metrics.get(self.primary_metric)
        return float(value) if isinstance(value, int | float) else None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "model_id": self.model_id,
            "problem": self.problem,
            "version": self.version,
            "algorithm": self.algorithm,
            "role": self.role,
            "trained_at": self.trained_at,
            "feature_schema_version": self.feature_schema_version,
            "feature_names": list(self.feature_names),
            "train_seasons": list(self.train_seasons),
            "validation_seasons": list(self.validation_seasons),
            "test_seasons": list(self.test_seasons),
            "fit_seasons": list(self.fit_seasons),
            "primary_metric": self.primary_metric,
            "random_seed": self.random_seed,
            "data_snapshot": self.data_snapshot,
            "data_fingerprint": self.data_fingerprint,
            "training_rows": self.training_rows,
            "feature_matrix_hash": self.feature_matrix_hash,
            "calibration_method": self.calibration_method,
            "hyperparameters": self.hyperparameters,
            "train_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
            "validation_folds": self.validation_folds,
            "test_metrics": self.test_metrics,
            "baseline_metrics": self.baseline_metrics,
            "benchmark_metrics": self.benchmark_metrics,
            "promotion": self.promotion,
            "split_rationale": self.split_rationale,
            "test_evaluations": self.test_evaluations,
            "library_versions": self.library_versions,
            "command": self.command,
            "notes": self.notes,
            "artifact_sha256": self.artifact_sha256,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, source: str = "<metadata>") -> ModelMetadata:
        try:
            return cls(
                schema=int(payload.get("schema", ARTIFACT_SCHEMA)),
                model_id=str(payload["model_id"]),
                problem=str(payload["problem"]),
                version=str(payload["version"]),
                algorithm=str(payload["algorithm"]),
                role=str(payload["role"]),
                trained_at=str(payload["trained_at"]),
                feature_schema_version=str(payload["feature_schema_version"]),
                feature_names=tuple(str(name) for name in payload["feature_names"]),
                train_seasons=tuple(int(s) for s in payload.get("train_seasons", [])),
                validation_seasons=tuple(int(s) for s in payload.get("validation_seasons", [])),
                test_seasons=tuple(int(s) for s in payload.get("test_seasons", [])),
                fit_seasons=tuple(int(s) for s in payload.get("fit_seasons", [])),
                primary_metric=str(payload.get("primary_metric", "")),
                random_seed=int(payload.get("random_seed", 0)),
                data_snapshot=payload.get("data_snapshot"),
                data_fingerprint=payload.get("data_fingerprint"),
                training_rows=int(payload.get("training_rows", 0)),
                feature_matrix_hash=payload.get("feature_matrix_hash"),
                calibration_method=payload.get("calibration_method"),
                hyperparameters=dict(payload.get("hyperparameters", {})),
                train_metrics=dict(payload.get("train_metrics", {})),
                validation_metrics=dict(payload.get("validation_metrics", {})),
                validation_folds=list(payload.get("validation_folds", [])),
                test_metrics=dict(payload.get("test_metrics", {})),
                baseline_metrics=dict(payload.get("baseline_metrics", {})),
                benchmark_metrics=dict(payload.get("benchmark_metrics", {})),
                promotion=dict(payload.get("promotion", {})),
                split_rationale=str(payload.get("split_rationale", "")),
                test_evaluations=int(payload.get("test_evaluations", 0)),
                library_versions=dict(payload.get("library_versions", {})),
                command=payload.get("command"),
                notes=list(payload.get("notes", [])),
                artifact_sha256=payload.get("artifact_sha256"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelArtifactInvalidError(source, f"malformed metadata: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A fitted model paired with the record of how it came to exist."""

    model: FittedModel
    metadata: ModelMetadata

    def write(self, directory: Path) -> Path:
        """Write ``model.json`` and ``metadata.json``, checksumming the first.

        The model file is written before the metadata so that a crash between
        the two leaves an artifact without a record rather than a record
        pointing at a file that was never written.
        """
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / MODEL_FILE
        body = json.dumps(
            {"schema": ARTIFACT_SCHEMA, "model": self.model.serialise()},
            indent=2,
            sort_keys=True,
        )
        model_path.write_text(body, encoding="utf-8")

        checksum = hashlib.sha256(body.encode("utf-8")).hexdigest()
        payload = self.metadata.as_dict()
        payload["artifact_sha256"] = checksum
        (directory / METADATA_FILE).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return directory


def read_artifact(directory: Path) -> ModelArtifact:
    """Load and verify an artifact directory.

    Raises:
        ModelArtifactInvalidError: The files are missing, unparsable, fail the
            checksum, or declare a schema this build does not understand. Every
            one of those is refused rather than worked around: a model that
            loads from a damaged artifact still returns numbers, and no caller
            downstream has any way to notice they are wrong.
    """
    model_path = directory / MODEL_FILE
    metadata_path = directory / METADATA_FILE
    for path in (model_path, metadata_path):
        if not path.is_file():
            raise ModelArtifactInvalidError(str(directory), f"missing {path.name}")

    raw_model = model_path.read_text(encoding="utf-8")
    try:
        model_payload = json.loads(raw_model)
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelArtifactInvalidError(str(directory), f"invalid JSON: {exc}") from exc

    schema = int(model_payload.get("schema", 0))
    if schema != ARTIFACT_SCHEMA:
        raise ModelArtifactInvalidError(
            str(directory), f"artifact schema {schema}, this build reads {ARTIFACT_SCHEMA}"
        )

    metadata = ModelMetadata.from_dict(metadata_payload, source=str(metadata_path))
    expected = metadata.artifact_sha256
    actual = hashlib.sha256(raw_model.encode("utf-8")).hexdigest()
    if expected and expected != actual:
        raise ModelArtifactInvalidError(
            str(directory),
            "checksum mismatch; the model file does not match its metadata record",
        )

    try:
        model = load_model(dict(model_payload["model"]))
    except KeyError as exc:
        raise ModelArtifactInvalidError(str(directory), f"missing model section: {exc}") from exc

    if tuple(model.feature_names) != metadata.feature_names:
        raise ModelArtifactInvalidError(
            str(directory), "model features disagree with the metadata feature list"
        )
    return ModelArtifact(model=model, metadata=metadata)


def build_metadata(**values: Any) -> ModelMetadata:
    """Construct metadata, filling in the fields every run shares."""
    values.setdefault("trained_at", _utc_now())
    values.setdefault("library_versions", library_versions())
    return ModelMetadata(**values)
