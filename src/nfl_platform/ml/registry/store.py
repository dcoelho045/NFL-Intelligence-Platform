"""The on-disk model registry.

Resolution is by ``(problem, version)``, never by caller-supplied path. That is
both an ergonomics decision and a security one: a loader that accepts a path is
a loader an upstream caller can point anywhere, and "load the model at this
location" is the kind of parameter that reaches an HTTP boundary eventually.
Versions are validated as plain identifiers, so no input can traverse out of the
configured directory.

The manifest records which version is the default for each problem. Keeping that
in a file rather than inferring it from timestamps means promotion is an
explicit, reviewable act: training a new candidate does not silently change what
the API serves.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import (
    ModelArtifactInvalidError,
    ModelNotAvailableError,
    ModelVersionNotFoundError,
)
from nfl_platform.ml.models.base import FittedModel
from nfl_platform.ml.registry.artifacts import ModelArtifact, ModelMetadata, read_artifact

logger = get_logger(__name__)

MANIFEST_FILE = "manifest.json"
MANIFEST_SCHEMA = 1

# Version and problem names become path segments, so they are restricted to
# characters that cannot escape the model directory.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _validate_name(value: str, kind: str) -> str:
    if not _SAFE_NAME.match(value):
        raise ValueError(f"Invalid {kind} {value!r}: letters, digits, '.', '_' and '-' only.")
    return value


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """What callers get back: a scorer plus its provenance."""

    model: FittedModel
    metadata: ModelMetadata

    @property
    def model_id(self) -> str:
        return self.metadata.model_id


class ModelRegistry:
    """Reads, writes and resolves versioned artifacts under one directory."""

    def __init__(self, root: Path | str | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._root = Path(root) if root is not None else self._settings.resolved_model_dir()
        self._cache: dict[tuple[str, str], LoadedModel] = {}

    @property
    def root(self) -> Path:
        return self._root

    # --- manifest -------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self._root / MANIFEST_FILE

    def manifest(self) -> dict[str, Any]:
        path = self._manifest_path()
        if not path.is_file():
            return {"schema": MANIFEST_SCHEMA, "defaults": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelArtifactInvalidError(str(path), f"invalid manifest JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ModelArtifactInvalidError(str(path), "manifest is not an object")
        payload.setdefault("defaults", {})
        return payload

    def set_default(self, problem: str, version: str) -> None:
        """Promote one version to be what callers get when they ask for none."""
        _validate_name(problem, "problem")
        _validate_name(version, "version")
        payload = self.manifest()
        defaults = dict(payload.get("defaults", {}))
        defaults[problem] = version
        payload["defaults"] = defaults
        payload["schema"] = MANIFEST_SCHEMA
        payload["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        self._root.mkdir(parents=True, exist_ok=True)
        self._manifest_path().write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._cache.pop((problem, version), None)

    # --- discovery ------------------------------------------------------------

    def versions(self, problem: str) -> tuple[str, ...]:
        """Installed versions for a problem, newest name last."""
        _validate_name(problem, "problem")
        directory = self._root / problem
        if not directory.is_dir():
            return ()
        return tuple(
            sorted(child.name for child in directory.iterdir() if (child / "model.json").is_file())
        )

    def default_version(self, problem: str) -> str | None:
        """Configured pin, then the manifest default, then the only installed version."""
        pinned = self._pinned_version(problem)
        if pinned:
            return pinned
        manifest_default = self.manifest().get("defaults", {}).get(problem)
        if isinstance(manifest_default, str) and manifest_default:
            return manifest_default
        installed = self.versions(problem)
        return installed[-1] if len(installed) == 1 else None

    def _pinned_version(self, problem: str) -> str | None:
        pins = {
            "game_win": self._settings.game_model_version,
            "fantasy_points": self._settings.fantasy_model_version,
        }
        return pins.get(problem)

    def available(self, problem: str) -> bool:
        if not self._settings.ml_enable_predictions:
            return False
        version = self.default_version(problem)
        return bool(version) and (self._root / problem / str(version) / "model.json").is_file()

    def problems(self) -> tuple[str, ...]:
        """Problems with at least one installed artifact directory."""
        if not self._root.is_dir():
            return ()
        return tuple(sorted(child.name for child in self._root.iterdir() if child.is_dir()))

    def describe(self) -> list[ModelMetadata]:
        """Metadata for every default model that loads, skipping ones that do not.

        A status view must not fail because one artifact is damaged -- that is
        exactly the situation it exists to report.
        """
        found: list[ModelMetadata] = []
        for problem in self.problems():
            version = self.default_version(problem)
            if not version:
                continue
            try:
                found.append(self.load(problem, version).metadata)
            except (ModelArtifactInvalidError, ModelVersionNotFoundError, ValueError) as exc:
                logger.warning(
                    "Skipping unreadable model artifact",
                    extra={"problem": problem, "version": version, "reason": str(exc)},
                )
        return found

    # --- load / save ----------------------------------------------------------

    def load(self, problem: str, version: str | None = None) -> LoadedModel:
        """Resolve and load one artifact, verifying its integrity.

        Raises:
            ModelNotAvailableError: Predictions are switched off, or no version
                is installed for this problem.
            ModelVersionNotFoundError: A specific version was asked for and is
                not installed.
            ModelArtifactInvalidError: The artifact is damaged or unreadable.
        """
        _validate_name(problem, "problem")
        if not self._settings.ml_enable_predictions:
            raise ModelNotAvailableError(problem, "predictions are disabled by configuration")

        # An explicitly supplied version is validated before anything else, so a
        # malformed one is rejected rather than quietly falling through to the
        # default and serving a model the caller did not ask for.
        if version is not None:
            _validate_name(version, "version")

        resolved = version or self.default_version(problem)
        if not resolved:
            installed = self.versions(problem)
            reason = (
                f"{len(installed)} versions installed and no default selected"
                if installed
                else f"no artifacts found under {self._root / problem}"
            )
            raise ModelNotAvailableError(problem, reason)
        _validate_name(resolved, "version")

        cached = self._cache.get((problem, resolved))
        if cached is not None:
            return cached

        directory = self._root / problem / resolved
        if not directory.is_dir():
            raise ModelVersionNotFoundError(problem, resolved, self.versions(problem))

        artifact = read_artifact(directory)
        loaded = LoadedModel(model=artifact.model, metadata=artifact.metadata)
        self._cache[(problem, resolved)] = loaded
        logger.info(
            "Loaded model artifact",
            extra={"problem": problem, "version": resolved, "model_id": loaded.model_id},
        )
        return loaded

    def save(self, artifact: ModelArtifact, *, make_default: bool = False) -> Path:
        """Write an artifact and optionally promote it."""
        problem = _validate_name(artifact.metadata.problem, "problem")
        version = _validate_name(artifact.metadata.version, "version")
        directory = artifact.write(self._root / problem / version)
        self._cache.pop((problem, version), None)
        if make_default:
            self.set_default(problem, version)
        return directory

    def clear_cache(self) -> None:
        self._cache.clear()
