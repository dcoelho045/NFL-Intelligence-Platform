"""Model artifacts and the on-disk registry."""

from __future__ import annotations

from nfl_platform.ml.registry.artifacts import (
    ARTIFACT_SCHEMA,
    ModelArtifact,
    ModelMetadata,
)
from nfl_platform.ml.registry.store import LoadedModel, ModelRegistry

__all__ = [
    "ARTIFACT_SCHEMA",
    "LoadedModel",
    "ModelArtifact",
    "ModelMetadata",
    "ModelRegistry",
]
