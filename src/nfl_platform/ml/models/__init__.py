"""Fitted estimators.

This package is the quarantine. ``sklearn`` and ``lightgbm`` are imported here
and nowhere else in the codebase, so the question "which code can fit a model"
has a one-directory answer, the same way ``openai`` answers to ``*/providers/``
and ``mcp`` answers to ``mcp_server/``.

Two contracts hold for everything in here:

*Serialisation is pickle-free.* Linear models and baselines round-trip through
JSON; LightGBM uses its own plain-text dump. The result is diffable, portable
across library patch releases, safe to load from disk without executing
whatever happens to be in the file, and small enough that a fixture artifact can
be committed for tests.

*Preprocessing belongs to the model.* Imputation and scaling parameters are
fitted alongside the estimator, serialised with it, and reapplied on every
prediction. Keeping them in a separate object is how a training fold's medians
end up being silently reused across a fold boundary.
"""

from __future__ import annotations

# Imported for their registration side effect. ``load_model`` resolves an
# artifact's ``algorithm`` string against an explicit table, so a class that is
# never imported is a class an artifact cannot name -- which is the intended
# safety property, but it would make loading a perfectly good file fail.
from nfl_platform.ml.models import baselines, boosted, calibration, linear  # noqa: F401
from nfl_platform.ml.models.base import (
    FittedModel,
    ModelKind,
    Preprocessor,
    load_model,
    model_type,
    register_model,
    registered_algorithms,
)

__all__ = [
    "FittedModel",
    "ModelKind",
    "Preprocessor",
    "load_model",
    "model_type",
    "register_model",
    "registered_algorithms",
]
