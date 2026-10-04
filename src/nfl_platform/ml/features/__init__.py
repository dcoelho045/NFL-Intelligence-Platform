"""Feature construction.

Every feature in this package carries a :class:`FeatureSpec` declaring which
dataset it comes from and -- the part that matters -- *when it becomes known*.
A feature without that declaration cannot be audited for leakage, so the
builders refuse to emit one.
"""

from __future__ import annotations

from nfl_platform.ml.features.schema import (
    Availability,
    FeatureSet,
    FeatureSpec,
)

__all__ = ["Availability", "FeatureSet", "FeatureSpec"]
