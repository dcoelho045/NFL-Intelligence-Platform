"""Problem identifiers.

A separate module so the serving path can name a problem without importing
:mod:`nfl_platform.ml.training`. Two constants do not need their own file on
merit; they get one because the alternative is an import edge that would let the
API server pull in the training pipeline, and an architecture test exists
specifically to prevent that.
"""

from __future__ import annotations

from typing import Final

GAME_PROBLEM: Final = "game_win"
FANTASY_PROBLEM: Final = "fantasy_points"

PROBLEMS: Final = (GAME_PROBLEM, FANTASY_PROBLEM)
