"""Inference.

Loads a frozen artifact and scores it. Nothing here fits anything, and nothing
here imports :mod:`nfl_platform.ml.training`, :mod:`nfl_platform.ml.evaluation`
or :mod:`nfl_platform.ml.datasets` -- an architecture test enforces that, so a
convenience import cannot quietly make the API server capable of training.

Both predictors call the same feature builders training used. That is the whole
defence against train/serve skew: there is one implementation, so there is
nothing for a second one to disagree with.
"""

from __future__ import annotations

from nfl_platform.ml.predict.fantasy import FantasyPointsPredictor
from nfl_platform.ml.predict.game import GameWinPredictor
from nfl_platform.ml.predict.result import PredictionResult

__all__ = ["FantasyPointsPredictor", "GameWinPredictor", "PredictionResult"]
