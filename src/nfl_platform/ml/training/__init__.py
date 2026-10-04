"""Offline training. Nothing in the serving path may import this package."""

from __future__ import annotations

from nfl_platform.ml.problems import FANTASY_PROBLEM, GAME_PROBLEM
from nfl_platform.ml.training.pipeline import (
    TrainingOutcome,
    summarise,
    train_fantasy_models,
    train_game_models,
)

__all__ = [
    "FANTASY_PROBLEM",
    "GAME_PROBLEM",
    "TrainingOutcome",
    "summarise",
    "train_fantasy_models",
    "train_game_models",
]
