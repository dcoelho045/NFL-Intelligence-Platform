"""Dataset assembly and time-aware splitting."""

from __future__ import annotations

from nfl_platform.ml.datasets.builder import TrainingDataset
from nfl_platform.ml.datasets.splits import SplitPlan, TimeFold, complete_seasons, plan_splits

__all__ = ["SplitPlan", "TimeFold", "TrainingDataset", "complete_seasons", "plan_splits"]
