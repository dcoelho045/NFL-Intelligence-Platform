"""Game-prediction inference.

Builds the feature row for one scheduled game and scores the frozen artifact.
The history window is drawn from the seasons immediately before the target, so a
prediction for week 2 still has the previous season's form behind it rather than
one game.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import InsufficientHistoryError
from nfl_platform.ml.features.games import (
    GAME_FEATURES,
    build_game_features,
)
from nfl_platform.ml.predict.result import (
    PredictionResult,
    feature_matrix,
    optional_float,
    top_contributions,
)
from nfl_platform.ml.problems import GAME_PROBLEM
from nfl_platform.ml.registry.store import LoadedModel, ModelRegistry

logger = get_logger(__name__)

#: Seasons of history loaded behind the target season. Two is enough to fill an
#: eight-game window in week 1 while keeping the download small.
HISTORY_SEASONS: Final = 2


class GameWinPredictor:
    """Scores the home team's win probability for one game."""

    problem: Final = GAME_PROBLEM

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        settings: Settings | None = None,
        *,
        version: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._registry = registry or ModelRegistry(settings=self._settings)
        self._version = version

    @property
    def available(self) -> bool:
        return self._registry.available(self.problem)

    def load(self) -> LoadedModel:
        """Resolve the artifact. Raises when none is installed or it is damaged."""
        return self._registry.load(self.problem, self._version)

    def baseline(self) -> LoadedModel | None:
        """The home-field baseline, for contrast. None when it is not installed."""
        try:
            return self._registry.load(self.problem, "v1_baseline")
        except Exception:
            return None

    def feature_frame(
        self,
        schedules: pl.DataFrame,
        team_stats: pl.DataFrame | None,
        *,
        season: int,
        week: int,
    ) -> pl.DataFrame:
        """Feature rows for every game in one week, played or not."""
        features = build_game_features(schedules, team_stats, completed_only=False)
        return features.filter((pl.col("season") == season) & (pl.col("week") == week))

    def predict_row(self, row: pl.DataFrame) -> PredictionResult:
        """Score one prepared feature row.

        Raises:
            InsufficientHistoryError: Fewer completed games behind either team
                than ``NFL_ML_MIN_HISTORY_GAMES``.
            FeatureSchemaMismatchError: The row's columns are not the ones the
                artifact was fitted on.
        """
        loaded = self.load()
        minimum = self._settings.ml_min_history_games
        history_min = int(row.item(0, "history_games_min") or 0)
        if history_min < minimum:
            home = str(row.item(0, "home_team"))
            away = str(row.item(0, "away_team"))
            raise InsufficientHistoryError(f"{away} at {home}", history_min, minimum)

        names = tuple(loaded.metadata.feature_names)
        loaded.model.check_features(names)
        matrix = feature_matrix(row, names)
        probability = float(loaded.model.predict(matrix)[0])

        baseline_value: float | None = None
        baseline = self.baseline()
        if baseline is not None and baseline.metadata.version != loaded.metadata.version:
            baseline.model.check_features(tuple(baseline.metadata.feature_names))
            baseline_value = float(
                baseline.model.predict(feature_matrix(row, tuple(baseline.metadata.feature_names)))[
                    0
                ]
            )

        home_history = int(row.item(0, "home_history_games") or 0)
        away_history = int(row.item(0, "away_history_games") or 0)
        caveats = _caveats(row, history_min, loaded)

        return PredictionResult(
            value=probability,
            metadata=loaded.metadata,
            feature_values={
                name: optional_float(row.item(0, name)) for name in GAME_FEATURES.names
            },
            contributions=top_contributions(loaded.model, matrix, names),
            baseline_value=baseline_value,
            history={
                str(row.item(0, "home_team")): home_history,
                str(row.item(0, "away_team")): away_history,
            },
            caveats=caveats,
            extras={
                "spread_line": optional_float(row.item(0, "spread_line")),
                "total_line": optional_float(row.item(0, "total_line")),
                "completed": bool(row.item(0, "completed")),
            },
        )


def _caveats(row: pl.DataFrame, history_min: int, loaded: LoadedModel) -> list[str]:
    caveats = ["A model estimate for a game that has not been played, not a measured statistic."]
    if history_min < 6:
        caveats.append(
            f"Form is built on as few as {history_min} completed games for one of these "
            "teams, so it is noisier than a mid-season prediction."
        )
    if bool(row.item(0, "completed")):
        caveats.append(
            "This game has already been played. The prediction uses only information "
            "available before kickoff, so it is a backtest of the model rather than a "
            "forecast."
        )
    if not loaded.metadata.calibration_method:
        caveats.append(
            "Probabilities are uncalibrated; treat them as a ranking rather than a "
            "literal frequency."
        )
    season = int(row.item(0, "season"))
    if loaded.metadata.test_seasons and season in loaded.metadata.test_seasons:
        caveats.append(
            f"Season {season} is in this model's frozen test period; it was not trained on it."
        )
    elif loaded.metadata.fit_seasons and season in loaded.metadata.fit_seasons:
        caveats.append(
            f"Season {season} is inside this model's training window, so the probability "
            "is in-sample and flatters the model."
        )
    return caveats
