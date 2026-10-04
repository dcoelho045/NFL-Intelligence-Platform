"""Fantasy-projection inference.

Scores the player-week feature row for an upcoming game. ``played_only=False``
when building those rows, because whether the player takes a snap is exactly
what is not yet known -- which is also why the resulting projection is labelled
conditional on playing everywhere it surfaces.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import InsufficientHistoryError
from nfl_platform.ml.features.fantasy import (
    FANTASY_FEATURES,
    build_fantasy_features,
)
from nfl_platform.ml.predict.result import (
    PredictionResult,
    feature_matrix,
    optional_float,
    top_contributions,
)
from nfl_platform.ml.problems import FANTASY_PROBLEM
from nfl_platform.ml.registry.store import LoadedModel, ModelRegistry

logger = get_logger(__name__)

HISTORY_SEASONS: Final = 2
BASELINE_VERSION: Final = "v1_rolling_average"


class FantasyPointsPredictor:
    """Projects PPR points for one player in one week, given that he plays."""

    problem: Final = FANTASY_PROBLEM

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
        return self._registry.load(self.problem, self._version)

    def baseline(self) -> LoadedModel | None:
        try:
            return self._registry.load(self.problem, BASELINE_VERSION)
        except Exception:
            return None

    def feature_frame(
        self,
        player_stats: pl.DataFrame,
        schedules: pl.DataFrame,
        *,
        season: int,
        week: int,
        team_stats: pl.DataFrame | None = None,
        snap_counts: pl.DataFrame | None = None,
        ff_opportunity: pl.DataFrame | None = None,
        players: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        features = build_fantasy_features(
            player_stats,
            schedules,
            team_stats=team_stats,
            snap_counts=snap_counts,
            ff_opportunity=ff_opportunity,
            players=players,
            played_only=False,
        )
        return features.filter((pl.col("season") == season) & (pl.col("week") == week))

    def predict_row(self, row: pl.DataFrame) -> PredictionResult:
        """Score one prepared player-week row.

        Raises:
            InsufficientHistoryError: Fewer games played than
                ``NFL_ML_MIN_HISTORY_GAMES``. A player with no history has an
                all-null feature row, and imputing it would produce a projection
                for "an average player" wearing this player's name.
        """
        loaded = self.load()
        minimum = self._settings.ml_min_history_games
        history = int(row.item(0, "history_games") or 0)
        if history < minimum:
            subject = str(row.item(0, "player_display_name") or row.item(0, "player_id"))
            raise InsufficientHistoryError(subject, history, minimum)

        names = tuple(loaded.metadata.feature_names)
        loaded.model.check_features(names)
        matrix = feature_matrix(row, names)
        projection = float(loaded.model.predict(matrix)[0])

        baseline_value: float | None = None
        baseline = self.baseline()
        if baseline is not None and baseline.metadata.version != loaded.metadata.version:
            baseline.model.check_features(tuple(baseline.metadata.feature_names))
            baseline_value = float(
                baseline.model.predict(feature_matrix(row, tuple(baseline.metadata.feature_names)))[
                    0
                ]
            )

        return PredictionResult(
            value=projection,
            metadata=loaded.metadata,
            feature_values={
                name: optional_float(row.item(0, name)) for name in FANTASY_FEATURES.names
            },
            contributions=top_contributions(loaded.model, matrix, names),
            baseline_value=baseline_value,
            history={str(row.item(0, "player_id")): history},
            caveats=_caveats(row, history, loaded),
            extras={
                "typical_error": _validation_mae(loaded),
                "played": bool(row.item(0, "played")),
                "completed": bool(row.item(0, "completed")),
            },
        )


def _validation_mae(loaded: LoadedModel) -> float | None:
    value = loaded.metadata.validation_metrics.get("mae")
    if isinstance(value, int | float) and np.isfinite(float(value)):
        return float(value)
    return None


def _caveats(row: pl.DataFrame, history: int, loaded: LoadedModel) -> list[str]:
    caveats = [
        "A model projection, not a measured statistic.",
        "Conditional on the player taking an offensive snap. It contains no estimate of "
        "the chance he is inactive, so check injury status separately before starting him.",
    ]
    if history < 5:
        caveats.append(f"Built on {history} prior game(s), so the usage picture behind it is thin.")
    if optional_float(row.item(0, "snap_share_recent")) is None:
        caveats.append(
            "No recent snap-share data, so the role estimate rests on box-score usage alone."
        )
    weeks_out = optional_float(row.item(0, "weeks_since_last_game"))
    if weeks_out is not None and weeks_out > 2:
        caveats.append(
            f"The player last took a snap {int(weeks_out)} weeks ago; the window may not "
            "reflect his current role."
        )
    season = int(row.item(0, "season"))
    if loaded.metadata.fit_seasons and season in loaded.metadata.fit_seasons:
        caveats.append(
            f"Season {season} is inside this model's training window, so the projection "
            "is in-sample."
        )
    return caveats
