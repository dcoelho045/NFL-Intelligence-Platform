"""Prediction service.

The boundary between fitted models and the rest of the application. It does
four things and delegates everything else:

1. validates and resolves the request (seasons, weeks, teams, player names),
2. loads the data windows the feature builders need,
3. calls the predictors,
4. converts the result into a DTO that states what kind of claim it is.

It computes no numbers of its own. A probability or a projected point total that
did not come out of a fitted artifact must not exist anywhere in this file --
that is the architectural difference between this milestone and the analytics
services beside it, and it is the one that would be easiest to erode.

No language model is reachable from here. The LLM may read these DTOs; it may
never produce one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.domain import normalize_team, validate_season, validate_week
from nfl_platform.exceptions import (
    InsufficientHistoryError,
    NflPlatformError,
    TeamNotFoundError,
)
from nfl_platform.ml.features.fantasy import MODELLED_POSITIONS
from nfl_platform.ml.predict.fantasy import HISTORY_SEASONS, FantasyPointsPredictor
from nfl_platform.ml.predict.game import GameWinPredictor
from nfl_platform.ml.predict.result import PredictionResult
from nfl_platform.ml.problems import FANTASY_PROBLEM, GAME_PROBLEM
from nfl_platform.ml.registry.artifacts import ModelMetadata
from nfl_platform.ml.registry.store import ModelRegistry
from nfl_platform.schemas.common import GameContext, PlayerRef
from nfl_platform.schemas.fantasy import FantasyScoring
from nfl_platform.schemas.prediction import (
    FantasyProjection,
    FantasyProjectionComparison,
    FeatureContribution,
    GamePrediction,
    GamePredictionSlate,
    ModelInfo,
    ModelRegistryStatus,
    ModelRole,
    PredictionBasis,
    PredictionProblem,
)
from nfl_platform.services.directory import PlayerDirectoryService

logger = get_logger(__name__)

#: The only scoring format modelled in this version. Stated as a constant rather
#: than silently assumed, so a standard-scoring request gets an explicit refusal
#: instead of a PPR number wearing the wrong label.
MODELLED_SCORING = FantasyScoring.PPR

MAX_COMPARED_PLAYERS = 6


class PredictionService:
    """Serves model predictions as typed DTOs."""

    def __init__(
        self,
        loader: NflDataLoader,
        *,
        directory: PlayerDirectoryService,
        registry: ModelRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._loader = loader
        self._directory = directory
        self._settings = settings or get_settings()
        self._registry = registry or ModelRegistry(settings=self._settings)
        self._games = GameWinPredictor(self._registry, self._settings)
        self._fantasy = FantasyPointsPredictor(self._registry, self._settings)

    # --- status ---------------------------------------------------------------

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def available(self) -> bool:
        """True when at least one prediction model is installed and enabled."""
        return self._games.available or self._fantasy.available

    def model_status(self) -> ModelRegistryStatus:
        """What is installed, at which version, trained on what.

        Never raises for a missing or damaged artifact: reporting that a model
        is unavailable is this method's job, so failing would leave the caller
        unable to distinguish "not installed" from "status check broken".
        """
        enabled = self._settings.ml_enable_predictions
        notes: list[str] = []
        models: list[ModelInfo] = []
        if not enabled:
            notes.append(
                "Predictions are switched off by configuration (NFL_ML_ENABLE_PREDICTIONS). "
                "This is a deployment setting, not a statement about any game."
            )
        else:
            models = [_model_info(metadata) for metadata in self._registry.describe()]
            if not models:
                notes.append(
                    "No model artifacts are installed. Train them with: "
                    "python scripts/train_models.py"
                )
        return ModelRegistryStatus(
            available=bool(models),
            enabled=enabled,
            model_dir=str(self._registry.root),
            models=models,
            notes=notes,
        )

    # --- game prediction ------------------------------------------------------

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        *,
        season: int | None = None,
        week: int | None = None,
    ) -> GamePrediction:
        """Probability that ``home_team`` beats ``away_team``.

        Args:
            home_team: Home team abbreviation.
            away_team: Away team abbreviation.
            season: Season of the game. Defaults to the loader's default season.
            week: Week of the game. Defaults to the earliest week in which the
                two teams are scheduled to meet that season.

        Raises:
            ModelNotAvailableError: No game model is installed or enabled.
            TeamNotFoundError: A team does not appear in that season's schedule.
            InsufficientHistoryError: Too few completed games behind either team.
        """
        home = normalize_team(home_team)
        away = normalize_team(away_team)
        resolved_season = self._resolve_season(season)
        schedules, team_stats = self._game_inputs(resolved_season)

        resolved_week = (
            validate_week(week)
            if week is not None
            else self._find_week(schedules, resolved_season, home, away)
        )
        features = self._games.feature_frame(
            schedules, team_stats, season=resolved_season, week=resolved_week
        )
        row = features.filter((pl.col("home_team") == home) & (pl.col("away_team") == away))
        if row.is_empty():
            row = self._synthetic_game_row(
                schedules, team_stats, resolved_season, resolved_week, home, away
            )
        return self._to_game_prediction(
            self._games.predict_row(row.head(1)), row.head(1), schedules
        )

    def predict_slate(self, season: int, week: int) -> GamePredictionSlate:
        """Every game in one week that the model can score."""
        resolved_season = validate_season(season, latest_season=self._loader.current_season)
        resolved_week = validate_week(week)
        schedules, team_stats = self._game_inputs(resolved_season)
        features = self._games.feature_frame(
            schedules, team_stats, season=resolved_season, week=resolved_week
        )

        predictions: list[GamePrediction] = []
        skipped: list[str] = []
        for index in range(features.height):
            row = features.slice(index, 1)
            label = f"{row.item(0, 'away_team')} at {row.item(0, 'home_team')}"
            try:
                predictions.append(
                    self._to_game_prediction(self._games.predict_row(row), row, schedules)
                )
            except InsufficientHistoryError as exc:
                # One unpredictable game must not void the rest of the slate, but
                # it also must not disappear: the caller is told which and why.
                skipped.append(f"{label}: {exc}")

        if not predictions and not skipped:
            raise TeamNotFoundError(
                "", f"the week {resolved_week} schedule for season {resolved_season}"
            )
        model_info = (
            predictions[0].model_info if predictions else _model_info(self._games.load().metadata)
        )
        return GamePredictionSlate(
            season=resolved_season,
            week=resolved_week,
            predictions=predictions,
            game_count=len(predictions),
            skipped=skipped,
            model_info=model_info,
        )

    # --- fantasy projection ---------------------------------------------------

    def project_fantasy_points(
        self,
        player: str,
        *,
        season: int | None = None,
        week: int | None = None,
        scoring: FantasyScoring = MODELLED_SCORING,
    ) -> FantasyProjection:
        """Project PPR points for one player in one week, given that he plays.

        Raises:
            ValueError: A scoring format other than PPR was requested. Only PPR
                is modelled in this version, and returning a PPR number labelled
                as standard scoring would be worse than refusing.
            PlayerNotFoundError: The identifier resolves to nobody.
            AmbiguousPlayerError: The name matches several players.
            InsufficientHistoryError: Too few games played to build features.
        """
        self._require_modelled_scoring(scoring)
        player_id = self._directory.resolve(player)
        resolved_season = self._resolve_season(season)
        row, resolved_week = self._fantasy_row(player_id, resolved_season, week)
        return self._to_projection(self._fantasy.predict_row(row), row, resolved_week)

    def compare_projections(
        self,
        players: list[str],
        *,
        season: int | None = None,
        week: int | None = None,
        scoring: FantasyScoring = MODELLED_SCORING,
    ) -> FantasyProjectionComparison:
        """Project several players for one week and name the highest.

        Players who cannot be projected become caveats rather than errors, so a
        start/sit question about three players still gets an answer about the
        two the model can speak to.
        """
        self._require_modelled_scoring(scoring)
        if len(players) < 2:
            raise ValueError("Comparing projections needs at least two players.")
        if len(players) > MAX_COMPARED_PLAYERS:
            raise ValueError(f"At most {MAX_COMPARED_PLAYERS} players can be compared at once.")

        resolved_season = self._resolve_season(season)
        projections: list[FantasyProjection] = []
        caveats: list[str] = []
        resolved_week = week

        for entry in players:
            try:
                projection = self.project_fantasy_points(
                    entry, season=resolved_season, week=resolved_week, scoring=scoring
                )
            except (InsufficientHistoryError, NflPlatformError, ValueError) as exc:
                caveats.append(f"{entry}: {exc}")
                continue
            projections.append(projection)
            resolved_week = projection.week

        ranked = sorted(projections, key=lambda item: item.projected_points, reverse=True)
        margin = (
            round(ranked[0].projected_points - ranked[1].projected_points, 2)
            if len(ranked) >= 2
            else None
        )
        if margin is not None and margin < 1.0:
            caveats.append(
                f"The projections differ by {margin:.2f} points, which is well inside the "
                "model's typical error. Treat this as a coin flip rather than a recommendation."
            )
        caveats.append(
            "All projections are conditional on the player playing. Confirm availability "
            "before acting on them."
        )

        model_info = ranked[0].model_info if ranked else _model_info(self._fantasy.load().metadata)
        return FantasyProjectionComparison(
            season=resolved_season,
            week=resolved_week if resolved_week is not None else 0,
            scoring=scoring,
            projections=ranked,
            recommended_player_id=ranked[0].player.player_id if ranked else None,
            margin=margin,
            model_info=model_info,
            caveats=caveats,
        )

    # --- input assembly -------------------------------------------------------

    def _require_modelled_scoring(self, scoring: FantasyScoring) -> None:
        if scoring is not MODELLED_SCORING:
            raise ValueError(
                f"Only {MODELLED_SCORING.value} projections are modelled in this version. "
                "Use the deterministic fantasy endpoints for standard-scoring production."
            )

    def _resolve_season(self, season: int | None) -> int:
        if season is None:
            return self._loader.default_season
        return validate_season(season, latest_season=self._loader.current_season)

    def _history_seasons(self, season: int) -> list[int]:
        """The target season plus enough earlier ones to fill the form windows."""
        earliest = max(1999, season - HISTORY_SEASONS)
        return list(range(earliest, season + 1))

    def _game_inputs(self, season: int) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        seasons = self._history_seasons(season)
        schedules = self._loader.load_schedules(seasons)
        try:
            team_stats = self._loader.load_team_stats(seasons)
        except NflPlatformError:
            # Efficiency features degrade to null; the scoreboard features and
            # the baseline still work, and the caveat list says so.
            logger.warning(
                "Team stats unavailable; game features degrade", extra={"season": season}
            )
            team_stats = None
        return schedules, team_stats

    def _find_week(self, schedules: pl.DataFrame, season: int, home: str, away: str) -> int:
        match = schedules.filter(
            (pl.col("season") == season)
            & (pl.col("home_team") == home)
            & (pl.col("away_team") == away)
        ).sort("week")
        if match.is_empty():
            raise TeamNotFoundError(
                home, f"the season {season} schedule as the home team against {away}"
            )
        return int(match.item(0, "week"))

    def _synthetic_game_row(
        self,
        schedules: pl.DataFrame,
        team_stats: pl.DataFrame | None,
        season: int,
        week: int,
        home: str,
        away: str,
    ) -> pl.DataFrame:
        """Build a feature row for a matchup that is not on the schedule.

        Hypothetical matchups are a legitimate question ("who would win if these
        two met this week"), and answering them by appending a synthetic fixture
        keeps the feature path identical to the real one. The venue columns
        default to an outdoor grass stadium with equal rest, which is recorded
        as a caveat rather than presented as fact.
        """
        template = schedules.filter(pl.col("season") == season)
        if template.is_empty():
            raise TeamNotFoundError(home, f"the season {season} schedule")
        for team in (home, away):
            appears = template.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))
            if appears.is_empty():
                raise TeamNotFoundError(team, f"the season {season} schedule")

        synthetic = template.head(1).with_columns(
            pl.lit(f"{season}_{week:02d}_{away}_{home}_hypothetical").alias("game_id"),
            pl.lit(week).cast(pl.Int32).alias("week"),
            pl.lit(home).alias("home_team"),
            pl.lit(away).alias("away_team"),
            pl.lit(None, dtype=pl.Float64).alias("home_score"),
            pl.lit(None, dtype=pl.Float64).alias("away_score"),
            pl.lit(None, dtype=pl.Float64).alias("result"),
            pl.lit(7.0).alias("home_rest"),
            pl.lit(7.0).alias("away_rest"),
            pl.lit(0).alias("div_game"),
            pl.lit("outdoors").alias("roof"),
            pl.lit("grass").alias("surface"),
            pl.lit("Home").alias("location"),
            pl.lit(None, dtype=pl.Float64).alias("spread_line"),
            pl.lit(None, dtype=pl.Float64).alias("total_line"),
        )
        combined = pl.concat([schedules, synthetic], how="diagonal_relaxed")
        features = self._games.feature_frame(combined, team_stats, season=season, week=week)
        row = features.filter((pl.col("home_team") == home) & (pl.col("away_team") == away))
        if row.is_empty():
            raise TeamNotFoundError(home, f"the season {season} week {week} schedule")
        return row

    def _fantasy_row(
        self, player_id: str, season: int, week: int | None
    ) -> tuple[pl.DataFrame, int]:
        seasons = self._history_seasons(season)
        player_stats = self._loader.load_player_stats(seasons)
        schedules = self._loader.load_schedules(seasons)
        team_stats = self._optional(lambda: self._loader.load_team_stats(seasons))
        snap_counts = self._optional(lambda: self._loader.load_snap_counts(seasons))
        opportunity = self._optional(lambda: self._loader.load_fantasy_opportunity(seasons))
        players = self._optional(self._loader.load_players)

        resolved_week = (
            week if week is not None else self._latest_player_week(player_stats, player_id, season)
        )
        features = self._fantasy.feature_frame(
            player_stats,
            schedules,
            season=season,
            week=validate_week(resolved_week),
            team_stats=team_stats,
            snap_counts=snap_counts,
            ff_opportunity=opportunity,
            players=players,
        )
        row = features.filter(pl.col("player_id") == player_id)
        if row.is_empty():
            raise InsufficientHistoryError(
                f"player {player_id} in {season} week {resolved_week}",
                0,
                self._settings.ml_min_history_games,
            )
        return row.head(1), int(resolved_week)

    def _latest_player_week(self, player_stats: pl.DataFrame, player_id: str, season: int) -> int:
        rows = player_stats.filter(
            (pl.col("player_id") == player_id) & (pl.col("season") == season)
        )
        if rows.is_empty():
            raise InsufficientHistoryError(
                f"player {player_id} in season {season}", 0, self._settings.ml_min_history_games
            )
        latest = _optional_number(rows.get_column("week").max())
        return int(latest) if latest is not None else 1

    @staticmethod
    def _optional(load: Any) -> pl.DataFrame | None:
        """Load an auxiliary dataset, degrading to None when it is unavailable.

        Snap counts, opportunity data and the player crosswalk each improve the
        feature row but none is required. Failing the whole projection because
        one optional dataset is unreachable would trade a good answer for no
        answer.
        """
        try:
            return load()  # type: ignore[no-any-return]
        except NflPlatformError:
            return None

    # --- DTO conversion -------------------------------------------------------

    def _to_game_prediction(
        self, result: PredictionResult, row: pl.DataFrame, schedules: pl.DataFrame
    ) -> GamePrediction:
        home = str(row.item(0, "home_team"))
        away = str(row.item(0, "away_team"))
        probability = round(float(result.value), 4)
        spread = result.extras.get("spread_line")

        market: float | None = None
        try:
            benchmark = self._registry.load(GAME_PROBLEM, "v1_market_benchmark")
        except NflPlatformError:
            benchmark = None
        if benchmark is not None and spread is not None:
            market = round(
                float(benchmark.model.predict(np.array([[float(spread)]], dtype=np.float64))[0]), 4
            )

        return GamePrediction(
            season=int(row.item(0, "season")),
            week=int(row.item(0, "week")),
            home_team=home,
            away_team=away,
            game=self._game_context(schedules, str(row.item(0, "game_id"))),
            home_win_probability=probability,
            away_win_probability=round(1.0 - probability, 4),
            predicted_winner=home if probability >= 0.5 else away,
            baseline_home_win_probability=(
                None if result.baseline_value is None else round(result.baseline_value, 4)
            ),
            market_home_win_probability=market,
            top_factors=_contributions(result),
            model_info=_model_info(result.metadata),
            basis=PredictionBasis(
                as_of_season=int(row.item(0, "season")),
                as_of_week=int(row.item(0, "week")),
                feature_count=len(result.metadata.feature_names),
                history_games=result.history,
                notes=[
                    "Features use only games completed before this week.",
                    "Betting lines are not model inputs; the market probability is shown "
                    "for reference only.",
                ],
            ),
            caveats=result.caveats,
        )

    def _game_context(self, schedules: pl.DataFrame, game_id: str) -> GameContext | None:
        match = schedules.filter(pl.col("game_id") == game_id)
        if match.is_empty():
            return None
        row = match.head(1).to_dicts()[0]
        return GameContext(
            game_id=str(row.get("game_id")),
            season=int(row["season"]),
            week=int(row["week"]),
            game_type=_optional_str(row.get("game_type")),
            gameday=_optional_str(row.get("gameday")),
            home_team=str(row.get("home_team")),
            away_team=str(row.get("away_team")),
            roof=_optional_str(row.get("roof")),
            surface=_optional_str(row.get("surface")),
            spread_line=_optional_number(row.get("spread_line")),
            total_line=_optional_number(row.get("total_line")),
        )

    def _to_projection(
        self, result: PredictionResult, row: pl.DataFrame, week: int
    ) -> FantasyProjection:
        player_id = str(row.item(0, "player_id"))
        name, position, team = self._directory.reference(player_id)
        return FantasyProjection(
            player=PlayerRef(player_id=player_id, name=name, position=position, team=team),
            season=int(row.item(0, "season")),
            week=week,
            opponent=_optional_str(row.item(0, "opponent")),
            scoring=MODELLED_SCORING,
            projected_points=round(float(result.value), 2),
            conditional_on_playing=True,
            baseline_projected_points=(
                None if result.baseline_value is None else round(result.baseline_value, 2)
            ),
            typical_error=(
                None
                if result.extras.get("typical_error") is None
                else round(float(result.extras["typical_error"]), 2)
            ),
            top_factors=_contributions(result),
            model_info=_model_info(result.metadata),
            basis=PredictionBasis(
                as_of_season=int(row.item(0, "season")),
                as_of_week=week,
                feature_count=len(result.metadata.feature_names),
                history_games=result.history,
                notes=[
                    "Usage and production windows cover only games played before this week.",
                    "Positions modelled: " + ", ".join(MODELLED_POSITIONS) + ".",
                ],
            ),
            caveats=result.caveats,
        )


def _model_info(metadata: ModelMetadata) -> ModelInfo:
    return ModelInfo(
        model_id=metadata.model_id,
        problem=PredictionProblem(metadata.problem),
        version=metadata.version,
        algorithm=metadata.algorithm,
        role=ModelRole(metadata.role),
        trained_at=metadata.trained_at,
        training_seasons=list(metadata.fit_seasons or metadata.train_seasons),
        validation_seasons=list(metadata.validation_seasons),
        test_seasons=list(metadata.test_seasons),
        feature_count=len(metadata.feature_names),
        feature_schema_version=metadata.feature_schema_version,
        data_snapshot=metadata.data_snapshot,
        primary_metric=metadata.primary_metric,
        validation_score=metadata.validation_score,
        calibration_method=metadata.calibration_method,
    )


def _contributions(result: PredictionResult) -> list[FeatureContribution]:
    return [
        FeatureContribution(
            feature=name,
            value=None if value is None else round(value, 4),
            contribution=round(contribution, 4),
            direction="increases" if contribution >= 0 else "decreases",
        )
        for name, value, contribution in result.contributions
    ]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


__all__ = ["FANTASY_PROBLEM", "GAME_PROBLEM", "MODELLED_SCORING", "PredictionService"]
