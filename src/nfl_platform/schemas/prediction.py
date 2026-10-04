"""Prediction DTOs.

These are deliberately *not* shaped like the analytics DTOs. A
:class:`~nfl_platform.schemas.team.TeamOffenseSummary` reports what happened and
needs no provenance beyond the window it covers. A prediction is an estimate
about a game that has not been played, so three things travel with every one of
them and none is optional:

``model_info``
    Which artifact produced the number, trained on which seasons, scoring what
    on validation. Without it "61%" is indistinguishable from a guess.

``basis``
    How much history the features were actually built from. A week-1 prediction
    and a week-12 prediction are not the same kind of claim even when the
    probability is identical.

``caveats``
    Stated in the payload rather than left to the presentation layer, because
    the two consumers most likely to drop them -- a language model writing prose
    and a JSON client -- are exactly the ones that most need them.

Every DTO here carries ``model_info``, and no DTO outside this module does. That
asymmetry is the machine-readable form of "deterministic analytics and model
output are different kinds of claim".
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import ConfigDict, Field

from nfl_platform.schemas.common import BaseDTO, GameContext, PlayerRef
from nfl_platform.schemas.fantasy import FantasyScoring


class PredictionProblem(StrEnum):
    """The prediction problems this platform models."""

    GAME_WIN = "game_win"
    FANTASY_POINTS = "fantasy_points"


class ModelRole(StrEnum):
    """What a fitted estimator is for.

    ``BASELINE`` exists as a first-class role rather than a footnote: a baseline
    that is only ever mentioned in a report is easy to quietly stop comparing
    against, and a baseline that wins its comparison has to be shippable for the
    promotion rule to mean anything.
    """

    BASELINE = "baseline"
    CANDIDATE = "candidate"
    BENCHMARK = "benchmark"


class PredictionDTO(BaseDTO):
    """Base for prediction payloads.

    ``protected_namespaces`` is cleared so fields may be named ``model_id`` and
    ``model_info``. Pydantic reserves the ``model_`` prefix for its own methods;
    renaming the fields to dodge it would make the wire format worse to read in
    order to make the library quieter.
    """

    model_config = ConfigDict(protected_namespaces=())


class ModelInfo(PredictionDTO):
    """Which artifact produced a prediction, and how well it did on validation."""

    model_id: str = Field(description="Stable artifact identifier, '<problem>/<version>'.")
    problem: PredictionProblem
    version: str = Field(description="Artifact version, e.g. 'v1_logistic'.")
    algorithm: str = Field(description="Estimator family, e.g. 'logistic_regression'.")
    role: ModelRole = Field(
        description="Whether this artifact is the simple baseline or the stronger candidate."
    )
    trained_at: str = Field(description="UTC ISO-8601 timestamp of the training run.")
    training_seasons: list[int] = Field(description="Seasons the final fit was trained on.")
    validation_seasons: list[int] = Field(description="Seasons used for model selection.")
    test_seasons: list[int] = Field(
        description="Frozen final test period, scored once and never tuned against."
    )
    feature_count: int = Field(ge=0)
    feature_schema_version: str = Field(
        description="Feature-builder contract version. A change invalidates the artifact."
    )
    data_snapshot: str | None = Field(
        default=None, description="Pinned input snapshot the training run read."
    )
    primary_metric: str = Field(description="Metric model selection was decided on.")
    validation_score: float | None = Field(
        default=None, description="Score of primary_metric on the validation folds."
    )
    calibration_method: str | None = Field(
        default=None,
        description="How probabilities were calibrated. Null for regression models.",
    )


class ModelRegistryStatus(PredictionDTO):
    """Which prediction capabilities are usable on this deployment."""

    available: bool = Field(description="True when at least one model is installed and loadable.")
    enabled: bool = Field(description="False when predictions are switched off by configuration.")
    model_dir: str
    models: list[ModelInfo] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PredictionBasis(PredictionDTO):
    """What the feature row was actually built from.

    Two predictions with the same probability are not equally informative if one
    was built from twelve games and the other from two, so the counts are part
    of the payload rather than something a caller has to infer.
    """

    as_of_season: int
    as_of_week: int = Field(description="Features use only games completed before this week.")
    feature_count: int = Field(ge=0)
    history_games: dict[str, int] = Field(
        default_factory=dict,
        description="Completed games available per subject, e.g. {'KC': 11, 'BUF': 10}.",
    )
    notes: list[str] = Field(default_factory=list)


class FeatureContribution(PredictionDTO):
    """One feature's push on a single prediction.

    SHAP values for tree models, standardised coefficient products for linear
    ones. Both answer "why this number", which keeps the explanation in the
    prediction layer where it is derived, rather than in a language model where
    it would be invented.
    """

    feature: str
    value: float | None = Field(default=None, description="The feature's value for this subject.")
    contribution: float = Field(description="Signed push on the model output, in model units.")
    direction: str = Field(description="'increases' or 'decreases' the predicted value.")


class GamePrediction(PredictionDTO):
    """Estimated probability that the home team wins one game."""

    season: int
    week: int
    home_team: str
    away_team: str
    game: GameContext | None = Field(
        default=None, description="Schedule row, when the game is on the published schedule."
    )
    home_win_probability: float = Field(ge=0.0, le=1.0)
    away_win_probability: float = Field(ge=0.0, le=1.0)
    predicted_winner: str = Field(description="Team abbreviation with the higher probability.")
    baseline_home_win_probability: float | None = Field(
        default=None,
        description="What the simple baseline says, so the model's contribution is visible.",
    )
    market_home_win_probability: float | None = Field(
        default=None,
        description=(
            "Reference probability implied by the closing spread, when one is published. "
            "A benchmark only: market information is not an input to the model."
        ),
    )
    top_factors: list[FeatureContribution] = Field(default_factory=list)
    model_info: ModelInfo
    basis: PredictionBasis
    caveats: list[str] = Field(default_factory=list)


class GamePredictionSlate(PredictionDTO):
    """Every modelled game in one week."""

    season: int
    week: int
    predictions: list[GamePrediction] = Field(default_factory=list)
    game_count: int = Field(ge=0)
    skipped: list[str] = Field(
        default_factory=list,
        description="Games that could not be predicted, with the reason for each.",
    )
    model_info: ModelInfo


class FantasyProjection(PredictionDTO):
    """Estimated fantasy production for one player in one week.

    ``conditional_on_playing`` is the load-bearing field. The model was trained
    only on games the player actually took a snap in, so the projection answers
    "how much if he plays", not "how much in expectation". Those differ by the
    entire probability of being inactive, and a consumer that conflates them
    will start an injured player on a confident-looking number.
    """

    player: PlayerRef
    season: int
    week: int
    opponent: str | None = None
    scoring: FantasyScoring
    projected_points: float
    floor_points: float | None = Field(
        default=None, description="Lower quantile of the projection, when quantile models exist."
    )
    ceiling_points: float | None = Field(
        default=None, description="Upper quantile of the projection, when quantile models exist."
    )
    interval_level: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Coverage of the floor-to-ceiling interval."
    )
    conditional_on_playing: bool = Field(
        default=True,
        description=(
            "True: this is production given the player takes a snap. It does not "
            "include any probability that he is inactive."
        ),
    )
    baseline_projected_points: float | None = Field(
        default=None, description="What the simple baseline projects, for contrast."
    )
    typical_error: float | None = Field(
        default=None,
        description=(
            "Validation mean absolute error for this model, as a scale reference. "
            "Not an interval for this specific player."
        ),
    )
    top_factors: list[FeatureContribution] = Field(default_factory=list)
    model_info: ModelInfo
    basis: PredictionBasis
    caveats: list[str] = Field(default_factory=list)


class FantasyProjectionComparison(PredictionDTO):
    """Side-by-side projections for a start/sit decision."""

    season: int
    week: int
    scoring: FantasyScoring
    projections: list[FantasyProjection] = Field(default_factory=list)
    recommended_player_id: str | None = Field(
        default=None, description="Highest projected player. Null when none could be projected."
    )
    margin: float | None = Field(
        default=None, description="Projected points between the top two players."
    )
    conditional_on_playing: bool = True
    model_info: ModelInfo
    caveats: list[str] = Field(default_factory=list)
