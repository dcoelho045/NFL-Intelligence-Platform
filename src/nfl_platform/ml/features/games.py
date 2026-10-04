"""Game-prediction features.

One row per game, from the home team's point of view, built in three stages:

``team_game_frame``
    Schedules and team box scores reshaped into one row per team per game. The
    schedule is the spine, so unplayed games survive and become predictable
    rows; the box score is a left join, so a missing one degrades to nulls
    instead of dropping the game.

``team_form``
    Each team's accumulated state *after* every completed game. Computed only
    from finished games, so it cannot contain anything about a game that has not
    happened.

``build_game_features``
    Joins both teams' form as of strictly before kickoff and reduces every pair
    to a home-minus-away difference.

Everything the model sees is a difference. Two reasons: with roughly 250 usable
rows per season, halving the feature count is worth more than the absolute
levels would add; and a difference is sign-symmetric in a way that matches the
question -- "who is better here" rather than "how good is each".

Market lines are read and carried through for the benchmark in
:mod:`nfl_platform.ml.models.baselines`, but no market column appears in
:data:`GAME_FEATURES`. A model given the closing spread mostly learns to restate
it, which would measure the betting market rather than this pipeline.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from nfl_platform.ml.features.schema import Availability, FeatureSet, FeatureSpec
from nfl_platform.ml.features.windows import (
    TIME_INDEX,
    join_as_of,
    safe_ratio,
    time_index,
    trailing_count,
    trailing_observed,
    trailing_sums,
)

#: Bump when any feature's meaning changes. Artifacts record it and refuse to
#: load against a different one, so a silent redefinition cannot be scored.
GAME_FEATURE_SCHEMA_VERSION: Final = "game-v1"

#: Long window: roughly half a season, long enough to be stable and short enough
#: to track a team that changed in-season.
FORM_WINDOW: Final = 8
#: Short window for "current form", where recency is the point.
RECENT_WINDOW: Final = 5
#: Rows below this are dropped from training. A one-game window is noise, and
#: teaching the model to trust it would make week 1 actively misleading.
MIN_TRAINING_HISTORY: Final = 4

LABEL: Final = "home_win"
GAME_KEY: Final = "game_id"

_DOME_ROOFS: Final = frozenset({"dome", "closed", "retractable"})
_TURF_SURFACES: Final = frozenset(
    {"astroturf", "fieldturf", "matrixturf", "sportturf", "a_turf", "astroplay"}
)

# Box-score columns read from team_stats. Narrow on purpose: every extra column
# is another thing that can change shape upstream and break a training run.
_BOX_COLUMNS: Final = (
    "attempts",
    "sacks_suffered",
    "carries",
    "passing_epa",
    "rushing_epa",
    "passing_yards",
    "sack_yards_lost",
    "rushing_yards",
    "passing_first_downs",
    "rushing_first_downs",
    "passing_interceptions",
    "rushing_fumbles_lost",
    "sack_fumbles_lost",
    "receiving_fumbles_lost",
    "penalty_yards",
)

# Quantities accumulated per team. Rates are formed from these afterwards, so a
# window is a ratio of totals rather than a mean of per-game ratios.
_ACCUMULATED: Final = (
    "points_for",
    "points_against",
    "wins",
    "plays_off",
    "epa_off",
    "yards_off",
    "first_downs_off",
    "turnovers_off",
    "dropbacks_off",
    "sacks_allowed",
    "penalty_yards_off",
    "plays_def",
    "epa_def",
    "yards_def",
    "first_downs_def",
    "turnovers_def",
    "dropbacks_def",
    "sacks_forced",
)

_LONG: Final = f"_l{FORM_WINDOW}"
_SHORT: Final = f"_r{RECENT_WINDOW}"
_SEEN: Final = f"_seen_l{FORM_WINDOW}"

#: Per-game rates whose source is the box score rather than the scoreboard. Their
#: denominator is the number of games that actually carried a box score, so a
#: missing team_stats dataset yields "unknown" rather than a confident zero.
_BOX_RATES: Final = ("turnovers_off", "turnovers_def", "penalty_yards_off")

#: Form columns, named as they appear on a team row before differencing.
_TEAM_FORM_COLUMNS: Final = (
    "off_epa_per_play",
    "def_epa_per_play",
    "points_for_per_game",
    "points_against_per_game",
    "off_yards_per_play",
    "def_yards_per_play",
    "off_first_down_rate",
    "def_first_down_rate",
    "turnovers_per_game",
    "takeaways_per_game",
    "sack_rate_allowed",
    "sack_rate_forced",
    "penalty_yards_per_game",
    "recent_win_rate",
    "recent_point_margin",
)

GAME_FEATURES: Final = FeatureSet(
    GAME_FEATURE_SCHEMA_VERSION,
    (
        FeatureSpec(
            "off_epa_per_play_diff",
            "Home minus away offensive expected points added per play.",
            "team_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "def_epa_per_play_diff",
            "Home minus away expected points added allowed per defensive play. "
            "Negative favours the home team.",
            "team_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "points_for_per_game_diff",
            "Home minus away points scored per game.",
            "schedules",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "points_against_per_game_diff",
            "Home minus away points allowed per game. Negative favours the home team.",
            "schedules",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "off_yards_per_play_diff",
            "Home minus away net yards per offensive play, sack yardage deducted.",
            "team_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "def_yards_per_play_diff",
            "Home minus away net yards allowed per defensive play.",
            "team_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "off_first_down_rate_diff",
            "Home minus away offensive first downs per play. A success-rate proxy that "
            "needs no play-by-play data.",
            "team_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "def_first_down_rate_diff",
            "Home minus away first downs allowed per defensive play.",
            "team_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "turnovers_per_game_diff",
            "Home minus away giveaways per game: interceptions plus lost fumbles.",
            "team_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "takeaways_per_game_diff",
            "Home minus away takeaways forced per game.",
            "team_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "sack_rate_allowed_diff",
            "Home minus away sacks allowed per dropback.",
            "team_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "sack_rate_forced_diff",
            "Home minus away sacks forced per opponent dropback.",
            "team_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "penalty_yards_per_game_diff",
            "Home minus away penalty yards conceded per game.",
            "team_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} completed games",
        ),
        FeatureSpec(
            "recent_win_rate_diff",
            "Home minus away win rate. Ties count as half a win.",
            "schedules",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} completed games",
        ),
        FeatureSpec(
            "recent_point_margin_diff",
            "Home minus away average scoring margin.",
            "schedules",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} completed games",
        ),
        FeatureSpec(
            "rest_days_diff",
            "Home minus away days since each team's previous game.",
            "schedules",
            Availability.SCHEDULE,
        ),
        FeatureSpec(
            "div_game",
            "1 when the teams share a division.",
            "schedules",
            Availability.SCHEDULE,
        ),
        FeatureSpec(
            "is_dome",
            "1 when the roof is closed, fixed or retractable.",
            "schedules",
            Availability.SCHEDULE,
        ),
        FeatureSpec(
            "is_turf",
            "1 when the playing surface is artificial.",
            "schedules",
            Availability.SCHEDULE,
        ),
        FeatureSpec(
            "neutral_site",
            "1 when neither team is really at home, so the home-field term should not apply.",
            "schedules",
            Availability.SCHEDULE,
        ),
        FeatureSpec(
            "week",
            "Week number. Lets the model discount early-season form built on few games.",
            "schedules",
            Availability.SCHEDULE,
        ),
        FeatureSpec(
            "history_games_min",
            "Completed games behind the less-established of the two teams.",
            "schedules",
            Availability.PRIOR_GAMES,
            f"capped at {FORM_WINDOW}",
        ),
    ),
)


def _flag(column: str, members: frozenset[str], alias: str) -> pl.Expr:
    """Case-insensitive membership flag, 0 when the column is null."""
    return (
        (pl.col(column).cast(pl.Utf8).str.to_lowercase().str.strip_chars().is_in(list(members)))
        .fill_null(False)
        .cast(pl.Int8)
        .alias(alias)
    )


def _sum_of(columns: tuple[str, ...]) -> pl.Expr:
    return pl.sum_horizontal([pl.col(c).fill_null(0) for c in columns])


def build_team_game_frame(
    schedules: pl.DataFrame, team_stats: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Reshape schedules and box scores into one row per team per game.

    The schedule is the spine so that scheduled-but-unplayed games remain in the
    frame. They are what inference predicts, and keeping them here means
    training and inference traverse the same code rather than two near-copies.

    Args:
        schedules: nflverse schedules, one row per game.
        team_stats: nflverse weekly team stats. Optional; without it the
            efficiency columns are null and only the scoreboard features exist.

    Returns:
        One row per team per game with scoreboard, venue and box-score columns,
        plus ``completed`` marking which rows may contribute to history.
    """
    shared = [
        pl.col("game_id"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("game_type"),
        pl.col("spread_line").cast(pl.Float64, strict=False),
        pl.col("total_line").cast(pl.Float64, strict=False),
        pl.col("div_game").cast(pl.Int8, strict=False).fill_null(0),
        _flag("roof", _DOME_ROOFS, "is_dome"),
        _flag("surface", _TURF_SURFACES, "is_turf"),
        (pl.col("location").cast(pl.Utf8).str.to_lowercase() == "neutral")
        .fill_null(False)
        .cast(pl.Int8)
        .alias("neutral_site"),
    ]

    def side(
        team: str, opponent: str, score: str, against: str, rest: str, home: int
    ) -> pl.DataFrame:
        return schedules.select(
            *shared,
            pl.col(team).alias("team"),
            pl.col(opponent).alias("opponent"),
            pl.lit(home, dtype=pl.Int8).alias("is_home"),
            pl.col(score).cast(pl.Float64, strict=False).alias("points_for"),
            pl.col(against).cast(pl.Float64, strict=False).alias("points_against"),
            pl.col(rest).cast(pl.Float64, strict=False).alias("rest_days"),
        )

    spine = pl.concat(
        [
            side("home_team", "away_team", "home_score", "away_score", "home_rest", 1),
            side("away_team", "home_team", "away_score", "home_score", "away_rest", 0),
        ],
        how="vertical",
    ).with_columns(
        (pl.col("points_for").is_not_null() & pl.col("points_against").is_not_null()).alias(
            "completed"
        ),
        (pl.col("points_for") - pl.col("points_against")).alias("point_margin"),
        time_index(),
    )
    # A tie is half a win. Dropping ties from the label is right -- binary
    # classification has no honest home for them -- but erasing them from a
    # team's form record would be a different and wrong claim about its season.
    spine = spine.with_columns(
        pl.when(pl.col("point_margin") > 0)
        .then(1.0)
        .when(pl.col("point_margin") < 0)
        .then(0.0)
        .otherwise(0.5)
        .alias("wins")
    )

    return _attach_box_scores(spine, team_stats)


def _attach_box_scores(spine: pl.DataFrame, team_stats: pl.DataFrame | None) -> pl.DataFrame:
    """Join each team's offensive box score, and its opponent's as defence allowed."""
    offence_columns = (
        "plays_off",
        "epa_off",
        "yards_off",
        "first_downs_off",
        "turnovers_off",
        "dropbacks_off",
        "sacks_allowed",
        "penalty_yards_off",
    )
    if team_stats is None or team_stats.is_empty():
        return spine.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in offence_columns]
            + [
                pl.lit(None, dtype=pl.Float64).alias(c)
                for c in ("plays_def", "epa_def", "yards_def", "first_downs_def")
            ]
            + [
                pl.lit(None, dtype=pl.Float64).alias(c)
                for c in ("turnovers_def", "dropbacks_def", "sacks_forced")
            ]
        )

    available = [c for c in _BOX_COLUMNS if c in team_stats.columns]
    box = team_stats.select(
        pl.col("game_id"),
        pl.col("team"),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in available],
    )
    for column in _BOX_COLUMNS:
        if column not in available:
            box = box.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))

    box = box.select(
        "game_id",
        "team",
        (_sum_of(("attempts", "sacks_suffered"))).alias("dropbacks_off"),
        (_sum_of(("attempts", "sacks_suffered", "carries"))).alias("plays_off"),
        (_sum_of(("passing_epa", "rushing_epa"))).alias("epa_off"),
        (
            pl.col("passing_yards").fill_null(0)
            - pl.col("sack_yards_lost").fill_null(0)
            + pl.col("rushing_yards").fill_null(0)
        ).alias("yards_off"),
        (_sum_of(("passing_first_downs", "rushing_first_downs"))).alias("first_downs_off"),
        (
            _sum_of(
                (
                    "passing_interceptions",
                    "rushing_fumbles_lost",
                    "sack_fumbles_lost",
                    "receiving_fumbles_lost",
                )
            )
        ).alias("turnovers_off"),
        pl.col("sacks_suffered").fill_null(0).alias("sacks_allowed"),
        pl.col("penalty_yards").fill_null(0).alias("penalty_yards_off"),
    )

    defence = box.rename(
        {
            "team": "opponent",
            "dropbacks_off": "dropbacks_def",
            "plays_off": "plays_def",
            "epa_off": "epa_def",
            "yards_off": "yards_def",
            "first_downs_off": "first_downs_def",
            "turnovers_off": "turnovers_def",
            "sacks_allowed": "sacks_forced",
            "penalty_yards_off": "penalty_yards_def",
        }
    )

    return spine.join(box, on=["game_id", "team"], how="left").join(
        defence, on=["game_id", "opponent"], how="left"
    )


def build_team_form(team_game: pl.DataFrame) -> pl.DataFrame:
    """Each team's accumulated state after every completed game.

    Only completed games enter. That single filter is what makes the result
    leak-proof by construction: a frame that never saw an unplayed game cannot
    encode anything about one.
    """
    completed = team_game.filter(pl.col("completed")).sort(["team", TIME_INDEX])
    if completed.is_empty():
        return completed

    long_form = trailing_sums(
        completed,
        group_by=["team"],
        order_by=[TIME_INDEX],
        columns=_ACCUMULATED,
        window=FORM_WINDOW,
        suffix=_LONG,
    )
    long_form = trailing_count(
        long_form,
        group_by=["team"],
        order_by=[TIME_INDEX],
        alias="history_games",
        window=FORM_WINDOW,
    )
    form = trailing_sums(
        long_form,
        group_by=["team"],
        order_by=[TIME_INDEX],
        columns=("wins", "point_margin"),
        window=RECENT_WINDOW,
        suffix=_SHORT,
    )
    form = trailing_count(
        form,
        group_by=["team"],
        order_by=[TIME_INDEX],
        alias="recent_games",
        window=RECENT_WINDOW,
    )
    form = trailing_observed(
        form,
        group_by=["team"],
        order_by=[TIME_INDEX],
        columns=_BOX_RATES,
        window=FORM_WINDOW,
        suffix=_SEEN,
    )

    return form.with_columns(
        safe_ratio(f"epa_off{_LONG}", f"plays_off{_LONG}", "off_epa_per_play"),
        safe_ratio(f"epa_def{_LONG}", f"plays_def{_LONG}", "def_epa_per_play"),
        safe_ratio(f"points_for{_LONG}", "history_games", "points_for_per_game"),
        safe_ratio(f"points_against{_LONG}", "history_games", "points_against_per_game"),
        safe_ratio(f"yards_off{_LONG}", f"plays_off{_LONG}", "off_yards_per_play"),
        safe_ratio(f"yards_def{_LONG}", f"plays_def{_LONG}", "def_yards_per_play"),
        safe_ratio(f"first_downs_off{_LONG}", f"plays_off{_LONG}", "off_first_down_rate"),
        safe_ratio(f"first_downs_def{_LONG}", f"plays_def{_LONG}", "def_first_down_rate"),
        safe_ratio(f"turnovers_off{_LONG}", f"turnovers_off{_SEEN}", "turnovers_per_game"),
        safe_ratio(f"turnovers_def{_LONG}", f"turnovers_def{_SEEN}", "takeaways_per_game"),
        safe_ratio(f"sacks_allowed{_LONG}", f"dropbacks_off{_LONG}", "sack_rate_allowed"),
        safe_ratio(f"sacks_forced{_LONG}", f"dropbacks_def{_LONG}", "sack_rate_forced"),
        safe_ratio(
            f"penalty_yards_off{_LONG}",
            f"penalty_yards_off{_SEEN}",
            "penalty_yards_per_game",
        ),
        safe_ratio(f"wins{_SHORT}", "recent_games", "recent_win_rate"),
        safe_ratio(f"point_margin{_SHORT}", "recent_games", "recent_point_margin"),
    )


def build_game_features(
    schedules: pl.DataFrame,
    team_stats: pl.DataFrame | None = None,
    *,
    completed_only: bool = True,
) -> pl.DataFrame:
    """Build the one-row-per-game feature matrix.

    Args:
        schedules: nflverse schedules covering the target games **and** enough
            earlier seasons to fill the form windows.
        team_stats: nflverse weekly team stats over the same span.
        completed_only: Keep only games with a final score. True for training
            and evaluation; False when predicting an upcoming week.

    Returns:
        One row per game with identity columns, every name in
        :data:`GAME_FEATURES`, the ``home_win`` label where known, and the
        market columns used by the benchmark baseline.
    """
    team_game = build_team_game_frame(schedules, team_stats)
    form = build_team_form(team_game)

    targets = team_game.filter(pl.col("is_home") == 1)
    if completed_only:
        targets = targets.filter(pl.col("completed"))

    carried = [*_TEAM_FORM_COLUMNS, "history_games"]
    home = join_as_of(
        targets.select(
            GAME_KEY,
            "season",
            "week",
            "game_type",
            TIME_INDEX,
            pl.col("team").alias("home_team"),
            pl.col("opponent").alias("away_team"),
            pl.col("points_for").alias("home_score"),
            pl.col("points_against").alias("away_score"),
            pl.col("rest_days").alias("home_rest_days"),
            "div_game",
            "is_dome",
            "is_turf",
            "neutral_site",
            "spread_line",
            "total_line",
            "completed",
            pl.col("team"),
        ),
        form,
        by=["team"],
        columns=carried,
    ).rename({name: f"home_{name}" for name in carried})

    away_rest = targets.select(
        GAME_KEY, pl.col("rest_days").cast(pl.Float64).alias("away_rest_days")
    )
    away_targets = team_game.filter(pl.col("is_home") == 0).select(
        GAME_KEY, TIME_INDEX, pl.col("team")
    )
    away = join_as_of(away_targets, form, by=["team"], columns=carried).rename(
        {name: f"away_{name}" for name in carried}
    )

    merged = (
        home.drop("team")
        .join(away.drop("team", TIME_INDEX), on=GAME_KEY, how="left")
        .join(away_rest, on=GAME_KEY, how="left")
    )

    differences = [
        (pl.col(f"home_{name}") - pl.col(f"away_{name}")).cast(pl.Float64).alias(f"{name}_diff")
        for name in _TEAM_FORM_COLUMNS
    ]

    features = merged.with_columns(
        *differences,
        (pl.col("home_rest_days") - pl.col("away_rest_days"))
        .cast(pl.Float64)
        .alias("rest_days_diff"),
        pl.min_horizontal(
            pl.col("home_history_games").fill_null(0), pl.col("away_history_games").fill_null(0)
        )
        .cast(pl.Int32)
        .alias("history_games_min"),
        pl.when(pl.col("home_score") > pl.col("away_score"))
        .then(1)
        .when(pl.col("home_score") < pl.col("away_score"))
        .then(0)
        .otherwise(None)
        .cast(pl.Int8)
        .alias(LABEL),
    )

    identity = [
        GAME_KEY,
        "season",
        "week",
        "game_type",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "completed",
        "spread_line",
        "total_line",
        "home_history_games",
        "away_history_games",
        TIME_INDEX,
    ]
    # ``week`` is both an identity column and a feature, so the projection is
    # deduplicated rather than concatenated.
    projection = list(dict.fromkeys([*identity, *GAME_FEATURES.names, LABEL]))
    return features.select(projection).sort([TIME_INDEX, GAME_KEY])


def training_rows(
    features: pl.DataFrame, *, min_history: int = MIN_TRAINING_HISTORY
) -> pl.DataFrame:
    """Rows eligible for fitting and scoring.

    Three filters, each one a decision worth stating:

    *Completed only.* A game with no final score has no label.

    *Ties dropped.* Roughly two games in a thousand. Binary classification has
    no honest encoding for them, and inventing one to avoid losing 0.2% of rows
    would be a worse trade than losing them.

    *Minimum history.* A team with one prior game has a form window that is
    noise. Keeping those rows teaches the model that noisy windows are
    trustworthy, which is precisely wrong in week 1 where it matters most.
    """
    return features.filter(
        pl.col("completed")
        & pl.col(LABEL).is_not_null()
        & (pl.col("history_games_min") >= min_history)
    )
