"""Fantasy-projection features.

One row per player per game, restricted to the population the model is actually
asked about: quarterbacks, running backs, receivers and tight ends who took an
offensive snap.

## Why the projection is conditional on playing

Rows where a player did not play are **excluded** rather than labelled zero.
Including them would quietly change the problem. A player who is inactive scores
zero, so a model trained on those rows spends most of its capacity learning who
is likely to be inactive -- and its mean absolute error then measures
availability prediction rather than production. Two consequences follow, and
both are stated in the DTO rather than left for a reader to work out:

* the projection means "points if he plays", not "points in expectation";
* it must be combined with an availability judgement, which this milestone does
  not model, before it answers a start/sit question on its own.

Excluding the rows is itself a decision with a cost: the population is selected
using a fact known only after kickoff. That is acceptable precisely because the
evaluation population is defined the same way, so the reported error is the
error on the rows the projection claims to cover.

## Expected points

``total_fantasy_points_exp`` from nflverse's opportunity data is the single most
useful input here. Realised fantasy points are dominated by touchdown variance
that does not persist week to week; expected points describe the opportunity
that does. Carrying both, plus their difference, lets the model learn the
regression to the mean rather than chase the noise.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from nfl_platform.ml.features.games import build_team_form, build_team_game_frame
from nfl_platform.ml.features.schema import Availability, FeatureSet, FeatureSpec
from nfl_platform.ml.features.windows import (
    TIME_INDEX,
    join_as_of,
    safe_ratio,
    time_index,
    trailing_count,
    trailing_observed,
    trailing_std,
    trailing_sums,
)

FANTASY_FEATURE_SCHEMA_VERSION: Final = "fantasy-v1"

#: Short window: current role. Long window: established role.
RECENT_WINDOW: Final = 3
FORM_WINDOW: Final = 8
#: Opponent-versus-position defence needs a longer look; a single week of points
#: allowed to tight ends is almost entirely one player's touchdown.
OPPONENT_WINDOW: Final = 8
MIN_TRAINING_HISTORY: Final = 3

LABEL: Final = "fantasy_points_ppr"
STANDARD_LABEL: Final = "fantasy_points"
MODELLED_POSITIONS: Final = ("QB", "RB", "WR", "TE")

_RECENT: Final = f"_r{RECENT_WINDOW}"
_LONG: Final = f"_l{FORM_WINDOW}"
_OPP: Final = f"_o{OPPONENT_WINDOW}"
_SEEN_RECENT: Final = f"_seen_r{RECENT_WINDOW}"
_SEEN_LONG: Final = f"_seen_l{FORM_WINDOW}"

_PLAYER_ACCUMULATED: Final = (
    "fantasy_points_ppr",
    "exp_fantasy_points",
    "offense_pct",
    "target_share",
    "air_yards_share",
    "wopr",
    "targets",
    "carries",
    "attempts",
    "receptions",
)

#: Columns whose average is only meaningful over the games that actually
#: recorded them. See :func:`trailing_observed` for why the game count is the
#: wrong denominator for these.
_PLAYER_RATES: Final = (
    "exp_fantasy_points",
    "offense_pct",
    "target_share",
    "air_yards_share",
    "wopr",
)

FANTASY_FEATURES: Final = FeatureSet(
    FANTASY_FEATURE_SCHEMA_VERSION,
    (
        FeatureSpec(
            "ppr_per_game_recent",
            "Mean PPR points over recent games played.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} games played",
        ),
        FeatureSpec(
            "ppr_per_game_long",
            "Mean PPR points over the longer window.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} games played",
        ),
        FeatureSpec(
            "ppr_volatility",
            "Standard deviation of PPR points; separates a steady floor from a boom-bust role.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} games played",
        ),
        FeatureSpec(
            "exp_ppr_per_game_recent",
            "Mean expected PPR points from opportunity data over recent games.",
            "ff_opportunity",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} games played",
        ),
        FeatureSpec(
            "exp_ppr_per_game_long",
            "Mean expected PPR points over the longer window.",
            "ff_opportunity",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} games played",
        ),
        FeatureSpec(
            "ppr_minus_expected_long",
            "Realised minus expected points per game. Positive means recent scoring ran "
            "ahead of opportunity, which historically regresses.",
            "ff_opportunity",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} games played",
        ),
        FeatureSpec(
            "snap_share_recent",
            "Mean share of offensive snaps over recent games.",
            "snap_counts",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} games played",
        ),
        FeatureSpec(
            "snap_share_long",
            "Mean share of offensive snaps over the longer window.",
            "snap_counts",
            Availability.PRIOR_GAMES,
            f"last {FORM_WINDOW} games played",
        ),
        FeatureSpec(
            "target_share_recent",
            "Mean share of team targets over recent games.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} games played",
        ),
        FeatureSpec(
            "air_yards_share_recent",
            "Mean share of team air yards over recent games.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} games played",
        ),
        FeatureSpec(
            "wopr_recent",
            "Weighted opportunity rating: target share and air-yards share combined.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} games played",
        ),
        FeatureSpec(
            "targets_per_game_recent",
            "Mean targets over recent games.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} games played",
        ),
        FeatureSpec(
            "carries_per_game_recent",
            "Mean rushing attempts over recent games.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} games played",
        ),
        FeatureSpec(
            "attempts_per_game_recent",
            "Mean pass attempts over recent games. Separates starting quarterbacks.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} games played",
        ),
        FeatureSpec(
            "receptions_per_game_recent",
            "Mean receptions over recent games.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"last {RECENT_WINDOW} games played",
        ),
        FeatureSpec(
            "history_games",
            "Games played before this one, capped at the long window.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"capped at {FORM_WINDOW}",
        ),
        FeatureSpec(
            "weeks_since_last_game",
            "Weeks since the player last took a snap. Flags a bye, a return from injury "
            "or a long absence without needing an injury report.",
            "player_stats",
            Availability.PRIOR_GAMES,
        ),
        FeatureSpec(
            "opponent_ppr_allowed_to_position",
            "Mean PPR points the opponent has allowed per game to this position.",
            "player_stats",
            Availability.PRIOR_GAMES,
            f"last {OPPONENT_WINDOW} opponent games",
        ),
        FeatureSpec(
            "opponent_def_epa_per_play",
            "Expected points added the opponent allows per defensive play.",
            "team_stats",
            Availability.PRIOR_GAMES,
            "last 8 opponent games",
        ),
        FeatureSpec(
            "opponent_points_allowed_per_game",
            "Points the opponent allows per game.",
            "schedules",
            Availability.PRIOR_GAMES,
            "last 8 opponent games",
        ),
        FeatureSpec(
            "team_plays_per_game",
            "The player's own team's offensive plays per game; a pace and volume proxy.",
            "team_stats",
            Availability.PRIOR_GAMES,
            "last 8 team games",
        ),
        FeatureSpec(
            "is_home", "1 when the player's team is at home.", "schedules", Availability.SCHEDULE
        ),
        FeatureSpec(
            "rest_days",
            "Days since the team's previous game.",
            "schedules",
            Availability.SCHEDULE,
        ),
        FeatureSpec("is_dome", "1 when the roof is closed.", "schedules", Availability.SCHEDULE),
        FeatureSpec(
            "is_turf", "1 when the surface is artificial.", "schedules", Availability.SCHEDULE
        ),
        FeatureSpec("div_game", "1 for a divisional game.", "schedules", Availability.SCHEDULE),
        FeatureSpec("week", "Week number.", "schedules", Availability.SCHEDULE),
        FeatureSpec("is_qb", "Position indicator.", "player_stats", Availability.SCHEDULE),
        FeatureSpec("is_rb", "Position indicator.", "player_stats", Availability.SCHEDULE),
        FeatureSpec("is_wr", "Position indicator.", "player_stats", Availability.SCHEDULE),
        FeatureSpec("is_te", "Position indicator.", "player_stats", Availability.SCHEDULE),
        FeatureSpec(
            "implied_team_total",
            "Points the closing line implies for the player's team: half the total, "
            "adjusted by half the spread. A scoring-environment estimate, not a "
            "statement about this player.",
            "schedules",
            Availability.MARKET_CLOSE,
        ),
    ),
)


def _column(frame: pl.DataFrame, name: str, dtype: pl.DataType | type[pl.DataType]) -> pl.Expr:
    """Read a column, or a typed null when the dataset does not carry it."""
    if name in frame.columns:
        return pl.col(name).cast(dtype, strict=False).alias(name)
    return pl.lit(None, dtype=dtype).alias(name)


def _player_spine(player_stats: pl.DataFrame) -> pl.DataFrame:
    numeric = (
        "attempts",
        "carries",
        "targets",
        "receptions",
        "target_share",
        "air_yards_share",
        "wopr",
        "fantasy_points",
        "fantasy_points_ppr",
    )
    return (
        player_stats.filter(
            pl.col("position").is_in(list(MODELLED_POSITIONS))
            & pl.col("game_id").is_not_null()
            & pl.col("player_id").is_not_null()
        )
        .select(
            pl.col("player_id"),
            _column(player_stats, "player_display_name", pl.Utf8),
            pl.col("position"),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("game_id"),
            pl.col("team"),
            pl.col("opponent_team").alias("opponent"),
            *[_column(player_stats, name, pl.Float64) for name in numeric],
        )
        .unique(subset=["player_id", "game_id"], keep="first")
    )


def _attach_snaps(
    frame: pl.DataFrame, snap_counts: pl.DataFrame | None, players: pl.DataFrame | None
) -> pl.DataFrame:
    """Join snap share, bridging Pro-Football-Reference ids to GSIS ids.

    ``snap_counts`` is keyed on a PFR identifier while everything else here uses
    GSIS, so the player table is the crosswalk. Without it the join would have to
    match on display name, which fails silently for every player who shares one.
    """
    missing = [
        pl.lit(None, dtype=pl.Float64).alias("offense_pct"),
        pl.lit(None, dtype=pl.Float64).alias("offense_snaps"),
    ]
    if snap_counts is None or snap_counts.is_empty() or players is None or players.is_empty():
        return frame.with_columns(missing)
    if "pfr_id" not in players.columns or "pfr_player_id" not in snap_counts.columns:
        return frame.with_columns(missing)

    crosswalk = (
        players.select(pl.col("gsis_id").alias("player_id"), pl.col("pfr_id"))
        .drop_nulls()
        .unique(subset=["pfr_id"], keep="first")
    )
    snaps = (
        snap_counts.select(
            pl.col("game_id"),
            pl.col("pfr_player_id").alias("pfr_id"),
            (pl.col("offense_pct").cast(pl.Float64, strict=False)).alias("offense_pct"),
            (pl.col("offense_snaps").cast(pl.Float64, strict=False)).alias("offense_snaps"),
        )
        .join(crosswalk, on="pfr_id", how="inner")
        .drop("pfr_id")
        .unique(subset=["game_id", "player_id"], keep="first")
    )
    return frame.join(snaps, on=["game_id", "player_id"], how="left")


def _attach_expected_points(
    frame: pl.DataFrame, ff_opportunity: pl.DataFrame | None
) -> pl.DataFrame:
    if (
        ff_opportunity is None
        or ff_opportunity.is_empty()
        or "total_fantasy_points_exp" not in ff_opportunity.columns
    ):
        return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("exp_fantasy_points"))
    expected = (
        ff_opportunity.select(
            pl.col("game_id"),
            pl.col("player_id"),
            pl.col("total_fantasy_points_exp")
            .cast(pl.Float64, strict=False)
            .alias("exp_fantasy_points"),
        )
        .drop_nulls(subset=["game_id", "player_id"])
        .unique(subset=["game_id", "player_id"], keep="first")
    )
    return frame.join(expected, on=["game_id", "player_id"], how="left")


def _game_context(team_game: pl.DataFrame) -> pl.DataFrame:
    """Per team-game venue, rest and market context."""
    return team_game.select(
        "game_id",
        "team",
        "game_type",
        "is_home",
        "rest_days",
        "is_dome",
        "is_turf",
        "div_game",
        "completed",
        TIME_INDEX,
        # nflverse states spread_line from the home team's perspective: positive
        # means the home team is favoured. The sign therefore flips for the away
        # side, and getting that backwards would invert the single most
        # informative environment feature.
        (
            pl.col("total_line") / 2.0
            + pl.when(pl.col("is_home") == 1)
            .then(pl.col("spread_line") / 2.0)
            .otherwise(-pl.col("spread_line") / 2.0)
        )
        .cast(pl.Float64)
        .alias("implied_team_total"),
    )


def _opponent_position_history(played: pl.DataFrame) -> pl.DataFrame:
    """How many fantasy points each defence has allowed to each position, per game."""
    per_game = (
        played.group_by(["opponent", "position", "game_id", TIME_INDEX])
        .agg(pl.col(LABEL).fill_null(0).sum().alias("ppr_allowed"))
        .sort(["opponent", "position", TIME_INDEX])
    )
    rolled = trailing_sums(
        per_game,
        group_by=["opponent", "position"],
        order_by=[TIME_INDEX],
        columns=("ppr_allowed",),
        window=OPPONENT_WINDOW,
        suffix=_OPP,
    )
    rolled = trailing_count(
        rolled,
        group_by=["opponent", "position"],
        order_by=[TIME_INDEX],
        alias="opponent_games",
        window=OPPONENT_WINDOW,
    )
    return rolled.with_columns(
        safe_ratio(f"ppr_allowed{_OPP}", "opponent_games", "opponent_ppr_allowed_to_position")
    )


def _player_history(played: pl.DataFrame) -> pl.DataFrame:
    """Each player's accumulated production after every game they played in."""
    ordered = played.sort(["player_id", TIME_INDEX])
    history = trailing_sums(
        ordered,
        group_by=["player_id"],
        order_by=[TIME_INDEX],
        columns=_PLAYER_ACCUMULATED,
        window=RECENT_WINDOW,
        suffix=_RECENT,
    )
    history = trailing_sums(
        history,
        group_by=["player_id"],
        order_by=[TIME_INDEX],
        columns=_PLAYER_ACCUMULATED,
        window=FORM_WINDOW,
        suffix=_LONG,
    )
    history = trailing_count(
        history,
        group_by=["player_id"],
        order_by=[TIME_INDEX],
        alias="recent_games",
        window=RECENT_WINDOW,
    )
    history = trailing_count(
        history,
        group_by=["player_id"],
        order_by=[TIME_INDEX],
        alias="history_games",
        window=FORM_WINDOW,
    )
    history = trailing_observed(
        history,
        group_by=["player_id"],
        order_by=[TIME_INDEX],
        columns=_PLAYER_RATES,
        window=RECENT_WINDOW,
        suffix=_SEEN_RECENT,
    )
    history = trailing_observed(
        history,
        group_by=["player_id"],
        order_by=[TIME_INDEX],
        columns=_PLAYER_RATES,
        window=FORM_WINDOW,
        suffix=_SEEN_LONG,
    )
    history = trailing_std(
        history,
        group_by=["player_id"],
        order_by=[TIME_INDEX],
        column=LABEL,
        window=FORM_WINDOW,
        alias="ppr_volatility",
    )

    return history.with_columns(
        safe_ratio(f"fantasy_points_ppr{_RECENT}", "recent_games", "ppr_per_game_recent"),
        safe_ratio(f"fantasy_points_ppr{_LONG}", "history_games", "ppr_per_game_long"),
        safe_ratio(
            f"exp_fantasy_points{_RECENT}",
            f"exp_fantasy_points{_SEEN_RECENT}",
            "exp_ppr_per_game_recent",
        ),
        safe_ratio(
            f"exp_fantasy_points{_LONG}",
            f"exp_fantasy_points{_SEEN_LONG}",
            "exp_ppr_per_game_long",
        ),
        safe_ratio(f"offense_pct{_RECENT}", f"offense_pct{_SEEN_RECENT}", "snap_share_recent"),
        safe_ratio(f"offense_pct{_LONG}", f"offense_pct{_SEEN_LONG}", "snap_share_long"),
        safe_ratio(f"target_share{_RECENT}", f"target_share{_SEEN_RECENT}", "target_share_recent"),
        safe_ratio(
            f"air_yards_share{_RECENT}",
            f"air_yards_share{_SEEN_RECENT}",
            "air_yards_share_recent",
        ),
        safe_ratio(f"wopr{_RECENT}", f"wopr{_SEEN_RECENT}", "wopr_recent"),
        safe_ratio(f"targets{_RECENT}", "recent_games", "targets_per_game_recent"),
        safe_ratio(f"carries{_RECENT}", "recent_games", "carries_per_game_recent"),
        safe_ratio(f"attempts{_RECENT}", "recent_games", "attempts_per_game_recent"),
        safe_ratio(f"receptions{_RECENT}", "recent_games", "receptions_per_game_recent"),
        pl.col(TIME_INDEX).alias("last_game_time_index"),
    ).with_columns(
        (pl.col("ppr_per_game_long") - pl.col("exp_ppr_per_game_long")).alias(
            "ppr_minus_expected_long"
        )
    )


_PLAYER_HISTORY_COLUMNS: Final = (
    "ppr_per_game_recent",
    "ppr_per_game_long",
    "ppr_volatility",
    "exp_ppr_per_game_recent",
    "exp_ppr_per_game_long",
    "ppr_minus_expected_long",
    "snap_share_recent",
    "snap_share_long",
    "target_share_recent",
    "air_yards_share_recent",
    "wopr_recent",
    "targets_per_game_recent",
    "carries_per_game_recent",
    "attempts_per_game_recent",
    "receptions_per_game_recent",
    "history_games",
    "last_game_time_index",
)

_TEAM_FORM_CARRIED: Final = ("def_epa_per_play", "points_against_per_game")


def build_player_week_frame(
    player_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    team_stats: pl.DataFrame | None = None,
    snap_counts: pl.DataFrame | None = None,
    ff_opportunity: pl.DataFrame | None = None,
    players: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per modelled player per game, with usage, context and the label."""
    team_game = build_team_game_frame(schedules, team_stats)
    spine = _player_spine(player_stats).with_columns(time_index())
    spine = _attach_snaps(spine, snap_counts, players)
    spine = _attach_expected_points(spine, ff_opportunity)

    context = _game_context(team_game).drop(TIME_INDEX)
    merged = spine.join(context, on=["game_id", "team"], how="left")

    # Snap data starts in 2012; before that, and for any gap, fall back to
    # whether the player recorded an offensive touch. Treating a missing snap
    # count as "did not play" would silently delete two decades of rows.
    played = (
        pl.when(pl.col("offense_snaps").is_not_null())
        .then(pl.col("offense_snaps") >= 1)
        .otherwise(
            pl.sum_horizontal(
                pl.col("attempts").fill_null(0),
                pl.col("carries").fill_null(0),
                pl.col("targets").fill_null(0),
            )
            > 0
        )
        .fill_null(False)
        .alias("played")
    )
    return merged.with_columns(
        played,
        pl.col("position").cast(pl.Utf8),
    ).with_columns(
        [
            (pl.col("position") == position).cast(pl.Int8).alias(f"is_{position.lower()}")
            for position in MODELLED_POSITIONS
        ]
    )


def build_fantasy_features(
    player_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    team_stats: pl.DataFrame | None = None,
    snap_counts: pl.DataFrame | None = None,
    ff_opportunity: pl.DataFrame | None = None,
    players: pl.DataFrame | None = None,
    played_only: bool = True,
) -> pl.DataFrame:
    """Build the player-week feature matrix.

    Args:
        player_stats: nflverse weekly player stats covering the targets and
            enough earlier weeks to fill the windows.
        schedules: nflverse schedules over the same span.
        team_stats: weekly team stats, for opponent defence and team pace.
        snap_counts: snap counts, for snap share.
        ff_opportunity: opportunity data, for expected fantasy points.
        players: player table, used only as the PFR-to-GSIS crosswalk.
        played_only: Keep only rows where the player took a snap. True for
            training and evaluation, which is what makes the projection
            conditional on playing. False when scoring an upcoming week, where
            whether he plays is not yet known.

    Returns:
        One row per player-game with identity columns, every name in
        :data:`FANTASY_FEATURES`, and the PPR and standard labels.
    """
    frame = build_player_week_frame(
        player_stats,
        schedules,
        team_stats=team_stats,
        snap_counts=snap_counts,
        ff_opportunity=ff_opportunity,
        players=players,
    )

    # History is built from played, completed games only. Everything downstream
    # inherits leak-freedom from this filter plus the strict as-of join.
    history_source = frame.filter(pl.col("played") & pl.col("completed").fill_null(False))
    targets = frame.filter(pl.col("played")) if played_only else frame

    if history_source.is_empty() or targets.is_empty():
        return _empty_features(targets)

    player_history = _player_history(history_source)
    opponent_history = _opponent_position_history(history_source)
    team_form = build_team_form(build_team_game_frame(schedules, team_stats))

    enriched = join_as_of(
        targets, player_history, by=["player_id"], columns=_PLAYER_HISTORY_COLUMNS
    )
    enriched = join_as_of(
        enriched,
        opponent_history,
        by=["opponent", "position"],
        columns=("opponent_ppr_allowed_to_position",),
    )
    enriched = join_as_of(
        enriched,
        team_form.select("team", TIME_INDEX, *_TEAM_FORM_CARRIED).rename({"team": "opponent"}),
        by=["opponent"],
        columns=_TEAM_FORM_CARRIED,
    ).rename(
        {
            "def_epa_per_play": "opponent_def_epa_per_play",
            "points_against_per_game": "opponent_points_allowed_per_game",
        }
    )
    enriched = join_as_of(
        enriched,
        team_form.select(
            "team",
            TIME_INDEX,
            safe_ratio("plays_off_l8", "history_games", "team_plays_per_game"),
        ),
        by=["team"],
        columns=("team_plays_per_game",),
    )

    enriched = enriched.with_columns(
        (pl.col(TIME_INDEX) - pl.col("last_game_time_index"))
        .cast(pl.Float64)
        .alias("weeks_since_last_game"),
        pl.col("history_games").fill_null(0).cast(pl.Int32).alias("history_games"),
    )

    identity = [
        "player_id",
        "player_display_name",
        "position",
        "season",
        "week",
        "game_id",
        "game_type",
        "team",
        "opponent",
        "played",
        "completed",
        TIME_INDEX,
    ]
    projection = list(dict.fromkeys([*identity, *FANTASY_FEATURES.names, LABEL, STANDARD_LABEL]))
    return enriched.select(projection).sort([TIME_INDEX, "player_id"])


def _empty_features(targets: pl.DataFrame) -> pl.DataFrame:
    """A correctly shaped empty frame, so callers need no special case."""
    columns = {
        name: pl.Series(name, [], dtype=pl.Float64)
        for name in (*FANTASY_FEATURES.names, LABEL, STANDARD_LABEL)
    }
    base = targets.head(0)
    for name, series in columns.items():
        if name not in base.columns:
            base = base.with_columns(series)
    return base


def training_rows(
    features: pl.DataFrame, *, min_history: int = MIN_TRAINING_HISTORY
) -> pl.DataFrame:
    """Rows eligible for fitting and scoring.

    The minimum-history filter is the one worth defending. A player with no
    prior games has an entirely null feature row, and imputing it to the league
    median produces a projection that is really just "an average player" wearing
    a specific name. Reporting that the history is insufficient is more useful
    than a number that looks personalised and is not.
    """
    return features.filter(
        pl.col("played")
        & pl.col("completed").fill_null(False)
        & pl.col(LABEL).is_not_null()
        & (pl.col("history_games") >= min_history)
    )
