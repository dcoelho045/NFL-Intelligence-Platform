"""As-of windowing.

Every historical feature in this project is built here. That is a deliberate
bottleneck: temporal leakage in a sports model almost always enters as a single
aggregate that forgot to exclude the present, and one audited primitive is
safer than twenty hand-written ``rolling_mean`` calls spread across two builders.

The mechanism is two steps, kept separate on purpose:

1. :func:`trailing_sums` accumulates a team's or player's state **after** each
   completed game. That table is a pure function of finished games and knows
   nothing about what is being predicted.
2. :func:`join_as_of` attaches that state to a target row, matching only history
   **strictly before** the target's own timestamp.

Doing it this way rather than shifting a rolling window by one row matters for
the case that breaks naive implementations: predicting week 18 when week 12 was
the last game played. A shifted row-window would count the six unplayed games as
rows and silently truncate the lookback. An as-of join cannot, because unplayed
games never enter the history table at all.

Windows count *games*, not weeks, matching the convention the analytics layer
already uses. A bye, a suspension or a mid-season trade must not quietly shrink
a "last eight games" window to five.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

_ONES = "__window_one"

#: Column name for the chronological ordinal every as-of join is keyed on.
TIME_INDEX = "time_index"


def time_index(season: str = "season", week: str = "week") -> pl.Expr:
    """A single sortable integer for (season, week).

    Postseason weeks continue the regular-season numbering (19-22), so plain
    arithmetic orders a whole career correctly without a calendar join, and the
    gap between any two consecutive games is at least one.
    """
    return (pl.col(season).cast(pl.Int64) * 100 + pl.col(week).cast(pl.Int64)).alias(TIME_INDEX)


def _ordered(frame: pl.DataFrame, group_by: Sequence[str], order_by: Sequence[str]) -> pl.DataFrame:
    """Sort into (group, chronological) order.

    ``over`` applies an expression to each group's rows *in frame order*, so the
    sort is not cosmetic -- it is what makes "the previous game" mean the
    previous game.
    """
    missing = [c for c in (*group_by, *order_by) if c not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing ordering column(s): {', '.join(missing)}")
    return frame.sort([*group_by, *order_by])


def trailing_sums(
    frame: pl.DataFrame,
    *,
    group_by: Sequence[str],
    order_by: Sequence[str],
    columns: Sequence[str],
    window: int,
    suffix: str,
) -> pl.DataFrame:
    """Inclusive trailing sums over the last ``window`` rows of each group.

    Sums rather than means because almost every football rate is a ratio of two
    totals. Averaging per-game ratios instead weights a 40-play game the same as
    a 75-play one, which quietly changes what "EPA per play over the last eight
    games" means.

    The result describes the group's state *after* each row, so it is only safe
    as a feature via :func:`join_as_of`, which is what enforces the gap.
    """
    ordered = _ordered(frame, group_by, order_by)
    return ordered.with_columns(
        [
            pl.col(column)
            .fill_null(0)
            .rolling_sum(window_size=window, min_samples=1)
            .over(group_by)
            .alias(f"{column}{suffix}")
            for column in columns
        ]
    )


def trailing_observed(
    frame: pl.DataFrame,
    *,
    group_by: Sequence[str],
    order_by: Sequence[str],
    columns: Sequence[str],
    window: int,
    suffix: str,
) -> pl.DataFrame:
    """How many rows in the window actually carried a value for each column.

    The denominator that makes "unknown" survive averaging. :func:`trailing_sums`
    treats a null as zero, which is right for a counting statistic -- a game with
    no recorded carries really is zero carries -- and wrong for a rate. Dividing
    a snap-share sum by the game count gives 0.0 for a player whose snap data
    does not exist, which reads as "never on the field" rather than "not
    measured". Seasons before snap counts were published would silently become a
    league of players who never play.

    Dividing by this count instead yields null, which is the truth.
    """
    ordered = _ordered(frame, group_by, order_by)
    return ordered.with_columns(
        [
            pl.col(column)
            .is_not_null()
            .cast(pl.Int32)
            .rolling_sum(window_size=window, min_samples=1)
            .over(group_by)
            .alias(f"{column}{suffix}")
            for column in columns
        ]
    )


def trailing_std(
    frame: pl.DataFrame,
    *,
    group_by: Sequence[str],
    order_by: Sequence[str],
    column: str,
    window: int,
    alias: str,
) -> pl.DataFrame:
    """Inclusive trailing standard deviation over the last ``window`` rows.

    A volatility feature, which for weekly fantasy is as informative as the mean:
    two players averaging twelve points are not interchangeable if one ranges
    8-16 and the other 0-34.
    """
    ordered = _ordered(frame, group_by, order_by)
    return ordered.with_columns(
        pl.col(column).rolling_std(window_size=window, min_samples=2).over(group_by).alias(alias)
    )


def trailing_count(
    frame: pl.DataFrame,
    *,
    group_by: Sequence[str],
    order_by: Sequence[str],
    alias: str,
    window: int | None = None,
) -> pl.DataFrame:
    """How many rows the group has up to and including each row.

    Surfaced as a feature rather than hidden, because "this rests on two games of
    history" is itself information the model and the caller both need. It is
    also the quantity the insufficient-history guard tests.
    """
    ordered = _ordered(frame, group_by, order_by).with_columns(
        pl.lit(1, dtype=pl.Int32).alias(_ONES)
    )
    counter = pl.col(_ONES)
    rolled = counter.cum_sum() if window is None else counter.rolling_sum(window, min_samples=1)
    return ordered.with_columns(rolled.over(group_by).cast(pl.Int32).alias(alias)).drop(_ONES)


def join_as_of(
    targets: pl.DataFrame,
    history: pl.DataFrame,
    *,
    by: Sequence[str],
    columns: Sequence[str],
) -> pl.DataFrame:
    """Attach to each target row the most recent history **strictly before** it.

    Both frames must carry :data:`TIME_INDEX`. The target key is decremented by
    one before matching, which is what makes the match strict: a team's own game
    at (2024, 7) can never supply the history for the prediction of that same
    game, while its game at (2024, 6) still can.

    Args:
        targets: Rows to be predicted or trained on, one per subject per event.
        history: Post-game state produced by :func:`trailing_sums`.
        by: Entity columns matched exactly, e.g. ``["team"]``.
        columns: History columns to bring across.

    Returns:
        ``targets`` with ``columns`` attached. Subjects with no prior history get
        nulls, never zeros -- an unknown value and a value of zero are different
        claims, and conflating them is how a debut becomes a confident prediction
        about nothing.
    """
    for frame, label in ((targets, "targets"), (history, "history")):
        if TIME_INDEX not in frame.columns:
            raise ValueError(f"{label} frame is missing the {TIME_INDEX!r} column")

    key = "__as_of_key"
    left = targets.with_columns((pl.col(TIME_INDEX) - 1).alias(key)).sort(key)
    right = history.select([*by, TIME_INDEX, *columns]).sort(TIME_INDEX)

    # Both sides are sorted on the line above; polars cannot verify that itself
    # once `by` groups are involved, and its warning would otherwise fire on
    # every feature build.
    joined = left.join_asof(
        right,
        left_on=key,
        right_on=TIME_INDEX,
        by=list(by),
        strategy="backward",
        check_sortedness=False,
    )
    return joined.drop(key).drop(f"{TIME_INDEX}_right", strict=False)


def safe_ratio(numerator: str, denominator: str, alias: str) -> pl.Expr:
    """``numerator / denominator``, null when the denominator is missing or zero.

    Null rather than zero on purpose. A team with no prior plays has an *unknown*
    EPA per play, and zero is a specific, wrong, highly confident claim about it.
    Imputation happens later, inside a training fold, where its effect is measured.
    """
    return (
        pl.when(pl.col(denominator).is_null() | (pl.col(denominator) <= 0))
        .then(None)
        .otherwise(pl.col(numerator) / pl.col(denominator))
        .cast(pl.Float64)
        .alias(alias)
    )
