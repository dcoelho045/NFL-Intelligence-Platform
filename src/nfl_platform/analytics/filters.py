"""Shared filtering helpers for analytics functions.

Centralised so that "most recent N games" and season-type filtering mean exactly
the same thing everywhere.
"""

from __future__ import annotations

import polars as pl

# Values callers pass as ``season_type``. ``schedules`` uses a finer-grained
# ``game_type`` (REG, WC, DIV, CON, SB); the stats tables collapse those to POST.
REGULAR_SEASON = "REG"
POSTSEASON = "POST"

SeasonTypeFilter = str | None


def filter_season_type(
    frame: pl.DataFrame, season_type: SeasonTypeFilter, column: str
) -> pl.DataFrame:
    """Filter by season type. ``None`` keeps every game type."""
    if season_type is None or column not in frame.columns:
        return frame
    return frame.filter(pl.col(column) == season_type)


def filter_weeks(frame: pl.DataFrame, weeks: list[int] | None) -> pl.DataFrame:
    """Restrict to explicit week numbers. ``None`` keeps every week."""
    if weeks is None or "week" not in frame.columns:
        return frame
    return frame.filter(pl.col("week").is_in(weeks))


def sort_most_recent_first(frame: pl.DataFrame) -> pl.DataFrame:
    """Order by season then week, most recent first.

    Sorting on (season, week) rather than a date column keeps ordering
    well-defined even when a game has no kickoff timestamp yet.
    """
    sort_columns = [column for column in ("season", "week") if column in frame.columns]
    if not sort_columns:
        return frame
    return frame.sort(sort_columns, descending=True, nulls_last=True)


def take_last_n_games(frame: pl.DataFrame, last_n_games: int | None) -> pl.DataFrame:
    """Return the ``last_n_games`` most recent rows of an already-sorted frame.

    Counts *games with data*, not calendar weeks, so a bye week or an inactive
    week does not silently shrink the sample.
    """
    if last_n_games is None:
        return frame
    return frame.head(last_n_games)
