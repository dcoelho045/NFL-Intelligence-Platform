"""Injury report analytics."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from nfl_platform.analytics.filters import filter_weeks, sort_most_recent_first
from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.domain import (
    SeasonInput,
    normalize_player_id,
    normalize_team,
    normalize_weeks,
)

_INJURY_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_type",
    "team",
    "gsis_id",
    "full_name",
    "position",
    "report_status",
    "report_primary_injury",
    "report_secondary_injury",
    "practice_status",
    "practice_primary_injury",
    "date_modified",
)


def get_injury_information(
    player_id: str,
    *,
    loader: NflDataLoader,
    seasons: SeasonInput = None,
    weeks: int | Iterable[int] | None = None,
    latest_only: bool = False,
) -> pl.DataFrame:
    """Return a player's injury report rows, newest first.

    Returns an empty frame rather than raising when a player has no injury
    entries: a healthy player legitimately has no rows, which is information,
    not an error. This differs deliberately from the player-stats functions,
    where an unknown id is a caller mistake.
    """
    normalized_id = normalize_player_id(player_id)
    resolved_weeks = normalize_weeks(weeks)

    injuries = loader.load_injuries(seasons)
    player_rows = injuries.filter(pl.col("gsis_id") == normalized_id)
    filtered = filter_weeks(player_rows, resolved_weeks)
    ordered = sort_most_recent_first(filtered)

    if latest_only:
        ordered = ordered.head(1)

    available = [column for column in _INJURY_COLUMNS if column in ordered.columns]
    return ordered.select(available)


def get_team_injuries(
    team: str,
    *,
    loader: NflDataLoader,
    seasons: SeasonInput = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Return every injury report row for a team, optionally for one week."""
    normalized_team = normalize_team(team)
    resolved_weeks = normalize_weeks(week)

    injuries = loader.load_injuries(seasons)
    team_rows = injuries.filter(pl.col("team") == normalized_team)
    filtered = filter_weeks(team_rows, resolved_weeks)
    ordered = sort_most_recent_first(filtered)

    available = [column for column in _INJURY_COLUMNS if column in ordered.columns]
    return ordered.select(available)
