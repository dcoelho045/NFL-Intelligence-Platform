"""Deterministic team analytics.

Team defence is derived from opponent rows in the team-stats dataset: the row
where ``opponent_team == X`` describes what an offence did *against* X, which is
exactly what X's defence allowed. This avoids parsing play-by-play for questions
that box-score data already answers.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from nfl_platform.analytics.filters import (
    SeasonTypeFilter,
    filter_season_type,
    filter_weeks,
    sort_most_recent_first,
    take_last_n_games,
)
from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.domain import SeasonInput, normalize_team, normalize_weeks, validate_game_count
from nfl_platform.exceptions import TeamNotFoundError

_TEAM_STAT_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "opponent_team",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_epa",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_epa",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "penalties",
    "penalty_yards",
)

_AGGREGATABLE_COLUMNS: tuple[str, ...] = (
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
)

_MEAN_ONLY_COLUMNS: tuple[str, ...] = ("passing_epa", "rushing_epa")


def _assert_team_present(frame: pl.DataFrame, team: str, column: str, scope: str) -> None:
    """Raise a helpful error when a team abbreviation does not exist in scope."""
    if column not in frame.columns:
        return
    known = tuple(sorted(frame.get_column(column).unique().drop_nulls().to_list()))
    if team not in known:
        raise TeamNotFoundError(team, scope, known)


def get_team_stats(
    team: str,
    *,
    loader: NflDataLoader,
    seasons: SeasonInput = None,
    last_n_games: int | None = 4,
    weeks: int | Iterable[int] | None = None,
    season_type: SeasonTypeFilter = None,
) -> pl.DataFrame:
    """Return a team's own per-game offensive production, newest first.

    ``season_type`` defaults to ``None`` (every game type); pass ``"REG"`` to
    exclude the postseason.

    Raises:
        TeamNotFoundError: The abbreviation does not appear in the requested seasons.
    """
    normalized_team = normalize_team(team)
    resolved_weeks = normalize_weeks(weeks)
    game_count = validate_game_count(last_n_games) if last_n_games is not None else None

    stats = loader.load_team_stats(seasons)
    _assert_team_present(stats, normalized_team, "team", "the team statistics dataset")

    team_rows = stats.filter(pl.col("team") == normalized_team)
    filtered = filter_weeks(team_rows, resolved_weeks)
    filtered = filter_season_type(filtered, season_type, "season_type")
    ordered = sort_most_recent_first(filtered)
    limited = take_last_n_games(ordered, game_count)

    available = [column for column in _TEAM_STAT_COLUMNS if column in limited.columns]
    return limited.select(available)


def get_team_defense(
    team: str,
    *,
    loader: NflDataLoader,
    seasons: SeasonInput = None,
    last_n_games: int | None = 4,
    weeks: int | Iterable[int] | None = None,
    season_type: SeasonTypeFilter = None,
) -> pl.DataFrame:
    """Return what a team's defence allowed per game, newest first.

    Columns are prefixed ``allowed_`` to make the perspective unambiguous:
    ``allowed_passing_yards`` is what opponents gained through the air against
    this team, not what this team gained.
    """
    normalized_team = normalize_team(team)
    resolved_weeks = normalize_weeks(weeks)
    game_count = validate_game_count(last_n_games) if last_n_games is not None else None

    stats = loader.load_team_stats(seasons)
    _assert_team_present(stats, normalized_team, "team", "the team statistics dataset")

    opponent_rows = stats.filter(pl.col("opponent_team") == normalized_team)
    filtered = filter_weeks(opponent_rows, resolved_weeks)
    filtered = filter_season_type(filtered, season_type, "season_type")
    ordered = sort_most_recent_first(filtered)
    limited = take_last_n_games(ordered, game_count)

    available = [column for column in _TEAM_STAT_COLUMNS if column in limited.columns]
    projected = limited.select(available)

    identity_columns = {"season", "week", "season_type", "game_id", "team", "opponent_team"}
    renames = {
        column: f"allowed_{column}"
        for column in projected.columns
        if column not in identity_columns
    }
    return projected.rename(renames).rename({"team": "offense_team", "opponent_team": "team"})


def summarize_team_stats(team_stats: pl.DataFrame, *, prefix: str = "") -> pl.DataFrame:
    """Aggregate per-game team stats into a single-row summary.

    ``prefix`` matches the column prefix in the input (``"allowed_"`` for the
    output of :func:`get_team_defense`).
    """
    aggregations: list[pl.Expr] = [pl.lit(team_stats.height, dtype=pl.Int64).alias("games")]

    for column in _AGGREGATABLE_COLUMNS:
        source = f"{prefix}{column}"
        if source in team_stats.columns:
            aggregations.append(pl.col(source).sum().alias(f"total_{source}"))
            aggregations.append(pl.col(source).mean().alias(f"avg_{source}"))

    for column in _MEAN_ONLY_COLUMNS:
        source = f"{prefix}{column}"
        if source in team_stats.columns:
            aggregations.append(pl.col(source).mean().alias(f"avg_{source}"))

    return team_stats.select(aggregations)
