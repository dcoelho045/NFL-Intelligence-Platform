"""Game-level analytics derived from the schedules dataset."""

from __future__ import annotations

import polars as pl

from nfl_platform.analytics.filters import SeasonTypeFilter, filter_season_type, take_last_n_games
from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.domain import SeasonInput, normalize_team, validate_game_count
from nfl_platform.exceptions import TeamNotFoundError


def get_recent_games(
    team: str,
    *,
    loader: NflDataLoader,
    seasons: SeasonInput = None,
    last_n_games: int | None = 4,
    season_type: SeasonTypeFilter = None,
    completed_only: bool = True,
) -> pl.DataFrame:
    """Return a team's games, newest first, from that team's perspective.

    Home/away rows are normalised into ``points_for`` / ``points_against`` so the
    caller never has to branch on which side of the schedule row the team sits.

    Args:
        completed_only: Drop games with no final score. Scheduled future games
            carry null scores and would otherwise pollute aggregates.
    """
    normalized_team = normalize_team(team)
    game_count = validate_game_count(last_n_games) if last_n_games is not None else None

    schedules = loader.load_schedules(seasons)
    teams_in_scope = set(schedules.get_column("home_team").unique().drop_nulls().to_list()) | set(
        schedules.get_column("away_team").unique().drop_nulls().to_list()
    )
    if normalized_team not in teams_in_scope:
        raise TeamNotFoundError(
            normalized_team, "the schedules dataset", tuple(sorted(teams_in_scope))
        )

    is_home = pl.col("home_team") == normalized_team
    team_games = schedules.filter(is_home | (pl.col("away_team") == normalized_team))

    if completed_only:
        team_games = team_games.filter(
            pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
        )

    team_games = filter_season_type(team_games, season_type, "game_type")

    normalized = team_games.with_columns(
        pl.lit(normalized_team).alias("team"),
        is_home.alias("is_home"),
        pl.when(is_home).then(pl.col("away_team")).otherwise(pl.col("home_team")).alias("opponent"),
        pl.when(is_home)
        .then(pl.col("home_score"))
        .otherwise(pl.col("away_score"))
        .alias("points_for"),
        pl.when(is_home)
        .then(pl.col("away_score"))
        .otherwise(pl.col("home_score"))
        .alias("points_against"),
    ).with_columns(
        (pl.col("points_for") - pl.col("points_against")).alias("point_differential"),
        pl.when(pl.col("points_for") > pl.col("points_against"))
        .then(pl.lit("W"))
        .when(pl.col("points_for") < pl.col("points_against"))
        .then(pl.lit("L"))
        .otherwise(pl.lit("T"))
        .alias("outcome"),
    )

    ordered = normalized.sort(["season", "week"], descending=True, nulls_last=True)
    limited = take_last_n_games(ordered, game_count)

    columns = [
        column
        for column in (
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "team",
            "opponent",
            "is_home",
            "points_for",
            "points_against",
            "point_differential",
            "outcome",
            "spread_line",
            "total_line",
            "roof",
            "surface",
        )
        if column in limited.columns
    ]
    return limited.select(columns)


def summarize_recent_games(recent_games: pl.DataFrame) -> pl.DataFrame:
    """Aggregate a team's recent games into a single-row record summary."""
    return recent_games.select(
        pl.lit(recent_games.height, dtype=pl.Int64).alias("games"),
        (pl.col("outcome") == "W").sum().alias("wins"),
        (pl.col("outcome") == "L").sum().alias("losses"),
        (pl.col("outcome") == "T").sum().alias("ties"),
        pl.col("points_for").mean().alias("avg_points_for"),
        pl.col("points_against").mean().alias("avg_points_against"),
        pl.col("point_differential").mean().alias("avg_point_differential"),
    )
