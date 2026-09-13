"""Deterministic player analytics.

Every function here is pure with respect to its inputs: given the same loader
data it always returns the same frame. No LLM, no heuristics, no estimation.
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
from nfl_platform.domain import (
    SeasonInput,
    normalize_player_id,
    normalize_weeks,
    validate_game_count,
)
from nfl_platform.exceptions import PlayerNotFoundError

# Columns surfaced by ``get_player_recent_stats``; the full 150-column frame is
# rarely what a caller wants and is expensive to pass around.
_RECENT_STATS_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "season_type",
    "game_id",
    "player_id",
    "player_display_name",
    "position",
    "team",
    "opponent_team",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "target_share",
    "air_yards_share",
    "wopr",
    "fantasy_points",
    "fantasy_points_ppr",
)

_USAGE_SUM_COLUMNS: tuple[str, ...] = (
    "attempts",
    "carries",
    "targets",
    "receptions",
    "receiving_yards",
    "rushing_yards",
    "passing_yards",
)

_USAGE_MEAN_COLUMNS: tuple[str, ...] = (
    "target_share",
    "air_yards_share",
    "wopr",
)


def get_player_recent_stats(
    player_id: str,
    *,
    loader: NflDataLoader,
    seasons: SeasonInput = None,
    last_n_games: int | None = 4,
    weeks: int | Iterable[int] | None = None,
    season_type: SeasonTypeFilter = None,
) -> pl.DataFrame:
    """Return a player's most recent per-game statistics, newest first.

    Args:
        player_id: nflverse GSIS id, for example ``"00-0033873"``.
        loader: Data loader providing the player-stats dataset.
        seasons: Season or seasons to search. ``None`` uses the configured default.
        last_n_games: Keep only this many most recent games. ``None`` keeps all.
        weeks: Optional explicit week filter applied before ``last_n_games``.
        season_type: ``"REG"`` or ``"POST"``. Defaults to ``None``, meaning every
            game type, so "most recent games" is never silently truncated at the
            end of the regular season.

    Raises:
        PlayerNotFoundError: The id has no rows in the requested seasons.
    """
    normalized_id = normalize_player_id(player_id)
    resolved_weeks = normalize_weeks(weeks)
    game_count = validate_game_count(last_n_games) if last_n_games is not None else None

    stats = loader.load_player_stats(seasons)
    player_rows = stats.filter(pl.col("player_id") == normalized_id)
    if player_rows.is_empty():
        raise PlayerNotFoundError(normalized_id, "the player statistics dataset")

    filtered = filter_weeks(player_rows, resolved_weeks)
    filtered = filter_season_type(filtered, season_type, "season_type")
    ordered = sort_most_recent_first(filtered)
    limited = take_last_n_games(ordered, game_count)

    available = [column for column in _RECENT_STATS_COLUMNS if column in limited.columns]
    return limited.select(available)


def summarize_player_stats(recent_stats: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-game player stats into a single-row summary.

    An empty input still yields one row reporting ``games = 0``, so callers never
    have to special-case a player who did not appear in the requested window.
    """
    aggregations: list[pl.Expr] = [pl.lit(recent_stats.height, dtype=pl.Int64).alias("games")]

    for column in (*_USAGE_SUM_COLUMNS, "fantasy_points", "fantasy_points_ppr"):
        if column in recent_stats.columns:
            aggregations.append(pl.col(column).sum().alias(f"total_{column}"))
            aggregations.append(pl.col(column).mean().alias(f"avg_{column}"))

    for column in _USAGE_MEAN_COLUMNS:
        if column in recent_stats.columns:
            aggregations.append(pl.col(column).mean().alias(f"avg_{column}"))

    return recent_stats.select(aggregations)


def get_player_usage(
    player_id: str,
    *,
    loader: NflDataLoader,
    seasons: SeasonInput = None,
    last_n_games: int | None = 4,
    season_type: SeasonTypeFilter = None,
    include_snap_share: bool = True,
) -> pl.DataFrame:
    """Return per-game opportunity metrics for a player, newest first.

    Usage is about opportunity (snaps, targets, carries, target share) rather
    than production. Snap counts are keyed by PFR id upstream, so they are joined
    through the players crosswalk; if that crosswalk is unavailable the snap
    columns are omitted rather than fabricated.
    """
    recent = get_player_recent_stats(
        player_id,
        loader=loader,
        seasons=seasons,
        last_n_games=last_n_games,
        season_type=season_type,
    )

    usage_columns = [
        column
        for column in (
            "season",
            "week",
            "game_id",
            "player_id",
            "player_display_name",
            "position",
            "team",
            "opponent_team",
            "attempts",
            "carries",
            "targets",
            "receptions",
            "target_share",
            "air_yards_share",
            "wopr",
        )
        if column in recent.columns
    ]
    usage = recent.select(usage_columns)

    if not include_snap_share or usage.is_empty():
        return usage

    snap_share = _snap_share_for_player(player_id, loader=loader, seasons=seasons)
    if snap_share is None:
        return usage
    return usage.join(snap_share, on=["season", "week"], how="left")


def _snap_share_for_player(
    player_id: str,
    *,
    loader: NflDataLoader,
    seasons: SeasonInput,
) -> pl.DataFrame | None:
    """Resolve per-week snap share for a GSIS id, or ``None`` if it cannot be linked."""
    pfr_id = resolve_pfr_id(player_id, loader=loader)
    if pfr_id is None:
        return None

    snaps = loader.load_snap_counts(seasons)
    player_snaps = snaps.filter(pl.col("pfr_player_id") == pfr_id)
    if player_snaps.is_empty():
        return None

    return player_snaps.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("offense_snaps"),
        pl.col("offense_pct").alias("offense_snap_pct"),
    ).unique(subset=["season", "week"], keep="first")


def resolve_pfr_id(player_id: str, *, loader: NflDataLoader) -> str | None:
    """Map a GSIS player id to its Pro Football Reference id, or ``None``."""
    normalized_id = normalize_player_id(player_id)
    players = loader.load_players()
    match = players.filter(pl.col("gsis_id") == normalized_id)
    if match.is_empty():
        return None
    value = match.get_column("pfr_id").drop_nulls().to_list()
    return str(value[0]) if value else None
