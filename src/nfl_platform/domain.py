"""Domain primitives: identifier normalisation and season/week validation.

These helpers exist so that every public function validates its inputs the same
way and raises the same explicit errors, rather than letting a bad argument
surface as an empty DataFrame or a Polars type error.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from nfl_platform.exceptions import (
    InvalidGameCountError,
    InvalidSeasonError,
    InvalidWeekError,
)

# nflverse publishes play-by-play from 1999 onward; earlier seasons are not available.
EARLIEST_SEASON = 1999

# Regular season (1-18) plus postseason rounds; 22 covers the 18-game era.
MIN_WEEK = 1
MAX_WEEK = 22

# nflverse player ids are GSIS ids formatted as ``00-0012345``.
_GSIS_ID_PATTERN = re.compile(r"^00-\d{7}$")

SeasonInput = int | Iterable[int] | None


def validate_season(season: object, *, latest_season: int) -> int:
    """Return ``season`` as an int, or raise :class:`InvalidSeasonError`."""
    if isinstance(season, bool) or not isinstance(season, int):
        raise InvalidSeasonError(season, EARLIEST_SEASON, latest_season)
    if not EARLIEST_SEASON <= season <= latest_season:
        raise InvalidSeasonError(season, EARLIEST_SEASON, latest_season)
    return season


def normalize_seasons(
    seasons: SeasonInput, *, default_season: int, latest_season: int
) -> list[int]:
    """Normalise a season argument into a sorted, de-duplicated list of ints.

    ``None`` resolves to ``default_season`` so that callers never depend on an
    implicit "current season" hidden inside the data layer.
    """
    if seasons is None:
        candidates: list[object] = [default_season]
    elif isinstance(seasons, int) and not isinstance(seasons, bool):
        candidates = [seasons]
    elif isinstance(seasons, Iterable) and not isinstance(seasons, str | bytes):
        candidates = list(seasons)
    else:
        raise InvalidSeasonError(seasons, EARLIEST_SEASON, latest_season)

    if not candidates:
        raise InvalidSeasonError(seasons, EARLIEST_SEASON, latest_season)

    validated = {validate_season(item, latest_season=latest_season) for item in candidates}
    return sorted(validated)


def validate_week(week: object) -> int:
    """Return ``week`` as an int, or raise :class:`InvalidWeekError`."""
    if isinstance(week, bool) or not isinstance(week, int):
        raise InvalidWeekError(week)
    if not MIN_WEEK <= week <= MAX_WEEK:
        raise InvalidWeekError(week)
    return week


def normalize_weeks(weeks: int | Iterable[int] | None) -> list[int] | None:
    """Normalise an optional week filter into a sorted list, or ``None`` for all weeks."""
    if weeks is None:
        return None
    if isinstance(weeks, int) and not isinstance(weeks, bool):
        return [validate_week(weeks)]
    if isinstance(weeks, Iterable) and not isinstance(weeks, str | bytes):
        values = list(weeks)
        if not values:
            raise InvalidWeekError(weeks)
        return sorted({validate_week(item) for item in values})
    raise InvalidWeekError(weeks)


def validate_game_count(last_n_games: object) -> int:
    """Return ``last_n_games`` as a positive int, or raise :class:`InvalidGameCountError`."""
    if isinstance(last_n_games, bool) or not isinstance(last_n_games, int):
        raise InvalidGameCountError(last_n_games)
    if last_n_games < 1:
        raise InvalidGameCountError(last_n_games)
    return last_n_games


def normalize_team(team: str) -> str:
    """Normalise a team abbreviation to the uppercase form nflverse uses.

    Deliberately does not map historical relocations (for example ``OAK`` to
    ``LV``): nflverse stores the abbreviation that was correct for the season in
    question, so rewriting it would corrupt historical queries. Validity is
    checked against the abbreviations actually present in the loaded data.
    """
    if not isinstance(team, str):
        raise TypeError(f"Team abbreviation must be a string, got {type(team).__name__}.")
    normalized = team.strip().upper()
    if not normalized:
        raise ValueError("Team abbreviation must not be empty.")
    return normalized


def normalize_player_id(player_id: str) -> str:
    """Normalise a GSIS player id, without asserting that the player exists."""
    if not isinstance(player_id, str):
        raise TypeError(f"Player id must be a string, got {type(player_id).__name__}.")
    normalized = player_id.strip()
    if not normalized:
        raise ValueError("Player id must not be empty.")
    return normalized


def looks_like_gsis_id(player_id: str) -> bool:
    """Report whether ``player_id`` matches the ``00-0012345`` GSIS format."""
    return bool(_GSIS_ID_PATTERN.match(player_id))


def format_season_range(seasons: Sequence[int]) -> str:
    """Render a season list compactly for log messages and error text."""
    if not seasons:
        return "<none>"
    if len(seasons) == 1:
        return str(seasons[0])
    return f"{seasons[0]}-{seasons[-1]}"
