"""Shared base for the analytics services.

Services are the stable, object-oriented interface over the Milestone 1
analytics functions. They own three responsibilities and nothing else:

1. resolve and validate a season/week window,
2. call the deterministic analytics functions,
3. convert the resulting Polars frames into typed DTOs.

The numerical work stays in ``nfl_platform.analytics``. Services compose and
translate; they do not compute.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

import polars as pl

from nfl_platform.analytics.filters import POSTSEASON, REGULAR_SEASON
from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.domain import (
    SeasonInput,
    normalize_seasons,
    normalize_weeks,
    validate_game_count,
)
from nfl_platform.schemas.common import SeasonType, SeasonWindow, StatWindow

T = TypeVar("T")

_SEASON_TYPE_TO_FILTER: dict[SeasonType, str | None] = {
    SeasonType.REGULAR: REGULAR_SEASON,
    SeasonType.POST: POSTSEASON,
    SeasonType.ALL: None,
}


class AnalyticsService:
    """Shared implementation base: holds the loader and the conversion helpers.

    Not an abstract interface -- subclasses inherit behaviour rather than fulfil
    a contract, so there is nothing to declare abstract.
    """

    def __init__(self, loader: NflDataLoader) -> None:
        self._loader = loader

    @property
    def loader(self) -> NflDataLoader:
        return self._loader

    # --- window helpers -------------------------------------------------------

    def resolve_seasons(self, seasons: SeasonInput) -> list[int]:
        """Normalise a season argument using the loader's configured defaults."""
        return normalize_seasons(
            seasons,
            default_season=self._loader.default_season,
            latest_season=self._loader.current_season,
        )

    @staticmethod
    def season_type_filter(season_type: SeasonType) -> str | None:
        """Translate the DTO enum into the analytics layer's filter value."""
        return _SEASON_TYPE_TO_FILTER[season_type]

    def build_season_window(
        self,
        seasons: SeasonInput,
        season_type: SeasonType,
        weeks: int | Iterable[int] | None = None,
    ) -> SeasonWindow:
        return SeasonWindow(
            seasons=self.resolve_seasons(seasons),
            season_type=season_type,
            weeks=normalize_weeks(weeks),
        )

    def build_stat_window(
        self,
        seasons: SeasonInput,
        season_type: SeasonType,
        games: int,
        *,
        last_n_games: int | None = None,
        weeks: int | Iterable[int] | None = None,
    ) -> StatWindow:
        """Describe the scope a result was computed over.

        ``games`` is the count that actually contributed, which can be lower than
        ``last_n_games`` after a bye week or an injury absence.
        """
        if last_n_games is not None:
            validate_game_count(last_n_games)
        return StatWindow(
            seasons=self.resolve_seasons(seasons),
            season_type=season_type,
            weeks=normalize_weeks(weeks),
            games=games,
            requested_last_n_games=last_n_games,
        )

    # --- frame conversion helpers ---------------------------------------------

    @staticmethod
    def cell(frame: pl.DataFrame, column: str, row: int = 0) -> Any:
        """Read one value, returning None when the column or row is absent.

        Analytics frames vary in width by position and dataset, so callers would
        otherwise need a membership check before every single field.
        """
        if column not in frame.columns or frame.height <= row:
            return None
        return frame.item(row, column)

    @staticmethod
    def optional_int(value: Any) -> int | None:
        """Coerce a frame value to int, tolerating nulls and float-typed integers."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def optional_float(value: Any) -> float | None:
        """Coerce a frame value to float, tolerating nulls."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def optional_str(value: Any) -> str | None:
        """Coerce a frame value to a non-empty string, or None."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def row_int(cls, row: dict[str, Any], column: str) -> int | None:
        return cls.optional_int(row.get(column))

    @classmethod
    def row_float(cls, row: dict[str, Any], column: str) -> float | None:
        return cls.optional_float(row.get(column))

    @classmethod
    def row_str(cls, row: dict[str, Any], column: str) -> str | None:
        return cls.optional_str(row.get(column))

    @staticmethod
    def column_mean(frame: pl.DataFrame, column: str) -> float | None:
        """Mean of a column, ignoring nulls. None when the column is absent or empty."""
        if column not in frame.columns or frame.is_empty():
            return None
        return AnalyticsService.optional_float(frame.get_column(column).drop_nulls().mean())

    @staticmethod
    def column_sum(frame: pl.DataFrame, column: str) -> float | None:
        """Sum of a column, ignoring nulls. None when the column is absent or empty."""
        if column not in frame.columns or frame.is_empty():
            return None
        value = frame.get_column(column).drop_nulls().sum()
        return AnalyticsService.optional_float(value)
