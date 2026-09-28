"""The public data-loading entry point.

``NflDataLoader`` owns season normalisation, schema validation, and per-instance
memoisation. Every analytics function takes a loader, which is what makes data
access explicit and testable rather than an import-time side effect.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC
from typing import Any

import polars as pl

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.data.ingestion.datasets import Dataset, get_dataset_spec
from nfl_platform.data.ingestion.sources import DataSource, NflreadpyDataSource
from nfl_platform.data.ingestion.validation import (
    normalize_key_columns,
    select_columns,
    validate_schema,
)
from nfl_platform.domain import SeasonInput, format_season_range, normalize_seasons

logger = get_logger(__name__)

_CacheKey = tuple[str, tuple[int, ...] | None, tuple[tuple[str, Any], ...]]


def _current_season() -> int:
    """Resolve the current NFL season via nflreadpy, falling back to the calendar year."""
    try:
        import nflreadpy

        return int(nflreadpy.get_current_season())
    except Exception:  # pragma: no cover - only hit when nflreadpy is unavailable
        from datetime import datetime

        now = datetime.now(UTC)
        # The NFL season is labelled by the year it starts; it begins in September.
        return now.year if now.month >= 9 else now.year - 1


class NflDataLoader:
    """Loads validated nflverse datasets.

    Results are memoised per instance so that an analytics call chain touching
    the same dataset several times downloads it once.
    """

    def __init__(
        self,
        source: DataSource | None = None,
        settings: Settings | None = None,
        *,
        current_season: int | None = None,
        enable_cache: bool = True,
    ) -> None:
        self._settings = settings or get_settings()
        self._source = source or NflreadpyDataSource(self._settings)
        self._current_season = current_season if current_season is not None else _current_season()
        self._enable_cache = enable_cache
        self._cache: dict[_CacheKey, pl.DataFrame] = {}

    @property
    def current_season(self) -> int:
        return self._current_season

    @property
    def default_season(self) -> int:
        """Season used when a caller does not specify one."""
        configured = self._settings.default_season
        return configured if configured is not None else self._current_season

    def clear_cache(self) -> None:
        """Drop the per-instance dataset cache."""
        self._cache.clear()

    def _seasons(self, seasons: SeasonInput) -> list[int]:
        return normalize_seasons(
            seasons,
            default_season=self.default_season,
            latest_season=self._current_season,
        )

    def load(
        self,
        dataset: Dataset | str,
        seasons: SeasonInput = None,
        *,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Load, validate, and optionally project a registered dataset."""
        spec = get_dataset_spec(dataset)
        resolved_seasons = self._seasons(seasons) if spec.season_aware else None

        cache_key: _CacheKey = (
            spec.name.value,
            tuple(resolved_seasons) if resolved_seasons is not None else None,
            tuple(sorted(kwargs.items())),
        )
        cached = self._cache.get(cache_key) if self._enable_cache else None
        if cached is None:
            frame = self._source.fetch(spec.name, resolved_seasons, **kwargs)
            frame = validate_schema(frame, spec)
            frame = normalize_key_columns(frame)
            logger.info(
                "Loaded dataset",
                extra={
                    "dataset": spec.name.value,
                    "seasons": format_season_range(resolved_seasons or []),
                    "rows": frame.height,
                    "columns": frame.width,
                },
            )
            if self._enable_cache:
                self._cache[cache_key] = frame
        else:
            frame = cached

        return select_columns(frame, columns, spec)

    # --- Dataset-specific accessors -------------------------------------------------
    # Thin, explicitly typed wrappers so callers get discoverable, documented entry
    # points instead of passing dataset name strings around.

    def load_schedules(self, seasons: SeasonInput = None) -> pl.DataFrame:
        """Game schedules and results."""
        return self.load(Dataset.SCHEDULES, seasons)

    def load_player_stats(
        self, seasons: SeasonInput = None, *, summary_level: str = "week"
    ) -> pl.DataFrame:
        """Weekly player statistics."""
        return self.load(Dataset.PLAYER_STATS, seasons, summary_level=summary_level)

    def load_team_stats(
        self, seasons: SeasonInput = None, *, summary_level: str = "week"
    ) -> pl.DataFrame:
        """Weekly team statistics."""
        return self.load(Dataset.TEAM_STATS, seasons, summary_level=summary_level)

    def load_play_by_play(
        self, seasons: SeasonInput = None, *, columns: list[str] | None = None
    ) -> pl.DataFrame:
        """Play-by-play data. Pass ``columns`` to avoid materialising all 370+ columns."""
        return self.load(Dataset.PLAY_BY_PLAY, seasons, columns=columns)

    def load_rosters_weekly(self, seasons: SeasonInput = None) -> pl.DataFrame:
        """Weekly rosters."""
        return self.load(Dataset.ROSTERS_WEEKLY, seasons)

    def load_players(self) -> pl.DataFrame:
        """Player master table and identifier crosswalk."""
        return self.load(Dataset.PLAYERS)

    def load_injuries(self, seasons: SeasonInput = None) -> pl.DataFrame:
        """Weekly injury reports."""
        return self.load(Dataset.INJURIES, seasons)

    def load_snap_counts(self, seasons: SeasonInput = None) -> pl.DataFrame:
        """Per-game snap counts."""
        return self.load(Dataset.SNAP_COUNTS, seasons)

    def load_fantasy_opportunity(
        self, seasons: SeasonInput = None, *, stat_type: str = "weekly"
    ) -> pl.DataFrame:
        """Expected fantasy points and opportunity metrics."""
        return self.load(Dataset.FF_OPPORTUNITY, seasons, stat_type=stat_type)

    def available_seasons(self, dataset: Dataset | str, seasons: Iterable[int]) -> list[int]:
        """Return the subset of ``seasons`` that actually produced rows."""
        frame = self.load(dataset, list(seasons))
        if "season" not in frame.columns:
            return []
        return sorted(frame.get_column("season").unique().drop_nulls().to_list())
