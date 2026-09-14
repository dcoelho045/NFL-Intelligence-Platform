"""Reference data about what the platform can answer for."""

from __future__ import annotations

import polars as pl

from nfl_platform.data.ingestion.datasets import Dataset
from nfl_platform.domain import SeasonInput
from nfl_platform.exceptions import NflPlatformError
from nfl_platform.services.base import AnalyticsService


class MetaService(AnalyticsService):
    """Valid teams and populated seasons, so callers need not guess."""

    def teams(self, seasons: SeasonInput = None) -> list[str]:
        """Team abbreviations appearing in the requested seasons.

        Derived from the data rather than a hard-coded list, so relocations and
        rebrands stay correct without maintenance.
        """
        schedules = self._loader.load_schedules(seasons)
        home = schedules.get_column("home_team").drop_nulls().to_list()
        away = schedules.get_column("away_team").drop_nulls().to_list()
        return sorted({str(team) for team in (*home, *away)})

    def seasons(self) -> list[int]:
        """Seasons with schedule data, most recent first."""
        newest = self._loader.current_season
        available: list[int] = []
        for season in range(newest, newest - 6, -1):
            try:
                frame = self._loader.load(Dataset.SCHEDULES, season)
            except NflPlatformError:
                # A season may not be published yet, which is not an error here.
                continue
            if not frame.is_empty():
                available.append(season)
        return available

    def current_season(self) -> int:
        return self._loader.current_season

    def team_exists(self, team: str, seasons: SeasonInput = None) -> bool:
        schedules = self._loader.load_schedules(seasons)
        return not schedules.filter(
            (pl.col("home_team") == team) | (pl.col("away_team") == team)
        ).is_empty()
