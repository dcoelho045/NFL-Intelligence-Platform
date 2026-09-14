"""Composition root for the service layer.

One container builds one set of services over one shared loader. Sharing the
loader matters: its per-instance memoisation is what stops five services from
downloading the same dataset five times.

This is what Milestones 3, 5, and 7 construct. They call these services
in-process rather than over HTTP, so FastAPI is one adapter beside them rather
than a dependency of them.
"""

from __future__ import annotations

from functools import cached_property

from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.services.directory import PlayerDirectoryService
from nfl_platform.services.fantasy import FantasyAnalyticsService
from nfl_platform.services.matchups import MatchupAnalyticsService
from nfl_platform.services.meta import MetaService
from nfl_platform.services.players import PlayerAnalyticsService
from nfl_platform.services.teams import TeamAnalyticsService


class ServiceContainer:
    """Lazily builds and holds the analytics services."""

    def __init__(
        self, loader: NflDataLoader | None = None, settings: Settings | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._loader = loader or NflDataLoader(settings=self._settings)

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def loader(self) -> NflDataLoader:
        return self._loader

    @cached_property
    def directory(self) -> PlayerDirectoryService:
        return PlayerDirectoryService(self._loader)

    @cached_property
    def players(self) -> PlayerAnalyticsService:
        return PlayerAnalyticsService(self._loader, directory=self.directory)

    @cached_property
    def teams(self) -> TeamAnalyticsService:
        return TeamAnalyticsService(self._loader, players=self.players)

    @cached_property
    def matchups(self) -> MatchupAnalyticsService:
        return MatchupAnalyticsService(self._loader, teams=self.teams)

    @cached_property
    def fantasy(self) -> FantasyAnalyticsService:
        return FantasyAnalyticsService(self._loader, directory=self.directory, teams=self.teams)

    @cached_property
    def meta(self) -> MetaService:
        return MetaService(self._loader)
