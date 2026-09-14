"""Analytics services: the stable, object-oriented interface over the data layer.

Services return typed DTOs from ``nfl_platform.schemas``. Polars DataFrames do
not cross this boundary.

This package imports no web framework and no LLM SDK, so later milestones can
expose it as HTTP endpoints, LLM tools, or MCP tools without duplicating logic.
"""

from nfl_platform.services.base import AnalyticsService
from nfl_platform.services.container import ServiceContainer
from nfl_platform.services.directory import PlayerDirectoryService, normalize_name
from nfl_platform.services.fantasy import FantasyAnalyticsService
from nfl_platform.services.matchups import MatchupAnalyticsService
from nfl_platform.services.meta import MetaService
from nfl_platform.services.players import PlayerAnalyticsService
from nfl_platform.services.teams import TeamAnalyticsService

__all__ = [
    "AnalyticsService",
    "FantasyAnalyticsService",
    "MatchupAnalyticsService",
    "MetaService",
    "PlayerAnalyticsService",
    "PlayerDirectoryService",
    "ServiceContainer",
    "TeamAnalyticsService",
    "normalize_name",
]
