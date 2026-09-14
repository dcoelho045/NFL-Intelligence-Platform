"""Typed data-transfer objects: the project's stable public contract.

These models are the boundary where Polars DataFrames stop. Everything above
this layer -- the HTTP API today, LLM tools and MCP tools in later milestones --
speaks in these types.

This package deliberately imports no web framework and no LLM SDK, so it can be
imported by any adapter. ``model_json_schema()`` on these models produces the
JSON Schema that tool-calling and MCP tool definitions require.
"""

from nfl_platform.schemas.common import (
    BaseDTO,
    ErrorResponse,
    GameContext,
    HealthStatus,
    PlayerRef,
    ReadinessStatus,
    SeasonType,
    SeasonWindow,
    StatWindow,
)
from nfl_platform.schemas.fantasy import (
    FantasyComparison,
    FantasyMatchupContext,
    FantasyProduction,
    FantasyScoring,
    FantasyUsageTrend,
    FantasyWeek,
)
from nfl_platform.schemas.matchup import (
    HeadToHeadGame,
    MatchupAnalysis,
    MatchupSide,
)
from nfl_platform.schemas.player import (
    PlayerCandidate,
    PlayerComparison,
    PlayerGameLog,
    PlayerGameStats,
    PlayerInjuryReport,
    PlayerInjuryStatus,
    PlayerProfile,
    PlayerSearchResult,
    PlayerSeasonStats,
    PlayerStatAverages,
    PlayerStatLine,
    PlayerUsage,
    PlayerUsageGame,
)
from nfl_platform.schemas.team import (
    TeamDefenseSummary,
    TeamForm,
    TeamGameLog,
    TeamGameResult,
    TeamHomeAwaySplits,
    TeamInjuryEntry,
    TeamInjuryReport,
    TeamOffenseSummary,
    TeamSplit,
)

__all__ = [
    "BaseDTO",
    "ErrorResponse",
    "FantasyComparison",
    "FantasyMatchupContext",
    "FantasyProduction",
    "FantasyScoring",
    "FantasyUsageTrend",
    "FantasyWeek",
    "GameContext",
    "HeadToHeadGame",
    "HealthStatus",
    "MatchupAnalysis",
    "MatchupSide",
    "PlayerCandidate",
    "PlayerComparison",
    "PlayerGameLog",
    "PlayerGameStats",
    "PlayerInjuryReport",
    "PlayerInjuryStatus",
    "PlayerProfile",
    "PlayerRef",
    "PlayerSearchResult",
    "PlayerSeasonStats",
    "PlayerStatAverages",
    "PlayerStatLine",
    "PlayerUsage",
    "PlayerUsageGame",
    "ReadinessStatus",
    "SeasonType",
    "SeasonWindow",
    "StatWindow",
    "TeamDefenseSummary",
    "TeamForm",
    "TeamGameLog",
    "TeamGameResult",
    "TeamHomeAwaySplits",
    "TeamInjuryEntry",
    "TeamInjuryReport",
    "TeamOffenseSummary",
    "TeamSplit",
]
