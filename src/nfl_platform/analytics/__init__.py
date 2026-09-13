"""Deterministic NFL analytics.

These functions are the project's source of numerical truth. They contain no
LLM calls and no estimation: every value is computed from nflverse data.
"""

from nfl_platform.analytics.games import get_recent_games, summarize_recent_games
from nfl_platform.analytics.injuries import get_injury_information, get_team_injuries
from nfl_platform.analytics.players import (
    get_player_recent_stats,
    get_player_usage,
    resolve_pfr_id,
    summarize_player_stats,
)
from nfl_platform.analytics.teams import (
    get_team_defense,
    get_team_stats,
    summarize_team_stats,
)

__all__ = [
    "get_injury_information",
    "get_player_recent_stats",
    "get_player_usage",
    "get_recent_games",
    "get_team_defense",
    "get_team_injuries",
    "get_team_stats",
    "resolve_pfr_id",
    "summarize_player_stats",
    "summarize_recent_games",
    "summarize_team_stats",
]
