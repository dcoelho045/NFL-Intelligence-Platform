"""Argument fields shared across tools.

The definitions moved to :mod:`nfl_platform.schemas.fields` in Milestone 5 so the
MCP adapter could describe the same arguments in the same words without
importing a sibling adapter. They are re-exported here because this is where the
tool modules expect to find them.
"""

from __future__ import annotations

from nfl_platform.schemas.fields import (
    PLAYER_DESCRIPTION,
    TEAM_DESCRIPTION,
    ComparedPlayersField,
    LastNGamesField,
    PlayerField,
    SeasonsField,
    SeasonTypeField,
    TeamField,
    WeeksField,
)

__all__ = [
    "PLAYER_DESCRIPTION",
    "TEAM_DESCRIPTION",
    "ComparedPlayersField",
    "LastNGamesField",
    "PlayerField",
    "SeasonTypeField",
    "SeasonsField",
    "TeamField",
    "WeeksField",
]
