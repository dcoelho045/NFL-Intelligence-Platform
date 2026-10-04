"""Argument field definitions shared by every adapter.

Defined once, in the schema layer, so the LLM tool schemas and the MCP tool
schemas describe the same argument in the same words. Inconsistent phrasing
between two catalogues of the same capability is a real source of wrong
arguments, and it is invisible until a model gets it wrong.

This lives here rather than in ``llm/`` because ``llm/`` and ``mcp_server/`` are
siblings that must not import each other -- the same reasoning that put
:class:`~nfl_platform.schemas.evidence.SourceKind` in the schema layer.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from nfl_platform.domain import MAX_WEEK, MIN_WEEK
from nfl_platform.schemas.common import SeasonType

PLAYER_DESCRIPTION = (
    "Player full name (for example 'Josh Allen') or GSIS id (for example '00-0034857'). "
    "Prefer a GSIS id when you already have one, because names can be ambiguous."
)
TEAM_DESCRIPTION = (
    "Team abbreviation, for example 'KC', 'BUF', 'SF'. Call list_reference_data if unsure."
)

# Upper bound on the players one comparison call may take. Two is the minimum
# that makes a comparison, six is where a shared window stops being readable.
MAX_COMPARISON_PLAYERS = 6
# Upper bound on how many seasons one request may span. Each season is a
# separate download, so this bounds both latency and payload size.
MAX_SEASONS_PER_REQUEST = 6

PlayerField = Annotated[str, Field(description=PLAYER_DESCRIPTION, min_length=1)]
TeamField = Annotated[str, Field(description=TEAM_DESCRIPTION, min_length=2, max_length=4)]

SeasonsField = Annotated[
    list[int] | None,
    Field(
        default=None,
        description="Seasons to cover, for example [2024]. Omit for the current default season.",
    ),
]
LastNGamesField = Annotated[
    int | None,
    Field(
        default=4,
        ge=1,
        le=50,
        description="Number of most recent games with data to include. Null includes every game.",
    ),
]
WeeksField = Annotated[
    list[int] | None,
    Field(default=None, description="Specific week numbers (1-22). Omit for every week."),
]
SeasonTypeField = Annotated[
    SeasonType,
    Field(
        default=SeasonType.ALL,
        description="REG for regular season only, POST for playoffs only, ALL for both.",
    ),
]
WeekField = Annotated[
    int | None,
    Field(
        default=None,
        ge=MIN_WEEK,
        le=MAX_WEEK,
        description="A single week number (1-22). Omit for every week.",
    ),
]
SeasonField = Annotated[
    int | None,
    Field(
        default=None,
        description="A single season, for example 2024. Omit for the current default season.",
    ),
]
ComparedPlayersField = Annotated[
    list[str],
    Field(
        min_length=2,
        max_length=MAX_COMPARISON_PLAYERS,
        description="Two to six player names or GSIS ids to compare over one shared window.",
    ),
]
