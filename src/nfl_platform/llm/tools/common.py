"""Argument fields shared across tools.

Defined once so every tool describes seasons, windows, and identifiers to the
model in identical language. Inconsistent phrasing between tool schemas is a
real source of wrong arguments.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from nfl_platform.schemas.common import SeasonType

PLAYER_DESCRIPTION = (
    "Player full name (for example 'Josh Allen') or GSIS id (for example '00-0034857'). "
    "Prefer a GSIS id when you already have one, because names can be ambiguous."
)
TEAM_DESCRIPTION = (
    "Team abbreviation, for example 'KC', 'BUF', 'SF'. Call list_reference_data if unsure."
)

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
