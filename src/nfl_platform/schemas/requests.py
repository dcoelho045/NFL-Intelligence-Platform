"""Request models for operations whose arguments do not fit in a URL.

These live alongside the response DTOs rather than inside ``api/`` because
Milestones 3 and 5 need them as tool-argument schemas, and must be able to
import them without pulling in a web framework.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from nfl_platform.schemas.common import BaseDTO, SeasonType
from nfl_platform.schemas.fantasy import FantasyScoring

MAX_COMPARISON_PLAYERS = 6
MAX_LAST_N_GAMES = 50


class PlayerComparisonRequest(BaseDTO):
    """Compare two or more players over one shared window."""

    player_ids: list[str] = Field(
        min_length=2,
        max_length=MAX_COMPARISON_PLAYERS,
        description="Player ids or names. Names must resolve unambiguously.",
    )
    seasons: list[int] | None = Field(
        default=None, description="Seasons to cover. Omit to use the configured default."
    )
    last_n_games: int | None = Field(
        default=4, ge=1, le=MAX_LAST_N_GAMES, description="Games per player. Null for all."
    )
    season_type: SeasonType = SeasonType.ALL

    @field_validator("player_ids")
    @classmethod
    def _reject_blank_ids(cls, value: list[str]) -> list[str]:
        if any(not item or not item.strip() for item in value):
            raise ValueError("Player identifiers must not be blank.")
        return [item.strip() for item in value]


class FantasyComparisonRequest(PlayerComparisonRequest):
    """Compare fantasy production across players."""

    scoring: FantasyScoring = FantasyScoring.PPR
