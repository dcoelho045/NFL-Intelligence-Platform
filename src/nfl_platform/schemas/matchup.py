"""Matchup DTOs.

A matchup analysis is descriptive only. It reports what both teams have done and
the scheduling context. It deliberately contains no win probability or projected
score -- prediction is Milestone 6, and mixing the two would let a descriptive
endpoint be mistaken for a predictive one.
"""

from __future__ import annotations

from pydantic import Field

from nfl_platform.schemas.common import BaseDTO, GameContext, SeasonWindow
from nfl_platform.schemas.team import (
    TeamDefenseSummary,
    TeamForm,
    TeamOffenseSummary,
)


class MatchupSide(BaseDTO):
    """One team's profile within a matchup."""

    team: str
    is_home: bool | None = Field(
        default=None, description="Null when the fixture is not on the schedule."
    )
    offense: TeamOffenseSummary
    defense: TeamDefenseSummary
    form: TeamForm
    rest_days: int | None = None


class HeadToHeadGame(BaseDTO):
    """A previous meeting between the two teams."""

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    home_score: int | None = None
    away_score: int | None = None
    winner: str | None = Field(default=None, description="Team abbreviation, or null for a tie.")


class MatchupAnalysis(BaseDTO):
    """Side-by-side descriptive comparison of two teams."""

    window: SeasonWindow
    team_a: MatchupSide
    team_b: MatchupSide
    scheduled_game: GameContext | None = Field(
        default=None,
        description="The scheduled fixture, when the two teams meet in the requested week.",
    )
    head_to_head: list[HeadToHeadGame] = Field(
        default_factory=list, description="Previous meetings within the requested seasons."
    )
    note: str | None = Field(
        default=None,
        description="Caveats, such as a small sample or no scheduled meeting.",
    )
