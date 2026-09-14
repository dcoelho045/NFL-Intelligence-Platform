"""Fantasy DTOs.

Fantasy views are re-framings of the same deterministic data as the player
views: production, opportunity, and matchup context. Nothing here projects or
predicts.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from nfl_platform.schemas.common import BaseDTO, PlayerRef, StatWindow
from nfl_platform.schemas.team import TeamDefenseSummary


class FantasyScoring(StrEnum):
    """Which scoring column a fantasy view reports."""

    STANDARD = "standard"
    PPR = "ppr"


class FantasyWeek(BaseDTO):
    """One week of fantasy production."""

    season: int
    week: int
    opponent: str | None = None
    points: float | None = None
    targets: int | None = None
    carries: int | None = None
    receptions: int | None = None
    offense_snap_pct: float | None = None


class FantasyProduction(BaseDTO):
    """Recent fantasy scoring for a player."""

    player: PlayerRef
    window: StatWindow
    scoring: FantasyScoring
    weeks: list[FantasyWeek]
    total_points: float | None = None
    avg_points: float | None = None
    best_week: float | None = None
    worst_week: float | None = None
    std_dev: float | None = Field(
        default=None,
        description="Week-to-week standard deviation; a consistency measure, not a projection.",
    )


class FantasyUsageTrend(BaseDTO):
    """Direction of a player's opportunity over the window.

    ``direction`` is computed from a simple comparison of the two halves of the
    window, not a regression. It is a description of what happened, never a
    forecast of what will happen.
    """

    player: PlayerRef
    window: StatWindow
    weeks: list[FantasyWeek]
    avg_target_share_early: float | None = None
    avg_target_share_recent: float | None = None
    avg_snap_pct_early: float | None = None
    avg_snap_pct_recent: float | None = None
    target_share_direction: str | None = Field(
        default=None, description="'rising', 'falling', 'stable', or null when unknown."
    )
    snap_share_direction: str | None = None


class FantasyMatchupContext(BaseDTO):
    """What the upcoming or requested opponent's defence has allowed."""

    player: PlayerRef
    opponent: str
    window: StatWindow
    opponent_defense: TeamDefenseSummary
    note: str | None = None


class FantasyComparison(BaseDTO):
    """Compare fantasy production across players over one shared window."""

    window: StatWindow
    scoring: FantasyScoring
    players: list[FantasyProduction]
    note: str | None = None
