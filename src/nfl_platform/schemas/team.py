"""Team-facing DTOs."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from nfl_platform.schemas.common import BaseDTO, PlayerRef, SeasonWindow, StatWindow
from nfl_platform.schemas.player import PlayerInjuryStatus


class TeamSplit(StrEnum):
    """Which subset of a team's games a summary covers."""

    ALL = "all"
    HOME = "home"
    AWAY = "away"


class TeamOffenseSummary(BaseDTO):
    """What a team's offence produced, per game and in total."""

    team: str
    window: StatWindow
    split: TeamSplit = TeamSplit.ALL
    total_passing_yards: int | None = None
    total_rushing_yards: int | None = None
    total_passing_tds: int | None = None
    total_rushing_tds: int | None = None
    total_turnovers: int | None = None
    avg_passing_yards: float | None = None
    avg_rushing_yards: float | None = None
    avg_total_yards: float | None = None
    avg_passing_epa: float | None = None
    avg_rushing_epa: float | None = None


class TeamDefenseSummary(BaseDTO):
    """What a team's defence allowed.

    Fields are prefixed ``allowed_`` so this can never be confused with the
    team's own production. Derived from opponent rows in the team-stats dataset.
    """

    team: str
    window: StatWindow
    split: TeamSplit = TeamSplit.ALL
    total_allowed_passing_yards: int | None = None
    total_allowed_rushing_yards: int | None = None
    total_allowed_passing_tds: int | None = None
    total_allowed_rushing_tds: int | None = None
    avg_allowed_passing_yards: float | None = None
    avg_allowed_rushing_yards: float | None = None
    avg_allowed_total_yards: float | None = None
    avg_allowed_passing_epa: float | None = None
    avg_allowed_rushing_epa: float | None = None


class TeamGameResult(BaseDTO):
    """One game from a single team's perspective."""

    game_id: str
    season: int
    week: int
    game_type: str | None = None
    gameday: str | None = None
    opponent: str
    is_home: bool
    points_for: int | None = None
    points_against: int | None = None
    point_differential: int | None = None
    outcome: str | None = Field(default=None, description="'W', 'L', or 'T'.")


class TeamGameLog(BaseDTO):
    """A team's games, newest first."""

    team: str
    window: StatWindow
    games: list[TeamGameResult]


class TeamForm(BaseDTO):
    """Recent won-lost record and scoring."""

    team: str
    window: StatWindow
    split: TeamSplit = TeamSplit.ALL
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    ties: int = Field(ge=0)
    avg_points_for: float | None = None
    avg_points_against: float | None = None
    avg_point_differential: float | None = None
    results: list[str] = Field(
        default_factory=list, description="Outcomes newest first, e.g. ['W','W','L']."
    )


class TeamHomeAwaySplits(BaseDTO):
    """A team's form at home versus on the road over the same window."""

    team: str
    window: SeasonWindow
    home: TeamForm
    away: TeamForm


class TeamInjuryEntry(BaseDTO):
    """One injury-report row, carrying the player identity a team view needs."""

    player: PlayerRef
    status: PlayerInjuryStatus


class TeamInjuryReport(BaseDTO):
    """Every reported injury for a team in a window."""

    team: str
    window: SeasonWindow
    entries: list[TeamInjuryEntry]
    player_count: int = Field(ge=0)
