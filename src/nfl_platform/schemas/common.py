"""Shared DTO building blocks."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nfl_platform.domain import EARLIEST_SEASON, MAX_WEEK, MIN_WEEK


class BaseDTO(BaseModel):
    """Base for every DTO.

    ``extra="forbid"`` matters more than it looks: LLM tool calls and MCP clients
    send model-generated arguments, and silently ignoring an unexpected field
    would hide a malformed call instead of rejecting it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class SeasonType(StrEnum):
    """Portion of the season a statistic covers."""

    REGULAR = "REG"
    POST = "POST"
    ALL = "ALL"


class PlayerRef(BaseDTO):
    """Minimal identity of a player, embedded wherever a player is referenced."""

    player_id: str = Field(description="nflverse GSIS id, for example '00-0033873'.")
    name: str = Field(description="Display name.")
    position: str | None = Field(default=None, description="Listed position, e.g. 'QB'.")
    team: str | None = Field(default=None, description="Team abbreviation.")


class SeasonWindow(BaseDTO):
    """The season/week scope a result was computed over.

    Every analytical response carries one of these so a caller -- or an LLM
    quoting the number -- can state exactly what the figure covers.
    """

    seasons: list[int] = Field(description="Seasons included, ascending.")
    season_type: SeasonType = Field(description="Game types included.")
    weeks: list[int] | None = Field(default=None, description="Explicit weeks requested, if any.")


class StatWindow(SeasonWindow):
    """A season window plus how many games actually contributed."""

    games: int = Field(ge=0, description="Number of games with data in this window.")
    requested_last_n_games: int | None = Field(
        default=None,
        description="The last_n_games limit requested, if any. May exceed 'games'.",
    )


class GameContext(BaseDTO):
    """Scheduling context for a single game."""

    game_id: str
    season: int = Field(ge=EARLIEST_SEASON)
    week: int = Field(ge=MIN_WEEK, le=MAX_WEEK)
    game_type: str | None = None
    gameday: str | None = Field(default=None, description="ISO date of kickoff.")
    home_team: str | None = None
    away_team: str | None = None
    roof: str | None = None
    surface: str | None = None
    spread_line: float | None = Field(
        default=None, description="Closing spread relative to the home team."
    )
    total_line: float | None = None


class HealthStatus(BaseDTO):
    """Liveness response. Performs no data access."""

    status: str = Field(default="ok")
    version: str


class ReadinessStatus(BaseDTO):
    """Readiness response. Confirms the data layer answers."""

    status: str = Field(description="'ready' or 'degraded'.")
    version: str
    data_layer_available: bool
    current_season: int | None = None
    detail: str | None = Field(default=None, description="Why the service is degraded, when it is.")


class ErrorResponse(BaseDTO):
    """Consistent error body for every failure the API returns."""

    error: str = Field(description="Machine-readable error type, e.g. 'player_not_found'.")
    message: str = Field(description="Human-readable explanation.")
    context: dict[str, Any] | None = Field(
        default=None,
        description="Structured detail, such as candidate players or valid team codes.",
    )
