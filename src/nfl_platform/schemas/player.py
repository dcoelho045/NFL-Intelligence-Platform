"""Player-facing DTOs."""

from __future__ import annotations

from pydantic import Field

from nfl_platform.schemas.common import BaseDTO, PlayerRef, SeasonWindow, StatWindow


class PlayerProfile(BaseDTO):
    """Identity and career context for one player."""

    player_id: str
    name: str
    position: str | None = None
    position_group: str | None = None
    latest_team: str | None = None
    status: str | None = None
    jersey_number: int | None = None
    height_inches: int | None = None
    weight_pounds: int | None = None
    college: str | None = None
    rookie_season: int | None = None
    last_season: int | None = None
    years_experience: int | None = None
    draft_year: int | None = None
    draft_round: int | None = None
    draft_pick: int | None = None
    pfr_id: str | None = Field(
        default=None, description="Pro Football Reference id; links to snap counts."
    )


class PlayerCandidate(BaseDTO):
    """One possible match for a player-name search.

    ``disambiguator`` is a short human-readable string such as
    "QB, BUF, 2018-2026" intended to be shown to a user -- or an LLM -- when a
    name is ambiguous.
    """

    player_id: str
    name: str
    position: str | None = None
    latest_team: str | None = None
    rookie_season: int | None = None
    last_season: int | None = None
    disambiguator: str


class PlayerSearchResult(BaseDTO):
    """Result of a player-name search."""

    query: str
    match_count: int = Field(ge=0)
    is_ambiguous: bool = Field(
        description="True when more than one distinct player matches the query."
    )
    candidates: list[PlayerCandidate]


class PlayerStatLine(BaseDTO):
    """Box-score production for a single game, or totals across a window.

    Counting stats are integers here. Per-game averages use
    :class:`PlayerStatAverages`, which types them as floats -- 5.67 carries per
    game is not an integer, and forcing it into one would silently round.
    """

    completions: int | None = None
    attempts: int | None = None
    passing_yards: int | None = None
    passing_tds: int | None = None
    passing_interceptions: int | None = None
    passing_epa: float | None = None
    carries: int | None = None
    rushing_yards: int | None = None
    rushing_tds: int | None = None
    rushing_epa: float | None = None
    targets: int | None = None
    receptions: int | None = None
    receiving_yards: int | None = None
    receiving_tds: int | None = None
    receiving_epa: float | None = None
    fantasy_points: float | None = None
    fantasy_points_ppr: float | None = None


class PlayerStatAverages(BaseDTO):
    """Per-game averages. Same statistics as :class:`PlayerStatLine`, all fractional."""

    completions: float | None = None
    attempts: float | None = None
    passing_yards: float | None = None
    passing_tds: float | None = None
    passing_interceptions: float | None = None
    passing_epa: float | None = None
    carries: float | None = None
    rushing_yards: float | None = None
    rushing_tds: float | None = None
    rushing_epa: float | None = None
    targets: float | None = None
    receptions: float | None = None
    receiving_yards: float | None = None
    receiving_tds: float | None = None
    receiving_epa: float | None = None
    fantasy_points: float | None = None
    fantasy_points_ppr: float | None = None


class PlayerGameStats(BaseDTO):
    """One player's production in one game."""

    season: int
    week: int
    season_type: str | None = None
    game_id: str | None = None
    team: str | None = None
    opponent: str | None = None
    stats: PlayerStatLine


class PlayerGameLog(BaseDTO):
    """Game-by-game performance, newest first."""

    player: PlayerRef
    window: StatWindow
    games: list[PlayerGameStats]


class PlayerSeasonStats(BaseDTO):
    """Aggregated production over a window, with per-game averages alongside totals.

    Totals and averages are both returned because quoting one when the other was
    meant is a common source of wrong answers.
    """

    player: PlayerRef
    window: StatWindow
    totals: PlayerStatLine
    per_game: PlayerStatAverages


class PlayerUsageGame(BaseDTO):
    """Opportunity metrics for one game.

    Usage is about opportunity (snaps, targets, carries), not production.
    """

    season: int
    week: int
    game_id: str | None = None
    team: str | None = None
    opponent: str | None = None
    attempts: int | None = None
    carries: int | None = None
    targets: int | None = None
    receptions: int | None = None
    target_share: float | None = Field(
        default=None, description="Share of the team's targets, 0-1."
    )
    air_yards_share: float | None = None
    wopr: float | None = Field(
        default=None, description="Weighted opportunity rating; combines target and air-yard share."
    )
    offense_snaps: int | None = None
    offense_snap_pct: float | None = Field(
        default=None,
        description="Share of team offensive snaps, 0-1. Null when the player has no PFR id.",
    )


class PlayerUsage(BaseDTO):
    """Per-game usage plus window averages."""

    player: PlayerRef
    window: StatWindow
    games: list[PlayerUsageGame]
    avg_offense_snap_pct: float | None = None
    avg_target_share: float | None = None
    avg_wopr: float | None = None
    total_targets: int | None = None
    total_carries: int | None = None
    snap_data_available: bool = Field(
        description="False when snap counts could not be linked via the PFR crosswalk."
    )


class PlayerInjuryStatus(BaseDTO):
    """One weekly injury-report entry."""

    season: int
    week: int
    team: str | None = None
    report_status: str | None = Field(
        default=None, description="Game designation: Out, Doubtful, Questionable."
    )
    report_primary_injury: str | None = None
    report_secondary_injury: str | None = None
    practice_status: str | None = None
    practice_primary_injury: str | None = None
    date_modified: str | None = None


class PlayerInjuryReport(BaseDTO):
    """Injury history for a player over a window.

    An empty ``entries`` list means no injury was reported, which is information,
    not an error.
    """

    player: PlayerRef
    window: SeasonWindow
    entries: list[PlayerInjuryStatus]
    has_injury_data: bool


class PlayerComparison(BaseDTO):
    """Side-by-side comparison of two or more players over an identical window.

    The window is shared deliberately: comparing players over different numbers
    of games is the most common way a comparison misleads.
    """

    window: StatWindow
    players: list[PlayerSeasonStats]
    note: str | None = Field(
        default=None,
        description="Caveats, e.g. a player having fewer games than requested.",
    )
