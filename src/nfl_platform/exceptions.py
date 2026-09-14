"""Exception hierarchy for the NFL platform.

A single root exception lets callers (and, later, tool-calling layers) distinguish
expected domain failures from genuine bugs.
"""

from __future__ import annotations


class NflPlatformError(Exception):
    """Base class for every error raised deliberately by this package."""


class DatasetNotAvailableError(NflPlatformError):
    """A requested nflverse dataset could not be retrieved."""

    def __init__(self, dataset: str, reason: str) -> None:
        self.dataset = dataset
        self.reason = reason
        super().__init__(f"Dataset {dataset!r} could not be loaded: {reason}")


class SchemaValidationError(NflPlatformError):
    """A loaded dataset is missing columns this project depends on."""

    def __init__(self, dataset: str, missing_columns: frozenset[str]) -> None:
        self.dataset = dataset
        self.missing_columns = missing_columns
        listed = ", ".join(sorted(missing_columns))
        super().__init__(
            f"Dataset {dataset!r} is missing required column(s): {listed}. "
            "The upstream nflverse schema may have changed."
        )


class UnknownDatasetError(NflPlatformError):
    """A dataset name was requested that is not in the registry."""

    def __init__(self, dataset: str, known: frozenset[str]) -> None:
        self.dataset = dataset
        listed = ", ".join(sorted(known))
        super().__init__(f"Unknown dataset {dataset!r}. Known datasets: {listed}.")


class InvalidSeasonError(NflPlatformError):
    """A season value is outside the range nflverse publishes."""

    def __init__(self, season: object, earliest: int, latest: int) -> None:
        self.season = season
        super().__init__(
            f"Invalid season {season!r}. Expected an integer between {earliest} and {latest}."
        )


class InvalidWeekError(NflPlatformError):
    """A week value is not a usable NFL week number."""

    def __init__(self, week: object) -> None:
        self.week = week
        super().__init__(f"Invalid week {week!r}. Expected an integer between 1 and 22.")


class InvalidGameCountError(NflPlatformError):
    """A ``last_n_games`` argument is not a positive integer."""

    def __init__(self, value: object) -> None:
        self.value = value
        super().__init__(f"Invalid game count {value!r}. Expected a positive integer.")


class PlayerNotFoundError(NflPlatformError):
    """No rows exist for the supplied player identifier in the requested scope."""

    def __init__(self, player_id: str, scope: str) -> None:
        self.player_id = player_id
        self.scope = scope
        super().__init__(f"Player {player_id!r} was not found in {scope}.")


class AmbiguousPlayerError(NflPlatformError):
    """A player name matches more than one player.

    Carries the candidates rather than guessing. Roughly 7.5% of nflverse
    players share a display name with someone else, so silently picking the
    first match would return real statistics for the wrong person -- a failure
    that looks like a correct answer.
    """

    def __init__(self, query: str, candidates: tuple[tuple[str, str], ...]) -> None:
        self.query = query
        self.candidates = candidates
        listed = "; ".join(f"{player_id} ({description})" for player_id, description in candidates)
        super().__init__(
            f"Player name {query!r} matches {len(candidates)} players: {listed}. "
            "Narrow the search by position, team, or season."
        )


class TeamNotFoundError(NflPlatformError):
    """A team abbreviation does not appear in the requested dataset scope."""

    def __init__(self, team: str, scope: str, known: tuple[str, ...] = ()) -> None:
        self.team = team
        self.scope = scope
        self.known = known
        suffix = f" Known teams in scope: {', '.join(known)}." if known else ""
        super().__init__(f"Team {team!r} was not found in {scope}.{suffix}")
