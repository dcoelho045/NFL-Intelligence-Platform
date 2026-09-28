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


class UnresolvedPlayersError(NflPlatformError):
    """Several identifiers in one batch could not be resolved to a single player.

    Reporting the first failure and stopping forces a caller to fix a
    multi-player request one name at a time, which is slow for a human and
    actively harmful for an LLM: each retry spends a turn, and enough failed
    turns withdraw the tool entirely. Every failure is carried at once so one
    correction is enough.
    """

    def __init__(
        self,
        ambiguous: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (),
        missing: tuple[str, ...] = (),
    ) -> None:
        self.ambiguous = ambiguous
        self.missing = missing
        parts: list[str] = []
        if ambiguous:
            named = "; ".join(f"{query!r} matches {len(items)}" for query, items in ambiguous)
            parts.append(f"ambiguous: {named}")
        if missing:
            parts.append(f"not found: {', '.join(repr(name) for name in missing)}")
        super().__init__(
            f"{len(ambiguous) + len(missing)} of the supplied players could not be "
            f"identified ({'; '.join(parts)}). Supply a player_id for each."
        )


class LlmError(NflPlatformError):
    """Base class for failures in the assistant layer."""


class LlmConfigurationError(LlmError):
    """The assistant was asked to run without usable provider configuration."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"LLM is not configured: {reason}")


class LlmProviderError(LlmError):
    """The model provider could not be reached or refused the request.

    Unlike a tool failure -- which is handed back to the model as data it can
    recover from -- this ends the run, because there is no model left to talk to.
    """

    def __init__(self, reason: str, *, retryable: bool = False) -> None:
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"LLM provider request failed: {reason}")


class UnknownToolError(LlmError):
    """The model requested a tool that is not registered."""

    def __init__(self, tool: str, known: tuple[str, ...]) -> None:
        self.tool = tool
        self.known = known
        super().__init__(f"Unknown tool {tool!r}. Available tools: {', '.join(known)}.")


class AgentLimitExceededError(LlmError):
    """A run hit its iteration or tool-call budget.

    Raised internally and converted into a truthful partial answer rather than
    surfacing to the user, because a bounded run that explains what it could not
    retrieve is more useful than an exception.
    """

    def __init__(self, limit: str, value: int) -> None:
        self.limit = limit
        self.value = value
        super().__init__(f"Agent stopped after reaching its {limit} limit of {value}.")


class RetrievalError(NflPlatformError):
    """Base class for failures in the retrieval layer."""


class WebSearchNotConfiguredError(RetrievalError):
    """Web retrieval was requested with no provider or key available.

    Distinct from a provider outage: nothing is wrong, the capability simply is
    not switched on, and the assistant should say so rather than retry.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Web search is not configured: {reason}")


class WebSearchError(RetrievalError):
    """The web-search provider could not be reached or refused the request."""

    def __init__(self, provider: str, reason: str, *, retryable: bool = True) -> None:
        self.provider = provider
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"Web search via {provider!r} failed: {reason}")


class WebSearchBudgetExceededError(RetrievalError):
    """A single run asked for more searches than its configured ceiling allows."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"This question has already used its budget of {limit} web search(es).")


class KnowledgeIndexUnavailableError(RetrievalError):
    """The local vector index is missing, empty, or unreadable."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Knowledge index at {path!r} is unavailable: {reason}")


class KnowledgeIndexStaleError(RetrievalError):
    """The index was built with different embedding settings than are configured now.

    Comparing vectors from two different models silently produces plausible
    nonsense, so this refuses to serve rather than degrade.
    """

    def __init__(self, path: str, expected: str, found: str) -> None:
        self.path = path
        self.expected = expected
        self.found = found
        super().__init__(
            f"Knowledge index at {path!r} was built with {found}, but {expected} is "
            "configured. Rebuild it with scripts/build_knowledge_index.py."
        )


class CorpusValidationError(RetrievalError):
    """A knowledge document is missing required front-matter or is unreadable."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Knowledge document {path!r} is invalid: {reason}")


class EmbeddingError(RetrievalError):
    """The embedding provider could not be reached or refused the request."""

    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        self.reason = reason
        super().__init__(f"Embedding via {provider!r} failed: {reason}")
