"""The feature contract.

Milestone 6 requires documenting "exactly when each feature becomes available".
Prose in a design document drifts away from the code within one refactor, so the
declaration lives here as data: every builder publishes a :class:`FeatureSet`,
the documentation is generated from it, and a test asserts the built matrix and
the declared set agree column for column.

:class:`Availability` is the field that does the work. It answers the only
question that distinguishes a legitimate feature from a leaked one -- could a
caller standing at the prediction timestamp have known this value?
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum


class Availability(StrEnum):
    """When a feature's value becomes knowable, relative to kickoff."""

    SCHEDULE = "at_scheduling"
    """Fixed when the season schedule is published: opponent, venue, rest, week."""

    PRIOR_GAMES = "end_of_prior_game"
    """Aggregated only from games that finished before the one being predicted."""

    MARKET_CLOSE = "market_close"
    """Published betting line. Known before kickoff, but a different information
    set from football data: it embeds news, injuries and public money. Features
    with this availability are never inputs to the primary game model."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One model input, with the provenance needed to audit it."""

    name: str
    description: str
    dataset: str
    availability: Availability
    window: str | None = None
    """Lookback used, e.g. 'last 8 completed games'. None for point-in-time values."""

    @property
    def is_market_derived(self) -> bool:
        return self.availability is Availability.MARKET_CLOSE


class FeatureSet:
    """An ordered, named collection of feature specifications.

    Order is part of the contract, not an implementation detail: estimators are
    fitted on positional arrays, so a set that reorders itself between training
    and inference silently scores the right model against the wrong columns.
    """

    def __init__(self, version: str, specs: Iterable[FeatureSpec]) -> None:
        self._version = version
        self._specs = tuple(specs)
        seen: set[str] = set()
        for spec in self._specs:
            if spec.name in seen:
                raise ValueError(f"Duplicate feature {spec.name!r} in set {version!r}.")
            seen.add(spec.name)

    @property
    def version(self) -> str:
        """Contract version. Changing any feature's meaning must change this."""
        return self._version

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        return self._specs

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs)

    @property
    def market_features(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs if spec.is_market_derived)

    def without_market(self) -> FeatureSet:
        """The same set with market-derived inputs removed.

        Used to answer "how much of this model is football and how much is the
        betting line" with a trained comparison rather than an assertion.
        """
        kept = [spec for spec in self._specs if not spec.is_market_derived]
        return FeatureSet(f"{self._version}-nomarket", kept)

    def select(self, names: Iterable[str]) -> FeatureSet:
        index = {spec.name: spec for spec in self._specs}
        return FeatureSet(self._version, [index[name] for name in names])

    def spec(self, name: str) -> FeatureSpec:
        for spec in self._specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self._specs)

    def __contains__(self, name: object) -> bool:
        return any(spec.name == name for spec in self._specs)
