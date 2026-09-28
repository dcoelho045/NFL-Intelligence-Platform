"""In-process TTL cache for web searches.

Deliberately memory-only and bounded. Persisting retrieved article text would
mean storing third-party content we have no need to keep, and the milestone asks
for no more web content to be stored than the request requires.

What it buys: a run that searches for the same player twice, or a benchmark that
repeats a question, pays the provider once.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from nfl_platform.retrieval.models import WebResult


@dataclass(frozen=True)
class CacheKey:
    """Everything that changes a result set."""

    query: str
    max_results: int
    recency_days: int | None
    provider: str

    @classmethod
    def build(
        cls, query: str, max_results: int, recency_days: int | None, provider: str
    ) -> CacheKey:
        return cls(
            query=" ".join(query.lower().split()),
            max_results=max_results,
            recency_days=recency_days,
            provider=provider,
        )


class SearchCache:
    """Bounded LRU cache with a time-to-live."""

    def __init__(self, *, ttl_seconds: int, max_entries: int) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._entries: OrderedDict[CacheKey, tuple[float, list[WebResult]]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._ttl > 0 and self._max_entries > 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: CacheKey) -> list[WebResult] | None:
        if not self.enabled:
            return None
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, results = entry
        if time.monotonic() - stored_at > self._ttl:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return list(results)

    def put(self, key: CacheKey, results: list[WebResult]) -> None:
        if not self.enabled:
            return
        self._entries[key] = (time.monotonic(), list(results))
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
