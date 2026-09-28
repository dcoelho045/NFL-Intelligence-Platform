"""Search policy: everything that is true regardless of which vendor answers.

Caching, sanitisation, truncation, domain filtering, per-run budgets and cost
accounting live here rather than in an adapter. That split is what makes the
provider genuinely swappable: changing vendor changes where results come from,
never how they are bounded or how they are treated.
"""

from __future__ import annotations

import time
from contextvars import ContextVar

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import WebSearchBudgetExceededError, WebSearchNotConfiguredError
from nfl_platform.retrieval.models import SearchStats, WebResult
from nfl_platform.retrieval.sanitize import sanitise
from nfl_platform.retrieval.web.cache import CacheKey, SearchCache
from nfl_platform.retrieval.web.provider import (
    NullSearchProvider,
    WebSearchProvider,
    build_search_provider,
)

logger = get_logger(__name__)

# Searches already spent by the question currently being answered. A ContextVar
# rather than instance state because the client is process-wide (one per
# container) while the budget is per run, and it must not leak between
# concurrent requests.
_searches_used: ContextVar[int] = ContextVar("web_searches_used", default=0)


class WebSearchClient:
    """Bounded, sanitised, cached access to a web-search provider."""

    def __init__(
        self,
        provider: WebSearchProvider | None = None,
        settings: Settings | None = None,
        *,
        cache: SearchCache | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider = provider or build_search_provider(self._settings)
        self._cache = cache or SearchCache(
            ttl_seconds=self._settings.web_search_cache_ttl_seconds,
            max_entries=self._settings.web_search_cache_max_entries,
        )

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def available(self) -> bool:
        return self._provider.available and self._settings.web_search_max_calls_per_run > 0

    @staticmethod
    def begin_run() -> None:
        """Reset the per-question search budget."""
        _searches_used.set(0)

    @staticmethod
    def searches_used() -> int:
        return _searches_used.get()

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        recency_days: int | None = None,
    ) -> tuple[list[WebResult], SearchStats]:
        """Run one bounded search and return sanitised results with provenance."""
        text = " ".join(query.split())
        if not text:
            raise ValueError("Search query must not be empty.")

        limit = self._clamp_results(max_results)
        window = (
            recency_days if recency_days is not None else self._settings.web_search_recency_days
        )
        stats = SearchStats(provider=self._provider.name)
        started = time.perf_counter()

        if isinstance(self._provider, NullSearchProvider) or not self._provider.available:
            raise WebSearchNotConfiguredError(
                f"provider {self._provider.name!r} has no usable configuration"
            )

        key = CacheKey.build(text, limit, window, self._provider.name)
        cached = self._cache.get(key)
        if cached is not None:
            stats.cache_hits = 1
            stats.results_returned = len(cached)
            stats.duration_ms = (time.perf_counter() - started) * 1000
            return cached, stats

        self._spend_budget()
        raw = self._provider.search(
            text,
            max_results=limit,
            recency_days=window,
            include_domains=self._settings.web_search_allowed_domains or None,
            exclude_domains=self._settings.web_search_blocked_domains or None,
        )
        stats.queries = 1

        results = self._apply_policy(raw, stats)
        self._cache.put(key, results)
        stats.results_returned = len(results)
        stats.duration_ms = (time.perf_counter() - started) * 1000
        return results, stats

    def _clamp_results(self, requested: int | None) -> int:
        ceiling = self._settings.web_search_max_results
        if requested is None:
            return ceiling
        return max(1, min(requested, ceiling))

    def _spend_budget(self) -> None:
        limit = self._settings.web_search_max_calls_per_run
        used = _searches_used.get()
        if used >= limit:
            raise WebSearchBudgetExceededError(limit)
        _searches_used.set(used + 1)

    def _apply_policy(self, raw: list[WebResult], stats: SearchStats) -> list[WebResult]:
        """Domain filtering, sanitisation and injection handling, in that order."""
        allowed = {domain.lower() for domain in self._settings.web_search_allowed_domains}
        blocked = {domain.lower() for domain in self._settings.web_search_blocked_domains}
        max_chars = self._settings.web_search_max_chars_per_result
        kept: list[WebResult] = []

        for result in raw:
            # Re-checked locally even when the provider was asked to filter:
            # trusting a vendor to have enforced our policy is not enforcement.
            if not _domain_permitted(result.domain, allowed, blocked):
                stats.results_dropped += 1
                continue

            text, flagged = sanitise(result.text, max_chars=max_chars)
            title, title_flagged = sanitise(result.title, max_chars=200)
            flagged = flagged or title_flagged

            if flagged:
                stats.injection_flags += 1
                logger.warning(
                    "Retrieved content contains prompt-injection patterns",
                    extra={"url": result.url, "domain": result.domain},
                )
                if self._settings.web_search_drop_flagged_results:
                    stats.results_dropped += 1
                    continue

            kept.append(
                WebResult(
                    url=result.url,
                    title=title or result.url,
                    domain=result.domain,
                    text=text,
                    retrieved_at=result.retrieved_at,
                    provider=result.provider,
                    published_at=result.published_at,
                    provider_score=result.provider_score,
                    injection_flagged=flagged,
                )
            )

        if stats.injection_flags:
            stats.notes.append(
                f"{stats.injection_flags} result(s) contained text resembling instructions "
                "and were treated as untrusted."
            )
        return kept


def _domain_permitted(domain: str, allowed: set[str], blocked: set[str]) -> bool:
    """Suffix matching, so a rule for ``espn.com`` also covers ``www.espn.com``."""
    host = domain.lower()
    if any(host == rule or host.endswith(f".{rule}") for rule in blocked):
        return False
    if not allowed:
        return True
    return any(host == rule or host.endswith(f".{rule}") for rule in allowed)
