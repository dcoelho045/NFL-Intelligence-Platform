"""Test doubles shared across the suite.

Kept out of ``conftest.py`` so test modules can import these directly without
relying on the conftest module being importable as a package.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from nfl_platform.data.ingestion import Dataset, ParquetDataSource
from nfl_platform.exceptions import WebSearchError, WebSearchNotConfiguredError
from nfl_platform.retrieval.models import WebResult
from nfl_platform.retrieval.web.provider import domain_of, parse_timestamp

FIXTURE_SEASON = 2024

# Identifiers that exist in the committed fixture slice.
MAHOMES_ID = "00-0033873"
ALLEN_ID = "00-0034857"
UNKNOWN_PLAYER_ID = "00-9999999"

WEB_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "web"
KNOWLEDGE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "knowledge"


class CountingDataSource:
    """Wraps a data source and records how many times each dataset was fetched."""

    def __init__(self, inner: ParquetDataSource) -> None:
        self._inner = inner
        self.calls: list[Dataset] = []

    def fetch(self, dataset: Dataset, seasons: list[int] | None, **kwargs: Any) -> pl.DataFrame:
        self.calls.append(dataset)
        return self._inner.fetch(dataset, seasons, **kwargs)


class StubDataSource:
    """Returns caller-supplied frames, for exercising validation and error paths."""

    def __init__(self, frames: dict[Dataset, pl.DataFrame]) -> None:
        self._frames = frames

    def fetch(
        self,
        dataset: Dataset,
        seasons: list[int] | None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        return self._frames[dataset]


def load_web_fixture(name: str) -> list[WebResult]:
    """Read a recorded provider response and normalise it the way an adapter would.

    The payloads are real Tavily response shapes, trimmed and sanitised of keys
    we never read. Keeping the real shape is what makes these tests able to
    catch a normalisation regression that a hand-written dict would not.
    """
    payload = json.loads((WEB_FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    now = datetime.now(UTC)
    results: list[WebResult] = []
    for item in payload.get("results", []):
        url = item["url"]
        # Unusable URLs are dropped here exactly as a real adapter drops them.
        if not url.startswith(("http://", "https://")):
            continue
        published = item.get("published_date")
        # Some fixtures express age in days so they never become stale.
        if published is None and item.get("published_days_ago") is not None:
            published_at = now - timedelta(days=int(item["published_days_ago"]))
        else:
            published_at = parse_timestamp(published)
        results.append(
            WebResult(
                url=url,
                title=item.get("title") or url,
                domain=domain_of(url),
                text=item.get("content", ""),
                retrieved_at=now,
                provider="stub",
                published_at=published_at,
                provider_score=item.get("score"),
            )
        )
    return results


class StubWebSearchProvider:
    """A web-search provider driven by fixtures instead of a network.

    Satisfies the same Protocol as the real adapters, so everything above it --
    caching, sanitisation, domain policy, budgets, the tool, the agent -- runs
    exactly as it does in production.
    """

    def __init__(
        self,
        results: list[WebResult] | None = None,
        *,
        fixture: str | None = None,
        error: Exception | None = None,
        available: bool = True,
        name: str = "stub",
    ) -> None:
        if fixture is not None:
            results = load_web_fixture(fixture)
        self._results = results if results is not None else []
        self._error = error
        self._available = available
        self._name = name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def available(self) -> bool:
        return self._available

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[WebResult]:
        self.calls.append(
            {
                "query": query,
                "max_results": max_results,
                "recency_days": recency_days,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
            }
        )
        if self._error is not None:
            raise self._error
        if not self._available:
            raise WebSearchNotConfiguredError("stub provider is switched off")
        return self._results[:max_results]


def failing_web_provider(reason: str = "upstream is down") -> StubWebSearchProvider:
    return StubWebSearchProvider(error=WebSearchError("stub", reason))
