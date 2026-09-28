"""The web-search provider contract.

A Protocol rather than a base class, matching ``ChatClient`` and ``DataSource``:
a test double needs no inheritance, and the adapters stay unaware of each other.

This module imports no HTTP client on purpose. It is what everything above the
adapters depends on, so it must stay free of vendor concerns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import WebSearchNotConfiguredError
from nfl_platform.retrieval.models import WebResult


@runtime_checkable
class WebSearchProvider(Protocol):
    """One web-search vendor, normalised to :class:`WebResult`."""

    @property
    def name(self) -> str:
        """Vendor identifier recorded in provenance."""
        ...

    @property
    def available(self) -> bool:
        """Whether this provider can actually run a search right now."""
        ...

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[WebResult]:
        """Run one search. Raises WebSearchError on provider failure."""
        ...


class NullSearchProvider:
    """The provider used when web retrieval is switched off.

    A first-class implementation rather than a None check, so the whole stack --
    services, tools, API, tests -- runs unchanged on a machine with no key. The
    failure is a clear "not configured", never a crash.
    """

    def __init__(self, reason: str = "no provider is configured") -> None:
        self._reason = reason

    @property
    def name(self) -> str:
        return "none"

    @property
    def available(self) -> bool:
        return False

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[WebResult]:
        raise WebSearchNotConfiguredError(self._reason)


def domain_of(url: str) -> str:
    """Registrable host for a URL, lowercased and stripped of ``www.``."""
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def utc_now() -> datetime:
    """Retrieval timestamp. Always UTC so provenance is comparable across hosts."""
    return datetime.now(UTC)


def parse_timestamp(value: object) -> datetime | None:
    """Best-effort parse of the many date shapes search vendors return.

    Returns None rather than guessing. A wrong publication date is worse than an
    absent one, because freshness is one of the things the answer is judged on.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    for parse in (
        datetime.fromisoformat,
        # RFC 1123 with a numeric offset, then with a named zone: Tavily returns
        # "Sat, 26 Sep 2026 23:00:00 GMT", and %z does not match "GMT".
        lambda raw: datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %z"),
        lambda raw: datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %Z"),
        lambda raw: datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S"),
        lambda raw: datetime.strptime(raw, "%Y-%m-%d"),
        lambda raw: datetime.strptime(raw, "%d %b %Y"),
    ):
        try:
            parsed = parse(text)
        except (ValueError, TypeError):
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def build_search_provider(settings: Settings | None = None) -> WebSearchProvider:
    """Build the configured provider.

    Imports the adapter lazily so a deployment that never searches the web never
    imports an HTTP client.
    """
    settings = settings or get_settings()
    name = settings.web_search_provider

    if name == "none":
        return NullSearchProvider("NFL_WEB_SEARCH_PROVIDER is set to 'none'")

    key = settings.web_search_api_key
    if key is None or not key.get_secret_value().strip():
        return NullSearchProvider(f"no API key for provider {name!r}")

    if name == "tavily":
        from nfl_platform.retrieval.web.tavily import TavilySearchProvider

        return TavilySearchProvider(settings)

    from nfl_platform.retrieval.web.brave import BraveSearchProvider

    return BraveSearchProvider(settings)
