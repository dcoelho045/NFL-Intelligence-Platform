"""Brave Search adapter.

Second implementation, and the reason the provider abstraction is credible: an
interface with one implementation is a class. Brave returns descriptions rather
than extracted page text, so it grounds answers less well than Tavily and is
offered as the alternative rather than the default.
"""

from __future__ import annotations

import httpx

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import WebSearchError, WebSearchNotConfiguredError
from nfl_platform.retrieval.models import WebResult
from nfl_platform.retrieval.web.provider import domain_of, parse_timestamp, utc_now

logger = get_logger(__name__)

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

# Brave expresses recency as a coarse bucket rather than a day count.
_FRESHNESS_BUCKETS: tuple[tuple[int, str], ...] = (
    (1, "pd"),
    (7, "pw"),
    (31, "pm"),
    (365, "py"),
)


def _freshness(recency_days: int | None) -> str | None:
    if not recency_days:
        return None
    for limit, code in _FRESHNESS_BUCKETS:
        if recency_days <= limit:
            return code
    return "py"


class BraveSearchProvider:
    """Calls the Brave Search API and normalises the response."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        key = self._settings.web_search_api_key
        self._api_key = key.get_secret_value().strip() if key is not None else ""

    @property
    def name(self) -> str:
        return "brave"

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[WebResult]:
        if not self._api_key:
            raise WebSearchNotConfiguredError("no Brave API key is set")

        # Brave has no include/exclude parameters, so domain policy is applied
        # locally in WebSearchClient. That is also why the client re-checks it
        # for every provider rather than trusting the vendor to have done it.
        params: dict[str, str | int] = {"q": query, "count": max_results}
        freshness = _freshness(recency_days)
        if freshness:
            params["freshness"] = freshness

        body = self._get(params)
        web = body.get("web")
        results = web.get("results") if isinstance(web, dict) else None
        if not isinstance(results, list):
            raise WebSearchError(self.name, "response contained no results array")
        return [item for item in (self._normalise(raw) for raw in results) if item is not None]

    def _get(self, params: dict[str, str | int]) -> dict[str, object]:
        headers = {
            "X-Subscription-Token": self._api_key,
            "Accept": "application/json",
        }
        timeout = self._settings.web_search_timeout_seconds
        try:
            if self._client is not None:
                response = self._client.get(
                    BRAVE_URL, params=params, headers=headers, timeout=timeout
                )
            else:
                with httpx.Client(timeout=timeout) as client:
                    response = client.get(BRAVE_URL, params=params, headers=headers)
        except httpx.TimeoutException as exc:
            raise WebSearchError(self.name, f"request timed out after {timeout}s") from exc
        except httpx.HTTPError as exc:
            raise WebSearchError(self.name, f"request failed: {exc}") from exc

        if response.status_code in (401, 403):
            raise WebSearchNotConfiguredError("the Brave API key was rejected")
        if response.status_code >= 400:
            retryable = response.status_code == 429 or response.status_code >= 500
            raise WebSearchError(
                self.name,
                f"provider returned HTTP {response.status_code}",
                retryable=retryable,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise WebSearchError(self.name, "response was not valid JSON", retryable=False) from exc
        if not isinstance(body, dict):
            raise WebSearchError(self.name, "response was not a JSON object", retryable=False)
        return body

    def _normalise(self, raw: object) -> WebResult | None:
        if not isinstance(raw, dict):
            return None
        url = raw.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return None
        description = raw.get("description")
        return WebResult(
            url=url,
            title=str(raw.get("title") or url),
            domain=domain_of(url),
            text=description if isinstance(description, str) else "",
            retrieved_at=utc_now(),
            provider=self.name,
            published_at=parse_timestamp(raw.get("page_age") or raw.get("age")),
        )
