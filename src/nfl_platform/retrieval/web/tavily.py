"""Tavily search adapter.

Chosen as the default because it returns extracted page text rather than a
search-engine description. Snippet-only APIs force you to fetch and parse the
page yourself, which is the crawler this milestone deliberately does not build.

One of only two modules permitted to import an HTTP client; the architecture
test enforces that.
"""

from __future__ import annotations

import httpx

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import WebSearchError, WebSearchNotConfiguredError
from nfl_platform.retrieval.models import WebResult
from nfl_platform.retrieval.web.provider import domain_of, parse_timestamp, utc_now

logger = get_logger(__name__)

TAVILY_URL = "https://api.tavily.com/search"

# Tavily only honours the `days` recency filter on the news topic, so the topic
# is derived from the caller's intent rather than configured separately.
_NEWS_TOPIC = "news"
_GENERAL_TOPIC = "general"


class TavilySearchProvider:
    """Calls the Tavily Search API and normalises the response."""

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
        return "tavily"

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
            raise WebSearchNotConfiguredError("no Tavily API key is set")

        payload: dict[str, object] = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            # Both suppress extra billable work we do not use: the answer would be
            # a second model's opinion, and raw content is the page before cleaning.
            "include_answer": False,
            "include_raw_content": False,
            "topic": _NEWS_TOPIC if recency_days else _GENERAL_TOPIC,
        }
        if recency_days:
            payload["days"] = recency_days
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        body = self._post(payload)
        results = body.get("results")
        if not isinstance(results, list):
            raise WebSearchError(self.name, "response contained no results array")
        return [item for item in (self._normalise(raw) for raw in results) if item is not None]

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        timeout = self._settings.web_search_timeout_seconds
        try:
            if self._client is not None:
                response = self._client.post(
                    TAVILY_URL, json=payload, headers=headers, timeout=timeout
                )
            else:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(TAVILY_URL, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise WebSearchError(self.name, f"request timed out after {timeout}s") from exc
        except httpx.HTTPError as exc:
            raise WebSearchError(self.name, f"request failed: {exc}") from exc

        self._raise_for_status(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise WebSearchError(self.name, "response was not valid JSON", retryable=False) from exc
        if not isinstance(body, dict):
            raise WebSearchError(self.name, "response was not a JSON object", retryable=False)
        return body

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        # 401/403 are configuration problems and 4xx generally will not fix
        # itself, so only rate limiting and server faults are worth retrying.
        if response.status_code in (401, 403):
            raise WebSearchNotConfiguredError("the Tavily API key was rejected")
        retryable = response.status_code == 429 or response.status_code >= 500
        raise WebSearchError(
            self.name,
            f"provider returned HTTP {response.status_code}",
            retryable=retryable,
        )

    def _normalise(self, raw: object) -> WebResult | None:
        """Vendor shape to :class:`WebResult`; unusable entries are dropped."""
        if not isinstance(raw, dict):
            return None
        url = raw.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return None
        content = raw.get("content")
        score = raw.get("score")
        return WebResult(
            url=url,
            title=str(raw.get("title") or url),
            domain=domain_of(url),
            text=content if isinstance(content, str) else "",
            retrieved_at=utc_now(),
            provider=self.name,
            published_at=parse_timestamp(raw.get("published_date")),
            provider_score=float(score) if isinstance(score, int | float) else None,
        )
