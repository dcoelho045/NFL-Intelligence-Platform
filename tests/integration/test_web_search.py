"""Live web-retrieval checks.

Deselected by default: these spend provider credits and depend on the state of
the internet. Run them deliberately::

    pytest -m web

What they verify is the contract we cannot verify from fixtures -- that the real
provider still returns the shape the adapter expects, and that a current
question actually finds current information.
"""

from __future__ import annotations

import pytest

from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.retrieval.web.provider import NullSearchProvider, build_search_provider
from nfl_platform.retrieval.web.search import WebSearchClient

pytestmark = pytest.mark.web


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    settings = get_settings()
    provider = build_search_provider(settings)
    if isinstance(provider, NullSearchProvider):
        pytest.skip("No web-search provider is configured; set TAVILY_API_KEY to run these.")
    return settings


@pytest.fixture
def live_client(live_settings: Settings) -> WebSearchClient:
    client = WebSearchClient(settings=live_settings)
    client.begin_run()
    return client


class TestLiveWebSearch:
    def test_a_current_question_returns_results(self, live_client: WebSearchClient):
        results, stats = live_client.search("NFL injury report this week", recency_days=7)
        assert results
        assert stats.queries == 1

    def test_every_result_carries_provenance(self, live_client: WebSearchClient):
        results, _ = live_client.search("NFL news today", recency_days=3)
        for item in results:
            assert item.url.startswith("http")
            assert item.title
            assert item.domain
            assert item.retrieved_at.tzinfo is not None

    def test_results_are_bounded_by_configuration(
        self, live_settings: Settings, live_client: WebSearchClient
    ):
        results, _ = live_client.search("NFL news today")
        assert len(results) <= live_settings.web_search_max_results

    def test_text_is_truncated_to_the_configured_cap(
        self, live_settings: Settings, live_client: WebSearchClient
    ):
        results, _ = live_client.search("NFL news today")
        cap = live_settings.web_search_max_chars_per_result
        assert all(len(item.text) <= cap + 8 for item in results)

    def test_recency_filtering_returns_recent_material(self, live_client: WebSearchClient):
        from datetime import UTC, datetime, timedelta

        results, _ = live_client.search("NFL news", recency_days=7)
        dated = [item for item in results if item.published_at is not None]
        if not dated:
            pytest.skip("Provider returned no publication dates for this query.")
        cutoff = datetime.now(UTC) - timedelta(days=30)
        assert any(item.published_at >= cutoff for item in dated)

    def test_an_absurd_query_does_not_invent_results(self, live_client: WebSearchClient):
        """No-result behaviour is a contract, not an accident."""
        results, _ = live_client.search(
            "zzzqqq nonexistent nfl player xyzzy plugh quarterback 1823"
        )
        assert isinstance(results, list)

    def test_repeating_a_query_costs_nothing(self, live_client: WebSearchClient):
        live_client.search("NFL injury report")
        _, stats = live_client.search("NFL injury report")
        assert stats.cache_hits == 1
        assert stats.queries == 0
