"""Search-adapter tests.

These exercise the real adapters against ``httpx.MockTransport``: the request
construction, status handling and normalisation all run, but nothing leaves the
process. That is the difference between testing the adapter and testing a mock
of it.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from nfl_platform.config.settings import Settings
from nfl_platform.exceptions import WebSearchError, WebSearchNotConfiguredError
from nfl_platform.retrieval.web.brave import BRAVE_URL, BraveSearchProvider
from nfl_platform.retrieval.web.provider import WebSearchProvider, domain_of, parse_timestamp
from nfl_platform.retrieval.web.tavily import TAVILY_URL, TavilySearchProvider

TAVILY_BODY = {
    "query": "mahomes injury",
    "results": [
        {
            "title": "Mahomes full participant",
            "url": "https://www.espn.com/nfl/story/_/id/1/mahomes",
            "content": "He practised fully on Wednesday.",
            "score": 0.93,
            "published_date": "2026-09-25T14:00:00Z",
        },
        {
            "title": "No URL here",
            "url": "not-a-url",
            "content": "should be discarded",
            "score": 0.1,
        },
    ],
}

BRAVE_BODY = {
    "web": {
        "results": [
            {
                "title": "Chiefs injury report",
                "url": "https://www.nfl.com/news/chiefs-report",
                "description": "Three starters were limited.",
                "page_age": "2026-09-25T14:00:00Z",
            }
        ]
    }
}


@pytest.fixture
def keyed_settings(settings: Settings) -> Settings:
    # model_copy skips validation, so the secret has to be constructed here.
    return settings.model_copy(update={"web_search_api_key": SecretStr("test-key")})


def transport_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestTavilyRequest:
    def test_posts_to_the_search_endpoint_with_a_bearer_token(self, keyed_settings: Settings):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("authorization")
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=TAVILY_BODY)

        with transport_client(handler) as client:
            TavilySearchProvider(keyed_settings, client=client).search(
                "mahomes injury", max_results=3
            )

        assert captured["url"] == TAVILY_URL
        assert captured["auth"] == "Bearer test-key"
        assert captured["body"]["query"] == "mahomes injury"
        assert captured["body"]["max_results"] == 3

    def test_does_not_pay_for_features_it_does_not_use(self, keyed_settings: Settings):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=TAVILY_BODY)

        with transport_client(handler) as client:
            TavilySearchProvider(keyed_settings, client=client).search("q", max_results=3)

        assert captured["body"]["include_answer"] is False
        assert captured["body"]["include_raw_content"] is False

    def test_recency_switches_to_the_news_topic(self, keyed_settings: Settings):
        """Tavily only honours `days` on the news topic, so the two travel together."""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=TAVILY_BODY)

        with transport_client(handler) as client:
            TavilySearchProvider(keyed_settings, client=client).search(
                "q", max_results=3, recency_days=7
            )

        assert captured["body"]["topic"] == "news"
        assert captured["body"]["days"] == 7

    def test_no_recency_uses_the_general_topic(self, keyed_settings: Settings):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=TAVILY_BODY)

        with transport_client(handler) as client:
            TavilySearchProvider(keyed_settings, client=client).search("q", max_results=3)

        assert captured["body"]["topic"] == "general"
        assert "days" not in captured["body"]

    def test_domain_filters_are_forwarded(self, keyed_settings: Settings):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=TAVILY_BODY)

        with transport_client(handler) as client:
            TavilySearchProvider(keyed_settings, client=client).search(
                "q", max_results=3, include_domains=["espn.com"], exclude_domains=["bad.test"]
            )

        assert captured["body"]["include_domains"] == ["espn.com"]
        assert captured["body"]["exclude_domains"] == ["bad.test"]


class TestTavilyNormalisation:
    def test_maps_the_vendor_shape_onto_web_result(self, keyed_settings: Settings):
        with transport_client(lambda _: httpx.Response(200, json=TAVILY_BODY)) as client:
            results = TavilySearchProvider(keyed_settings, client=client).search("q", max_results=5)

        assert len(results) == 1
        result = results[0]
        assert result.title == "Mahomes full participant"
        assert result.domain == "espn.com"
        assert result.text == "He practised fully on Wednesday."
        assert result.provider == "tavily"
        assert result.provider_score == pytest.approx(0.93)
        assert result.published_at is not None

    def test_entries_without_a_usable_url_are_discarded(self, keyed_settings: Settings):
        with transport_client(lambda _: httpx.Response(200, json=TAVILY_BODY)) as client:
            results = TavilySearchProvider(keyed_settings, client=client).search("q", max_results=5)
        assert all(item.url.startswith("https://") for item in results)

    def test_missing_content_becomes_empty_text_not_none(self, keyed_settings: Settings):
        body = {"results": [{"url": "https://a.test/x", "title": "T"}]}
        with transport_client(lambda _: httpx.Response(200, json=body)) as client:
            results = TavilySearchProvider(keyed_settings, client=client).search("q", max_results=5)
        assert results[0].text == ""


class TestTavilyFailures:
    def test_timeouts_become_a_retryable_domain_error(self, keyed_settings: Settings):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("too slow", request=request)

        with transport_client(handler) as client:
            provider = TavilySearchProvider(keyed_settings, client=client)
            with pytest.raises(WebSearchError) as excinfo:
                provider.search("q", max_results=3)
        assert excinfo.value.retryable is True

    def test_connection_errors_become_a_domain_error(self, keyed_settings: Settings):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with transport_client(handler) as client, pytest.raises(WebSearchError):
            TavilySearchProvider(keyed_settings, client=client).search("q", max_results=3)

    @pytest.mark.parametrize("status", [401, 403])
    def test_rejected_keys_are_a_configuration_problem(self, keyed_settings: Settings, status: int):
        with (
            transport_client(lambda _: httpx.Response(status, json={})) as client,
            pytest.raises(WebSearchNotConfiguredError),
        ):
            TavilySearchProvider(keyed_settings, client=client).search("q", max_results=3)

    def test_rate_limiting_is_retryable(self, keyed_settings: Settings):
        with transport_client(lambda _: httpx.Response(429, json={})) as client:
            provider = TavilySearchProvider(keyed_settings, client=client)
            with pytest.raises(WebSearchError) as excinfo:
                provider.search("q", max_results=3)
        assert excinfo.value.retryable is True

    def test_server_errors_are_retryable(self, keyed_settings: Settings):
        with transport_client(lambda _: httpx.Response(503, json={})) as client:
            provider = TavilySearchProvider(keyed_settings, client=client)
            with pytest.raises(WebSearchError) as excinfo:
                provider.search("q", max_results=3)
        assert excinfo.value.retryable is True

    def test_client_errors_are_not_retryable(self, keyed_settings: Settings):
        """Retrying a malformed request just spends the credit twice."""
        with transport_client(lambda _: httpx.Response(400, json={})) as client:
            provider = TavilySearchProvider(keyed_settings, client=client)
            with pytest.raises(WebSearchError) as excinfo:
                provider.search("q", max_results=3)
        assert excinfo.value.retryable is False

    def test_non_json_responses_fail_cleanly(self, keyed_settings: Settings):
        response = httpx.Response(200, text="<html>nope</html>")
        with (
            transport_client(lambda _: response) as client,
            pytest.raises(WebSearchError, match="not valid JSON"),
        ):
            TavilySearchProvider(keyed_settings, client=client).search("q", max_results=3)

    def test_a_missing_results_array_fails_cleanly(self, keyed_settings: Settings):
        with (
            transport_client(lambda _: httpx.Response(200, json={"query": "q"})) as client,
            pytest.raises(WebSearchError, match="no results array"),
        ):
            TavilySearchProvider(keyed_settings, client=client).search("q", max_results=3)

    def test_a_missing_key_is_reported_before_any_request(self, settings: Settings):
        provider = TavilySearchProvider(settings.model_copy(update={"web_search_api_key": None}))
        assert provider.available is False
        with pytest.raises(WebSearchNotConfiguredError):
            provider.search("q", max_results=3)


class TestBraveAdapter:
    def test_sends_the_subscription_token_header(self, keyed_settings: Settings):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["token"] = request.headers.get("x-subscription-token")
            captured["url"] = str(request.url)
            return httpx.Response(200, json=BRAVE_BODY)

        with transport_client(handler) as client:
            BraveSearchProvider(keyed_settings, client=client).search("q", max_results=3)

        assert captured["token"] == "test-key"
        assert str(captured["url"]).startswith(BRAVE_URL)

    def test_normalises_to_the_same_web_result_type(self, keyed_settings: Settings):
        with transport_client(lambda _: httpx.Response(200, json=BRAVE_BODY)) as client:
            results = BraveSearchProvider(keyed_settings, client=client).search("q", max_results=3)

        assert len(results) == 1
        assert results[0].domain == "nfl.com"
        assert results[0].provider == "brave"
        assert results[0].text == "Three starters were limited."

    @pytest.mark.parametrize(
        ("days", "expected"), [(1, "pd"), (7, "pw"), (30, "pm"), (200, "py"), (400, "py")]
    )
    def test_recency_maps_onto_braves_coarse_buckets(
        self, keyed_settings: Settings, days: int, expected: str
    ):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["freshness"] = request.url.params.get("freshness")
            return httpx.Response(200, json=BRAVE_BODY)

        with transport_client(handler) as client:
            BraveSearchProvider(keyed_settings, client=client).search(
                "q", max_results=3, recency_days=days
            )

        assert captured["freshness"] == expected

    def test_rejected_keys_are_a_configuration_problem(self, keyed_settings: Settings):
        with (
            transport_client(lambda _: httpx.Response(401, json={})) as client,
            pytest.raises(WebSearchNotConfiguredError),
        ):
            BraveSearchProvider(keyed_settings, client=client).search("q", max_results=3)

    def test_both_adapters_satisfy_the_protocol(self, keyed_settings: Settings):
        assert isinstance(TavilySearchProvider(keyed_settings), WebSearchProvider)
        assert isinstance(BraveSearchProvider(keyed_settings), WebSearchProvider)


class TestSharedHelpers:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.espn.com/nfl/x", "espn.com"),
            ("https://espn.com/nfl/x", "espn.com"),
            ("http://SUB.Example.CO.UK/a", "sub.example.co.uk"),
            ("not-a-url", ""),
        ],
    )
    def test_domain_extraction(self, url: str, expected: str):
        assert domain_of(url) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "2026-09-25T14:00:00Z",
            "2026-09-25T14:00:00+00:00",
            "2026-09-25",
            "Thu, 25 Sep 2026 14:00:00 +0000",
            # The shape Tavily actually returns; %z does not match "GMT".
            "Sat, 26 Sep 2026 23:00:00 GMT",
            "Thu, 25 Sep 2026 14:00:00",
            "25 Sep 2026",
        ],
    )
    def test_timestamp_parsing_accepts_the_common_vendor_shapes(self, value: str):
        parsed = parse_timestamp(value)
        assert parsed is not None
        assert parsed.tzinfo is not None

    @pytest.mark.parametrize("value", [None, "", "   ", "last Tuesday", 12345, "not a date"])
    def test_unparsable_timestamps_are_none_rather_than_guessed(self, value: object):
        """A wrong publication date is worse than an absent one: freshness is judged on it."""
        assert parse_timestamp(value) is None
