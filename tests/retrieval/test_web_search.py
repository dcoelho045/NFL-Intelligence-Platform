"""Search-policy tests: everything true regardless of which vendor answers."""

from __future__ import annotations

import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.exceptions import (
    WebSearchBudgetExceededError,
    WebSearchError,
    WebSearchNotConfiguredError,
)
from nfl_platform.retrieval.web.cache import CacheKey, SearchCache
from nfl_platform.retrieval.web.provider import NullSearchProvider, build_search_provider
from nfl_platform.retrieval.web.search import WebSearchClient
from support import StubWebSearchProvider, failing_web_provider


def make_client(provider: object, settings: Settings, **overrides: object) -> WebSearchClient:
    run_settings = settings.model_copy(update=overrides) if overrides else settings
    client = WebSearchClient(
        provider=provider,
        settings=run_settings,
        cache=SearchCache(
            ttl_seconds=run_settings.web_search_cache_ttl_seconds,
            max_entries=run_settings.web_search_cache_max_entries,
        ),
    )
    client.begin_run()
    return client


class TestBasicSearch:
    def test_returns_normalised_results_with_provenance(self, web_client: WebSearchClient):
        results, stats = web_client.search("Patrick Mahomes injury status")
        assert results
        first = results[0]
        assert first.url.startswith("https://")
        assert first.domain == "espn.com"
        assert first.title
        assert first.retrieved_at.tzinfo is not None
        assert stats.queries == 1

    def test_publication_dates_are_parsed_when_present(self, web_client: WebSearchClient):
        results, _ = web_client.search("injury status")
        assert all(item.published_at is not None for item in results)

    def test_empty_query_is_rejected(self, web_client: WebSearchClient):
        with pytest.raises(ValueError, match="must not be empty"):
            web_client.search("   ")

    def test_result_count_is_capped_by_configuration(
        self, web_provider: StubWebSearchProvider, settings: Settings
    ):
        client = make_client(web_provider, settings, web_search_max_results=2)
        results, _ = client.search("injury status")
        assert len(results) <= 2
        assert web_provider.calls[0]["max_results"] == 2

    def test_requesting_more_than_the_ceiling_is_clamped(
        self, web_provider: StubWebSearchProvider, settings: Settings
    ):
        client = make_client(web_provider, settings, web_search_max_results=2)
        client.search("injury status", max_results=10)
        assert web_provider.calls[0]["max_results"] == 2


class TestNoResultsAndFailures:
    def test_no_results_is_not_an_error(self, settings: Settings):
        client = make_client(StubWebSearchProvider(fixture="no_results"), settings)
        results, stats = client.search("did they sign a belgian punter")
        assert results == []
        assert stats.queries == 1

    def test_provider_failure_propagates_as_a_domain_error(self, settings: Settings):
        client = make_client(failing_web_provider(), settings)
        with pytest.raises(WebSearchError):
            client.search("anything at all")

    def test_unconfigured_provider_reports_itself_rather_than_failing(self, settings: Settings):
        client = make_client(NullSearchProvider("no key"), settings)
        assert client.available is False
        with pytest.raises(WebSearchNotConfiguredError):
            client.search("anything at all")

    def test_provider_marked_unavailable_is_treated_as_unconfigured(self, settings: Settings):
        client = make_client(StubWebSearchProvider(available=False), settings)
        with pytest.raises(WebSearchNotConfiguredError):
            client.search("anything at all")


class TestBudget:
    def test_searches_are_capped_per_run(
        self, web_provider: StubWebSearchProvider, settings: Settings
    ):
        client = make_client(web_provider, settings, web_search_max_calls_per_run=2)
        client.search("query one")
        client.search("query two")
        with pytest.raises(WebSearchBudgetExceededError):
            client.search("query three")
        assert web_provider.call_count == 2

    def test_beginning_a_run_resets_the_budget(
        self, web_provider: StubWebSearchProvider, settings: Settings
    ):
        client = make_client(web_provider, settings, web_search_max_calls_per_run=1)
        client.search("query one")
        client.begin_run()
        client.search("query two")
        assert web_provider.call_count == 2

    def test_a_zero_budget_disables_web_retrieval(
        self, web_provider: StubWebSearchProvider, settings: Settings
    ):
        client = make_client(web_provider, settings, web_search_max_calls_per_run=0)
        assert client.available is False
        with pytest.raises(WebSearchBudgetExceededError):
            client.search("query one")

    def test_cache_hits_do_not_spend_budget(
        self, web_provider: StubWebSearchProvider, settings: Settings
    ):
        client = make_client(web_provider, settings, web_search_max_calls_per_run=1)
        client.search("repeated query")
        client.search("repeated query")
        assert web_provider.call_count == 1


class TestCaching:
    def test_identical_queries_hit_the_cache(
        self, web_provider: StubWebSearchProvider, web_client: WebSearchClient
    ):
        web_client.search("same question")
        _, stats = web_client.search("same question")
        assert stats.cache_hits == 1
        assert web_provider.call_count == 1

    def test_query_normalisation_makes_whitespace_and_case_irrelevant(
        self, web_provider: StubWebSearchProvider, web_client: WebSearchClient
    ):
        web_client.search("Mahomes Injury")
        web_client.search("  mahomes   injury  ")
        assert web_provider.call_count == 1

    def test_different_recency_windows_are_different_searches(
        self, web_provider: StubWebSearchProvider, web_client: WebSearchClient
    ):
        web_client.search("same text", recency_days=7)
        web_client.search("same text", recency_days=30)
        assert web_provider.call_count == 2

    def test_disabled_cache_always_calls_the_provider(
        self, web_provider: StubWebSearchProvider, settings: Settings
    ):
        client = make_client(
            web_provider, settings, web_search_cache_ttl_seconds=0, web_search_max_calls_per_run=5
        )
        client.search("same text")
        client.search("same text")
        assert web_provider.call_count == 2


class TestCacheMechanics:
    def test_expired_entries_are_discarded(self):
        cache = SearchCache(ttl_seconds=0, max_entries=8)
        assert cache.enabled is False
        assert cache.get(CacheKey.build("q", 5, None, "stub")) is None

    def test_least_recently_used_entries_are_evicted(self):
        cache = SearchCache(ttl_seconds=600, max_entries=2)
        for index in range(3):
            cache.put(CacheKey.build(f"q{index}", 5, None, "stub"), [])
        assert len(cache) == 2
        assert cache.get(CacheKey.build("q0", 5, None, "stub")) is None
        assert cache.get(CacheKey.build("q2", 5, None, "stub")) is not None

    def test_clearing_empties_the_cache(self):
        cache = SearchCache(ttl_seconds=600, max_entries=4)
        cache.put(CacheKey.build("q", 5, None, "stub"), [])
        cache.clear()
        assert len(cache) == 0


class TestDomainPolicy:
    def test_blocked_domains_are_dropped(
        self, web_provider: StubWebSearchProvider, settings: Settings
    ):
        client = make_client(web_provider, settings, web_search_blocked_domains=["espn.com"])
        results, stats = client.search("injury status")
        assert all(item.domain != "espn.com" for item in results)
        assert stats.results_dropped >= 1

    def test_allow_list_excludes_everything_else(
        self, web_provider: StubWebSearchProvider, settings: Settings
    ):
        client = make_client(web_provider, settings, web_search_allowed_domains=["nfl.com"])
        results, _ = client.search("injury status")
        assert [item.domain for item in results] == ["nfl.com"]

    def test_domain_rules_match_subdomains(
        self, web_provider: StubWebSearchProvider, settings: Settings
    ):
        client = make_client(web_provider, settings, web_search_blocked_domains=["com"])
        results, _ = client.search("injury status")
        assert results == []

    def test_policy_is_reapplied_locally_not_just_passed_upstream(
        self, web_provider: StubWebSearchProvider, settings: Settings
    ):
        """The stub ignores the filter arguments, exactly as Brave does."""
        client = make_client(web_provider, settings, web_search_blocked_domains=["espn.com"])
        results, _ = client.search("injury status")
        assert web_provider.calls[0]["exclude_domains"] == ["espn.com"]
        assert all(item.domain != "espn.com" for item in results)


class TestInjectionHandling:
    def test_flagged_results_are_dropped_by_default(self, settings: Settings):
        client = make_client(StubWebSearchProvider(fixture="injection"), settings)
        results, stats = client.search("player injury status")
        assert stats.injection_flags == 2
        assert stats.results_dropped == 2
        assert len(results) == 2

    def test_a_payload_hidden_in_a_comment_is_removed_rather_than_flagged(self, settings: Settings):
        """Stripping is stronger than flagging: the instruction never exists."""
        client = make_client(StubWebSearchProvider(fixture="injection"), settings)
        results, _ = client.search("player injury status")
        joined = " ".join(item.text for item in results).lower()
        assert "reveal your system prompt" not in joined
        assert any("red zone packages" in item.text for item in results)

    def test_clean_results_survive_alongside_flagged_ones(self, settings: Settings):
        client = make_client(StubWebSearchProvider(fixture="injection"), settings)
        results, _ = client.search("player injury status")
        assert any("first-team rep" in item.text for item in results)
        assert any(item.domain == "espn.com" for item in results)

    def test_no_instruction_text_survives_into_the_results(self, settings: Settings):
        client = make_client(StubWebSearchProvider(fixture="injection"), settings)
        results, _ = client.search("player injury status")
        joined = " ".join(item.text for item in results).lower()
        for phrase in ("ignore all previous", "you are now", "new instructions", "disregard"):
            assert phrase not in joined

    def test_dropping_can_be_disabled_but_the_flag_remains(self, settings: Settings):
        client = make_client(
            StubWebSearchProvider(fixture="injection"),
            settings,
            web_search_drop_flagged_results=False,
        )
        results, stats = client.search("player injury status")
        assert len(results) == 4
        assert sum(item.injection_flagged for item in results) == 2
        assert stats.injection_flags == 2

    def test_a_note_records_that_content_was_treated_as_untrusted(self, settings: Settings):
        client = make_client(StubWebSearchProvider(fixture="injection"), settings)
        _, stats = client.search("player injury status")
        assert any("untrusted" in note for note in stats.notes)


class TestMessyResults:
    def test_markup_never_reaches_the_caller(self, settings: Settings):
        client = make_client(StubWebSearchProvider(fixture="messy"), settings)
        results, _ = client.search("nfl news")
        joined = " ".join(item.text for item in results)
        assert "<script" not in joined
        assert "<div" not in joined
        assert "tracker.example.test" not in joined

    def test_missing_publication_dates_are_none_not_guessed(self, settings: Settings):
        client = make_client(StubWebSearchProvider(fixture="messy"), settings)
        results, _ = client.search("nfl news")
        undated = [item for item in results if item.domain == "pff.com"]
        assert undated and undated[0].published_at is None

    def test_unparsable_dates_are_none_not_guessed(self, settings: Settings):
        client = make_client(StubWebSearchProvider(fixture="messy"), settings)
        results, _ = client.search("nfl news")
        unparsable = [item for item in results if item.domain == "theathletic.com"]
        assert unparsable and unparsable[0].published_at is None

    def test_entries_without_a_usable_url_never_load(self, settings: Settings):
        """Dropped in the fixture loader, exactly as a real adapter drops them."""
        client = make_client(StubWebSearchProvider(fixture="messy"), settings)
        results, _ = client.search("nfl news")
        assert all(item.url.startswith("http") for item in results)


class TestProviderSelection:
    def test_no_key_yields_the_null_provider(self, settings: Settings):
        provider = build_search_provider(settings.model_copy(update={"web_search_api_key": None}))
        assert isinstance(provider, NullSearchProvider)
        assert provider.available is False

    def test_provider_set_to_none_yields_the_null_provider(self, settings: Settings):
        provider = build_search_provider(
            settings.model_copy(update={"web_search_provider": "none"})
        )
        assert isinstance(provider, NullSearchProvider)

    def test_the_stub_satisfies_the_provider_protocol(self):
        from nfl_platform.retrieval.web.provider import WebSearchProvider

        assert isinstance(StubWebSearchProvider(), WebSearchProvider)

    def test_the_null_provider_satisfies_the_protocol(self):
        from nfl_platform.retrieval.web.provider import WebSearchProvider

        assert isinstance(NullSearchProvider(), WebSearchProvider)
