"""The two retrieval tools, and how the agent routes between evidence sources.

The model is scripted throughout: these assert what the tools do and what the
agent records, not what a particular model happens to choose. Routing quality
with a real model is measured by the benchmark, not here.
"""

from __future__ import annotations

from pathlib import Path

from nfl_platform.config.settings import Settings
from nfl_platform.llm.registry import ToolRegistry, build_default_registry
from nfl_platform.retrieval.knowledge.embeddings import FakeEmbedder
from nfl_platform.retrieval.knowledge.retriever import KnowledgeRetriever
from nfl_platform.retrieval.knowledge.store import VectorStore
from nfl_platform.retrieval.web.cache import SearchCache
from nfl_platform.retrieval.web.provider import NullSearchProvider
from nfl_platform.retrieval.web.search import WebSearchClient
from nfl_platform.schemas.evidence import RETRIEVAL_KINDS, SourceKind
from nfl_platform.services import ServiceContainer
from support import MAHOMES_ID, StubWebSearchProvider, failing_web_provider


def container_with(loader, settings: Settings, provider=None, retriever=None) -> ServiceContainer:
    client = None
    if provider is not None:
        client = WebSearchClient(
            provider=provider,
            settings=settings,
            cache=SearchCache(ttl_seconds=600, max_entries=16),
        )
        client.begin_run()
    return ServiceContainer(
        loader=loader, settings=settings, web_client=client, knowledge_retriever=retriever
    )


class TestToolSchemas:
    def test_both_tools_are_registered(self, registry: ToolRegistry):
        assert "search_web" in registry
        assert "search_knowledge_base" in registry

    def test_web_tool_declares_itself_as_web_evidence(self, registry: ToolRegistry):
        assert registry.source_kind("search_web") is SourceKind.WEB

    def test_knowledge_tool_declares_itself_as_knowledge_base_evidence(
        self, registry: ToolRegistry
    ):
        assert registry.source_kind("search_knowledge_base") is SourceKind.KNOWLEDGE_BASE

    def test_analytics_tools_remain_structured(self, registry: ToolRegistry):
        assert registry.source_kind("get_player_stats") is SourceKind.STRUCTURED

    def test_web_description_steers_statistics_away_from_the_web(self, registry: ToolRegistry):
        description = registry.get("search_web").description.lower()
        assert "current" in description
        assert "do not use this for statistics" in description

    def test_knowledge_description_says_it_holds_no_statistics(self, registry: ToolRegistry):
        description = registry.get("search_knowledge_base").description.lower()
        assert "no statistics" in description
        assert "not current" in description

    def test_there_is_no_tool_that_fetches_an_arbitrary_url(self, registry: ToolRegistry):
        """Removing this capability removes the inject-a-link escalation path."""
        for name in registry.names:
            schema = registry.get(name).json_schema()
            properties = schema["function"]["parameters"].get("properties", {})
            assert "url" not in properties, f"{name} accepts a caller-supplied URL"


class TestWebTool:
    def test_returns_results_with_provenance(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute(
            "search_web", {"query": "Patrick Mahomes injury status"}, services
        )
        assert result.ok
        assert result.source_kind is SourceKind.WEB
        assert result.data["results"][0]["url"].startswith("https://")

    def test_recency_is_forwarded_to_the_provider(
        self,
        web_provider: StubWebSearchProvider,
        registry: ToolRegistry,
        services: ServiceContainer,
    ):
        registry.execute("search_web", {"query": "injury status", "recency_days": 3}, services)
        assert web_provider.calls[0]["recency_days"] == 3

    def test_a_too_short_query_is_rejected_before_any_request(
        self,
        web_provider: StubWebSearchProvider,
        registry: ToolRegistry,
        services: ServiceContainer,
    ):
        result = registry.execute("search_web", {"query": "a"}, services)
        assert not result.ok
        assert result.error_code == "invalid_arguments"
        assert web_provider.call_count == 0

    def test_an_unknown_argument_is_rejected(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute(
            "search_web", {"query": "injury status", "url": "https://evil.test"}, services
        )
        assert not result.ok
        assert result.error_code == "invalid_arguments"

    def test_no_results_is_a_success_with_an_empty_list(
        self, loader, settings: Settings, registry: ToolRegistry
    ):
        services = container_with(
            loader, settings, provider=StubWebSearchProvider(fixture="no_results")
        )
        result = registry.execute("search_web", {"query": "nothing to find here"}, services)
        assert result.ok
        assert result.data["result_count"] == 0

    def test_provider_failure_is_recoverable_and_forbids_invention(
        self, loader, settings: Settings, registry: ToolRegistry
    ):
        services = container_with(loader, settings, provider=failing_web_provider())
        result = registry.execute("search_web", {"query": "injury status"}, services)
        assert not result.ok
        assert result.error_code == "web_search_error"
        assert "do not invent" in result.error["recovery"].lower()

    def test_unconfigured_web_search_tells_the_model_not_to_retry(
        self, loader, settings: Settings, registry: ToolRegistry
    ):
        services = container_with(loader, settings, provider=NullSearchProvider("no key"))
        result = registry.execute("search_web", {"query": "injury status"}, services)
        assert not result.ok
        assert result.error_code == "web_search_not_configured"
        assert result.error["retryable"] is False

    def test_exceeding_the_budget_is_reported_as_recoverable_guidance(
        self, loader, settings: Settings, registry: ToolRegistry
    ):
        services = container_with(
            loader,
            settings.model_copy(update={"web_search_max_calls_per_run": 1}),
            provider=StubWebSearchProvider(fixture="injury_status"),
        )
        assert registry.execute("search_web", {"query": "first query"}, services).ok
        result = registry.execute("search_web", {"query": "second query"}, services)
        assert not result.ok
        assert result.error_code == "web_search_budget_exhausted"


class TestKnowledgeTool:
    def test_returns_chunks_with_metadata(self, registry: ToolRegistry, services: ServiceContainer):
        result = registry.execute(
            "search_knowledge_base", {"query": "what does PPR scoring mean"}, services
        )
        assert result.ok
        assert result.source_kind is SourceKind.KNOWLEDGE_BASE
        first = result.data["results"][0]
        assert first["document_id"]
        assert first["updated"]

    def test_top_k_is_honoured(self, registry: ToolRegistry, services: ServiceContainer):
        result = registry.execute(
            "search_knowledge_base", {"query": "scoring", "top_k": 1}, services
        )
        assert result.data["result_count"] <= 1

    def test_a_missing_index_tells_the_model_not_to_retry(
        self, loader, settings: Settings, registry: ToolRegistry, tmp_path: Path
    ):
        services = container_with(
            loader,
            settings,
            provider=StubWebSearchProvider(),
            retriever=KnowledgeRetriever(
                store=VectorStore(tmp_path / "absent.parquet"),
                embedder=FakeEmbedder(),
                settings=settings,
            ),
        )
        result = registry.execute("search_knowledge_base", {"query": "what is PPR"}, services)
        assert not result.ok
        assert result.error_code == "knowledge_base_unavailable"
        assert result.error["retryable"] is False

    def test_an_off_corpus_question_succeeds_with_nothing(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        """An empty result is data: the corpus does not cover this."""
        result = registry.execute(
            "search_knowledge_base",
            {"query": "how many touchdowns did he throw in week 3"},
            services,
        )
        assert result.ok
        assert result.data["result_count"] == 0


class TestAnalyticsStillOwnsStatistics:
    def test_structured_tools_are_unchanged(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute(
            "get_player_stats", {"player": MAHOMES_ID, "last_n_games": 3}, services
        )
        assert result.ok
        assert result.source_kind is SourceKind.STRUCTURED

    def test_the_analytics_tool_set_is_untouched_by_this_milestone(self):
        registry = build_default_registry()
        structured = [
            name for name in registry.names if registry.source_kind(name) is SourceKind.STRUCTURED
        ]
        assert len(structured) == 17

    def test_only_two_retrieval_tools_were_added(self):
        registry = build_default_registry()
        retrieval = [
            name for name in registry.names if registry.source_kind(name) in RETRIEVAL_KINDS
        ]
        assert sorted(retrieval) == ["search_knowledge_base", "search_web"]
