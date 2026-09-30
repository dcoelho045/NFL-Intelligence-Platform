"""Retrieval services: DTO conversion, framing, and degraded behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.exceptions import (
    KnowledgeDocumentNotFoundError,
    KnowledgeIndexUnavailableError,
    WebSearchError,
    WebSearchNotConfiguredError,
)
from nfl_platform.retrieval.knowledge.embeddings import FakeEmbedder
from nfl_platform.retrieval.knowledge.retriever import KnowledgeRetriever
from nfl_platform.retrieval.knowledge.store import VectorStore
from nfl_platform.retrieval.web.cache import SearchCache
from nfl_platform.retrieval.web.provider import NullSearchProvider
from nfl_platform.retrieval.web.search import WebSearchClient
from nfl_platform.services import ServiceContainer
from nfl_platform.services.knowledge import KnowledgeService
from nfl_platform.services.research import WebResearchService
from support import StubWebSearchProvider, failing_web_provider


def research_service(provider, settings: Settings) -> WebResearchService:
    client = WebSearchClient(
        provider=provider,
        settings=settings,
        cache=SearchCache(ttl_seconds=600, max_entries=16),
    )
    client.begin_run()
    return WebResearchService(client=client, settings=settings)


class TestWebResearchService:
    def test_returns_a_typed_evidence_dto(self, services: ServiceContainer):
        evidence = services.research.search("Patrick Mahomes injury status")
        assert evidence.query == "Patrick Mahomes injury status"
        assert evidence.result_count == len(evidence.results)
        assert evidence.result_count > 0

    def test_every_result_carries_full_provenance(self, services: ServiceContainer):
        for item in services.research.search("injury status").results:
            assert item.url.startswith("https://")
            assert item.title
            assert item.domain
            assert item.retrieved_at

    def test_results_are_labelled_as_untrusted(self, services: ServiceContainer):
        evidence = services.research.search("injury status")
        assert any("untrusted" in note for note in evidence.notes)

    def test_no_results_tells_the_model_not_to_substitute_memory(self, settings: Settings):
        service = research_service(StubWebSearchProvider(fixture="no_results"), settings)
        evidence = service.search("a question with no answer online")
        assert evidence.result_count == 0
        assert any("Do not substitute remembered" in note for note in evidence.notes)

    def test_recency_is_echoed_back(self, services: ServiceContainer):
        assert services.research.search("injury", recency_days=7).recency_days == 7

    def test_cache_hits_are_reported(self, services: ServiceContainer):
        services.research.search("same query")
        assert services.research.search("same query").cache_hit is True

    def test_dropped_results_are_counted(self, settings: Settings):
        service = research_service(StubWebSearchProvider(fixture="injection"), settings)
        assert service.search("player injury status").dropped_count == 2

    def test_provider_failure_surfaces_as_a_domain_error(self, settings: Settings):
        service = research_service(failing_web_provider(), settings)
        with pytest.raises(WebSearchError):
            service.search("anything")

    def test_unconfigured_provider_reports_itself(self, settings: Settings):
        service = research_service(NullSearchProvider("no key"), settings)
        assert service.available is False
        with pytest.raises(WebSearchNotConfiguredError):
            service.search("anything")

    def test_beginning_a_run_is_safe_when_unconfigured(self, settings: Settings):
        """The agent calls this on every question, key or no key."""
        research_service(NullSearchProvider("no key"), settings).begin_run()


class TestKnowledgeService:
    def test_returns_a_typed_evidence_dto(self, services: ServiceContainer):
        evidence = services.knowledge.search("what does PPR mean", min_score=0.0)
        assert evidence.result_count == len(evidence.results)
        assert evidence.results

    def test_every_chunk_carries_its_document_metadata(self, services: ServiceContainer):
        for item in services.knowledge.search("PPR scoring", min_score=0.0).results:
            assert item.document_id
            assert item.chunk_id
            assert item.title
            assert item.document_type
            assert item.source
            assert item.updated
            assert 0.0 <= item.score <= 1.0

    def test_results_are_labelled_as_stable_not_current(self, services: ServiceContainer):
        evidence = services.knowledge.search("PPR scoring", min_score=0.0)
        assert any("not current information" in note for note in evidence.notes)

    def test_an_off_corpus_question_returns_nothing(self, services: ServiceContainer):
        evidence = services.knowledge.search("how many yards did Mahomes throw", min_score=0.95)
        assert evidence.result_count == 0
        assert any("does not cover" in note for note in evidence.notes)

    def test_the_applied_floor_is_reported(self, services: ServiceContainer):
        assert services.knowledge.search("PPR", min_score=0.4).min_score == pytest.approx(0.4)

    def test_a_missing_index_surfaces_as_a_domain_error(self, tmp_path: Path, settings: Settings):
        service = KnowledgeService(
            retriever=KnowledgeRetriever(
                store=VectorStore(tmp_path / "absent.parquet"),
                embedder=FakeEmbedder(),
                settings=settings,
            ),
            settings=settings,
        )
        assert service.available is False
        with pytest.raises(KnowledgeIndexUnavailableError):
            service.search("what does PPR mean")

    def test_status_reports_index_provenance(self, services: ServiceContainer):
        documents, chunks, model, built_at = services.knowledge.status()
        assert documents == 3
        assert chunks and chunks > 0
        assert model == FakeEmbedder().model
        assert built_at

    def test_status_is_none_when_there_is_no_index(self, tmp_path: Path, settings: Settings):
        service = KnowledgeService(
            retriever=KnowledgeRetriever(
                store=VectorStore(tmp_path / "absent.parquet"),
                embedder=FakeEmbedder(),
                settings=settings,
            ),
            settings=settings,
        )
        assert service.status() == (None, None, None, None)


class TestKnowledgeCorpusBrowsing:
    """Reading a document is not searching for one: no index, no embedding call."""

    def test_documents_are_listed_without_their_bodies(self, services: ServiceContainer):
        index = services.knowledge.documents()
        assert index.document_count == 3
        assert {document.document_id for document in index.documents} == {
            "test-scoring",
            "test-usage",
            "test-conventions",
        }
        for document in index.documents:
            assert document.characters > 0

    def test_a_document_is_returned_with_its_text_and_provenance(self, services: ServiceContainer):
        document = services.knowledge.document("test-scoring")
        assert document.title == "Test Scoring Reference"
        assert document.updated == "2024-01-01"
        assert "PPR means points per reception" in document.text
        assert document.characters == len(document.text)

    def test_an_unknown_id_names_the_ones_that_exist(self, services: ServiceContainer):
        with pytest.raises(KnowledgeDocumentNotFoundError) as caught:
            services.knowledge.document("no-such-document")
        assert "test-scoring" in caught.value.known

    @pytest.mark.parametrize(
        "document_id",
        ["../../../etc/passwd", "/etc/passwd", "test-scoring/../secrets", "C:/Windows/win.ini"],
    )
    def test_an_identifier_is_never_treated_as_a_path(
        self, services: ServiceContainer, document_id: str
    ):
        """Lookup is a dictionary hit against loaded documents, not a file open."""
        with pytest.raises(KnowledgeDocumentNotFoundError):
            services.knowledge.document(document_id)

    def test_browsing_works_without_a_built_index(self, tmp_path: Path, settings: Settings):
        """The corpus on disk is the source; the vector index is only for search."""
        service = KnowledgeService(
            retriever=KnowledgeRetriever(
                store=VectorStore(tmp_path / "absent.parquet"),
                embedder=FakeEmbedder(),
                settings=settings,
            ),
            settings=settings,
        )
        assert service.available is False
        assert service.documents().document_count == 3


class TestContainerWiring:
    def test_retrieval_services_are_exposed(self, services: ServiceContainer):
        assert services.research is not None
        assert services.knowledge is not None

    def test_services_are_cached(self, services: ServiceContainer):
        assert services.research is services.research
        assert services.knowledge is services.knowledge

    def test_injected_doubles_are_used_rather_than_built(self, services: ServiceContainer):
        """If the injection failed these would try to reach the network."""
        assert services.research.provider_name == "stub"
        assert services.knowledge.available is True

    def test_a_container_without_retrieval_still_builds(self, loader, settings: Settings):
        """Analytics must stay usable on a deployment with no retrieval configured."""
        container = ServiceContainer(
            loader=loader, settings=settings.model_copy(update={"web_search_provider": "none"})
        )
        assert container.research.available is False
        assert container.players is not None
