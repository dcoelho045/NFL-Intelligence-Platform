"""Shared pytest fixtures.

The default suite is fully offline. Structured data comes from the committed
Parquet fixtures in ``tests/fixtures/nflverse``; web search is driven by
recorded payloads in ``tests/fixtures/web``; and the knowledge index is built at
session scope from the miniature corpus in ``tests/fixtures/knowledge`` using a
deterministic hash embedder. No API key, no network, and no test that can fail
because a third party changed something.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.data.ingestion import NflDataLoader, ParquetDataSource
from nfl_platform.retrieval.knowledge.chunking import chunk_document
from nfl_platform.retrieval.knowledge.corpus import corpus_fingerprint, load_corpus
from nfl_platform.retrieval.knowledge.embeddings import FakeEmbedder
from nfl_platform.retrieval.knowledge.retriever import KnowledgeRetriever
from nfl_platform.retrieval.knowledge.store import IndexManifest, VectorStore
from nfl_platform.retrieval.web.cache import SearchCache
from nfl_platform.retrieval.web.search import WebSearchClient
from nfl_platform.services import ServiceContainer
from support import (
    FIXTURE_SEASON,
    KNOWLEDGE_FIXTURE_DIR,
    CountingDataSource,
    StubWebSearchProvider,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "nflverse"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture
def settings() -> Settings:
    """Settings isolated from the developer's local .env file."""
    return Settings(
        _env_file=None,
        default_season=FIXTURE_SEASON,
        embedding_provider="fake",
        knowledge_dir=KNOWLEDGE_FIXTURE_DIR,
        # The hash embedder's scores sit in a different range from a real model's,
        # so the production floor would reject everything. The floor is a property
        # of the embedding model, not of the retriever.
        rag_min_score=0.15,
    )


@pytest.fixture
def fixture_source() -> ParquetDataSource:
    return ParquetDataSource(FIXTURE_DIR)


@pytest.fixture
def loader(fixture_source: ParquetDataSource, settings: Settings) -> NflDataLoader:
    """Offline loader pinned to the fixture season."""
    return NflDataLoader(
        source=fixture_source,
        settings=settings,
        current_season=FIXTURE_SEASON,
    )


@pytest.fixture(scope="session")
def knowledge_index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the test index once per session with the deterministic embedder."""
    documents = load_corpus(KNOWLEDGE_FIXTURE_DIR)
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    embedder = FakeEmbedder()
    result = embedder.embed([chunk.text for chunk in chunks])

    path = tmp_path_factory.mktemp("rag") / "index.parquet"
    VectorStore(path).write(
        chunks,
        result.vectors,
        IndexManifest.build(
            model=embedder.model,
            dimensions=embedder.dimensions,
            chunks=len(chunks),
            documents=len(documents),
            corpus_fingerprint=corpus_fingerprint(documents),
            embedding_tokens=result.total_tokens,
        ),
    )
    return path


@pytest.fixture
def web_provider() -> StubWebSearchProvider:
    """Default stub returning the recorded injury-status payload."""
    return StubWebSearchProvider(fixture="injury_status")


@pytest.fixture
def web_client(web_provider: StubWebSearchProvider, settings: Settings) -> WebSearchClient:
    # A fresh cache per test; a shared one would leak results between cases.
    client = WebSearchClient(
        provider=web_provider,
        settings=settings,
        cache=SearchCache(
            ttl_seconds=settings.web_search_cache_ttl_seconds,
            max_entries=settings.web_search_cache_max_entries,
        ),
    )
    client.begin_run()
    return client


@pytest.fixture
def knowledge_retriever(knowledge_index_path: Path, settings: Settings) -> KnowledgeRetriever:
    return KnowledgeRetriever(
        store=VectorStore(knowledge_index_path),
        embedder=FakeEmbedder(),
        settings=settings,
    )


@pytest.fixture
def services(
    loader: NflDataLoader,
    settings: Settings,
    web_client: WebSearchClient,
    knowledge_retriever: KnowledgeRetriever,
) -> ServiceContainer:
    """Offline container sharing the fixture-backed loader and the retrieval doubles."""
    return ServiceContainer(
        loader=loader,
        settings=settings,
        web_client=web_client,
        knowledge_retriever=knowledge_retriever,
    )


@pytest.fixture
def counting_source(fixture_source: ParquetDataSource) -> CountingDataSource:
    return CountingDataSource(fixture_source)
