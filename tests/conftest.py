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
from nfl_platform.data.ingestion.datasets import Dataset
from nfl_platform.ml.features.fantasy import build_fantasy_features
from nfl_platform.ml.features.games import build_game_features
from nfl_platform.ml.registry.store import ModelRegistry
from nfl_platform.ml.training.pipeline import train_fantasy_models, train_game_models
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
from synthetic import SyntheticLeague, build_league

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


# --- prediction fixtures (Milestone 6) ---------------------------------------
# The committed nflverse fixture is two teams over six weeks, which is fine for
# analytics assertions and useless for a time-aware model needing ten complete
# seasons. These fixtures stand up a generated league instead, so the prediction
# surface is exercisable end to end with no network and no real training run.


@pytest.fixture(scope="session")
def synthetic_league() -> SyntheticLeague:
    return build_league()


@pytest.fixture(scope="session")
def synthetic_data_dir(
    synthetic_league: SyntheticLeague, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The synthetic league written out as nflverse-shaped Parquet files."""
    directory = tmp_path_factory.mktemp("synthetic-nflverse")
    for dataset, frame in (
        (Dataset.SCHEDULES, synthetic_league.schedules),
        (Dataset.TEAM_STATS, synthetic_league.team_stats),
        (Dataset.PLAYER_STATS, synthetic_league.player_stats),
        (Dataset.SNAP_COUNTS, synthetic_league.snap_counts),
        (Dataset.FF_OPPORTUNITY, synthetic_league.ff_opportunity),
        (Dataset.PLAYERS, synthetic_league.players),
    ):
        frame.write_parquet(directory / f"{dataset.value}.parquet")
    return directory


@pytest.fixture(scope="session")
def trained_model_dir(
    synthetic_league: SyntheticLeague, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Train both problems on the synthetic league once per session.

    Real training, not stubs: the artifacts under test are produced by the same
    pipeline production uses, so a change that breaks serialisation or promotion
    fails here rather than in a hand-written double that quietly drifted.
    """
    directory = tmp_path_factory.mktemp("models")
    settings = Settings(_env_file=None, model_dir=directory)
    registry = ModelRegistry(root=directory, settings=settings)

    game = build_game_features(synthetic_league.schedules, synthetic_league.team_stats)
    fantasy = build_fantasy_features(
        synthetic_league.player_stats,
        synthetic_league.schedules,
        team_stats=synthetic_league.team_stats,
        snap_counts=synthetic_league.snap_counts,
        ff_opportunity=synthetic_league.ff_opportunity,
        players=synthetic_league.players,
    )
    for outcome in (
        train_game_models(game, settings=settings, snapshot="synthetic"),
        train_fantasy_models(fantasy, settings=settings, snapshot="synthetic"),
    ):
        for artifact in outcome.artifacts:
            registry.save(
                artifact, make_default=artifact.metadata.version == outcome.default_version
            )
    return directory


@pytest.fixture
def prediction_settings(trained_model_dir: Path, synthetic_league: SyntheticLeague) -> Settings:
    return Settings(
        _env_file=None,
        model_dir=trained_model_dir,
        default_season=synthetic_league.seasons[-1],
        embedding_provider="fake",
        knowledge_dir=KNOWLEDGE_FIXTURE_DIR,
        ml_min_history_games=3,
    )


@pytest.fixture
def prediction_loader(
    synthetic_data_dir: Path, prediction_settings: Settings, synthetic_league: SyntheticLeague
) -> NflDataLoader:
    return NflDataLoader(
        source=ParquetDataSource(synthetic_data_dir),
        settings=prediction_settings,
        current_season=synthetic_league.seasons[-1],
    )


@pytest.fixture
def prediction_services(
    prediction_loader: NflDataLoader,
    prediction_settings: Settings,
    trained_model_dir: Path,
) -> ServiceContainer:
    """A container whose prediction service has real artifacts and real data."""
    return ServiceContainer(
        loader=prediction_loader,
        settings=prediction_settings,
        model_registry=ModelRegistry(root=trained_model_dir, settings=prediction_settings),
    )


@pytest.fixture
def empty_model_registry(tmp_path: Path, settings: Settings) -> ModelRegistry:
    """A registry pointing at a directory with no artifacts in it."""
    return ModelRegistry(root=tmp_path / "no-models", settings=settings)
