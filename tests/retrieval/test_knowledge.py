"""Embeddings, the vector store, and the retriever."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.exceptions import (
    KnowledgeIndexStaleError,
    KnowledgeIndexUnavailableError,
)
from nfl_platform.retrieval.knowledge.chunking import chunk_document
from nfl_platform.retrieval.knowledge.corpus import corpus_fingerprint, load_corpus
from nfl_platform.retrieval.knowledge.embeddings import (
    EmbeddingClient,
    FakeEmbedder,
    build_embedding_client,
    normalise,
)
from nfl_platform.retrieval.knowledge.retriever import KnowledgeRetriever
from nfl_platform.retrieval.knowledge.store import (
    IndexManifest,
    VectorStore,
    manifest_path_for,
)
from support import KNOWLEDGE_FIXTURE_DIR


def build_index(path: Path, embedder: EmbeddingClient | None = None) -> VectorStore:
    embedder = embedder or FakeEmbedder()
    documents = load_corpus(KNOWLEDGE_FIXTURE_DIR)
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    result = embedder.embed([chunk.text for chunk in chunks])
    store = VectorStore(path)
    store.write(
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
    return store


class TestNormalisation:
    def test_vectors_are_scaled_to_unit_length(self):
        vector = normalise([3.0, 4.0])
        assert math.isclose(math.sqrt(sum(value**2 for value in vector)), 1.0)

    def test_a_zero_vector_is_left_alone_rather_than_dividing_by_zero(self):
        assert normalise([0.0, 0.0]) == [0.0, 0.0]


class TestFakeEmbedder:
    def test_satisfies_the_protocol(self):
        assert isinstance(FakeEmbedder(), EmbeddingClient)

    def test_is_deterministic(self):
        embedder = FakeEmbedder()
        assert embedder.embed_query("target share") == embedder.embed_query("target share")

    def test_different_text_gives_different_vectors(self):
        embedder = FakeEmbedder()
        assert embedder.embed_query("target share") != embedder.embed_query("snap share")

    def test_shared_words_score_higher_than_unrelated_text(self):
        """Enough signal to test ranking order, which is all it claims to be."""
        embedder = FakeEmbedder()
        query = embedder.embed_query("target share percentage")
        related = embedder.embed_query("target share of pass attempts")
        unrelated = embedder.embed_query("overtime kickoff rules")
        assert sum(a * b for a, b in zip(query, related, strict=True)) > sum(
            a * b for a, b in zip(query, unrelated, strict=True)
        )

    def test_reports_its_width(self):
        assert FakeEmbedder(128).dimensions == 128

    def test_embedding_an_empty_batch_is_not_an_error(self):
        assert FakeEmbedder().embed([]).vectors == []

    def test_the_factory_honours_the_fake_provider(self, settings: Settings):
        assert isinstance(build_embedding_client(settings), FakeEmbedder)


class TestVectorStore:
    def test_round_trips_chunks_and_metadata(self, tmp_path: Path):
        build_index(tmp_path / "index.parquet")
        store = VectorStore(tmp_path / "index.parquet")
        store.load()
        assert len(store) > 0

    def test_writes_a_manifest_beside_the_index(self, tmp_path: Path):
        path = tmp_path / "index.parquet"
        build_index(path)
        manifest = json.loads(manifest_path_for(path).read_text(encoding="utf-8"))
        assert manifest["model"] == FakeEmbedder().model
        assert manifest["chunks"] > 0
        assert manifest["corpus_fingerprint"]

    def test_metadata_survives_the_round_trip(self, tmp_path: Path):
        store = build_index(tmp_path / "index.parquet")
        reloaded = VectorStore(tmp_path / "index.parquet")
        reloaded.load()
        hits = reloaded.search(FakeEmbedder().embed_query("PPR scoring"), top_k=1)
        chunk = hits[0].chunk
        assert chunk.document_id
        assert chunk.title
        assert chunk.document_type
        assert chunk.source
        assert chunk.updated
        assert store.manifest is not None

    def test_results_are_ordered_by_score(self, tmp_path: Path):
        store = build_index(tmp_path / "index.parquet")
        hits = store.search(FakeEmbedder().embed_query("target share"), top_k=5)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_bounds_the_result_count(self, tmp_path: Path):
        store = build_index(tmp_path / "index.parquet")
        assert len(store.search(FakeEmbedder().embed_query("scoring"), top_k=2)) <= 2

    def test_the_score_floor_removes_weak_matches(self, tmp_path: Path):
        store = build_index(tmp_path / "index.parquet")
        assert store.search(FakeEmbedder().embed_query("scoring"), top_k=5, min_score=1.1) == []

    def test_a_missing_index_reports_how_to_build_it(self, tmp_path: Path):
        with pytest.raises(KnowledgeIndexUnavailableError, match="build_knowledge_index"):
            VectorStore(tmp_path / "absent.parquet").load()

    def test_an_unreadable_index_is_reported_not_crashed(self, tmp_path: Path):
        path = tmp_path / "index.parquet"
        path.write_text("this is not parquet", encoding="utf-8")
        with pytest.raises(KnowledgeIndexUnavailableError, match="unreadable"):
            VectorStore(path).load()

    def test_refuses_to_write_an_empty_index(self, tmp_path: Path):
        manifest = IndexManifest.build(
            model="m", dimensions=4, chunks=0, documents=0, corpus_fingerprint="x"
        )
        with pytest.raises(ValueError, match="empty index"):
            VectorStore(tmp_path / "index.parquet").write([], [], manifest)

    def test_mismatched_chunk_and_vector_counts_are_rejected(self, tmp_path: Path):
        documents = load_corpus(KNOWLEDGE_FIXTURE_DIR)
        chunks = [chunk for document in documents for chunk in chunk_document(document)]
        manifest = IndexManifest.build(
            model="m", dimensions=4, chunks=1, documents=1, corpus_fingerprint="x"
        )
        with pytest.raises(ValueError, match="must match"):
            VectorStore(tmp_path / "index.parquet").write(chunks, [[0.1] * 4], manifest)

    def test_an_index_built_with_another_model_is_refused(self, tmp_path: Path):
        """Comparing vectors across models yields plausible nonsense, not an error."""
        path = tmp_path / "index.parquet"
        build_index(path)
        with pytest.raises(KnowledgeIndexStaleError):
            VectorStore(path).load(expected_signature="text-embedding-3-small@1536d")

    def test_a_query_of_the_wrong_width_is_refused(self, tmp_path: Path):
        store = build_index(tmp_path / "index.parquet")
        with pytest.raises(KnowledgeIndexStaleError):
            store.search([0.1, 0.2, 0.3], top_k=3)

    def test_a_missing_manifest_does_not_prevent_loading(self, tmp_path: Path):
        path = tmp_path / "index.parquet"
        build_index(path)
        manifest_path_for(path).unlink()
        store = VectorStore(path)
        store.load()
        assert len(store) > 0
        assert store.manifest is None

    def test_index_size_stays_small(self, tmp_path: Path):
        """The storage claim behind choosing Parquet over a database."""
        path = tmp_path / "index.parquet"
        store = build_index(path)
        bytes_per_chunk = path.stat().st_size / len(store)
        assert bytes_per_chunk < 8000


class TestRetriever:
    @pytest.fixture
    def retriever(self, tmp_path: Path, settings: Settings) -> KnowledgeRetriever:
        path = tmp_path / "index.parquet"
        build_index(path)
        return KnowledgeRetriever(
            store=VectorStore(path), embedder=FakeEmbedder(), settings=settings
        )

    def test_finds_the_document_that_answers_the_question(self, retriever: KnowledgeRetriever):
        hits, _ = retriever.search("what does PPR mean in fantasy scoring", min_score=0.0)
        assert hits[0].chunk.document_id == "test-scoring"

    def test_finds_the_usage_document_for_a_usage_question(self, retriever: KnowledgeRetriever):
        hits, _ = retriever.search("what is target share", min_score=0.0)
        assert hits[0].chunk.document_id == "test-usage"

    def test_reports_the_embedding_it_used(self, retriever: KnowledgeRetriever):
        _, stats = retriever.search("PPR scoring", min_score=0.0)
        assert stats.provider == FakeEmbedder().model
        assert stats.queries == 1

    def test_an_off_corpus_question_returns_nothing(self, retriever: KnowledgeRetriever):
        """The floor is what stops the least-irrelevant chunk looking like an answer."""
        hits, stats = retriever.search(
            "how many passing yards did Mahomes throw for in week three", min_score=0.9
        )
        assert hits == []
        assert any("does not cover" in note for note in stats.notes)

    def test_top_k_is_honoured(self, retriever: KnowledgeRetriever):
        hits, _ = retriever.search("scoring", top_k=1, min_score=0.0)
        assert len(hits) == 1

    def test_empty_queries_are_rejected(self, retriever: KnowledgeRetriever):
        with pytest.raises(ValueError, match="must not be empty"):
            retriever.search("  ")

    def test_availability_reflects_whether_the_index_exists(
        self, tmp_path: Path, settings: Settings
    ):
        absent = KnowledgeRetriever(
            store=VectorStore(tmp_path / "nope.parquet"),
            embedder=FakeEmbedder(),
            settings=settings,
        )
        assert absent.available is False

    def test_an_injected_store_is_not_silently_replaced(self, tmp_path: Path, settings: Settings):
        """VectorStore defines __len__, so `store or default` would discard an empty one."""
        path = tmp_path / "custom.parquet"
        build_index(path)
        retriever = KnowledgeRetriever(
            store=VectorStore(path), embedder=FakeEmbedder(), settings=settings
        )
        assert retriever.store.path == path
