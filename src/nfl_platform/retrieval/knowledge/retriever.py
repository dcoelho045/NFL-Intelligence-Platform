"""Query-side entry point for the local knowledge base.

Embeds the question, scans the index, and applies the score floor. The floor is
the part that matters: without it an off-corpus question ("how many yards did
Mahomes throw for?") returns the least-irrelevant chunk, and a confident answer
built on it looks exactly like a correct one.
"""

from __future__ import annotations

import time
from pathlib import Path

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.retrieval.knowledge.embeddings import EmbeddingClient, build_embedding_client
from nfl_platform.retrieval.knowledge.store import VectorStore
from nfl_platform.retrieval.models import RetrievedChunk, SearchStats

logger = get_logger(__name__)


class KnowledgeRetriever:
    """Semantic search over the curated corpus."""

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: EmbeddingClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        # `store or default` would discard an injected store: VectorStore defines
        # __len__, so one that has not loaded yet is falsy.
        self._store = (
            store
            if store is not None
            else VectorStore(Path(self._settings.resolved_knowledge_index_path()))
        )
        self._embedder = embedder
        self._checked = False

    @property
    def embedder(self) -> EmbeddingClient:
        """Built on first use so no key is needed until a search actually runs."""
        if self._embedder is None:
            self._embedder = build_embedding_client(self._settings)
        return self._embedder

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def available(self) -> bool:
        return self._store.exists

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> tuple[list[RetrievedChunk], SearchStats]:
        """Return the chunks above the score floor, best first."""
        text = " ".join(query.split())
        if not text:
            raise ValueError("Search query must not be empty.")

        limit = top_k if top_k is not None else self._settings.rag_top_k
        limit = max(1, min(limit, self._settings.rag_top_k * 2))
        floor = min_score if min_score is not None else self._settings.rag_min_score

        started = time.perf_counter()
        embedder = self.embedder
        self._ensure_compatible(embedder)

        vector = embedder.embed_query(text)
        hits = self._store.search(vector, top_k=limit, min_score=floor)

        stats = SearchStats(
            provider=embedder.model,
            queries=1,
            results_returned=len(hits),
            embedding_tokens=max(1, len(text.split())),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        if not hits:
            stats.notes.append(
                f"No chunk scored above the {floor} relevance floor; "
                "the knowledge base does not cover this question."
            )
        return hits, stats

    def _ensure_compatible(self, embedder: EmbeddingClient) -> None:
        """Refuse to serve an index built with different embedding settings."""
        if self._checked:
            return
        self._store.load(expected_signature=f"{embedder.model}@{embedder.dimensions}d")
        self._checked = True

        manifest = self._store.manifest
        if manifest is not None:
            logger.info(
                "Knowledge index loaded",
                extra={
                    "chunks": manifest.chunks,
                    "documents": manifest.documents,
                    "model": manifest.model,
                },
            )
