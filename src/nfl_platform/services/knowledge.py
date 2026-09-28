"""Knowledge-base service.

Wraps the local RAG retriever and converts hits into DTOs. Keeps the relevance
floor and the "this is stable, not current" framing in one place, so every
caller gets the same treatment.
"""

from __future__ import annotations

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.retrieval.knowledge.retriever import KnowledgeRetriever
from nfl_platform.schemas.evidence import KnowledgeEvidence, KnowledgeEvidenceItem

logger = get_logger(__name__)


class KnowledgeService:
    """Curated, stable domain knowledge."""

    def __init__(
        self,
        retriever: KnowledgeRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._retriever = retriever or KnowledgeRetriever(settings=self._settings)

    @property
    def available(self) -> bool:
        return self._retriever.available

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> KnowledgeEvidence:
        """Return the corpus chunks that answer this question, best first."""
        hits, stats = self._retriever.search(query, top_k=top_k, min_score=min_score)
        floor = min_score if min_score is not None else self._settings.rag_min_score

        notes = list(stats.notes)
        if hits:
            notes.append(
                "Knowledge-base content is stable reference material, not current "
                "information. Check the 'updated' date before implying it is recent."
            )

        return KnowledgeEvidence(
            query=query,
            results=[
                KnowledgeEvidenceItem(
                    document_id=hit.chunk.document_id,
                    chunk_id=hit.chunk.chunk_id,
                    title=hit.chunk.title,
                    section=hit.chunk.section or None,
                    text=hit.chunk.text,
                    document_type=hit.chunk.document_type,
                    source=hit.chunk.source,
                    updated=hit.chunk.updated,
                    score=round(hit.score, 4),
                )
                for hit in hits
            ],
            result_count=len(hits),
            min_score=floor,
            notes=notes,
        )

    def status(self) -> tuple[int | None, int | None, str | None, str | None]:
        """Index size and provenance for the readiness endpoint, or Nones if absent."""
        store = self._retriever.store
        if not store.exists:
            return None, None, None, None
        try:
            store.load()
        except Exception:
            logger.info("Knowledge index could not be loaded for status", exc_info=True)
            return None, None, None, None
        manifest = store.manifest
        if manifest is None:
            return None, len(store), None, None
        return manifest.documents, manifest.chunks, manifest.model, manifest.built_at
