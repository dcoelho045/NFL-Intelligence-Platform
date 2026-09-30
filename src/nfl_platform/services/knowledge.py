"""Knowledge-base service.

Wraps the local RAG retriever and converts hits into DTOs. Keeps the relevance
floor and the "this is stable, not current" framing in one place, so every
caller gets the same treatment.
"""

from __future__ import annotations

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import KnowledgeDocumentNotFoundError
from nfl_platform.retrieval.knowledge.corpus import KnowledgeDocument as CorpusDocument
from nfl_platform.retrieval.knowledge.corpus import load_corpus
from nfl_platform.retrieval.knowledge.retriever import KnowledgeRetriever
from nfl_platform.schemas.evidence import (
    KnowledgeCorpusIndex,
    KnowledgeDocument,
    KnowledgeDocumentRef,
    KnowledgeEvidence,
    KnowledgeEvidenceItem,
)

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

    # --- corpus browsing --------------------------------------------------
    #
    # Reading a whole document is a different operation from searching for a
    # passage: no embedding call, no relevance floor, and no index required.
    # It reads the curated source of truth on disk, which is what lets an MCP
    # client attach a definition verbatim instead of paraphrasing a chunk.

    def documents(self) -> KnowledgeCorpusIndex:
        """List every curated document, without bodies."""
        refs = [self._to_ref(document) for document in self._load_corpus()]
        return KnowledgeCorpusIndex(document_count=len(refs), documents=refs)

    def document(self, document_id: str) -> KnowledgeDocument:
        """Read one curated document in full.

        Resolution is by identifier against the loaded corpus. No part of the
        caller's input is ever joined to a filesystem path, so an id shaped like
        a traversal is simply an id that does not exist.
        """
        documents = self._load_corpus()
        for document in documents:
            if document.document_id == document_id:
                ref = self._to_ref(document)
                return KnowledgeDocument(**ref.model_dump(), text=document.text)
        raise KnowledgeDocumentNotFoundError(
            document_id, tuple(sorted(item.document_id for item in documents))
        )

    def _load_corpus(self) -> list[CorpusDocument]:
        return load_corpus(self._settings.resolved_knowledge_dir())

    @staticmethod
    def _to_ref(document: CorpusDocument) -> KnowledgeDocumentRef:
        return KnowledgeDocumentRef(
            document_id=document.document_id,
            title=document.title,
            document_type=document.document_type,
            source=document.source,
            updated=document.updated,
            characters=len(document.text),
        )
