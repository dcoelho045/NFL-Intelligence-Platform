"""Local RAG over a small curated corpus.

Stable knowledge only: definitions, rules, and this project's own methodology.
It is explicitly not current -- that is what web retrieval is for -- and every
chunk carries its ``updated`` date so the assistant cannot present it as fresh.
"""

from __future__ import annotations

from nfl_platform.retrieval.knowledge.chunking import chunk_document
from nfl_platform.retrieval.knowledge.corpus import (
    KnowledgeDocument,
    load_corpus,
    load_document,
)
from nfl_platform.retrieval.knowledge.embeddings import (
    EmbeddingClient,
    EmbeddingResult,
    FakeEmbedder,
    build_embedding_client,
)
from nfl_platform.retrieval.knowledge.retriever import KnowledgeRetriever
from nfl_platform.retrieval.knowledge.store import IndexManifest, VectorStore

__all__ = [
    "EmbeddingClient",
    "EmbeddingResult",
    "FakeEmbedder",
    "IndexManifest",
    "KnowledgeDocument",
    "KnowledgeRetriever",
    "VectorStore",
    "build_embedding_client",
    "chunk_document",
    "load_corpus",
    "load_document",
]
