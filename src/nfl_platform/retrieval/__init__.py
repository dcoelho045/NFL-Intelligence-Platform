"""Retrieval layer: web search and a small local knowledge base.

Sits beside :mod:`nfl_platform.analytics` rather than inside ``llm/``. Both are
capabilities the service layer composes; neither knows a model exists. That
placement is what keeps the tool layer free of storage concerns and keeps
retrieval usable without an LLM.

Two things live here and are deliberately kept apart:

*Web retrieval* is the primary source for anything current -- injuries, news,
transactions. It is untrusted external input and is treated as such.

*Local RAG* is a small curated corpus of stable knowledge -- definitions, rules,
this project's own methodology. It is trusted, versioned, and explicitly not
current.

Neither replaces the deterministic analytics layer, which remains the only
source of NFL statistics.
"""

from __future__ import annotations

from nfl_platform.retrieval.models import (
    KnowledgeChunk,
    RetrievedChunk,
    SourceKind,
    WebResult,
)

__all__ = [
    "KnowledgeChunk",
    "RetrievedChunk",
    "SourceKind",
    "WebResult",
]
