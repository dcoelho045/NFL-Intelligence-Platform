"""Provider-neutral retrieval types.

Vendor payloads are normalised into these inside each adapter and never escape
it, which is what makes swapping a search provider a one-directory change.

Every type carries its provenance. A retrieved fact whose origin has been lost
is indistinguishable from an invented one, so origin is part of the value rather
than something attached later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nfl_platform.schemas.evidence import SourceKind

__all__ = [
    "KnowledgeChunk",
    "RetrievedChunk",
    "SearchStats",
    "SourceKind",
    "WebResult",
]


@dataclass(frozen=True)
class WebResult:
    """One normalised web-search hit.

    ``text`` is untrusted. It has been sanitised and truncated by the time it
    reaches here, but it originated outside this system and must never be
    treated as an instruction.
    """

    url: str
    title: str
    domain: str
    text: str
    retrieved_at: datetime
    provider: str
    published_at: datetime | None = None
    provider_score: float | None = None
    injection_flagged: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "text": self.text,
            "retrieved_at": self.retrieved_at.isoformat(),
            "provider": self.provider,
        }
        if self.published_at is not None:
            payload["published_at"] = self.published_at.isoformat()
        if self.provider_score is not None:
            payload["provider_score"] = round(self.provider_score, 4)
        if self.injection_flagged:
            payload["injection_flagged"] = True
        return payload


@dataclass(frozen=True)
class KnowledgeChunk:
    """One indexed slice of a curated document.

    ``updated`` is carried all the way to the model on purpose: curated
    knowledge is stable, not current, and presenting it as current is the main
    way a small local corpus misleads.
    """

    document_id: str
    chunk_id: str
    title: str
    section: str
    text: str
    document_type: str
    source: str
    updated: str | None = None
    chunk_index: int = 0
    content_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "document_type": self.document_type,
            "source": self.source,
        }
        if self.updated:
            payload["updated"] = self.updated
        return payload


@dataclass(frozen=True)
class RetrievedChunk:
    """A knowledge chunk with the similarity score that surfaced it."""

    chunk: KnowledgeChunk
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {**self.chunk.as_dict(), "score": round(self.score, 4)}


@dataclass
class SearchStats:
    """What one retrieval call cost, for the run trace.

    Web search is billed per query and embeddings per token, so neither shows up
    in the LLM token counters. Tracking them here is what lets Milestone 9
    compare retrieval spend against model spend.
    """

    provider: str = ""
    queries: int = 0
    cache_hits: int = 0
    results_returned: int = 0
    results_dropped: int = 0
    injection_flags: int = 0
    embedding_tokens: int = 0
    duration_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def merge(self, other: SearchStats) -> None:
        self.provider = other.provider or self.provider
        self.queries += other.queries
        self.cache_hits += other.cache_hits
        self.results_returned += other.results_returned
        self.results_dropped += other.results_dropped
        self.injection_flags += other.injection_flags
        self.embedding_tokens += other.embedding_tokens
        self.duration_ms += other.duration_ms
        self.notes.extend(other.notes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "queries": self.queries,
            "cache_hits": self.cache_hits,
            "results_returned": self.results_returned,
            "results_dropped": self.results_dropped,
            "injection_flags": self.injection_flags,
            "embedding_tokens": self.embedding_tokens,
            "duration_ms": round(self.duration_ms, 1),
        }
