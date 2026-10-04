"""DTOs for retrieved evidence.

Provenance is part of the contract rather than something the presentation layer
adds later. A retrieved claim whose origin has been lost cannot be
distinguished from an invented one, so origin travels with the content all the
way to the caller.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from nfl_platform.schemas.common import BaseDTO

SourceKindName = Literal["structured", "web", "knowledge_base", "model"]


class SourceKind(StrEnum):
    """Where a piece of evidence came from.

    Defined in the schema layer because the retrieval layer, the prediction
    layer and the LLM layer all need it, and they are not allowed to import each
    other. The grounding check treats these differently: STRUCTURED and MODEL
    values are computed by this system and may ground a number, while WEB and
    KNOWLEDGE_BASE are retrieved text and may not.

    MODEL is kept separate from STRUCTURED even though both ground numbers,
    because the distinction is the point of the prediction layer: a measured
    statistic and an estimate of an unplayed game must never be presented as the
    same kind of claim.
    """

    STRUCTURED = "structured"
    WEB = "web"
    KNOWLEDGE_BASE = "knowledge_base"
    MODEL = "model"


#: Channels this system computes itself, and may therefore cite as the origin of
#: a number. Retrieved text is excluded: it is quoted, never relied on for arithmetic.
GROUNDING_KINDS = frozenset({SourceKind.STRUCTURED, SourceKind.MODEL})

#: Channels that return third-party content with its own provenance fields.
RETRIEVAL_KINDS = frozenset({SourceKind.WEB, SourceKind.KNOWLEDGE_BASE})


class SourceRef(BaseDTO):
    """One source that contributed to an answer.

    Built from the run trace, never written by the model, so the assistant
    cannot cite something it did not retrieve.
    """

    kind: SourceKindName = Field(description="Which evidence channel produced this.")
    label: str = Field(description="Human-readable identifier: a title, or a tool name.")
    url: str | None = Field(default=None, description="Source URL, for web evidence.")
    domain: str | None = Field(default=None, description="Publisher domain, for web evidence.")
    document_id: str | None = Field(
        default=None, description="Corpus document id, for knowledge-base evidence."
    )
    published_at: str | None = Field(default=None, description="Publication date, if known.")
    updated: str | None = Field(
        default=None, description="Last-reviewed date, for knowledge-base evidence."
    )
    retrieved_at: str | None = Field(default=None, description="When this was fetched.")


class WebEvidenceItem(BaseDTO):
    """One web result, sanitised and bounded."""

    url: str
    title: str
    domain: str
    snippet: str = Field(description="Sanitised, truncated extract. Untrusted external content.")
    retrieved_at: str
    published_at: str | None = None
    relevance: float | None = Field(
        default=None, description="Provider's own relevance score, where available."
    )


class WebEvidence(BaseDTO):
    """The result of one web search."""

    query: str
    provider: str
    results: list[WebEvidenceItem] = Field(default_factory=list)
    recency_days: int | None = None
    result_count: int = Field(ge=0)
    dropped_count: int = Field(
        default=0,
        ge=0,
        description="Results removed by domain policy or injection filtering.",
    )
    cache_hit: bool = False
    notes: list[str] = Field(default_factory=list)


class KnowledgeEvidenceItem(BaseDTO):
    """One retrieved chunk of the curated corpus."""

    document_id: str
    chunk_id: str
    title: str
    section: str | None = None
    text: str
    document_type: str
    source: str
    updated: str | None = Field(
        default=None,
        description=(
            "When this content was last reviewed. Curated knowledge is stable, not current."
        ),
    )
    score: float = Field(description="Cosine similarity to the query, 0 to 1.")


class KnowledgeEvidence(BaseDTO):
    """The result of one knowledge-base search."""

    query: str
    results: list[KnowledgeEvidenceItem] = Field(default_factory=list)
    result_count: int = Field(ge=0)
    min_score: float = Field(description="Relevance floor applied to this search.")
    notes: list[str] = Field(default_factory=list)


class RetrievalStatus(BaseDTO):
    """Whether each retrieval capability is usable on this deployment."""

    web_search_available: bool
    web_search_provider: str
    knowledge_base_available: bool
    knowledge_base_documents: int | None = None
    knowledge_base_chunks: int | None = None
    embedding_model: str | None = None
    knowledge_base_built_at: str | None = None


class KnowledgeDocumentRef(BaseDTO):
    """One curated document, without its body.

    The listing carries no text so an application can decide what to open
    before paying the context cost of opening it.
    """

    document_id: str = Field(description="Stable identifier, also the resource address.")
    title: str
    document_type: str
    source: str
    updated: str | None = Field(default=None, description="When this content was last reviewed.")
    characters: int = Field(ge=0, description="Length of the document body.")


class KnowledgeDocument(KnowledgeDocumentRef):
    """One curated document with its full text."""

    text: str = Field(description="The document body, verbatim.")


class KnowledgeCorpusIndex(BaseDTO):
    """Every curated document available to read."""

    document_count: int = Field(ge=0)
    documents: list[KnowledgeDocumentRef] = Field(default_factory=list)
