"""The embedding contract, plus a deterministic offline stand-in.

No SDK is imported here. The OpenAI implementation lives in
``retrieval/providers/``, the second of the two enumerated vendor-adapter
packages, so the property that matters -- every file that talks to a vendor SDK
can be listed -- still holds, without retrieval having to import upward into
``llm/`` to borrow a client.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from nfl_platform.config.settings import Settings, get_settings

FAKE_MODEL = "fake-hash-embedding"
FAKE_DIMENSIONS = 256

# Ignored when building a fake vector. Without this, a short query and a long
# chunk share "the" and "of" and score as weakly related, which puts every
# similarity in the same narrow band and makes the relevance floor untestable.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)


@dataclass(frozen=True)
class EmbeddingResult:
    """Vectors plus what they cost, so index builds can be priced."""

    vectors: list[list[float]]
    model: str
    dimensions: int
    total_tokens: int = 0


@runtime_checkable
class EmbeddingClient(Protocol):
    """Turns text into vectors."""

    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed a batch of documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query."""
        ...


def normalise(vector: Sequence[float]) -> list[float]:
    """Scale to unit length so cosine similarity is a plain dot product."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return list(vector)
    return [value / magnitude for value in vector]


class FakeEmbedder:
    """Deterministic hash-based embeddings for tests.

    Not semantic, and not pretending to be: it exists so ingestion, storage,
    scoring and metadata handling can be tested exactly, offline, with no API key
    and no fixture that goes stale when a provider retrains a model.

    Token overlap does produce some signal -- vectors are built from word
    hashes -- so a query sharing words with a chunk scores higher than one that
    does not, which is enough to test ranking order.
    """

    def __init__(self, dimensions: int = FAKE_DIMENSIONS) -> None:
        self._dimensions = dimensions

    @property
    def model(self) -> str:
        return FAKE_MODEL

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[self._vector(text) for text in texts],
            model=self.model,
            dimensions=self._dimensions,
            total_tokens=sum(len(text.split()) for text in texts),
        )

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for raw in text.lower().split():
            word = raw.strip(".,;:!?()[]\"'—-")
            if not word or word in _STOPWORDS:
                continue
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimensions] += 1.0
        return normalise(vector)


def build_embedding_client(settings: Settings | None = None) -> EmbeddingClient:
    """Build the configured embedding client.

    Imported lazily so nothing pulls in the OpenAI SDK on a deployment that only
    ever uses the offline embedder.
    """
    settings = settings or get_settings()
    if settings.embedding_provider == "fake":
        return FakeEmbedder(settings.embedding_dimensions or FAKE_DIMENSIONS)

    from nfl_platform.retrieval.providers.openai_embeddings import OpenAIEmbeddingClient

    return OpenAIEmbeddingClient(settings)
