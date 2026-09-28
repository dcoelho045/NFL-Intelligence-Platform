"""OpenAI embeddings adapter.

``text-embedding-3-small`` at $0.02 per 1M tokens makes a full rebuild of this
corpus cost roughly a tenth of a cent, and it supports truncated dimensions, so
storage can be cut threefold later without changing models.

The SDK is imported here and in ``llm/providers/`` only; an architecture test
enforces that.
"""

from __future__ import annotations

from collections.abc import Sequence

import openai
from openai import OpenAI

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import EmbeddingError, LlmConfigurationError
from nfl_platform.retrieval.knowledge.embeddings import EmbeddingResult, normalise

logger = get_logger(__name__)

# Native widths, so a configured dimension can be validated before spending a call.
MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
DEFAULT_DIMENSIONS = 1536

# USD per 1M tokens. Reporting estimates, not billing.
MODEL_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def estimate_embedding_cost_usd(model: str, tokens: int) -> float | None:
    """Approximate spend for an embedding run, or None for an unpriced model."""
    rate = MODEL_PRICING.get(model)
    return None if rate is None else tokens * rate / 1_000_000


class OpenAIEmbeddingClient:
    """An :class:`EmbeddingClient` backed by the OpenAI embeddings endpoint."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: OpenAI | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model = self._settings.embedding_model
        native = MODEL_DIMENSIONS.get(self._model, DEFAULT_DIMENSIONS)
        configured = self._settings.embedding_dimensions
        if configured is not None and configured > native:
            raise LlmConfigurationError(
                f"NFL_EMBEDDING_DIMENSIONS={configured} exceeds the {native} dimensions "
                f"{self._model!r} produces."
            )
        self._dimensions = configured or native
        self._truncated = configured is not None and configured < native

        if client is not None:
            self._client = client
            return
        api_key = self._settings.llm_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise LlmConfigurationError(
                "no API key found for embeddings. Set NFL_LLM_API_KEY or OPENAI_API_KEY."
            )
        self._client = OpenAI(
            api_key=api_key.get_secret_value(),
            timeout=float(self._settings.llm_timeout_seconds),
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], model=self._model, dimensions=self._dimensions)

        vectors: list[list[float]] = []
        total_tokens = 0
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            payload, tokens = self._request(batch)
            vectors.extend(payload)
            total_tokens += tokens

        return EmbeddingResult(
            vectors=vectors,
            model=self._model,
            dimensions=self._dimensions,
            total_tokens=total_tokens,
        )

    def embed_query(self, text: str) -> list[float]:
        vectors, _ = self._request([text])
        return vectors[0]

    def _request(self, batch: list[str]) -> tuple[list[list[float]], int]:
        kwargs: dict[str, object] = {"model": self._model, "input": batch}
        if self._truncated:
            kwargs["dimensions"] = self._dimensions
        try:
            response = self._client.embeddings.create(**kwargs)  # type: ignore[arg-type]
        except openai.AuthenticationError as exc:
            raise LlmConfigurationError(f"the embeddings API key was rejected: {exc}") from exc
        except openai.OpenAIError as exc:
            raise EmbeddingError("openai", str(exc)) from exc

        # Normalised on arrival so the store only ever holds unit vectors and
        # cosine similarity is a dot product at query time.
        vectors = [normalise(item.embedding) for item in response.data]
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return vectors, tokens
