"""Environment-driven settings.

All configuration is read from the environment (optionally via a local ``.env``
file) so that no machine-specific paths or secrets are committed to source.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CacheMode = Literal["filesystem", "memory", "off"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LlmProvider = Literal["openai"]
WebSearchProviderName = Literal["tavily", "brave", "none"]
EmbeddingProviderName = Literal["openai", "fake"]


class Settings(BaseSettings):
    """Runtime configuration for the data foundation."""

    model_config = SettingsConfigDict(
        env_prefix="NFL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cache_dir: Path = Field(
        default=Path("data/cache"),
        description="Directory nflreadpy uses for its filesystem download cache.",
    )
    snapshot_dir: Path = Field(
        default=Path("data/snapshots"),
        description="Directory for reproducible local Parquet snapshots.",
    )
    cache_mode: CacheMode = Field(
        default="filesystem",
        description="nflreadpy cache strategy.",
    )
    cache_duration_seconds: int = Field(
        default=86_400,
        ge=0,
        description="How long nflreadpy may reuse a cached download.",
    )
    request_timeout_seconds: int = Field(
        default=60,
        gt=0,
        description="HTTP timeout for nflverse downloads.",
    )
    default_season: int | None = Field(
        default=None,
        description="Season used when a caller omits one. None resolves to the current season.",
    )
    log_level: LogLevel = Field(default="INFO")
    log_json: bool = Field(default=False)

    llm_provider: LlmProvider = Field(
        default="openai",
        description="LLM vendor. Only OpenAI is implemented.",
    )
    llm_model: str = Field(
        default="gpt-4.1",
        description="Chat model used by the analyst assistant.",
    )
    # Explicit aliases bypass the NFL_ prefix so the SDK's own OPENAI_API_KEY also works.
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("NFL_LLM_API_KEY", "OPENAI_API_KEY"),
        description="Provider API key. Never logged or written to a run trace.",
    )
    llm_temperature: float | None = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature. 0 keeps tool selection as repeatable as possible. "
            "Leave blank to omit it entirely, which reasoning models such as o4-mini "
            "and gpt-5-mini require."
        ),
    )
    llm_max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum model round-trips per question before the loop stops.",
    )
    llm_max_tool_calls: int = Field(
        default=12,
        ge=1,
        le=60,
        description="Maximum tool executions per question.",
    )
    llm_timeout_seconds: int = Field(
        default=60,
        gt=0,
        description="Per-request timeout for provider calls.",
    )

    # --- Web retrieval (Milestone 4) -------------------------------------
    web_search_provider: WebSearchProviderName = Field(
        default="tavily",
        description=(
            "Web-search vendor. 'none' disables web retrieval and makes the tool "
            "report itself unavailable instead of failing."
        ),
    )
    # Vendor-native names are accepted so an existing key works without renaming.
    web_search_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("NFL_WEB_SEARCH_API_KEY", "TAVILY_API_KEY", "BRAVE_API_KEY"),
        description="Web-search provider key. Never logged or written to a run trace.",
    )
    web_search_max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Results requested per search. Each result costs tokens downstream.",
    )
    web_search_max_calls_per_run: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            "Hard ceiling on paid searches per question, independent of what the model "
            "decides to do. 0 disables web retrieval for the run."
        ),
    )
    web_search_recency_days: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Default recency window. None lets the caller decide per search.",
    )
    web_search_max_chars_per_result: int = Field(
        default=1200,
        ge=200,
        le=8000,
        description=(
            "Characters of retrieved text kept per result. Bounds both cost and the "
            "amount of untrusted content that can enter a prompt."
        ),
    )
    web_search_timeout_seconds: int = Field(
        default=15,
        gt=0,
        description="HTTP timeout for one search request.",
    )
    web_search_cache_ttl_seconds: int = Field(
        default=900,
        ge=0,
        description="How long an identical query may be served from the in-process cache.",
    )
    web_search_cache_max_entries: int = Field(
        default=128,
        ge=0,
        description="Bound on cached searches. The cache is in-process and never written to disk.",
    )
    web_search_allowed_domains: list[str] = Field(
        default_factory=list,
        description=(
            "If non-empty, only these domains are kept. Applied locally as well as "
            "upstream, because trusting a vendor to enforce our policy is not enforcement."
        ),
    )
    web_search_blocked_domains: list[str] = Field(
        default_factory=list,
        description="Domains dropped from every result set.",
    )
    web_search_drop_flagged_results: bool = Field(
        default=True,
        description=(
            "Drop results whose text contains prompt-injection patterns. Set false to keep "
            "them flagged but visible, which is useful only for research."
        ),
    )

    # --- Local RAG (Milestone 4) -----------------------------------------
    embedding_provider: EmbeddingProviderName = Field(
        default="openai",
        description="Embedding vendor. 'fake' is a deterministic offline stand-in for tests.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model. Changing it invalidates the index.",
    )
    embedding_dimensions: int | None = Field(
        default=None,
        ge=64,
        le=3072,
        description=(
            "Truncated embedding width. None uses the model default (1536 for "
            "text-embedding-3-small). Lower values trade a little recall for storage."
        ),
    )
    embedding_batch_size: int = Field(
        default=64,
        ge=1,
        le=256,
        description="Documents embedded per provider request when building the index.",
    )
    knowledge_dir: Path = Field(
        default=Path("data/knowledge"),
        description="Directory of curated markdown documents for the local RAG corpus.",
    )
    knowledge_index_path: Path = Field(
        default=Path("data/rag/index.parquet"),
        description="Built vector index. Regenerate with scripts/build_knowledge_index.py.",
    )
    rag_top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Chunks returned per knowledge-base search.",
    )
    rag_min_score: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity below which a chunk is discarded. Stops an off-corpus "
            "question returning the least-irrelevant chunk as if it were an answer. "
            "Calibrated for text-embedding-3-small from the benchmark's golden set: "
            "off-corpus questions peak around 0.38, on-corpus ones bottom out around "
            "0.51. Recalibrate with scripts/retrieval_benchmark.py after changing model."
        ),
    )
    rag_chunk_tokens: int = Field(
        default=350,
        ge=50,
        le=2000,
        description="Approximate target chunk size, in tokens.",
    )
    rag_chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Approximate token overlap between adjacent chunks of one section.",
    )

    @field_validator("default_season", mode="before")
    @classmethod
    def _blank_season_is_none(cls, value: object) -> object:
        """Treat ``NFL_DEFAULT_SEASON=`` in a .env file as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("llm_api_key", "web_search_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("llm_temperature", mode="before")
    @classmethod
    def _blank_temperature_is_none(cls, value: object) -> object:
        """``NFL_LLM_TEMPERATURE=`` means "do not send temperature at all"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("embedding_dimensions", "web_search_recency_days", mode="before")
    @classmethod
    def _blank_int_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("web_search_allowed_domains", "web_search_blocked_domains", mode="before")
    @classmethod
    def _split_domain_list(cls, value: object) -> object:
        """Accept ``a.com,b.com`` from a .env file as well as a JSON list."""
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                return value
            return [part.strip().lower() for part in text.split(",") if part.strip()]
        return value

    def resolved_cache_dir(self) -> Path:
        return self.cache_dir.expanduser().resolve()

    def resolved_snapshot_dir(self) -> Path:
        return self.snapshot_dir.expanduser().resolve()

    def resolved_knowledge_dir(self) -> Path:
        return self.knowledge_dir.expanduser().resolve()

    def resolved_knowledge_index_path(self) -> Path:
        return self.knowledge_index_path.expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
