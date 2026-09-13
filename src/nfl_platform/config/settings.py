"""Environment-driven settings.

All configuration is read from the environment (optionally via a local ``.env``
file) so that no machine-specific paths or secrets are committed to source.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CacheMode = Literal["filesystem", "memory", "off"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


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

    @field_validator("default_season", mode="before")
    @classmethod
    def _blank_season_is_none(cls, value: object) -> object:
        """Treat ``NFL_DEFAULT_SEASON=`` in a .env file as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def resolved_cache_dir(self) -> Path:
        return self.cache_dir.expanduser().resolve()

    def resolved_snapshot_dir(self) -> Path:
        return self.snapshot_dir.expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
