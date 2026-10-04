"""DTOs for the MCP adapter's resources.

They live in the schema layer with every other DTO so the contract an MCP client
reads is defined in the same place as the contract an HTTP client reads.
"""

from __future__ import annotations

from pydantic import Field

from nfl_platform.schemas.common import BaseDTO


class ServerStatus(BaseDTO):
    """Which capabilities are usable on this deployment.

    Read this before deciding what to offer a user: a missing knowledge index or
    a switched-off web search is a configuration fact, never a finding about the
    world.
    """

    server_name: str
    version: str
    data_layer_available: bool = Field(
        description="Whether the deterministic nflverse layer answered."
    )
    current_season: int | None = None
    web_search_enabled: bool = Field(
        description="Whether this server is permitted to spend web-search credits."
    )
    web_search_available: bool = Field(
        description="Whether web search is both enabled and configured with a provider key."
    )
    web_search_provider: str
    knowledge_base_available: bool
    knowledge_base_documents: int | None = None
    knowledge_base_chunks: int | None = None
    embedding_model: str | None = None
    knowledge_base_built_at: str | None = None
    predictions_enabled: bool = Field(
        default=False,
        description="Whether the prediction tools may run on this deployment.",
    )
    prediction_models_available: bool = Field(
        default=False,
        description=(
            "Whether at least one trained model artifact is installed and loadable. "
            "False means predictions cannot be produced; it says nothing about any game."
        ),
    )
    prediction_models: list[str] = Field(
        default_factory=list,
        description="Installed model identifiers, as '<problem>/<version>'.",
    )


class ReferenceData(BaseDTO):
    """Valid teams and populated seasons, so a caller need not guess."""

    current_season: int = Field(description="The season 'this season' resolves to.")
    seasons: list[int] = Field(default_factory=list, description="Seasons in scope.")
    teams: list[str] | None = Field(
        default=None, description="Team abbreviations, when a season was specified."
    )
