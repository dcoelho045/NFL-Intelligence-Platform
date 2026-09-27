"""Construction helpers for the assistant.

Keeps provider selection in one place so the API router, the CLI and the
benchmark script all build the agent the same way.
"""

from __future__ import annotations

from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import LlmConfigurationError
from nfl_platform.llm.agent import NflAnalystAgent
from nfl_platform.llm.client import ChatClient
from nfl_platform.llm.prompts import DEFAULT_PROMPT_VERSION
from nfl_platform.llm.providers.openai_client import OpenAIChatClient
from nfl_platform.llm.registry import ToolRegistry
from nfl_platform.services.container import ServiceContainer


def build_chat_client(settings: Settings | None = None) -> ChatClient:
    """Build the configured provider client."""
    settings = settings or get_settings()
    if settings.llm_provider == "openai":
        return OpenAIChatClient(settings)
    raise LlmConfigurationError(f"unsupported provider {settings.llm_provider!r}.")


def build_agent(
    container: ServiceContainer | None = None,
    *,
    client: ChatClient | None = None,
    registry: ToolRegistry | None = None,
    settings: Settings | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> NflAnalystAgent:
    """Assemble an agent, creating a provider client only if one is not supplied."""
    settings = settings or get_settings()
    return NflAnalystAgent(
        client=client or build_chat_client(settings),
        container=container or ServiceContainer(settings=settings),
        registry=registry,
        settings=settings,
        prompt_version=prompt_version,
    )


def llm_is_configured(settings: Settings | None = None) -> bool:
    """Whether a provider key is present, without attempting a call."""
    settings = settings or get_settings()
    key = settings.llm_api_key
    return key is not None and bool(key.get_secret_value().strip())
