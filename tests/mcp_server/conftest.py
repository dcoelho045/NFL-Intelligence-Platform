"""Fixtures for the MCP tests.

The server under test is the real one, built over the fixture-backed container
from ``tests/conftest.py``: real tools, real validation, real service calls
against committed Parquet, a deterministic hash embedder, and recorded search
payloads. Nothing here is mocked except the outside world.

``Client(server)`` connects in-process. It is the genuine protocol -- handshake,
``tools/list``, ``tools/call`` -- with no subprocess and no port.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from mcp import Client

from nfl_platform.config.settings import Settings
from nfl_platform.mcp_server import build_mcp_server
from nfl_platform.services import ServiceContainer


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests on asyncio only; the SDK supports both, we need one."""
    return "asyncio"


@pytest.fixture
def mcp_settings(settings: Settings) -> Settings:
    """Offline settings with web search enabled, so the stub provider is reachable."""
    return settings.model_copy(update={"mcp_enable_web_search": True})


@pytest.fixture
def mcp_server(services: ServiceContainer, mcp_settings: Settings):
    return build_mcp_server(container=services, settings=mcp_settings)


@pytest.fixture
async def client(mcp_server) -> AsyncIterator[Client]:
    """A connected in-process client."""
    async with Client(mcp_server) as connected:
        yield connected
