"""The NFL MCP adapter.

A third adapter beside ``api/`` (HTTP) and ``llm/`` (the in-process assistant),
all three over one :class:`~nfl_platform.services.container.ServiceContainer`.
"""

from __future__ import annotations

from nfl_platform.mcp_server.server import INSTRUCTIONS, build_mcp_server

__all__ = ["INSTRUCTIONS", "build_mcp_server"]
