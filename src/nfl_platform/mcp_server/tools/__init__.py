"""The MCP tool catalogue.

Twenty tools over five modules, each one a thin adapter onto a service method.
The list is deliberately smaller than the full service surface: a catalogue a
client can read is worth more than one that mirrors every internal method.

The prediction tools are the only ones whose numbers describe games that have
not been played. They are registered last so the catalogue reads in order of
increasing uncertainty: measured, retrieved, then estimated.
"""

from __future__ import annotations

from mcp.server import MCPServer

from nfl_platform.config.settings import Settings
from nfl_platform.mcp_server.tools.fantasy import register_fantasy_tools
from nfl_platform.mcp_server.tools.players import register_player_tools
from nfl_platform.mcp_server.tools.predictions import register_prediction_tools
from nfl_platform.mcp_server.tools.research import register_research_tools
from nfl_platform.mcp_server.tools.teams import register_team_tools
from nfl_platform.services.container import ServiceContainer

__all__ = ["register_tools"]


def register_tools(mcp: MCPServer, container: ServiceContainer, settings: Settings) -> None:
    """Register every tool this server exposes."""
    register_player_tools(mcp, container)
    register_team_tools(mcp, container)
    register_fantasy_tools(mcp, container)
    register_research_tools(mcp, container, settings)
    register_prediction_tools(mcp, container, settings)
