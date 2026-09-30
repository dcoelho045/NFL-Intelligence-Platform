"""End-to-end: a real MCP client against the server as a separate process.

Every other MCP test connects in-process, which exercises the protocol but not
the packaging: the console entry point, the stdio transport, the settings read
from the environment, and the rule that nothing may write to stdout.

This one launches ``python -m nfl_platform.mcp_server`` as a subprocess and
speaks to it over its stdin and stdout, which is exactly what an editor or a
desktop host does. It is marked ``integration`` because the subprocess builds a
live loader and reaches nflverse for its data.

    pytest -m integration tests/integration/test_mcp_stdio.py
"""

from __future__ import annotations

import os
import sys

import pytest
from mcp import Client, StdioServerParameters
from mcp.types import TextContent

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

LIVE_SEASON = 2024


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def server_parameters() -> StdioServerParameters:
    """Launch the installed entry point with web search switched off.

    Disabled because this test must not spend search credits; the deterministic
    tools are what it is here to prove.
    """
    environment = dict(os.environ)
    environment["NFL_MCP_ENABLE_WEB_SEARCH"] = "false"
    environment["NFL_LOG_LEVEL"] = "WARNING"
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "nfl_platform.mcp_server"],
        env=environment,
    )


async def test_a_client_can_discover_and_call_a_tool_over_stdio(
    server_parameters: StdioServerParameters,
):
    async with Client(server_parameters) as client:
        assert client.server_info is not None
        assert client.server_info.name == "nfl-intelligence"

        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        assert "get_player_stats" in names

        result = await client.call_tool(
            "get_player_stats",
            {"player": "Patrick Mahomes", "seasons": [LIVE_SEASON], "last_n_games": 4},
        )

        assert not result.is_error, _text(result.content)
        assert result.structured_content is not None
        assert result.structured_content["player"]["name"] == "Patrick Mahomes"
        assert result.structured_content["window"]["seasons"] == [LIVE_SEASON]
        assert result.structured_content["totals"]["passing_yards"]


async def test_a_resource_and_a_prompt_are_usable_over_stdio(
    server_parameters: StdioServerParameters,
):
    async with Client(server_parameters) as client:
        status = await client.read_resource("nfl://status")
        assert status.contents

        rendered = await client.get_prompt("analyze_matchup", {"team_a": "KC", "team_b": "BUF"})
        assert rendered.messages


async def test_a_failure_arrives_as_a_result_not_a_crash(
    server_parameters: StdioServerParameters,
):
    """The process must survive a bad call and keep serving."""
    async with Client(server_parameters) as client:
        failed = await client.call_tool("get_player_profile", {"player": "00-9999999"})
        assert failed.is_error
        assert "player_not_found" in _text(failed.content)

        listed = await client.list_tools()
        assert listed.tools


def _text(content: list[object]) -> str:
    return " ".join(block.text for block in content if isinstance(block, TextContent))
