"""Initialisation and discovery.

These are the acceptance criteria "the server starts" and "tools are
discoverable", asserted over the real protocol rather than by importing the
module and looking at a dict.
"""

from __future__ import annotations

import pytest
from mcp import Client

from nfl_platform import __version__

pytestmark = pytest.mark.anyio

# The curated surface. Kept as a literal so adding or removing a tool is a
# deliberate, reviewable change rather than a silent one.
EXPECTED_TOOLS = {
    "search_players",
    "get_player_profile",
    "get_player_stats",
    "get_player_game_log",
    "get_player_usage",
    "get_player_injuries",
    "compare_players",
    "get_team_stats",
    "get_team_form",
    "get_team_injuries",
    "get_matchup",
    "get_fantasy_production",
    "compare_fantasy_players",
    "search_web",
    "search_knowledge_base",
}

EXPECTED_PROMPTS = {"analyze_matchup", "fantasy_start_sit", "player_performance_review"}


class TestInitialisation:
    async def test_server_reports_its_identity(self, client: Client):
        assert client.server_info is not None
        assert client.server_info.name == "nfl-intelligence"
        assert client.server_info.version == __version__

    async def test_server_advertises_every_capability_it_implements(self, client: Client):
        capabilities = client.server_capabilities
        assert capabilities is not None
        assert capabilities.tools is not None
        assert capabilities.resources is not None
        assert capabilities.prompts is not None

    async def test_instructions_state_the_evidence_policy(self, client: Client):
        instructions = client.instructions or ""
        assert "authoritative" in instructions
        assert "untrusted" in instructions
        assert "not current" in instructions

    async def test_instructions_rule_out_predictions(self, client: Client):
        """Milestone 6 has not happened; a host must not expect a projection."""
        assert "produces a prediction or a projection" in (client.instructions or "").lower()

    async def test_protocol_version_is_negotiated(self, client: Client):
        assert client.protocol_version


class TestToolDiscovery:
    async def test_the_expected_tools_are_listed(self, client: Client):
        listed = await client.list_tools()
        assert {tool.name for tool in listed.tools} == EXPECTED_TOOLS

    async def test_the_catalogue_is_the_size_the_documentation_claims(self, client: Client):
        """docs/mcp-design.md and the README both quote this number."""
        listed = await client.list_tools()
        assert len(listed.tools) == 15

    async def test_every_tool_describes_itself(self, client: Client):
        listed = await client.list_tools()
        for tool in listed.tools:
            assert tool.description, tool.name
            assert tool.title, tool.name

    async def test_every_tool_publishes_an_input_schema(self, client: Client):
        listed = await client.list_tools()
        for tool in listed.tools:
            schema = tool.input_schema
            assert schema["type"] == "object", tool.name
            assert "properties" in schema, tool.name

    async def test_every_tool_publishes_an_output_schema(self, client: Client):
        """The DTO discipline from Milestone 2 is what makes this free."""
        listed = await client.list_tools()
        for tool in listed.tools:
            assert tool.output_schema is not None, tool.name

    async def test_every_tool_is_read_only(self, client: Client):
        listed = await client.list_tools()
        for tool in listed.tools:
            assert tool.annotations is not None, tool.name
            assert tool.annotations.read_only_hint is True, tool.name

    async def test_only_web_search_reaches_outside_the_loaded_data(self, client: Client):
        listed = await client.list_tools()
        open_world = {
            tool.name
            for tool in listed.tools
            if tool.annotations is not None and tool.annotations.open_world_hint
        }
        assert open_world == {"search_web"}

    async def test_analytics_tools_are_described_as_authoritative(self, client: Client):
        """Routing goes wrong when a client cannot tell the channels apart."""
        listed = await client.list_tools()
        stats = next(tool for tool in listed.tools if tool.name == "get_player_stats")
        assert "authoritative" in (stats.description or "")


class TestPromptAndResourceDiscovery:
    async def test_the_expected_prompts_are_listed(self, client: Client):
        listed = await client.list_prompts()
        assert {prompt.name for prompt in listed.prompts} == EXPECTED_PROMPTS

    async def test_resources_and_templates_are_listed_separately(self, client: Client):
        resources = await client.list_resources()
        templates = await client.list_resource_templates()

        assert {str(resource.uri) for resource in resources.resources} == {
            "nfl://status",
            "nfl://reference/seasons",
            "nfl://knowledge/documents",
        }
        assert {template.uri_template for template in templates.resource_templates} == {
            "nfl://reference/teams/{season}",
            "nfl://knowledge/documents/{document_id}",
        }
