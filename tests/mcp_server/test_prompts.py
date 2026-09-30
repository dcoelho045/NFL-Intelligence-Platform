"""Prompts: rendered text, nothing else.

A prompt is the one place this server speaks to a model it does not control, so
the assertions are mostly about content: the rendered message must keep pointing
the host at the deterministic tools and must not invite a projection that no
tool here produces.
"""

from __future__ import annotations

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp.types import TextContent

pytestmark = pytest.mark.anyio


async def render(client: Client, name: str, arguments: dict[str, str]) -> str:
    result = await client.get_prompt(name, arguments)
    return " ".join(
        block.text
        for message in result.messages
        if isinstance(block := message.content, TextContent)
    )


class TestPromptDiscovery:
    async def test_arguments_are_described_for_a_form(self, client: Client):
        listed = await client.list_prompts()
        matchup = next(p for p in listed.prompts if p.name == "analyze_matchup")
        required = {arg.name for arg in matchup.arguments or [] if arg.required}
        assert required == {"team_a", "team_b"}

    async def test_every_prompt_has_a_title_and_description(self, client: Client):
        listed = await client.list_prompts()
        for prompt in listed.prompts:
            assert prompt.title, prompt.name
            assert prompt.description, prompt.name


class TestRendering:
    async def test_matchup_interpolates_both_teams(self, client: Client):
        text = await render(client, "analyze_matchup", {"team_a": "KC", "team_b": "BUF"})
        assert "KC" in text and "BUF" in text
        assert "get_matchup" in text

    async def test_the_optional_week_appears_only_when_given(self, client: Client):
        without = await render(client, "analyze_matchup", {"team_a": "KC", "team_b": "BUF"})
        with_week = await render(
            client, "analyze_matchup", {"team_a": "KC", "team_b": "BUF", "week": "5"}
        )
        assert "week 5" not in without
        assert "week 5" in with_week

    async def test_week_and_season_qualify_the_question_together(self, client: Client):
        text = await render(
            client,
            "analyze_matchup",
            {"team_a": "KC", "team_b": "BUF", "week": "5", "season": "2024"},
        )
        assert "week 5 of the 2024 season" in text

    async def test_a_season_alone_qualifies_the_question(self, client: Client):
        text = await render(
            client, "player_performance_review", {"player": "Josh Allen", "season": "2024"}
        )
        assert "the 2024 season" in text

    async def test_start_sit_points_at_the_shared_window_tool(self, client: Client):
        text = await render(
            client, "fantasy_start_sit", {"player_a": "Josh Allen", "player_b": "Patrick Mahomes"}
        )
        assert "compare_fantasy_players" in text
        assert "Josh Allen" in text

    async def test_player_review_points_at_usage_and_the_knowledge_base(self, client: Client):
        text = await render(client, "player_performance_review", {"player": "Josh Allen"})
        assert "get_player_usage" in text
        assert "search_knowledge_base" in text

    async def test_a_missing_required_argument_fails_the_request(self, client: Client):
        with pytest.raises(MCPError):
            await client.get_prompt("analyze_matchup", {"team_a": "KC"})


class TestPromptsCarryTheEvidencePolicy:
    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("analyze_matchup", {"team_a": "KC", "team_b": "BUF"}),
            ("fantasy_start_sit", {"player_a": "Josh Allen", "player_b": "Patrick Mahomes"}),
            ("player_performance_review", {"player": "Josh Allen"}),
        ],
    )
    async def test_every_prompt_requires_numbers_to_come_from_a_tool(
        self, client: Client, name: str, arguments: dict[str, str]
    ):
        text = await render(client, name, arguments)
        assert "tool result" in text

    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("analyze_matchup", {"team_a": "KC", "team_b": "BUF"}),
            ("fantasy_start_sit", {"player_a": "Josh Allen", "player_b": "Patrick Mahomes"}),
            ("player_performance_review", {"player": "Josh Allen"}),
        ],
    )
    async def test_no_prompt_asks_for_a_prediction(
        self, client: Client, name: str, arguments: dict[str, str]
    ):
        """Milestone 6 owns predictions; a prompt must not invite an invented one."""
        text = (await render(client, name, arguments)).lower()
        assert "projection" in text or "prediction" in text

    async def test_a_prompt_that_mentions_web_search_says_what_to_do_when_it_is_off(
        self, client: Client
    ):
        text = await render(client, "analyze_matchup", {"team_a": "KC", "team_b": "BUF"})
        assert "unavailable" in text
        assert "Do not fill the gap from memory" in text
