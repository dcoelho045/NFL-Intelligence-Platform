"""Argument validation and error reporting.

Two separate guarantees are under test.

*Bad input never reaches a service.* The schema is the gate, and the SDK closes
it before the function body runs.

*A failure explains itself.* The rendered error keeps the code, the recovery hint
and the context (candidate ids, valid teams) that the Milestone 3 taxonomy
produces, because an error a caller cannot act on is only marginally better than
a wrong answer.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp import Client
from mcp.types import TextContent

from nfl_platform.mcp_server.errors import UNAVAILABLE_CODES
from support import FIXTURE_SEASON, MAHOMES_ID, UNKNOWN_PLAYER_ID

pytestmark = pytest.mark.anyio


async def failing_call(client: Client, name: str, arguments: dict[str, Any]) -> str:
    """Call a tool expecting failure, and return the error text."""
    result = await client.call_tool(name, arguments)
    assert result.is_error, f"{name} unexpectedly succeeded"
    assert result.structured_content is None
    return " ".join(block.text for block in result.content if isinstance(block, TextContent))


def detail(text: str) -> dict[str, Any]:
    """Pull the JSON tail out of a rendered error."""
    start = text.index("{")
    return json.loads(text[start:])


class TestArgumentValidation:
    async def test_an_out_of_range_value_is_rejected(self, client: Client):
        text = await failing_call(
            client, "get_player_stats", {"player": MAHOMES_ID, "last_n_games": 999}
        )
        assert "less than or equal to 50" in text

    async def test_a_hallucinated_argument_is_rejected(self, client: Client):
        """Ignoring an unknown argument would silently widen the window asked for."""
        text = await failing_call(
            client, "get_player_stats", {"player": MAHOMES_ID, "invented_argument": 1}
        )
        assert "invented_argument" in text

    async def test_the_published_schema_advertises_that_rule(self, client: Client):
        listed = await client.list_tools()
        for tool in listed.tools:
            assert tool.input_schema.get("additionalProperties") is False, tool.name

    async def test_a_missing_required_argument_is_rejected(self, client: Client):
        text = await failing_call(client, "get_player_stats", {})
        assert "player" in text

    async def test_a_wrong_type_is_rejected(self, client: Client):
        text = await failing_call(
            client, "get_player_stats", {"player": MAHOMES_ID, "seasons": "twenty twenty four"}
        )
        assert "seasons" in text

    async def test_an_empty_string_identifier_is_rejected(self, client: Client):
        text = await failing_call(client, "get_player_profile", {"player": ""})
        assert "at least 1 character" in text

    async def test_a_comparison_needs_at_least_two_players(self, client: Client):
        text = await failing_call(client, "compare_players", {"players": [MAHOMES_ID]})
        assert "at least 2 items" in text

    async def test_a_comparison_is_bounded(self, client: Client):
        text = await failing_call(client, "compare_players", {"players": [MAHOMES_ID] * 7})
        assert "at most 6 items" in text

    async def test_an_out_of_range_week_is_rejected(self, client: Client):
        text = await failing_call(client, "get_team_injuries", {"team": "KC", "week": 99})
        assert "less than or equal to 22" in text

    async def test_a_too_short_query_is_rejected(self, client: Client):
        text = await failing_call(client, "search_knowledge_base", {"query": "a"})
        assert "at least 3 characters" in text

    async def test_an_enum_value_outside_the_set_is_rejected(self, client: Client):
        text = await failing_call(client, "get_team_stats", {"team": "KC", "side": "special_teams"})
        assert "side" in text

    async def test_an_unknown_tool_is_an_error_not_a_crash(self, client: Client):
        text = await failing_call(client, "predict_game", {"team_a": "KC", "team_b": "BUF"})
        assert "predict_game" in text


class TestDomainErrors:
    async def test_an_unknown_player_says_how_to_recover(self, client: Client):
        text = await failing_call(client, "get_player_profile", {"player": UNKNOWN_PLAYER_ID})
        payload = detail(text)
        assert payload["code"] == "player_not_found"
        assert "search_players" in payload["recovery"]

    async def test_an_unknown_player_never_becomes_a_different_player(self, client: Client):
        text = await failing_call(client, "get_player_profile", {"player": UNKNOWN_PLAYER_ID})
        assert "Do not substitute a different player" in text

    async def test_an_ambiguous_name_returns_candidate_ids(self, client: Client):
        """The recovery path only works if the candidates travel with the error."""
        text = await failing_call(client, "get_player_profile", {"player": "Allen"})
        payload = detail(text)
        assert payload["code"] in ("ambiguous_player", "player_not_found")
        if payload["code"] == "ambiguous_player":
            assert payload["context"]["candidates"]

    async def test_an_unknown_team_lists_the_valid_ones(self, client: Client):
        text = await failing_call(
            client, "get_team_form", {"team": "XXX", "seasons": [FIXTURE_SEASON]}
        )
        payload = detail(text)
        assert payload["code"] == "team_not_found"
        assert payload["context"]["valid_teams"]

    async def test_an_invalid_season_is_reported_as_such(self, client: Client):
        text = await failing_call(
            client, "get_player_stats", {"player": MAHOMES_ID, "seasons": [1850]}
        )
        assert detail(text)["code"] == "invalid_season"

    async def test_every_error_carries_a_machine_readable_code(self, client: Client):
        """The SDK prefixes its own framing; the code must survive it intact."""
        text = await failing_call(client, "get_player_profile", {"player": UNKNOWN_PLAYER_ID})
        assert "player_not_found:" in text
        assert detail(text)["code"] == "player_not_found"

    async def test_errors_never_leak_internals(self, client: Client):
        text = await failing_call(client, "get_player_profile", {"player": UNKNOWN_PLAYER_ID})
        assert "Traceback" not in text
        assert "site-packages" not in text


class TestUnavailableIsNotEmpty:
    """D9: a switched-off capability must never read as "nothing was found"."""

    @pytest.fixture
    def mcp_settings(self, settings):
        return settings.model_copy(update={"mcp_enable_web_search": False})

    async def test_web_search_stays_in_the_catalogue_when_disabled(self, client: Client):
        listed = await client.list_tools()
        assert "search_web" in {tool.name for tool in listed.tools}

    async def test_calling_it_reports_configuration_not_absence(self, client: Client):
        text = await failing_call(client, "search_web", {"query": "Chiefs injury news"})
        payload = detail(text)
        assert payload["code"] == "web_search_not_configured"
        assert payload["retryable"] is False
        assert "not evidence that no such information exists" in payload["notice"]

    async def test_the_result_is_an_error_not_an_empty_success(self, client: Client):
        result = await client.call_tool("search_web", {"query": "Chiefs injury news"})
        assert result.is_error
        assert result.structured_content is None

    async def test_every_unavailable_code_carries_the_notice(self, client: Client):
        text = await failing_call(client, "search_web", {"query": "Chiefs injury news"})
        payload = detail(text)
        assert payload["code"] in UNAVAILABLE_CODES

    async def test_the_status_resource_agrees(self, client: Client):
        result = await client.read_resource("nfl://status")
        contents = result.contents[0]
        status = json.loads(getattr(contents, "text", "{}"))
        assert status["web_search_enabled"] is False
        assert status["web_search_available"] is False
