"""Prediction tools and the models resource over the real MCP protocol."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from mcp import Client

from nfl_platform.config.settings import Settings
from nfl_platform.mcp_server import build_mcp_server
from nfl_platform.ml.registry.store import ModelRegistry
from nfl_platform.services import ServiceContainer
from synthetic import SYNTHETIC_TEAMS, SyntheticLeague, synthetic_player_id

pytestmark = pytest.mark.anyio

HOME, AWAY = SYNTHETIC_TEAMS[0], SYNTHETIC_TEAMS[1]

PREDICTION_TOOLS = frozenset(
    {
        "predict_game",
        "predict_slate",
        "project_fantasy_points",
        "compare_fantasy_projections",
    }
)


@pytest.fixture
def prediction_server(prediction_services: ServiceContainer, prediction_settings: Settings):
    return build_mcp_server(container=prediction_services, settings=prediction_settings)


@pytest.fixture
async def prediction_client(prediction_server) -> AsyncIterator[Client]:
    async with Client(prediction_server) as connected:
        yield connected


@pytest.fixture
def target(synthetic_league: SyntheticLeague) -> dict[str, int]:
    return {"season": synthetic_league.seasons[-1], "week": 8}


def _payload(result) -> dict:
    return json.loads(result.content[0].text) if result.content else {}


class TestCatalogue:
    async def test_every_prediction_tool_is_registered(self, prediction_client: Client):
        listed = await prediction_client.list_tools()
        assert {tool.name for tool in listed.tools} >= PREDICTION_TOOLS

    async def test_the_analytics_tools_are_still_registered(self, prediction_client: Client):
        """Milestone 6 adds to the catalogue; it must not disturb it."""
        listed = await prediction_client.list_tools()
        names = {tool.name for tool in listed.tools}
        assert {"get_player_stats", "get_team_stats", "search_knowledge_base"} <= names

    async def test_the_descriptions_mark_predictions_as_estimates(self, prediction_client: Client):
        listed = await prediction_client.list_tools()
        tools = {tool.name: tool for tool in listed.tools}
        assert "estimate" in (tools["predict_game"].description or "").lower()

    async def test_the_fantasy_description_states_the_playing_condition(
        self, prediction_client: Client
    ):
        listed = await prediction_client.list_tools()
        tools = {tool.name: tool for tool in listed.tools}
        description = (tools["project_fantasy_points"].description or "").lower()
        assert "conditional on the player playing" in description

    async def test_every_prediction_tool_publishes_an_output_schema(
        self, prediction_client: Client
    ):
        listed = await prediction_client.list_tools()
        for tool in listed.tools:
            if tool.name in PREDICTION_TOOLS:
                assert tool.output_schema is not None, tool.name

    async def test_unknown_arguments_are_rejected(self, prediction_client: Client):
        """The hardening from Milestone 5 must cover the new tools too."""
        result = await prediction_client.call_tool(
            "predict_game",
            {"home_team": HOME, "away_team": AWAY, "quarter": 3},
        )
        assert result.is_error


class TestPredictGameTool:
    async def test_returns_a_probability(self, prediction_client: Client, target):
        result = await prediction_client.call_tool(
            "predict_game", {"home_team": HOME, "away_team": AWAY, **target}
        )
        assert not result.is_error
        body = _payload(result)
        assert 0.0 <= body["home_win_probability"] <= 1.0

    async def test_the_result_names_the_model(self, prediction_client: Client, target):
        result = await prediction_client.call_tool(
            "predict_game", {"home_team": HOME, "away_team": AWAY, **target}
        )
        assert _payload(result)["model_info"]["model_id"].startswith("game_win/")

    async def test_too_little_history_is_an_error_not_a_guess(
        self, prediction_client: Client, synthetic_league: SyntheticLeague
    ):
        result = await prediction_client.call_tool(
            "predict_game",
            {
                "home_team": HOME,
                "away_team": AWAY,
                "season": synthetic_league.seasons[0],
                "week": 1,
            },
        )
        assert result.is_error
        assert "insufficient_history" in result.content[0].text

    async def test_an_error_carries_recovery_guidance(
        self, prediction_client: Client, synthetic_league: SyntheticLeague
    ):
        result = await prediction_client.call_tool(
            "predict_game",
            {
                "home_team": HOME,
                "away_team": AWAY,
                "season": synthetic_league.seasons[0],
                "week": 1,
            },
        )
        assert "Do not guess" in result.content[0].text

    async def test_an_unavailable_model_is_flagged_as_a_capability_state(
        self, prediction_loader, settings: Settings, tmp_path, target
    ):
        """Milestone 5's invariant: unavailable is never evidence about the world."""
        container = ServiceContainer(
            loader=prediction_loader,
            settings=settings,
            model_registry=ModelRegistry(root=tmp_path / "empty", settings=settings),
        )
        server = build_mcp_server(container=container, settings=settings)
        async with Client(server) as client:
            result = await client.call_tool(
                "predict_game", {"home_team": HOME, "away_team": AWAY, **target}
            )
        assert result.is_error
        text = result.content[0].text
        assert "model_not_available" in text
        assert "not evidence that no such information exists" in text


class TestFantasyTools:
    async def test_projects_a_player(self, prediction_client: Client, target):
        result = await prediction_client.call_tool(
            "project_fantasy_points",
            {"player": synthetic_player_id(SYNTHETIC_TEAMS[0], 1), **target},
        )
        assert not result.is_error
        body = _payload(result)
        assert body["conditional_on_playing"] is True

    async def test_compare_ranks_players(self, prediction_client: Client, target):
        result = await prediction_client.call_tool(
            "compare_fantasy_projections",
            {
                "players": [
                    synthetic_player_id(SYNTHETIC_TEAMS[0], 1),
                    synthetic_player_id(SYNTHETIC_TEAMS[1], 1),
                ],
                **target,
            },
        )
        assert not result.is_error
        assert _payload(result)["recommended_player_id"]

    async def test_a_single_player_comparison_is_rejected(self, prediction_client: Client, target):
        result = await prediction_client.call_tool(
            "compare_fantasy_projections",
            {"players": [synthetic_player_id(SYNTHETIC_TEAMS[0], 1)], **target},
        )
        assert result.is_error


class TestResources:
    async def test_the_models_resource_is_registered(self, prediction_client: Client):
        listed = await prediction_client.list_resources()
        assert "nfl://models" in {str(resource.uri) for resource in listed.resources}

    async def test_the_models_resource_lists_installed_artifacts(self, prediction_client: Client):
        result = await prediction_client.read_resource("nfl://models")
        body = json.loads(result.contents[0].text)
        assert body["available"] is True
        assert {entry["problem"] for entry in body["models"]} == {
            "game_win",
            "fantasy_points",
        }

    async def test_status_reports_prediction_availability(self, prediction_client: Client):
        result = await prediction_client.read_resource("nfl://status")
        body = json.loads(result.contents[0].text)
        assert body["predictions_enabled"] is True
        assert body["prediction_models_available"] is True
        assert body["prediction_models"]

    async def test_the_models_resource_survives_an_empty_registry(
        self, prediction_loader, settings: Settings, tmp_path
    ):
        container = ServiceContainer(
            loader=prediction_loader,
            settings=settings,
            model_registry=ModelRegistry(root=tmp_path / "empty", settings=settings),
        )
        server = build_mcp_server(container=container, settings=settings)
        async with Client(server) as client:
            result = await client.read_resource("nfl://models")
        assert json.loads(result.contents[0].text)["available"] is False


class TestPrompts:
    async def test_the_prediction_prompt_is_registered(self, prediction_client: Client):
        listed = await prediction_client.list_prompts()
        assert "predict_game" in {prompt.name for prompt in listed.prompts}

    async def test_the_prompt_routes_numbers_to_the_model(self, prediction_client: Client):
        result = await prediction_client.get_prompt(
            "predict_game", {"home_team": HOME, "away_team": AWAY}
        )
        text = result.messages[0].content.text
        assert "predict_game" in text
        assert "Do not invent a probability" in text

    async def test_the_prompt_separates_model_output_from_analytics(
        self, prediction_client: Client
    ):
        result = await prediction_client.get_prompt(
            "predict_game", {"home_team": HOME, "away_team": AWAY}
        )
        text = result.messages[0].content.text
        assert "not a measured statistic" in text


class TestServerInstructions:
    async def test_the_instructions_describe_the_prediction_channel(self, prediction_server):
        instructions = prediction_server.instructions or ""
        assert "predict_game" in instructions
        assert "never by\n   a language model" in instructions.replace("\r\n", "\n")
