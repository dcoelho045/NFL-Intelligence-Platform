"""Assistant HTTP endpoints, driven by a scripted model.

The container is the offline fixture container and the model is scripted, so
these exercise real routing, validation, serialisation and tool execution with
no network and no API key.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from nfl_platform.api.app import create_app
from nfl_platform.config.settings import Settings
from nfl_platform.llm.agent import NflAnalystAgent
from nfl_platform.llm.client import ChatMessage, Completion, ScriptedChatClient, ToolCall, Usage
from nfl_platform.llm.prompts import DEFAULT_PROMPT_VERSION
from nfl_platform.llm.registry import build_default_registry
from nfl_platform.services import ServiceContainer
from support import MAHOMES_ID

QUESTION = "How has Patrick Mahomes played recently?"


def tool_then_answer(answer: str) -> list[Completion]:
    call = ToolCall(
        id="call_1",
        name="get_player_stats",
        arguments=f'{{"player": "{MAHOMES_ID}", "last_n_games": 3}}',
    )
    return [
        Completion(
            message=ChatMessage.assistant(tool_calls=[call]),
            usage=Usage(prompt_tokens=120, completion_tokens=25),
            finish_reason="tool_calls",
        ),
        Completion(
            message=ChatMessage.assistant(content=answer),
            usage=Usage(prompt_tokens=200, completion_tokens=45),
            finish_reason="stop",
        ),
    ]


@pytest.fixture
def assistant_client(services: ServiceContainer, settings: Settings) -> Iterator[TestClient]:
    agent = NflAnalystAgent(
        client=ScriptedChatClient(
            tool_then_answer("Mahomes has been efficient over his last three games."),
            model="gpt-4-turbo",
        ),
        container=services,
        registry=build_default_registry(),
        settings=settings,
    )
    with TestClient(create_app(container=services, settings=settings, agent=agent)) as client:
        yield client


class TestAskEndpoint:
    def test_returns_an_answer(self, assistant_client: TestClient):
        response = assistant_client.post("/api/v1/assistant/ask", json={"question": QUESTION})
        assert response.status_code == 200
        assert response.json()["answer"]

    def test_evidence_records_the_tool_that_ran(self, assistant_client: TestClient):
        body = assistant_client.post("/api/v1/assistant/ask", json={"question": QUESTION}).json()
        assert body["evidence"][0]["tool"] == "get_player_stats"
        assert body["evidence"][0]["arguments"]["player"] == MAHOMES_ID

    def test_stats_report_cost_and_latency(self, assistant_client: TestClient):
        stats = assistant_client.post("/api/v1/assistant/ask", json={"question": QUESTION}).json()[
            "stats"
        ]
        assert stats["total_tokens"] == 390
        assert stats["estimated_cost_usd"] > 0
        assert stats["duration_ms"] > 0
        assert stats["tools_used"] == ["get_player_stats"]

    def test_prompt_version_is_reported(self, assistant_client: TestClient):
        stats = assistant_client.post("/api/v1/assistant/ask", json={"question": QUESTION}).json()[
            "stats"
        ]
        assert stats["prompt_version"] == DEFAULT_PROMPT_VERSION

    def test_grounding_findings_are_exposed(self, assistant_client: TestClient):
        body = assistant_client.post("/api/v1/assistant/ask", json={"question": QUESTION}).json()
        assert body["stats"]["ungrounded_numbers"] == []

    def test_sources_are_returned_and_labelled_by_kind(self, assistant_client: TestClient):
        body = assistant_client.post("/api/v1/assistant/ask", json={"question": QUESTION}).json()
        assert body["sources"] == [{"kind": "structured", "label": "get_player_stats"}]

    def test_retrieval_counters_are_reported(self, assistant_client: TestClient):
        """A question answered from analytics alone must spend no retrieval budget."""
        stats = assistant_client.post("/api/v1/assistant/ask", json={"question": QUESTION}).json()[
            "stats"
        ]
        assert stats["retrieval"]["web_searches"] == 0
        assert stats["retrieval"]["knowledge_searches"] == 0

    def test_unstructured_number_bucket_is_exposed(self, assistant_client: TestClient):
        stats = assistant_client.post("/api/v1/assistant/ask", json={"question": QUESTION}).json()[
            "stats"
        ]
        assert stats["numbers_from_unstructured_sources"] == []
        assert stats["fabricated_sources"] == []

    def test_history_is_accepted(self, assistant_client: TestClient):
        response = assistant_client.post(
            "/api/v1/assistant/ask",
            json={
                "question": QUESTION,
                "history": [
                    {"role": "user", "content": "Earlier"},
                    {"role": "assistant", "content": "Answer"},
                ],
            },
        )
        assert response.status_code == 200


class TestValidation:
    def test_empty_question_is_rejected(self, assistant_client: TestClient):
        response = assistant_client.post("/api/v1/assistant/ask", json={"question": ""})
        assert response.status_code == 422

    def test_missing_question_is_rejected(self, assistant_client: TestClient):
        assert assistant_client.post("/api/v1/assistant/ask", json={}).status_code == 422

    def test_unknown_field_is_rejected(self, assistant_client: TestClient):
        response = assistant_client.post(
            "/api/v1/assistant/ask", json={"question": QUESTION, "temperature": 2}
        )
        assert response.status_code == 422

    def test_overlong_question_is_rejected(self, assistant_client: TestClient):
        response = assistant_client.post("/api/v1/assistant/ask", json={"question": "x" * 5000})
        assert response.status_code == 422

    def test_invalid_history_role_is_rejected(self, assistant_client: TestClient):
        response = assistant_client.post(
            "/api/v1/assistant/ask",
            json={"question": QUESTION, "history": [{"role": "system", "content": "hi"}]},
        )
        assert response.status_code == 422


class TestToolCatalogue:
    def test_lists_every_tool(self, assistant_client: TestClient):
        body = assistant_client.get("/api/v1/assistant/tools").json()
        assert body["count"] == len(build_default_registry())
        assert {tool["name"] for tool in body["tools"]} == set(build_default_registry().names)

    def test_each_tool_exposes_its_argument_schema(self, assistant_client: TestClient):
        tools = assistant_client.get("/api/v1/assistant/tools").json()["tools"]
        stats_tool = next(tool for tool in tools if tool["name"] == "get_player_stats")
        assert "player" in stats_tool["parameters"]["properties"]


class TestWithoutAProvider:
    def test_analytics_endpoints_still_work(self, services: ServiceContainer, settings: Settings):
        """A missing key must not take down the deterministic API."""
        bare = Settings(_env_file=None, default_season=2024, llm_api_key=None)
        with TestClient(create_app(container=services, settings=bare)) as client:
            assert client.get(f"/api/v1/players/{MAHOMES_ID}").status_code == 200

    def test_assistant_reports_that_it_is_not_configured(self, services: ServiceContainer):
        bare = Settings(_env_file=None, default_season=2024, llm_api_key=None)
        with TestClient(
            create_app(container=services, settings=bare), raise_server_exceptions=False
        ) as client:
            response = client.post("/api/v1/assistant/ask", json={"question": QUESTION})
        assert response.status_code == 503
        assert response.json()["error"] == "llm_not_configured"

    def test_tool_catalogue_still_works(self, services: ServiceContainer):
        """Inspecting the tool boundary must not require paying for a model."""
        bare = Settings(_env_file=None, default_season=2024, llm_api_key=None)
        with TestClient(
            create_app(container=services, settings=bare), raise_server_exceptions=False
        ) as client:
            response = client.get("/api/v1/assistant/tools")
        assert response.status_code == 200
        assert response.json()["count"] == len(build_default_registry())
