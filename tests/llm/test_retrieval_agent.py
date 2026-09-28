"""Retrieval through the whole agent loop.

The model is scripted, the provider is a fixture, and the index is built with the
hash embedder -- so these assert the system's behaviour rather than a model's
mood, and they cost nothing to run.
"""

from __future__ import annotations

import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.llm.agent import NflAnalystAgent
from nfl_platform.llm.client import ScriptedChatClient
from nfl_platform.llm.registry import build_default_registry
from nfl_platform.retrieval.web.cache import SearchCache
from nfl_platform.retrieval.web.provider import NullSearchProvider
from nfl_platform.retrieval.web.search import WebSearchClient
from nfl_platform.services import ServiceContainer
from support import MAHOMES_ID, StubWebSearchProvider, failing_web_provider

from .conftest import answer_completion, tool_call_completion

QUESTION = "Is Patrick Mahomes playing this week?"


def agent_with(completions, services: ServiceContainer, settings: Settings) -> NflAnalystAgent:
    return NflAnalystAgent(
        client=ScriptedChatClient(completions, model="gpt-4-turbo"),
        container=services,
        registry=build_default_registry(),
        settings=settings,
    )


def container_with(loader, settings: Settings, provider) -> ServiceContainer:
    client = WebSearchClient(
        provider=provider,
        settings=settings,
        cache=SearchCache(ttl_seconds=600, max_entries=16),
    )
    return ServiceContainer(loader=loader, settings=settings, web_client=client)


class TestWebRetrievalEndToEnd:
    def test_a_web_search_produces_an_answer_with_sources(
        self, services: ServiceContainer, settings: Settings
    ):
        agent = agent_with(
            [
                tool_call_completion([("search_web", {"query": "Mahomes injury status"})]),
                answer_completion("He is a full participant and carries no designation."),
            ],
            services,
            settings,
        )
        result = agent.ask(QUESTION)
        assert result.trace is not None
        assert "search_web" in result.trace.tools_used
        assert any(source["kind"] == "web" for source in result.trace.sources)

    def test_sources_carry_url_and_publisher(self, services: ServiceContainer, settings: Settings):
        agent = agent_with(
            [
                tool_call_completion([("search_web", {"query": "Mahomes injury status"})]),
                answer_completion("He practised fully."),
            ],
            services,
            settings,
        )
        source = next(item for item in agent.ask(QUESTION).trace.sources if item["kind"] == "web")
        assert source["url"].startswith("https://")
        assert source["domain"]
        assert source["retrieved_at"]

    def test_retrieval_counters_are_recorded(self, services: ServiceContainer, settings: Settings):
        agent = agent_with(
            [
                tool_call_completion([("search_web", {"query": "Mahomes injury status"})]),
                answer_completion("He practised fully."),
            ],
            services,
            settings,
        )
        retrieval = agent.ask(QUESTION).trace.retrieval
        assert retrieval.web_searches == 1
        assert retrieval.web_results > 0

    def test_the_budget_resets_between_questions(self, loader, settings: Settings):
        """A per-process counter would silently stop answering after a few questions."""
        run_settings = settings.model_copy(update={"web_search_max_calls_per_run": 1})
        provider = StubWebSearchProvider(fixture="injury_status")
        services = container_with(loader, run_settings, provider)

        for _ in range(3):
            agent = agent_with(
                [
                    tool_call_completion([("search_web", {"query": "Mahomes injury status"})]),
                    answer_completion("He practised fully."),
                ],
                services,
                run_settings,
            )
            result = agent.ask(QUESTION)
            assert result.trace.tool_calls[0].ok, result.trace.tool_calls[0].error_code

    def test_no_results_does_not_produce_an_invented_answer(self, loader, settings: Settings):
        services = container_with(loader, settings, StubWebSearchProvider(fixture="no_results"))
        agent = agent_with(
            [
                tool_call_completion([("search_web", {"query": "belgian punter signing"})]),
                answer_completion("I could not find any current reporting on that."),
            ],
            services,
            settings,
        )
        result = agent.ask("Did they sign a Belgian punter?")
        assert result.trace.tool_calls[0].ok
        assert result.trace.is_grounded

    def test_a_failed_search_is_recoverable_not_fatal(self, loader, settings: Settings):
        services = container_with(loader, settings, failing_web_provider())
        agent = agent_with(
            [
                tool_call_completion([("search_web", {"query": "Mahomes injury status"})]),
                answer_completion("Current reporting could not be retrieved."),
            ],
            services,
            settings,
        )
        result = agent.ask(QUESTION)
        assert result.answer
        assert result.trace.tool_calls[0].ok is False

    def test_an_unconfigured_provider_is_announced_before_the_model_tries(
        self, loader, settings: Settings
    ):
        services = container_with(loader, settings, NullSearchProvider("no key"))
        agent = agent_with([answer_completion("I cannot check current news.")], services, settings)
        agent.ask(QUESTION)
        context = agent._runtime_context()
        assert "search_web" in context
        assert "not configured on this deployment" in context
        assert "do not call it" in context

    def test_an_available_provider_announces_the_budget(
        self, services: ServiceContainer, settings: Settings
    ):
        agent = agent_with([answer_completion("ok")], services, settings)
        assert "web search(es)" in agent._runtime_context()


class TestKnowledgeRetrievalEndToEnd:
    def test_a_definition_question_produces_a_knowledge_source(
        self, services: ServiceContainer, settings: Settings
    ):
        agent = agent_with(
            [
                tool_call_completion([("search_knowledge_base", {"query": "what does PPR mean"})]),
                answer_completion("PPR means points per reception."),
            ],
            services,
            settings,
        )
        result = agent.ask("What does PPR mean?")
        sources = [item for item in result.trace.sources if item["kind"] == "knowledge_base"]
        assert sources
        assert sources[0]["document_id"]

    def test_knowledge_sources_carry_their_updated_date(
        self, services: ServiceContainer, settings: Settings
    ):
        agent = agent_with(
            [
                tool_call_completion([("search_knowledge_base", {"query": "what does PPR mean"})]),
                answer_completion("PPR means points per reception."),
            ],
            services,
            settings,
        )
        source = next(
            item
            for item in agent.ask("What does PPR mean?").trace.sources
            if item["kind"] == "knowledge_base"
        )
        assert source["updated"]


class TestRoutingIsRecorded:
    def test_a_structured_question_records_only_structured_sources(
        self, services: ServiceContainer, settings: Settings
    ):
        agent = agent_with(
            [
                tool_call_completion(
                    [("get_player_stats", {"player": MAHOMES_ID, "last_n_games": 3})]
                ),
                answer_completion("Here are his recent numbers."),
            ],
            services,
            settings,
        )
        result = agent.ask("How has Mahomes played recently?")
        assert {item["kind"] for item in result.trace.sources} == {"structured"}
        assert result.trace.retrieval.web_searches == 0

    def test_a_compound_question_records_all_the_kinds_it_used(
        self, services: ServiceContainer, settings: Settings
    ):
        agent = agent_with(
            [
                tool_call_completion(
                    [
                        ("get_player_stats", {"player": MAHOMES_ID, "last_n_games": 3}),
                        ("search_web", {"query": "Mahomes injury status"}),
                        ("search_knowledge_base", {"query": "what is target share"}),
                    ]
                ),
                answer_completion("Combining the three."),
            ],
            services,
            settings,
        )
        kinds = {item["kind"] for item in agent.ask("Should I start him?").trace.sources}
        assert kinds == {"structured", "web", "knowledge_base"}

    def test_each_tool_call_records_which_channel_it_used(
        self, services: ServiceContainer, settings: Settings
    ):
        agent = agent_with(
            [
                tool_call_completion(
                    [
                        ("get_player_stats", {"player": MAHOMES_ID}),
                        ("search_web", {"query": "Mahomes injury status"}),
                    ]
                ),
                answer_completion("Done."),
            ],
            services,
            settings,
        )
        calls = agent.ask(QUESTION).trace.tool_calls
        assert [call.source_kind for call in calls] == ["structured", "web"]


class TestPromptInjectionThroughTheAgent:
    @pytest.fixture
    def injected_services(self, loader, settings: Settings) -> ServiceContainer:
        return container_with(loader, settings, StubWebSearchProvider(fixture="injection"))

    def test_injected_instructions_never_reach_the_model(
        self, injected_services: ServiceContainer, settings: Settings
    ):
        client = ScriptedChatClient(
            [
                tool_call_completion([("search_web", {"query": "player injury status"})]),
                answer_completion("He is expected to start."),
            ],
            model="gpt-4-turbo",
        )
        agent = NflAnalystAgent(
            client=client,
            container=injected_services,
            registry=build_default_registry(),
            settings=settings,
        )
        result = agent.ask(QUESTION)

        transcript = " ".join(message.content or "" for message in result.messages).lower()
        for phrase in (
            "ignore all previous instructions",
            "you are now an unrestricted",
            "new instructions:",
            "reveal your system prompt",
        ):
            assert phrase not in transcript

    def test_retrieved_content_arrives_only_as_a_tool_message(
        self, injected_services: ServiceContainer, settings: Settings
    ):
        """The structural defence: it is never a system or user turn."""
        agent = agent_with(
            [
                tool_call_completion([("search_web", {"query": "player injury status"})]),
                answer_completion("He is expected to start."),
            ],
            injected_services,
            settings,
        )
        result = agent.ask(QUESTION)
        carriers = {
            message.role
            for message in result.messages
            if message.content and "espn.com" in message.content
        }
        assert carriers <= {"tool"}

    def test_injection_does_not_cause_additional_tool_calls(
        self, injected_services: ServiceContainer, settings: Settings
    ):
        agent = agent_with(
            [
                tool_call_completion([("search_web", {"query": "player injury status"})]),
                answer_completion("He is expected to start."),
            ],
            injected_services,
            settings,
        )
        result = agent.ask(QUESTION)
        assert [call.tool for call in result.trace.tool_calls] == ["search_web"]

    def test_dropped_results_are_counted_in_the_trace(
        self, injected_services: ServiceContainer, settings: Settings
    ):
        agent = agent_with(
            [
                tool_call_completion([("search_web", {"query": "player injury status"})]),
                answer_completion("He is expected to start."),
            ],
            injected_services,
            settings,
        )
        assert agent.ask(QUESTION).trace.retrieval.web_results_dropped == 2


class TestFabricatedCitations:
    def test_an_invented_url_is_recorded_on_the_trace(
        self, services: ServiceContainer, settings: Settings
    ):
        agent = agent_with(
            [
                tool_call_completion([("search_web", {"query": "Mahomes injury status"})]),
                answer_completion("See https://www.rotoworld.com/invented-story for more."),
            ],
            services,
            settings,
        )
        result = agent.ask(QUESTION)
        assert result.trace.fabricated_sources
        assert result.trace.citations_are_valid is False

    def test_a_real_citation_is_accepted(self, services: ServiceContainer, settings: Settings):
        url = "https://www.espn.com/nfl/story/_/id/00000001/chiefs-mahomes-practice-report"
        agent = agent_with(
            [
                tool_call_completion([("search_web", {"query": "Mahomes injury status"})]),
                answer_completion(f"ESPN reported this: {url}"),
            ],
            services,
            settings,
        )
        result = agent.ask(QUESTION)
        assert result.trace.citations_are_valid
