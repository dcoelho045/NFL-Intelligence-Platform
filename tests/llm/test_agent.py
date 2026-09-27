"""Agent loop behaviour: tool selection, recovery, grounding, and budgets.

The model is scripted; the tools are real. That split is deliberate -- it makes
the loop's control flow and the tool layer's error handling the things under
test, rather than a mock of either.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from nfl_platform.exceptions import DatasetNotAvailableError
from nfl_platform.llm.client import ChatMessage
from nfl_platform.llm.registry import ToolRegistry, build_default_registry
from nfl_platform.llm.results import ToolResult
from nfl_platform.llm.tools.base import Tool, ToolArgs
from nfl_platform.services import ServiceContainer
from support import ALLEN_ID, MAHOMES_ID, UNKNOWN_PLAYER_ID

from .conftest import AgentFactory, answer_completion, tool_call_completion

QUESTION = "How has Patrick Mahomes played recently?"


class BrokenToolArgs(ToolArgs):
    pass


class BrokenTool(Tool[BrokenToolArgs]):
    """Stands in for an upstream outage so the loop's recovery path is exercised."""

    name: ClassVar[str] = "always_fails"
    description: ClassVar[str] = "A tool that always fails, used to test failure handling paths."
    args_model: ClassVar[type[ToolArgs]] = BrokenToolArgs

    def invoke(self, container: ServiceContainer, args: BrokenToolArgs) -> ToolResult:
        raise DatasetNotAvailableError("player_stats", "nflverse unreachable")


class TestToolSelection:
    def test_agent_calls_the_tool_the_model_selected(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                answer_completion("Mahomes has been productive."),
            ]
        )
        result = agent.ask(QUESTION)
        assert result.trace.tools_used == ["get_player_stats"]

    def test_tool_schemas_are_offered_to_the_model(self, make_agent: AgentFactory):
        agent = make_agent([answer_completion("No tools needed.")])
        agent.ask("Who are you?")
        offered = agent._client.tools_offered[0]
        assert {schema["function"]["name"] for schema in offered} == set(agent.registry.names)

    def test_multi_tool_question_executes_every_requested_call(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion(
                    [
                        ("get_fantasy_production", {"player": MAHOMES_ID}),
                        ("get_fantasy_production", {"player": ALLEN_ID}),
                    ]
                ),
                answer_completion("Both have been steady."),
            ]
        )
        result = agent.ask("Should I start Mahomes or Allen?")
        assert len(result.trace.tool_calls) == 2
        assert all(call.ok for call in result.trace.tool_calls)

    def test_answer_without_tools_records_no_evidence(self, make_agent: AgentFactory):
        agent = make_agent([answer_completion("I can help with NFL statistics.")])
        result = agent.ask("What can you do?")
        assert result.evidence == []
        assert not result.used_tools


class TestEvidence:
    def test_evidence_is_built_from_executed_calls(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                answer_completion("He threw well."),
            ]
        )
        result = agent.ask(QUESTION)
        assert [item.tool for item in result.evidence] == ["get_player_stats"]
        assert result.evidence[0].arguments == {"player": MAHOMES_ID}

    def test_failed_calls_do_not_become_evidence(self, make_agent: AgentFactory):
        """Evidence must cite data that actually arrived, not calls that were attempted."""
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": UNKNOWN_PLAYER_ID})]),
                answer_completion("I could not find that player."),
            ]
        )
        result = agent.ask(QUESTION)
        assert result.evidence == []
        assert result.trace.tool_calls[0].ok is False

    def test_the_model_cannot_fabricate_a_citation(self, make_agent: AgentFactory):
        """Evidence comes from the trace, so claiming a tool does not create one."""
        agent = make_agent(
            [answer_completion("According to get_matchup, the Chiefs are favoured.")]
        )
        result = agent.ask("Who wins?")
        assert result.evidence == []

    def test_duplicate_calls_are_reported_once(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion(
                    [
                        ("get_player_stats", {"player": MAHOMES_ID}),
                        ("get_player_stats", {"player": MAHOMES_ID}),
                    ]
                ),
                answer_completion("Steady."),
            ]
        )
        result = agent.ask(QUESTION)
        assert len(result.evidence) == 1


class TestRecovery:
    def test_agent_recovers_from_invalid_arguments(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID, "oops": 1})]),
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                answer_completion("Recovered and answered."),
            ]
        )
        result = agent.ask(QUESTION)
        assert result.trace.tool_calls[0].error_code == "invalid_arguments"
        assert result.trace.tool_calls[1].ok
        assert len(result.trace.successful_tool_calls) == 1

    def test_agent_recovers_from_an_ambiguous_player_name(self, make_agent: AgentFactory):
        """The flagship recovery path: candidates come back, the model picks one."""
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": "Mike Williams"})]),
                tool_call_completion([("search_players", {"name": "Mike Williams"})]),
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                answer_completion("Resolved the player and answered."),
            ]
        )
        result = agent.ask("How is Mike Williams doing?")
        assert result.trace.tool_calls[0].error_code == "ambiguous_player"
        assert result.trace.tool_calls[-1].ok

    def test_tool_error_is_given_to_the_model_not_raised(self, make_agent: AgentFactory):
        registry = build_default_registry(extra=[BrokenTool()])
        agent = make_agent(
            [
                tool_call_completion([("always_fails", {})]),
                answer_completion("That data is temporarily unavailable."),
            ],
            tools=registry,
        )
        result = agent.ask(QUESTION)
        assert result.trace.tool_calls[0].error_code == "data_source_unavailable"
        assert "unavailable" in result.answer.lower()

    def test_repeatedly_failing_tool_is_withdrawn(self, make_agent: AgentFactory):
        """A model that cannot satisfy a schema must not burn the whole budget."""
        registry = build_default_registry(extra=[BrokenTool()])
        agent = make_agent(
            [
                tool_call_completion([("always_fails", {})]),
                tool_call_completion([("always_fails", {})]),
                tool_call_completion([("always_fails", {})]),
                answer_completion("Giving up on that source."),
            ],
            tools=registry,
        )
        agent.ask(QUESTION)
        final_offer = agent._client.tools_offered[-1]
        assert "always_fails" not in {s["function"]["name"] for s in final_offer}

    def test_unknown_tool_name_does_not_end_the_run(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion([("get_vibes", {})]),
                answer_completion("Answered without that tool."),
            ]
        )
        result = agent.ask(QUESTION)
        assert result.trace.tool_calls[0].error_code == "unknown_tool"
        assert result.answer


class TestNumericGrounding:
    def test_numbers_quoted_from_a_tool_are_grounded(
        self, make_agent: AgentFactory, registry: ToolRegistry, services: ServiceContainer
    ):
        truth = registry.execute(
            "get_player_stats", {"player": MAHOMES_ID, "last_n_games": 3}, services
        )
        yards = truth.data["totals"]["passing_yards"]
        agent = make_agent(
            [
                tool_call_completion(
                    [("get_player_stats", {"player": MAHOMES_ID, "last_n_games": 3})]
                ),
                answer_completion(f"He threw for {yards} yards over those games."),
            ]
        )
        result = agent.ask(QUESTION)
        assert result.trace.ungrounded_numbers == []
        assert result.trace.is_grounded

    def test_invented_numbers_are_flagged_on_the_trace(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                answer_completion("He threw for 9999 yards and 77 touchdowns."),
            ]
        )
        result = agent.ask(QUESTION)
        assert "9999" in result.trace.ungrounded_numbers
        assert not result.trace.is_grounded

    def test_flagging_does_not_block_the_answer(self, make_agent: AgentFactory):
        """Reported, not gated: regenerating costs a second run for a noisy signal."""
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                answer_completion("He threw for 9999 yards."),
            ]
        )
        result = agent.ask(QUESTION)
        assert "9999" in result.answer

    def test_an_answer_with_no_numbers_is_grounded(self, make_agent: AgentFactory):
        agent = make_agent([answer_completion("I do not have that information.")])
        result = agent.ask(QUESTION)
        assert result.trace.is_grounded


class TestBudgets:
    def test_iteration_budget_produces_a_final_answer(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                tool_call_completion([("get_player_usage", {"player": MAHOMES_ID})]),
                answer_completion("Here is what I found before running out of steps."),
            ],
            max_iterations=2,
        )
        result = agent.ask(QUESTION)
        assert result.trace.stop_reason == "iteration_budget_exhausted"
        assert result.trace.iterations == 2
        assert result.answer

    def test_the_final_call_offers_no_tools(self, make_agent: AgentFactory):
        """Withdrawing the tools is what forces the model to actually answer."""
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                answer_completion("Final."),
            ],
            max_iterations=1,
        )
        agent.ask(QUESTION)
        assert agent._client.tools_offered[-1] == []

    def test_tool_budget_stops_the_loop(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion(
                    [
                        ("get_player_stats", {"player": MAHOMES_ID}),
                        ("get_player_usage", {"player": MAHOMES_ID}),
                    ]
                ),
                answer_completion("Stopping after the tool budget."),
            ],
            max_tool_calls=2,
        )
        result = agent.ask(QUESTION)
        assert result.trace.stop_reason == "tool_budget_exhausted"
        assert len(result.trace.tool_calls) == 2

    def test_budget_exhaustion_still_returns_text(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                answer_completion("Partial answer with what I have."),
            ],
            max_iterations=1,
        )
        result = agent.ask(QUESTION)
        assert "Partial answer" in result.answer

    def test_empty_model_response_falls_back_to_an_honest_message(self, make_agent: AgentFactory):
        agent = make_agent([answer_completion("")])
        result = agent.ask(QUESTION)
        assert result.trace.stop_reason == "empty_answer"
        assert "could not produce an answer" in result.answer


class TestConversation:
    def test_system_prompt_is_sent_first(self, make_agent: AgentFactory):
        agent = make_agent([answer_completion("Hello.")])
        agent.ask(QUESTION)
        first = agent._client.calls[0][0]
        assert first.role == "system"
        assert "do not know NFL statistics" in first.content

    def test_runtime_context_states_the_current_season(self, make_agent: AgentFactory):
        agent = make_agent([answer_completion("Hello.")])
        agent.ask(QUESTION)
        context = agent._client.calls[0][1]
        assert "2024" in context.content

    def test_history_is_passed_through(self, make_agent: AgentFactory):
        agent = make_agent([answer_completion("Following up.")])
        history = [ChatMessage.user("Earlier question"), ChatMessage.assistant("Earlier answer")]
        agent.ask(QUESTION, history=history)
        contents = [message.content for message in agent._client.calls[0]]
        assert "Earlier question" in contents

    def test_tool_results_are_appended_as_tool_messages(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                answer_completion("Done."),
            ]
        )
        agent.ask(QUESTION)
        second_turn = agent._client.calls[1]
        tool_messages = [message for message in second_turn if message.role == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "call_1"

    def test_empty_question_is_rejected(self, make_agent: AgentFactory):
        agent = make_agent([answer_completion("Hello.")])
        with pytest.raises(ValueError, match="must not be empty"):
            agent.ask("   ")


class TestTraceContents:
    def test_tokens_and_latency_are_recorded(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                answer_completion("Done."),
            ]
        )
        result = agent.ask(QUESTION)
        assert result.trace.usage.total_tokens > 0
        assert result.trace.duration_ms > 0
        assert result.trace.tool_calls[0].duration_ms >= 0

    def test_cost_is_estimated_for_a_known_model(self, make_agent: AgentFactory):
        agent = make_agent([answer_completion("Done.")])
        result = agent.ask(QUESTION)
        assert result.trace.estimated_cost_usd > 0

    def test_prompt_version_is_recorded(self, make_agent: AgentFactory):
        agent = make_agent([answer_completion("Done.")])
        result = agent.ask(QUESTION)
        assert result.trace.prompt_version == "analyst_v1"

    def test_trace_serialises_for_logging(self, make_agent: AgentFactory):
        agent = make_agent(
            [
                tool_call_completion([("get_player_stats", {"player": MAHOMES_ID})]),
                answer_completion("Done."),
            ]
        )
        payload = agent.ask(QUESTION).trace.as_dict()
        assert payload["tools_used"] == ["get_player_stats"]
        assert payload["model"] == "gpt-4-turbo"
