"""Opt-in tests against a live LLM provider.

Deselected by default and skipped without an API key, because they cost money
and depend on a third party. Run them with:

    pytest -m llm

They assert *properties* -- was a tool called, is every number traceable, is an
unanswerable question refused -- rather than exact wording, which no useful test
can pin down for a language model.
"""

from __future__ import annotations

import pytest

from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.llm.agent import NflAnalystAgent
from nfl_platform.llm.factory import build_agent, llm_is_configured
from nfl_platform.services import ServiceContainer

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not llm_is_configured(), reason="No LLM API key configured."),
]


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    """Real provider configuration, but pinned to the offline fixture season."""
    return get_settings().model_copy(update={"default_season": 2024})


@pytest.fixture(scope="module")
def live_agent(live_settings: Settings) -> NflAnalystAgent:
    """A real model over the committed fixture data.

    Fixtures rather than live nflverse keeps the assertions stable: the model is
    the only moving part.
    """
    from nfl_platform.data.ingestion import NflDataLoader, ParquetDataSource

    loader = NflDataLoader(
        source=ParquetDataSource("tests/fixtures/nflverse"),
        settings=live_settings,
        current_season=2024,
    )
    container = ServiceContainer(loader=loader, settings=live_settings)
    return build_agent(container=container, settings=live_settings)


class TestLiveToolUse:
    def test_statistical_question_triggers_a_tool_call(self, live_agent: NflAnalystAgent):
        result = live_agent.ask("How many passing yards did Patrick Mahomes have in 2024?")
        assert result.used_tools, "The model answered a statistical question from memory."

    def test_answer_is_numerically_grounded(self, live_agent: NflAnalystAgent):
        result = live_agent.ask("How many passing yards did Patrick Mahomes have in 2024?")
        assert result.trace.ungrounded_numbers == [], (
            f"Answer contains untraceable numbers: {result.trace.ungrounded_numbers}"
        )

    def test_comparison_question_uses_a_comparison_tool(self, live_agent: NflAnalystAgent):
        result = live_agent.ask(
            "Should I start Patrick Mahomes or Josh Allen in fantasy this week?"
        )
        assert result.used_tools
        assert any("compare" in tool or "fantasy" in tool for tool in result.trace.tools_used)

    def test_team_question_uses_a_team_tool(self, live_agent: NflAnalystAgent):
        result = live_agent.ask("How has the Kansas City defense performed recently?")
        assert any("team" in tool or "matchup" in tool for tool in result.trace.tools_used)

    def test_unknown_player_is_refused_not_invented(self, live_agent: NflAnalystAgent):
        """The decisive test: no data must produce no numbers."""
        result = live_agent.ask("How many rushing yards did Zxqwv Nonexistent have in 2024?")
        assert result.trace.ungrounded_numbers == []

    def test_cost_and_latency_are_recorded(self, live_agent: NflAnalystAgent):
        result = live_agent.ask("Who did Kansas City play most recently?")
        assert result.trace.usage.total_tokens > 0
        assert result.trace.duration_ms > 0
        assert result.trace.estimated_cost_usd is not None
