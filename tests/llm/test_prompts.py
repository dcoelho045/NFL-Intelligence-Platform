"""Prompt loading and the rules the prompt must contain.

The content assertions are deliberate: the grounding rules are a functional
requirement of this milestone, so deleting them should break a test rather than
silently change the assistant's behaviour.
"""

from __future__ import annotations

import pytest

from nfl_platform.llm.prompts import (
    DEFAULT_PROMPT_VERSION,
    available_versions,
    load_prompt,
)


class TestPromptLoading:
    def test_default_prompt_loads(self):
        prompt = load_prompt()
        assert prompt.version == DEFAULT_PROMPT_VERSION
        assert prompt.text

    def test_version_is_recorded_on_the_prompt(self):
        assert load_prompt("analyst_v1").version == "analyst_v1"

    def test_unknown_version_raises_with_the_available_list(self):
        with pytest.raises(FileNotFoundError, match="analyst_v1"):
            load_prompt("analyst_v99")

    def test_available_versions_are_discoverable(self):
        assert "analyst_v1" in available_versions()

    def test_orchestrator_prompt_is_deferred_to_milestone_seven(self):
        """Nothing to orchestrate yet; adding it now would be dead configuration."""
        assert "orchestrator_v1" not in available_versions()


class TestPromptContent:
    @pytest.fixture
    def text(self) -> str:
        return load_prompt().text.lower()

    def test_forbids_stating_untooled_numbers(self, text: str):
        assert "never state a statistic you did not receive from a tool" in text

    def test_forbids_computing_statistics(self, text: str):
        assert "do not average, sum" in text

    def test_permits_admitting_missing_data(self, text: str):
        assert "i don't have that data" in text

    def test_explains_ambiguous_player_recovery(self, text: str):
        assert "ambiguous_player" in text

    def test_distinguishes_empty_results_from_failures(self, text: str):
        assert "empty list is data" in text

    def test_requires_stating_the_window(self, text: str):
        assert "window" in text

    def test_separates_judgement_from_quoted_numbers(self, text: str):
        assert "judgement is yours" in text
