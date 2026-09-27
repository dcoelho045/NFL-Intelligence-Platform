"""Fixtures for the assistant tests.

Only the model is faked. Tools execute for real against the committed Parquet
fixtures, so these tests exercise genuine validation, service calls and error
mapping rather than mocks of them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.llm.agent import NflAnalystAgent
from nfl_platform.llm.client import ChatMessage, Completion, ScriptedChatClient, ToolCall, Usage
from nfl_platform.llm.registry import ToolRegistry, build_default_registry
from nfl_platform.services import ServiceContainer

AgentFactory = Callable[..., NflAnalystAgent]


def tool_call_completion(
    calls: Sequence[tuple[str, dict[str, object] | str]],
    *,
    usage: Usage | None = None,
) -> Completion:
    """A model turn that requests one or more tools."""
    tool_calls = [
        ToolCall(
            id=f"call_{index}",
            name=name,
            arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
        )
        for index, (name, arguments) in enumerate(calls, start=1)
    ]
    return Completion(
        message=ChatMessage.assistant(tool_calls=tool_calls),
        usage=usage or Usage(prompt_tokens=100, completion_tokens=20),
        finish_reason="tool_calls",
    )


def answer_completion(text: str, *, usage: Usage | None = None) -> Completion:
    """A model turn that ends the loop."""
    return Completion(
        message=ChatMessage.assistant(content=text),
        usage=usage or Usage(prompt_tokens=150, completion_tokens=40),
        finish_reason="stop",
    )


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_registry()


@pytest.fixture
def make_agent(
    services: ServiceContainer, registry: ToolRegistry, settings: Settings
) -> AgentFactory:
    """Build an agent driven by a scripted sequence of model responses."""

    def factory(
        completions: Sequence[Completion],
        *,
        tools: ToolRegistry | None = None,
        max_iterations: int | None = None,
        max_tool_calls: int | None = None,
    ) -> NflAnalystAgent:
        run_settings = settings
        if max_iterations is not None or max_tool_calls is not None:
            run_settings = settings.model_copy(
                update={
                    "llm_max_iterations": max_iterations or settings.llm_max_iterations,
                    "llm_max_tool_calls": max_tool_calls or settings.llm_max_tool_calls,
                }
            )
        return NflAnalystAgent(
            client=ScriptedChatClient(completions, model="gpt-4-turbo"),
            container=services,
            registry=tools or registry,
            settings=run_settings,
        )

    return factory
