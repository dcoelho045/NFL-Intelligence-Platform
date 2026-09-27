"""Assistant endpoints.

A thin adapter over :class:`~nfl_platform.llm.agent.NflAnalystAgent`, exactly as
the other routers are thin adapters over services. The agent is resolved from
application state so tests can inject one driven by a scripted model.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from nfl_platform.config.logging import get_logger
from nfl_platform.exceptions import LlmConfigurationError
from nfl_platform.llm.agent import AssistantAnswer, NflAnalystAgent
from nfl_platform.llm.client import ChatMessage
from nfl_platform.llm.registry import ToolRegistry, build_default_registry
from nfl_platform.schemas.assistant import (
    AssistantAnswerResponse,
    AssistantAskRequest,
    AssistantRunStats,
    EvidenceItem,
    ToolCatalogue,
    ToolDescription,
)
from nfl_platform.schemas.common import ErrorResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/assistant", tags=["assistant"])


def get_agent(request: Request) -> NflAnalystAgent:
    """Resolve the agent, building it on first use.

    Built lazily so the API still starts, and every non-assistant endpoint still
    works, on a machine with no API key configured.
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        from nfl_platform.llm.factory import build_agent

        agent = build_agent(
            container=request.app.state.container,
            settings=request.app.state.settings,
        )
        request.app.state.agent = agent
    return agent


AgentDep = Annotated[NflAnalystAgent, Depends(get_agent)]


def get_tool_registry(request: Request) -> ToolRegistry:
    """Resolve the tool catalogue without constructing a provider client.

    Listing the tools requires no model, so this endpoint must keep working on a
    machine with no API key -- it is the cheapest way to inspect the
    LLM/analytics boundary.
    """
    agent = getattr(request.app.state, "agent", None)
    return agent.registry if agent is not None else build_default_registry()


RegistryDep = Annotated[ToolRegistry, Depends(get_tool_registry)]


@router.post(
    "/ask",
    response_model=AssistantAnswerResponse,
    summary="Ask an NFL question",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "No LLM provider is configured.",
        }
    },
    description=(
        "Answers a natural-language NFL question by calling deterministic analytics tools. "
        "Every statistic in the answer comes from a tool result; the `evidence` list records "
        "which tools ran, and `stats.ungrounded_numbers` reports any number that could not be "
        "traced back to one."
    ),
)
def ask(request: AssistantAskRequest, agent: AgentDep) -> AssistantAnswerResponse:
    history = [ChatMessage(role=turn.role, content=turn.content) for turn in request.history]
    result = agent.ask(request.question, history=history)
    return _to_response(result)


@router.get(
    "/tools",
    response_model=ToolCatalogue,
    summary="List the tools the assistant can call",
    description=(
        "The tool catalogue with the JSON Schema for each tool's arguments. Requires no LLM "
        "provider, so it is usable for inspecting the LLM/analytics boundary."
    ),
)
def list_tools(registry: RegistryDep) -> ToolCatalogue:
    tools = [
        ToolDescription(
            name=schema["function"]["name"],
            description=schema["function"]["description"],
            parameters=schema["function"]["parameters"],
        )
        for schema in registry.schemas()
    ]
    return ToolCatalogue(count=len(tools), tools=tools)


def _to_response(result: AssistantAnswer) -> AssistantAnswerResponse:
    trace = result.trace
    if trace is None:  # pragma: no cover - the agent always attaches a trace
        raise LlmConfigurationError("the agent returned no run trace.")
    return AssistantAnswerResponse(
        answer=result.answer,
        evidence=[
            EvidenceItem(tool=item.tool, arguments=item.arguments) for item in result.evidence
        ],
        stats=AssistantRunStats(
            model=trace.model,
            prompt_version=trace.prompt_version,
            iterations=trace.iterations,
            stop_reason=trace.stop_reason,
            duration_ms=round(trace.duration_ms, 1),
            prompt_tokens=trace.usage.prompt_tokens,
            completion_tokens=trace.usage.completion_tokens,
            total_tokens=trace.usage.total_tokens,
            estimated_cost_usd=trace.estimated_cost_usd,
            tools_used=trace.tools_used,
            ungrounded_numbers=trace.ungrounded_numbers,
        ),
    )
