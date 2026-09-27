"""The assistant adapter: a tool-calling LLM over the Milestone 2 services.

This package is a sibling of ``api/``, not a layer above or below it. Both call
``ServiceContainer`` in-process. Nothing here computes a statistic; every number
the assistant can state arrives from a service call recorded in the run trace.
"""

from nfl_platform.llm.agent import AssistantAnswer, NflAnalystAgent
from nfl_platform.llm.client import ChatClient, ChatMessage, Completion, ToolCall, Usage
from nfl_platform.llm.registry import ToolRegistry, build_default_registry
from nfl_platform.llm.trace import RunTrace, ToolCallRecord

__all__ = [
    "AssistantAnswer",
    "ChatClient",
    "ChatMessage",
    "Completion",
    "NflAnalystAgent",
    "RunTrace",
    "ToolCall",
    "ToolCallRecord",
    "ToolRegistry",
    "Usage",
    "build_default_registry",
]
