"""Provider-agnostic chat contract.

The agent talks to this protocol, never to a vendor SDK. Two things fall out of
that: the entire test suite runs offline against :class:`ScriptedChatClient`
with no key and no network, and swapping providers touches one directory.

These are plain dataclasses rather than Pydantic models because nothing here
crosses a validation boundary -- they are internal plumbing between the agent
and a provider adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke one tool.

    ``arguments`` is the raw JSON string the model produced. It is deliberately
    not parsed here: malformed JSON is a normal, recoverable event that the
    registry reports back to the model, so parsing belongs where that error can
    be handled.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Usage:
    """Token counts for one provider call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class ChatMessage:
    """One entry in the conversation sent to the model."""

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: Sequence[ToolCall] = ()
    ) -> ChatMessage:
        return cls(role="assistant", content=content, tool_calls=tuple(tool_calls))

    @classmethod
    def tool_result(cls, tool_call_id: str, name: str, content: str) -> ChatMessage:
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass(frozen=True)
class Completion:
    """One model response."""

    message: ChatMessage
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str | None = None

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.message.tool_calls

    @property
    def text(self) -> str:
        return self.message.content or ""


@runtime_checkable
class ChatClient(Protocol):
    """Anything that can turn messages plus tool schemas into a completion."""

    @property
    def model(self) -> str: ...

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]] = (),
    ) -> Completion: ...


class ScriptedChatClient:
    """A deterministic stand-in for a provider, used by the test suite.

    Takes the completions it should return, in order, and records what it was
    asked. Real tools still execute against real fixture data -- only the model
    is faked -- so these tests exercise validation, error mapping, and the loop
    itself rather than a mock of them.
    """

    def __init__(self, completions: Sequence[Completion], *, model: str = "scripted") -> None:
        if not completions:
            raise ValueError("ScriptedChatClient needs at least one completion.")
        self._completions = list(completions)
        self._model = model
        self.calls: list[list[ChatMessage]] = []
        self.tools_offered: list[list[dict[str, Any]]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]] = (),
    ) -> Completion:
        self.calls.append(list(messages))
        self.tools_offered.append(list(tools))
        index = min(len(self.calls) - 1, len(self._completions) - 1)
        return self._completions[index]


def scripted_tool_call(name: str, arguments: str, *, call_id: str = "call_1") -> Completion:
    """Build a completion whose only content is a single tool call."""
    call = ToolCall(id=call_id, name=name, arguments=arguments)
    return Completion(
        message=ChatMessage.assistant(tool_calls=[call]),
        finish_reason="tool_calls",
    )


def scripted_answer(text: str) -> Completion:
    """Build a completion that ends the loop with a final answer."""
    return Completion(message=ChatMessage.assistant(content=text), finish_reason="stop")
