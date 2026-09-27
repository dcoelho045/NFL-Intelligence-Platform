"""OpenAI chat-completions adapter.

Translates between this project's `ChatMessage`/`ToolCall` types and the SDK's
wire format, and converts SDK exceptions into :class:`LlmProviderError`. An
architecture test asserts that `openai` is imported nowhere else, which is what
keeps the rest of the codebase provably free of model calls and makes swapping
providers a single-file change.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import openai
from openai import OpenAI

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import LlmConfigurationError, LlmProviderError
from nfl_platform.llm.client import ChatMessage, Completion, ToolCall, Usage

logger = get_logger(__name__)

MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 1.0

# Transient by nature: worth one more try. Anything else fails immediately,
# because retrying a malformed request just spends money twice.
_RETRYABLE = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)

# Billing failures arrive as 429s, so they look transient and are not. Retrying
# an exhausted balance just delays the same answer three times over.
_TERMINAL_CODES = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "billing_hard_limit_reached",
        "account_deactivated",
    }
)


def _provider_message(exc: Exception) -> str:
    """The provider's own explanation, which is the actionable part.

    The SDK exposes ``body`` as the error object itself; the raw envelope nests
    it under ``error``. Both are accepted so this cannot silently fall back to
    the stringified exception.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        if body.get("message"):
            return str(body["message"])
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(exc)


def _is_terminal(exc: Exception) -> bool:
    return str(getattr(exc, "code", "") or "") in _TERMINAL_CODES


class OpenAIChatClient:
    """A :class:`~nfl_platform.llm.client.ChatClient` backed by OpenAI."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: OpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model = model or self._settings.llm_model
        if client is not None:
            self._client = client
            return
        api_key = self._settings.llm_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise LlmConfigurationError(
                "no API key found. Set NFL_LLM_API_KEY or OPENAI_API_KEY in your environment "
                "or .env file."
            )
        self._client = OpenAI(
            api_key=api_key.get_secret_value(),
            timeout=float(self._settings.llm_timeout_seconds),
        )

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]] = (),
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [_to_wire(message) for message in messages],
        }
        # Reasoning models reject any explicit temperature, so a blank setting
        # omits the key rather than sending a value they will refuse.
        if self._settings.llm_temperature is not None:
            payload["temperature"] = self._settings.llm_temperature
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        response = self._request(payload)
        choice = response.choices[0]
        return Completion(
            message=_from_wire(choice.message),
            usage=_usage(response),
            model=getattr(response, "model", self._model),
            finish_reason=choice.finish_reason,
        )

    def _request(self, payload: dict[str, Any]) -> Any:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self._client.chat.completions.create(**payload)
            except _RETRYABLE as exc:
                if _is_terminal(exc):
                    raise LlmProviderError(_provider_message(exc)) from exc
                if attempt == MAX_ATTEMPTS:
                    raise LlmProviderError(
                        f"{type(exc).__name__} after {MAX_ATTEMPTS} attempts: "
                        f"{_provider_message(exc)}",
                        retryable=True,
                    ) from exc
                delay = _BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Provider call failed, retrying",
                    extra={"attempt": attempt, "delay_seconds": delay, "error": type(exc).__name__},
                )
                time.sleep(delay)
            except openai.AuthenticationError as exc:
                raise LlmConfigurationError("the API key was rejected by the provider.") from exc
            except openai.NotFoundError as exc:
                raise LlmConfigurationError(
                    f"model {self._model!r} is unavailable: {_provider_message(exc)}"
                ) from exc
            except openai.OpenAIError as exc:
                raise LlmProviderError(_provider_message(exc)) from exc
        raise LlmProviderError(f"no response after {MAX_ATTEMPTS} attempts.", retryable=True)


def _to_wire(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role}
    # An assistant turn that only calls tools has no content, and the API rejects
    # an explicit null there, so the key is omitted entirely.
    if message.content is not None:
        payload["content"] = message.content
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.role == "tool" and message.content is None:
        payload["content"] = ""
    return payload


def _from_wire(message: Any) -> ChatMessage:
    raw_calls = getattr(message, "tool_calls", None) or []
    calls = tuple(
        ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments or "{}")
        for call in raw_calls
        if getattr(call, "function", None) is not None
    )
    return ChatMessage(role="assistant", content=message.content, tool_calls=calls)


def _usage(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
