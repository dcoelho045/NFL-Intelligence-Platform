"""OpenAI adapter: wire translation, usage extraction, and failure mapping.

The SDK is stubbed. These tests are about the translation boundary -- the one
place where a provider's shape meets this project's types -- not about OpenAI.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.exceptions import LlmConfigurationError, LlmProviderError
from nfl_platform.llm.client import ChatMessage, ToolCall
from nfl_platform.llm.providers.openai_client import MAX_ATTEMPTS, OpenAIChatClient, _to_wire


def response(
    content: str | None = "Hello",
    tool_calls: list[Any] | None = None,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 4,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        model="gpt-4-turbo",
    )


def wire_tool_call(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=arguments))


class FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.payloads: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> Any:
        self.payloads.append(payload)
        outcome = self._outcomes[min(len(self.payloads) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeOpenAI:
    def __init__(self, outcomes: list[Any]) -> None:
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, llm_model="gpt-4-turbo", llm_api_key="test-key")


def client_with(outcomes: list[Any], settings: Settings) -> tuple[OpenAIChatClient, FakeOpenAI]:
    fake = FakeOpenAI(outcomes)
    return OpenAIChatClient(settings, client=fake), fake


class TestConfiguration:
    def test_missing_key_is_reported_clearly(self):
        bare = Settings(_env_file=None, llm_api_key=None)
        with pytest.raises(LlmConfigurationError, match="NFL_LLM_API_KEY"):
            OpenAIChatClient(bare)

    def test_blank_key_is_treated_as_missing(self):
        bare = Settings(_env_file=None, llm_api_key="   ")
        with pytest.raises(LlmConfigurationError):
            OpenAIChatClient(bare)

    def test_model_comes_from_settings(self, settings: Settings):
        client, _ = client_with([response()], settings)
        assert client.model == "gpt-4-turbo"

    def test_model_can_be_overridden(self, settings: Settings):
        client = OpenAIChatClient(settings, client=FakeOpenAI([]), model="gpt-4o")
        assert client.model == "gpt-4o"


class TestRequestTranslation:
    def test_messages_and_temperature_are_sent(self, settings: Settings):
        client, fake = client_with([response()], settings)
        client.complete([ChatMessage.user("Hi")])
        payload = fake.completions.payloads[0]
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]
        assert payload["temperature"] == settings.llm_temperature

    def test_blank_temperature_is_omitted_entirely(self):
        """Reasoning models reject an explicit temperature, even the default."""
        reasoning = Settings(_env_file=None, llm_api_key="k", llm_temperature=None)
        client, fake = client_with([response()], reasoning)
        client.complete([ChatMessage.user("Hi")])
        assert "temperature" not in fake.completions.payloads[0]

    def test_tools_are_sent_when_provided(self, settings: Settings):
        client, fake = client_with([response()], settings)
        schema = {"type": "function", "function": {"name": "t", "parameters": {}}}
        client.complete([ChatMessage.user("Hi")], [schema])
        payload = fake.completions.payloads[0]
        assert payload["tools"] == [schema]
        assert payload["tool_choice"] == "auto"

    def test_no_tool_keys_when_none_offered(self, settings: Settings):
        """The forced final turn relies on tools being genuinely absent."""
        client, fake = client_with([response()], settings)
        client.complete([ChatMessage.user("Hi")], [])
        assert "tools" not in fake.completions.payloads[0]

    def test_assistant_tool_call_turn_omits_null_content(self):
        message = ChatMessage.assistant(tool_calls=[ToolCall("id1", "t", "{}")])
        wire = _to_wire(message)
        assert "content" not in wire
        assert wire["tool_calls"][0]["function"]["name"] == "t"

    def test_tool_result_carries_its_call_id(self):
        wire = _to_wire(ChatMessage.tool_result("id1", "t", "{}"))
        assert wire["tool_call_id"] == "id1"
        assert wire["role"] == "tool"


class TestResponseTranslation:
    def test_text_response_is_translated(self, settings: Settings):
        client, _ = client_with([response("Mahomes played well.")], settings)
        completion = client.complete([ChatMessage.user("Hi")])
        assert completion.text == "Mahomes played well."
        assert completion.tool_calls == ()

    def test_tool_calls_are_translated(self, settings: Settings):
        raw = response(None, [wire_tool_call("get_player_stats", '{"player":"x"}')])
        client, _ = client_with([raw], settings)
        completion = client.complete([ChatMessage.user("Hi")])
        assert completion.tool_calls[0].name == "get_player_stats"
        assert completion.tool_calls[0].arguments == '{"player":"x"}'

    def test_missing_arguments_default_to_an_empty_object(self, settings: Settings):
        raw = response(None, [wire_tool_call("t", "")])
        client, _ = client_with([raw], settings)
        assert client.complete([ChatMessage.user("Hi")]).tool_calls[0].arguments == "{}"

    def test_usage_is_extracted(self, settings: Settings):
        client, _ = client_with([response(prompt_tokens=120, completion_tokens=30)], settings)
        usage = client.complete([ChatMessage.user("Hi")]).usage
        assert (usage.prompt_tokens, usage.completion_tokens) == (120, 30)

    def test_absent_usage_does_not_crash(self, settings: Settings):
        raw = response()
        raw.usage = None
        client, _ = client_with([raw], settings)
        assert client.complete([ChatMessage.user("Hi")]).usage.total_tokens == 0


class TestFailureHandling:
    @pytest.fixture(autouse=True)
    def _no_sleeping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nfl_platform.llm.providers.openai_client.time.sleep", lambda _: None)

    def _timeout(self) -> openai.APITimeoutError:
        return openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))

    def test_transient_failure_is_retried_then_succeeds(self, settings: Settings):
        client, fake = client_with([self._timeout(), response("Recovered")], settings)
        assert client.complete([ChatMessage.user("Hi")]).text == "Recovered"
        assert len(fake.completions.payloads) == 2

    def test_exhausted_retries_raise_a_provider_error(self, settings: Settings):
        client, fake = client_with([self._timeout()], settings)
        with pytest.raises(LlmProviderError, match="APITimeoutError"):
            client.complete([ChatMessage.user("Hi")])
        assert len(fake.completions.payloads) == MAX_ATTEMPTS

    def test_exhausted_retries_are_marked_retryable(self, settings: Settings):
        client, _ = client_with([self._timeout()], settings)
        with pytest.raises(LlmProviderError) as caught:
            client.complete([ChatMessage.user("Hi")])
        assert caught.value.retryable is True

    def test_non_transient_error_is_not_retried(self, settings: Settings):
        """Retrying a malformed request just spends the money twice."""
        client, fake = client_with([openai.OpenAIError("bad request")], settings)
        with pytest.raises(LlmProviderError):
            client.complete([ChatMessage.user("Hi")])
        assert len(fake.completions.payloads) == 1

    def _status_error(
        self, error_type: type, status_code: int, code: str, message: str
    ) -> Exception:
        # The SDK passes the error object itself as `body`, which is where it
        # reads `.code` from.
        body = {"message": message, "code": code, "type": code}
        return error_type(
            message,
            response=httpx.Response(
                status_code,
                request=httpx.Request("POST", "https://api.openai.com"),
                json={"error": body},
            ),
            body=body,
        )

    def test_exhausted_credits_are_not_retried(self, settings: Settings):
        """A 429 for billing looks transient and never is."""
        exhausted = self._status_error(
            openai.RateLimitError,
            429,
            "insufficient_quota",
            "You have no credits remaining.",
        )
        client, fake = client_with([exhausted], settings)
        with pytest.raises(LlmProviderError, match="no credits remaining"):
            client.complete([ChatMessage.user("Hi")])
        assert len(fake.completions.payloads) == 1

    def test_genuine_rate_limit_is_still_retried(self, settings: Settings):
        throttled = self._status_error(
            openai.RateLimitError, 429, "rate_limit_exceeded", "Slow down."
        )
        client, fake = client_with([throttled, response("Recovered")], settings)
        assert client.complete([ChatMessage.user("Hi")]).text == "Recovered"
        assert len(fake.completions.payloads) == 2

    def test_unavailable_model_is_a_configuration_error(self, settings: Settings):
        """A retired model is the operator's problem to fix, not a transient fault."""
        missing = self._status_error(
            openai.NotFoundError,
            404,
            "model_not_found",
            "The model `gpt-4-turbo` does not exist or you do not have access to it.",
        )
        client, fake = client_with([missing], settings)
        with pytest.raises(LlmConfigurationError, match="gpt-4-turbo"):
            client.complete([ChatMessage.user("Hi")])
        assert len(fake.completions.payloads) == 1

    def test_provider_explanation_survives_retry_exhaustion(self, settings: Settings):
        """'RateLimitError after 3 attempts' alone tells an operator nothing."""
        throttled = self._status_error(
            openai.RateLimitError, 429, "rate_limit_exceeded", "Slow down please."
        )
        client, _ = client_with([throttled], settings)
        with pytest.raises(LlmProviderError, match="Slow down please"):
            client.complete([ChatMessage.user("Hi")])
