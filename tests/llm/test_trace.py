"""Run traces, token accounting, and cost estimation."""

from __future__ import annotations

from nfl_platform.llm.client import Usage
from nfl_platform.llm.trace import MODEL_PRICING, RunTrace, ToolCallRecord, estimate_cost_usd


def record(tool: str, *, ok: bool = True, error: str | None = None) -> ToolCallRecord:
    return ToolCallRecord(tool=tool, arguments={}, ok=ok, duration_ms=1.0, error_code=error)


class TestUsage:
    def test_usage_totals(self):
        assert Usage(prompt_tokens=10, completion_tokens=5).total_tokens == 15

    def test_usage_accumulates_across_turns(self):
        total = Usage(10, 5) + Usage(20, 7)
        assert (total.prompt_tokens, total.completion_tokens) == (30, 12)


class TestCost:
    def test_known_model_is_priced(self):
        cost = estimate_cost_usd("gpt-4-turbo", Usage(1_000_000, 1_000_000))
        prompt_rate, completion_rate = MODEL_PRICING["gpt-4-turbo"]
        assert cost == prompt_rate + completion_rate

    def test_unknown_model_returns_none_rather_than_guessing(self):
        assert estimate_cost_usd("some-future-model", Usage(100, 100)) is None

    def test_zero_usage_costs_nothing(self):
        assert estimate_cost_usd("gpt-4-turbo", Usage()) == 0.0


class TestRunTrace:
    def test_tools_used_preserves_order_without_duplicates(self):
        trace = RunTrace(model="m", prompt_version="v")
        for tool in ("a", "b", "a", "c"):
            trace.record_tool_call(record(tool))
        assert trace.tools_used == ["a", "b", "c"]

    def test_successful_calls_are_separable_from_failures(self):
        trace = RunTrace(model="m", prompt_version="v")
        trace.record_tool_call(record("a"))
        trace.record_tool_call(record("b", ok=False, error="player_not_found"))
        assert [call.tool for call in trace.successful_tool_calls] == ["a"]

    def test_finish_records_duration_and_reason(self):
        trace = RunTrace(model="m", prompt_version="v")
        trace.finish("tool_budget_exhausted")
        assert trace.duration_ms > 0
        assert trace.stop_reason == "tool_budget_exhausted"

    def test_grounded_by_default(self):
        assert RunTrace(model="m", prompt_version="v").is_grounded

    def test_ungrounded_numbers_flip_the_flag(self):
        trace = RunTrace(model="m", prompt_version="v")
        trace.ungrounded_numbers = ["999"]
        assert not trace.is_grounded

    def test_as_dict_contains_the_reporting_fields(self):
        trace = RunTrace(model="gpt-4-turbo", prompt_version="analyst_v1")
        trace.record_usage(Usage(100, 50))
        trace.record_tool_call(record("get_player_stats"))
        trace.finish()
        payload = trace.as_dict()
        assert payload["total_tokens"] == 150
        assert payload["estimated_cost_usd"] > 0
        assert payload["tools_used"] == ["get_player_stats"]
        assert payload["stop_reason"] == "completed"

    def test_error_code_is_only_present_on_failures(self):
        assert "error_code" not in record("a").as_dict()
        assert record("a", ok=False, error="boom").as_dict()["error_code"] == "boom"

    def test_api_key_never_appears_in_a_trace(self):
        """Traces are logged and returned over HTTP; secrets must not travel."""
        trace = RunTrace(model="gpt-4-turbo", prompt_version="analyst_v1", question="hi")
        assert "api_key" not in str(trace.as_dict())
