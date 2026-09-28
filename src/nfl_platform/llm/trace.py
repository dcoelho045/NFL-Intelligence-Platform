"""Run traces: what the assistant did, what it cost, how long it took.

Milestone 3 requires basic latency and cost information. This is deliberately a
plain record rather than an observability stack -- Milestone 9 owns that. What
matters now is that the trace is the *authoritative* account of a run: the
evidence attached to an answer is derived from these records, not written by the
model, so a citation cannot be fabricated.

API keys never enter a trace. Tool arguments do, because they are needed to
reproduce a run and contain nothing sensitive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from nfl_platform.llm.client import Usage

# USD per 1M tokens, as (prompt, completion). Provider prices drift; these are
# estimates for reporting, not billing, and should be refreshed when the model
# changes. An unlisted model reports None rather than a wrong number.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
}
_TOKENS_PER_PRICING_UNIT = 1_000_000


def estimate_cost_usd(model: str, usage: Usage) -> float | None:
    """Approximate spend for one run, or None for an unpriced model."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None
    prompt_rate, completion_rate = pricing
    return (
        usage.prompt_tokens * prompt_rate + usage.completion_tokens * completion_rate
    ) / _TOKENS_PER_PRICING_UNIT


@dataclass
class RetrievalStats:
    """What retrieval cost this run.

    Web searches are billed per query and embeddings per token, so neither
    appears in the model's token counters. Counting them here is what lets
    retrieval spend be compared against model spend later.

    Declared locally rather than imported from the retrieval layer: the LLM
    layer reaches retrieval only through services, and an import here would be
    the first crack in that.
    """

    web_searches: int = 0
    web_cache_hits: int = 0
    web_results: int = 0
    web_results_dropped: int = 0
    knowledge_searches: int = 0
    knowledge_hits: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "web_searches": self.web_searches,
            "web_cache_hits": self.web_cache_hits,
            "web_results": self.web_results,
            "web_results_dropped": self.web_results_dropped,
            "knowledge_searches": self.knowledge_searches,
            "knowledge_hits": self.knowledge_hits,
        }

    def record(self, source_kind: str, payload: dict[str, Any] | None) -> None:
        """Fold one retrieval tool result into the totals."""
        if not payload:
            return
        if source_kind == "web":
            self.web_searches += 1
            self.web_cache_hits += 1 if payload.get("cache_hit") else 0
            self.web_results += int(payload.get("result_count") or 0)
            self.web_results_dropped += int(payload.get("dropped_count") or 0)
        elif source_kind == "knowledge_base":
            self.knowledge_searches += 1
            self.knowledge_hits += int(payload.get("result_count") or 0)


@dataclass
class ToolCallRecord:
    """One executed tool call."""

    tool: str
    arguments: dict[str, Any]
    ok: bool
    duration_ms: float
    error_code: str | None = None
    result: dict[str, Any] | None = None
    source_kind: str = "structured"

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": self.tool,
            "arguments": self.arguments,
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 1),
            "source_kind": self.source_kind,
        }
        if self.error_code:
            payload["error_code"] = self.error_code
        return payload


@dataclass
class RunTrace:
    """Everything one question produced."""

    model: str
    prompt_version: str
    question: str = ""
    iterations: int = 0
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)
    numbers_from_unstructured_sources: list[str] = field(default_factory=list)
    fabricated_sources: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieval: RetrievalStats = field(default_factory=RetrievalStats)
    stop_reason: str = "completed"
    duration_ms: float = 0.0
    _started: float = field(default_factory=time.perf_counter, repr=False)

    def record_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls.append(record)

    def finish(self, stop_reason: str = "completed") -> None:
        self.stop_reason = stop_reason
        self.duration_ms = (time.perf_counter() - self._started) * 1000

    @property
    def successful_tool_calls(self) -> list[ToolCallRecord]:
        return [call for call in self.tool_calls if call.ok]

    @property
    def tools_used(self) -> list[str]:
        """Distinct tool names in call order."""
        seen: list[str] = []
        for call in self.tool_calls:
            if call.tool not in seen:
                seen.append(call.tool)
        return seen

    @property
    def estimated_cost_usd(self) -> float | None:
        return estimate_cost_usd(self.model, self.usage)

    @property
    def is_grounded(self) -> bool:
        return not self.ungrounded_numbers

    @property
    def citations_are_valid(self) -> bool:
        return not self.fabricated_sources

    @property
    def retrieval_sources(self) -> list[dict[str, Any]]:
        """Only the web and knowledge-base entries, for reporting."""
        return [item for item in self.sources if item.get("kind") != "structured"]

    def as_dict(self) -> dict[str, Any]:
        cost = self.estimated_cost_usd
        return {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "iterations": self.iterations,
            "stop_reason": self.stop_reason,
            "duration_ms": round(self.duration_ms, 1),
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "estimated_cost_usd": None if cost is None else round(cost, 6),
            "tools_used": self.tools_used,
            "tool_calls": [call.as_dict() for call in self.tool_calls],
            "ungrounded_numbers": self.ungrounded_numbers,
            "numbers_from_unstructured_sources": self.numbers_from_unstructured_sources,
            "fabricated_sources": self.fabricated_sources,
            "sources": self.sources,
            "retrieval": self.retrieval.as_dict(),
        }
