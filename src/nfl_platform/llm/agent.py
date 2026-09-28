"""The tool-calling loop.

Deliberately a plain `while`: the control flow *is* the thing Milestone 3 exists
to demonstrate, and a framework would hide it behind configuration.

Three properties are load-bearing:

*Bounded.* Iteration and tool-call budgets stop a model that loops. Hitting a
budget produces a truthful partial answer, never silence and never a guess.

*Recoverable.* A failed tool becomes a tool message, not an exception, so the
model gets a chance to fix its arguments. Only the provider being unreachable
ends a run.

*Trace-derived evidence.* The evidence list attached to an answer is built from
the tool calls that actually executed. The model cannot cite a tool it never
called, because it does not author the citations.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.llm.client import ChatClient, ChatMessage, ToolCall
from nfl_platform.llm.grounding import check_citations, check_grounding, collect_sources
from nfl_platform.llm.prompts import DEFAULT_PROMPT_VERSION, load_prompt
from nfl_platform.llm.registry import ToolRegistry, build_default_registry
from nfl_platform.llm.results import ToolResult
from nfl_platform.llm.trace import RunTrace, ToolCallRecord
from nfl_platform.services.container import ServiceContainer

logger = get_logger(__name__)

# After this many identical failures the tool is withdrawn for the run. Without
# it, a model that cannot satisfy a schema will burn the whole budget retrying.
MAX_REPEATED_FAILURES = 2

_BUDGET_NOTICE = (
    "You have reached the tool budget for this question. Answer now using only the "
    "tool results already returned, and state plainly what you could not retrieve."
)


@dataclass(frozen=True)
class Evidence:
    """One tool call that contributed to an answer."""

    tool: str
    arguments: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": self.arguments}


@dataclass
class AssistantAnswer:
    """The assistant's reply plus the record backing it."""

    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    trace: RunTrace | None = None
    messages: list[ChatMessage] = field(default_factory=list)

    @property
    def used_tools(self) -> bool:
        return bool(self.evidence)

    @property
    def sources(self) -> list[dict[str, Any]]:
        return [] if self.trace is None else self.trace.sources

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence": [item.as_dict() for item in self.evidence],
            "sources": self.sources,
            "trace": None if self.trace is None else self.trace.as_dict(),
        }


class NflAnalystAgent:
    """Answers NFL questions by calling deterministic analytics tools."""

    def __init__(
        self,
        client: ChatClient,
        container: ServiceContainer,
        *,
        registry: ToolRegistry | None = None,
        settings: Settings | None = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        self._client = client
        self._container = container
        self._registry = registry or build_default_registry()
        self._settings = settings or get_settings()
        self._prompt = load_prompt(prompt_version)

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def prompt_version(self) -> str:
        return self._prompt.version

    def ask(self, question: str, history: Sequence[ChatMessage] = ()) -> AssistantAnswer:
        """Answer one question, calling tools as the model requests them."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        # Web searches are budgeted per question, so the counter resets here
        # rather than per process.
        self._container.research.begin_run()

        trace = RunTrace(
            model=self._client.model,
            prompt_version=self._prompt.version,
            question=question,
        )
        messages: list[ChatMessage] = [
            ChatMessage.system(self._prompt.text),
            ChatMessage.system(self._runtime_context()),
            *history,
            ChatMessage.user(question),
        ]

        schemas = self._registry.schemas()
        results: list[ToolResult] = []
        evidence: list[Evidence] = []
        failure_counts: dict[str, int] = {}
        stop_reason = "completed"
        answer_text = ""

        while trace.iterations < self._settings.llm_max_iterations:
            trace.iterations += 1
            completion = self._client.complete(messages, schemas)
            trace.record_usage(completion.usage)
            messages.append(completion.message)

            if not completion.tool_calls:
                answer_text = completion.text
                break

            for call in completion.tool_calls:
                result, record = self._run_tool(call)
                trace.record_tool_call(record)
                trace.retrieval.record(record.source_kind, result.data if result.ok else None)
                results.append(result)
                messages.append(ChatMessage.tool_result(call.id, call.name, result.to_json()))
                if result.ok:
                    evidence.append(Evidence(tool=record.tool, arguments=record.arguments))
                else:
                    failure_counts[call.name] = failure_counts.get(call.name, 0) + 1

            schemas = self._remaining_schemas(failure_counts)

            if len(trace.tool_calls) >= self._settings.llm_max_tool_calls:
                stop_reason = "tool_budget_exhausted"
                answer_text = self._force_final_answer(messages, trace)
                break
        else:
            stop_reason = "iteration_budget_exhausted"
            answer_text = self._force_final_answer(messages, trace)

        if not answer_text.strip():
            answer_text = (
                "I could not produce an answer from the available data. "
                "Please rephrase the question or narrow it to a specific player, team or week."
            )
            # A budget reason explains *why* there is no text, so it outranks this.
            if stop_reason == "completed":
                stop_reason = "empty_answer"

        report = check_grounding(answer_text, results, question)
        citations = check_citations(answer_text, results)
        trace.ungrounded_numbers = list(report.ungrounded)
        trace.numbers_from_unstructured_sources = list(report.from_unstructured)
        trace.fabricated_sources = list(citations.fabricated)
        trace.sources = collect_sources(results)
        trace.finish(stop_reason)

        if not report.is_grounded:
            # Reported, never blocked: see nfl_platform.llm.grounding.
            logger.warning(
                "Answer contains numbers not present in any tool result",
                extra={"ungrounded": trace.ungrounded_numbers, "tools": trace.tools_used},
            )
        if not citations.is_valid:
            logger.warning(
                "Answer cites sources that were never retrieved",
                extra={"fabricated": trace.fabricated_sources},
            )

        return AssistantAnswer(
            answer=answer_text.strip(),
            evidence=self._deduplicate(evidence),
            trace=trace,
            messages=messages,
        )

    def _run_tool(self, call: ToolCall) -> tuple[ToolResult, ToolCallRecord]:
        started = time.perf_counter()
        result = self._registry.execute(call.name, call.arguments, self._container)
        duration_ms = (time.perf_counter() - started) * 1000
        record = ToolCallRecord(
            tool=call.name,
            arguments=self._safe_arguments(call.arguments),
            ok=result.ok,
            duration_ms=duration_ms,
            error_code=result.error_code,
            source_kind=str(result.source_kind),
        )
        return result, record

    def _remaining_schemas(self, failure_counts: dict[str, int]) -> list[dict[str, Any]]:
        """Withdraw tools that keep failing so the budget is not wasted on them."""
        exhausted = {
            name for name, count in failure_counts.items() if count >= MAX_REPEATED_FAILURES
        }
        if not exhausted:
            return self._registry.schemas()
        return [
            schema
            for schema in self._registry.schemas()
            if schema["function"]["name"] not in exhausted
        ]

    def _force_final_answer(self, messages: list[ChatMessage], trace: RunTrace) -> str:
        """Ask once more with no tools available, so the run ends with a real answer."""
        messages.append(ChatMessage.system(_BUDGET_NOTICE))
        completion = self._client.complete(messages, [])
        trace.record_usage(completion.usage)
        messages.append(completion.message)
        return completion.text

    def _runtime_context(self) -> str:
        """Facts the model would otherwise assume."""
        try:
            current_season = self._container.meta.current_season()
        except Exception:
            logger.info("Current season unavailable for runtime context", exc_info=True)
            current_season = None

        lines = [f"Today's date is {datetime.now(UTC).date().isoformat()}."]
        if current_season is not None:
            lines.append(
                f"The current NFL season in the available data is {current_season}. "
                "Treat 'this season' as that season unless the user says otherwise."
            )
        lines.append(
            "Tool arguments that omit seasons use the configured default season. "
            "Call list_reference_data if you need the seasons that actually have data."
        )
        lines.append(
            "Statistical data is cached and may lag live results by up to a day, so a "
            "just-finished game may not appear yet. Anything that could have changed "
            "since yesterday is a search_web question, not a statistics question."
        )
        lines.append(self._retrieval_context())
        return " ".join(lines)

    def _retrieval_context(self) -> str:
        """Tell the model which retrieval channels actually work here.

        Discovering that a capability is switched off by calling it wastes a
        turn, and a model that has burned a turn on an unavailable tool is more
        likely to answer from memory instead.
        """
        available: list[str] = []
        unavailable: list[str] = []

        (available if self._container.research.available else unavailable).append("search_web")
        (available if self._container.knowledge.available else unavailable).append(
            "search_knowledge_base"
        )

        if not unavailable:
            budget = self._settings.web_search_max_calls_per_run
            return f"You may run at most {budget} web search(es) for this question."
        return (
            f"{', '.join(unavailable)} is not configured on this deployment; do not call it. "
            "Answer from the analytics tools and say plainly what you could not check."
        )

    @staticmethod
    def _safe_arguments(raw: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}

    @staticmethod
    def _deduplicate(evidence: list[Evidence]) -> list[Evidence]:
        seen: set[str] = set()
        unique: list[Evidence] = []
        for item in evidence:
            key = json.dumps(item.as_dict(), sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique
