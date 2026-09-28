"""Interactive command line for the assistant.

    python -m nfl_platform.llm.cli
    python -m nfl_platform.llm.cli --question "How has Mahomes played recently?"

Kept dependency-free: no REPL framework, just stdin. The `--trace` flag prints
the tool calls and cost behind an answer, which is the fastest way to confirm by
hand that numbers came from tools.
"""

from __future__ import annotations

import argparse
import json
import sys

from nfl_platform.config.logging import configure_logging
from nfl_platform.config.settings import get_settings
from nfl_platform.exceptions import LlmConfigurationError, LlmError
from nfl_platform.llm.agent import AssistantAnswer, NflAnalystAgent
from nfl_platform.llm.client import ChatMessage
from nfl_platform.llm.factory import build_agent

BANNER = "NFL analyst assistant. Type a question, or 'exit' to quit."


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the NFL analyst assistant a question.")
    parser.add_argument("--question", "-q", help="Ask one question and exit.")
    parser.add_argument(
        "--trace", action="store_true", help="Print tool calls, tokens, and estimated cost."
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Treat every question independently instead of continuing the conversation.",
    )
    return parser.parse_args(argv)


def _render(result: AssistantAnswer, *, show_trace: bool) -> str:
    lines = ["", result.answer]
    trace = result.trace

    if trace is not None and trace.sources:
        lines.append("")
        for source in trace.sources:
            kind = source.get("kind")
            if kind == "structured":
                lines.append(f"  [analytics]  {source['label']}")
            elif kind == "web":
                when = source.get("published_at") or "undated"
                lines.append(f"  [web]        {source['label']}")
                lines.append(f"               {source.get('domain')} | {when}")
                lines.append(f"               {source.get('url')}")
            else:
                updated = source.get("updated") or "undated"
                lines.append(
                    f"  [knowledge]  {source['label']} "
                    f"({source.get('document_id')}, updated {updated})"
                )
    elif not result.evidence:
        lines.append("\n[no tools were called - this answer contains no verified statistics]")

    if trace is not None:
        retrieval = trace.retrieval
        if retrieval.web_searches or retrieval.knowledge_searches:
            lines.append(
                f"\n[retrieval: {retrieval.web_searches} web search(es), "
                f"{retrieval.web_results} result(s), "
                f"{retrieval.web_results_dropped} dropped; "
                f"{retrieval.knowledge_searches} kb search(es), "
                f"{retrieval.knowledge_hits} chunk(s)]"
            )
        if trace.numbers_from_unstructured_sources:
            lines.append(
                "[from retrieved text, not the analytics layer: "
                + ", ".join(trace.numbers_from_unstructured_sources)
                + "]"
            )
        if trace.fabricated_sources:
            lines.append(
                "[warning: these sources were never retrieved: "
                + ", ".join(trace.fabricated_sources)
                + "]"
            )
        if trace.ungrounded_numbers:
            lines.append(
                "[warning: these numbers were not found in any tool result: "
                + ", ".join(trace.ungrounded_numbers)
                + "]"
            )

    if show_trace and trace is not None:
        lines.append(json.dumps(trace.as_dict(), indent=2))
    return "\n".join(lines)


def _ask(agent: NflAnalystAgent, question: str, history: list[ChatMessage], *, trace: bool) -> None:
    try:
        result = agent.ask(question, history=history)
    except LlmError as exc:
        print(f"\nAssistant unavailable: {exc}", file=sys.stderr)
        return
    print(_render(result, show_trace=trace))
    history.append(ChatMessage.user(question))
    history.append(ChatMessage.assistant(result.answer))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(get_settings())

    try:
        agent = build_agent()
    except LlmConfigurationError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    history: list[ChatMessage] = []
    if args.question:
        _ask(agent, args.question, history, trace=args.trace)
        return 0

    print(BANNER)
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if not question:
            continue
        _ask(agent, question, [] if args.no_history else history, trace=args.trace)


if __name__ == "__main__":
    raise SystemExit(main())
