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
    if result.evidence:
        tools = ", ".join(sorted({item.tool for item in result.evidence}))
        lines.append(f"\n[sources: {tools}]")
    else:
        lines.append("\n[no tools were called - this answer contains no verified statistics]")
    if show_trace and result.trace is not None:
        lines.append(json.dumps(result.trace.as_dict(), indent=2))
    if result.trace is not None and result.trace.ungrounded_numbers:
        lines.append(
            "[warning: these numbers were not found in any tool result: "
            + ", ".join(result.trace.ungrounded_numbers)
            + "]"
        )
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
