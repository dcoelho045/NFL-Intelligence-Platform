"""Measure the assistant on a fixed question set.

Milestone 3 requires basic latency and cost information to be recorded. This
runs the golden questions once against the configured model and prints a table
plus a JSON summary.

It is a *measurement*, not a comparison: the project is pinned to one model for
now. When a second model is added, the `--model` flag makes a comparison a
matter of running this twice.

    python scripts/llm_benchmark.py
    python scripts/llm_benchmark.py --model gpt-4o --output docs/llm-evaluation.json

Costs real money. Uses the committed fixtures so the data is stable and the only
moving part is the model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.data.ingestion import NflDataLoader, ParquetDataSource
from nfl_platform.exceptions import LlmError
from nfl_platform.llm.factory import build_agent, llm_is_configured
from nfl_platform.services import ServiceContainer

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "nflverse"
FIXTURE_SEASON = 2024


@dataclass(frozen=True)
class GoldenQuestion:
    """A question plus the tools a correct answer should have used.

    ``expected_tools`` is a set of acceptable choices, not a single right
    answer -- several tools can legitimately answer one question.
    """

    question: str
    expected_tools: frozenset[str]
    expects_numbers: bool = True


GOLDEN_SET: tuple[GoldenQuestion, ...] = (
    GoldenQuestion(
        "How many passing yards did Patrick Mahomes have in 2024?",
        frozenset({"get_player_stats", "get_player_game_log"}),
    ),
    GoldenQuestion(
        "How has Patrick Mahomes played over his last three games?",
        frozenset({"get_player_stats", "get_player_game_log"}),
    ),
    GoldenQuestion(
        "What is Patrick Mahomes' snap share and usage this season?",
        frozenset({"get_player_usage", "get_fantasy_usage_trend"}),
    ),
    GoldenQuestion(
        "Is Patrick Mahomes on the injury report?",
        frozenset({"get_player_injuries"}),
        expects_numbers=False,
    ),
    GoldenQuestion(
        "Should I start Patrick Mahomes or Josh Allen in PPR this week?",
        frozenset({"compare_fantasy_players", "get_fantasy_production", "compare_players"}),
    ),
    GoldenQuestion(
        "Compare Patrick Mahomes and Josh Allen this season.",
        frozenset({"compare_players", "compare_fantasy_players", "get_player_stats"}),
    ),
    GoldenQuestion(
        "How has the Kansas City offense performed recently?",
        frozenset({"get_team_stats"}),
    ),
    GoldenQuestion(
        "How well has Buffalo's defense played this season?",
        frozenset({"get_team_stats"}),
    ),
    GoldenQuestion(
        "What is Kansas City's recent record?",
        frozenset({"get_team_form", "get_recent_games"}),
    ),
    GoldenQuestion(
        "Who is more likely to win between Kansas City and Buffalo?",
        frozenset({"get_matchup", "get_team_stats", "get_team_form"}),
    ),
    GoldenQuestion(
        "Which teams and seasons do you have data for?",
        frozenset({"list_reference_data"}),
        expects_numbers=False,
    ),
    GoldenQuestion(
        "How many rushing yards did Zxqwv Nonexistent have in 2024?",
        frozenset(),
        expects_numbers=False,
    ),
)


def build_offline_agent(settings: Settings, model: str | None) -> Any:
    """An agent over committed fixtures, so only the model varies between runs."""
    run_settings = settings.model_copy(
        update={"default_season": FIXTURE_SEASON, **({"llm_model": model} if model else {})}
    )
    loader = NflDataLoader(
        source=ParquetDataSource(FIXTURE_DIR),
        settings=run_settings,
        current_season=FIXTURE_SEASON,
    )
    container = ServiceContainer(loader=loader, settings=run_settings)
    return build_agent(container=container, settings=run_settings)


def run_question(agent: Any, item: GoldenQuestion) -> dict[str, Any]:
    try:
        result = agent.ask(item.question)
    except LlmError as exc:
        return {"question": item.question, "error": str(exc)}

    trace = result.trace
    used = set(trace.tools_used)
    selected_well = bool(used & item.expected_tools) if item.expected_tools else not used
    return {
        "question": item.question,
        "tools_used": trace.tools_used,
        "expected_tools": sorted(item.expected_tools),
        "tool_selection_ok": selected_well,
        "grounded": trace.is_grounded,
        "ungrounded_numbers": trace.ungrounded_numbers,
        "iterations": trace.iterations,
        "stop_reason": trace.stop_reason,
        "duration_ms": round(trace.duration_ms, 1),
        "total_tokens": trace.usage.total_tokens,
        "estimated_cost_usd": trace.estimated_cost_usd,
        "answer": result.answer,
    }


def summarise(rows: list[dict[str, Any]], model: str, prompt_version: str) -> dict[str, Any]:
    scored = [row for row in rows if "error" not in row]
    latencies = [row["duration_ms"] for row in scored]
    return {
        "model": model,
        "prompt_version": prompt_version,
        "questions": len(rows),
        "errors": len(rows) - len(scored),
        "tool_selection_accuracy": _ratio(scored, "tool_selection_ok"),
        "grounding_rate": _ratio(scored, "grounded"),
        "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "total_tokens": sum(row["total_tokens"] for row in scored),
        "total_cost_usd": round(sum(row["estimated_cost_usd"] or 0 for row in scored), 4),
    }


def _ratio(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(1 for row in rows if row[key]) / len(rows), 3)


def print_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    print(f"\nModel: {summary['model']}   Prompt: {summary['prompt_version']}")
    print("-" * 92)
    print(f"{'question':<52}{'tools':<10}{'grounded':<10}{'ms':>8}{'tokens':>10}")
    print("-" * 92)
    for row in rows:
        if "error" in row:
            print(f"{row['question'][:50]:<52}{'ERROR':<10}{row['error'][:38]}")
            continue
        print(
            f"{row['question'][:50]:<52}"
            f"{('ok' if row['tool_selection_ok'] else 'MISS'):<10}"
            f"{('yes' if row['grounded'] else 'NO'):<10}"
            f"{row['duration_ms']:>8.0f}{row['total_tokens']:>10}"
        )
    print("-" * 92)
    for key in (
        "tool_selection_accuracy",
        "grounding_rate",
        "mean_latency_ms",
        "median_latency_ms",
        "total_tokens",
        "total_cost_usd",
        "errors",
    ):
        print(f"{key:<28}{summary[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Override the configured model.")
    parser.add_argument("--output", type=Path, help="Write the full JSON results here.")
    parser.add_argument(
        "--limit", type=int, help="Run only the first N questions, to keep spend down."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=8.0,
        help=(
            "Seconds to pause between questions. Each question costs 9-20k tokens because "
            "the tool schemas are resent every turn, so back-to-back runs trip per-minute "
            "token limits on lower account tiers."
        ),
    )
    args = parser.parse_args(argv)

    if not llm_is_configured():
        print("No API key configured. Set NFL_LLM_API_KEY or OPENAI_API_KEY.", file=sys.stderr)
        return 2

    agent = build_offline_agent(get_settings(), args.model)
    questions = GOLDEN_SET[: args.limit] if args.limit else GOLDEN_SET

    rows = []
    for index, item in enumerate(questions):
        if index and args.delay:
            time.sleep(args.delay)
        rows.append(run_question(agent, item))

    summary = summarise(rows, agent._client.model, agent.prompt_version)
    print_report(summary, rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
