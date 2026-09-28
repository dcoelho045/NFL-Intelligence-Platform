"""Retrieval benchmark.

Three suites, deliberately separated by what they cost:

*rag* -- offline and free. Hit rate, MRR, and negative controls against the
golden set. Runs in CI and is the number that decides whether a reranker or
query rewriting would earn its place.

*web* -- spends search credits. Checks provenance completeness, recency, source
quality and no-result behaviour against a live provider.

*routing* -- spends model tokens and search credits. Scores which evidence
source a real model chooses per question, including over-retrieval: calling web
search on a pure-statistics question is a failure, not a neutral outcome.

    python scripts/retrieval_benchmark.py                      # rag only (free)
    python scripts/retrieval_benchmark.py --query "what is WOPR"
    python scripts/retrieval_benchmark.py --suite web
    python scripts/retrieval_benchmark.py --suite routing --output docs/retrieval-evaluation.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.exceptions import RetrievalError
from nfl_platform.llm.factory import build_agent, llm_is_configured
from nfl_platform.retrieval.knowledge.retriever import KnowledgeRetriever
from nfl_platform.retrieval.web.provider import (
    NullSearchProvider,
    build_search_provider,
)
from nfl_platform.retrieval.web.search import WebSearchClient
from nfl_platform.services import ServiceContainer

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "docs" / "retrieval-golden.json"

# Domains we would consider authoritative for NFL reporting. Used only to
# describe a live result set, never to filter one.
QUALITY_DOMAINS = frozenset(
    {
        "nfl.com",
        "espn.com",
        "theathletic.com",
        "cbssports.com",
        "nbcsports.com",
        "profootballtalk.nbcsports.com",
        "si.com",
        "pff.com",
        "yahoo.com",
        "foxsports.com",
        "apnews.com",
        "reuters.com",
    }
)


@dataclass(frozen=True)
class RoutingCase:
    """One end-to-end question and the evidence it ought to use."""

    question: str
    expected: frozenset[str]
    # Questions where reaching for the web is a cost failure, not a style choice.
    web_is_wasteful: bool = False
    expects_abstention: bool = False
    # Some questions have more than one correct route. Scoring them would measure
    # model nondeterminism rather than routing quality.
    routing_scored: bool = True


ROUTING_CASES: tuple[RoutingCase, ...] = (
    RoutingCase(
        "How many passing yards did Patrick Mahomes have in 2024?",
        frozenset({"structured"}),
        web_is_wasteful=True,
    ),
    RoutingCase(
        "How did the Kansas City Chiefs defence perform in 2024?",
        frozenset({"structured"}),
        web_is_wasteful=True,
    ),
    RoutingCase(
        "Compare Patrick Mahomes and Josh Allen over their last five games of 2024.",
        frozenset({"structured"}),
        web_is_wasteful=True,
    ),
    RoutingCase(
        "What were Josh Allen's rushing touchdowns in the 2024 regular season?",
        frozenset({"structured"}),
        web_is_wasteful=True,
    ),
    RoutingCase("What does PPR scoring mean?", frozenset({"knowledge_base"}), web_is_wasteful=True),
    RoutingCase("What is target share?", frozenset({"knowledge_base"}), web_is_wasteful=True),
    RoutingCase(
        "What does a questionable injury designation mean?",
        frozenset({"knowledge_base"}),
        web_is_wasteful=True,
    ),
    RoutingCase(
        "Why does this platform use last N games rather than last N weeks?",
        frozenset({"knowledge_base"}),
        web_is_wasteful=True,
    ),
    RoutingCase("What is Patrick Mahomes' current injury status?", frozenset({"web"})),
    RoutingCase("What is the latest news about the Kansas City Chiefs?", frozenset({"web"})),
    RoutingCase("Has there been any NFL trade news this week?", frozenset({"web"})),
    RoutingCase(
        "Should I start Patrick Mahomes or Josh Allen this week?",
        frozenset({"structured", "web"}),
    ),
    RoutingCase(
        "What is Travis Kelce's recent production and is he healthy right now?",
        frozenset({"structured", "web"}),
    ),
    RoutingCase(
        "How many receiving yards did a player named Zzyzx Nonexistent have in 2024?",
        # Looking the player up before abstaining and abstaining directly are both
        # defensible, so routing is not scored here. What is scored is that no
        # number is invented.
        frozenset({"structured"}),
        web_is_wasteful=True,
        expects_abstention=True,
        routing_scored=False,
    ),
)

_ABSTENTION_MARKERS = (
    "could not find",
    "couldn't find",
    "no data",
    "not found",
    "do not have",
    "don't have",
    "unable to",
    "no record",
    "not available",
    "there are no",
    "there is no",
    "no such player",
    "does not appear",
    "did not appear",
)


def load_golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def build_retriever(settings: Settings) -> KnowledgeRetriever:
    return KnowledgeRetriever(settings=settings)


def run_rag_suite(settings: Settings) -> dict[str, Any]:
    """Offline, free, deterministic. The suite that decides tuning questions."""
    golden = load_golden()
    retriever = build_retriever(settings)
    rows: list[dict[str, Any]] = []
    reciprocal_ranks: list[float] = []

    for case in golden["positive"]:
        started = time.perf_counter()
        hits, _ = retriever.search(case["question"], top_k=5)
        duration_ms = (time.perf_counter() - started) * 1000
        found = [hit.chunk.document_id for hit in hits]
        expected = set(case["expected_documents"])

        rank = next(
            (index for index, document in enumerate(found, start=1) if document in expected), 0
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        rows.append(
            {
                "question": case["question"],
                "expected_documents": sorted(expected),
                "retrieved_documents": found,
                "hit_at_1": bool(found[:1]) and found[0] in expected,
                "hit_at_3": any(document in expected for document in found[:3]),
                "hit_at_5": any(document in expected for document in found[:5]),
                "rank": rank,
                "top_score": round(hits[0].score, 4) if hits else None,
                "duration_ms": round(duration_ms, 1),
            }
        )

    negatives: list[dict[str, Any]] = []
    for case in golden["negative"]:
        hits, _ = retriever.search(case["question"], top_k=5)
        negatives.append(
            {
                "question": case["question"],
                "reason": case["reason"],
                "returned": len(hits),
                "correctly_empty": not hits,
                "top_score": round(hits[0].score, 4) if hits else None,
            }
        )

    positives = len(rows)
    return {
        "suite": "rag",
        "summary": {
            "embedding_model": retriever.embedder.model,
            "dimensions": retriever.embedder.dimensions,
            "min_score": settings.rag_min_score,
            "questions": positives,
            "hit_at_1": round(sum(row["hit_at_1"] for row in rows) / positives, 4),
            "hit_at_3": round(sum(row["hit_at_3"] for row in rows) / positives, 4),
            "hit_at_5": round(sum(row["hit_at_5"] for row in rows) / positives, 4),
            "mrr": round(statistics.fmean(reciprocal_ranks), 4),
            "negative_controls": len(negatives),
            "negative_control_pass_rate": round(
                sum(row["correctly_empty"] for row in negatives) / max(len(negatives), 1), 4
            ),
            "median_latency_ms": round(statistics.median(row["duration_ms"] for row in rows), 1),
        },
        "results": rows,
        "negative_controls": negatives,
    }


def run_web_suite(settings: Settings) -> dict[str, Any]:
    """Live provider checks. Spends one credit per query."""
    provider = build_search_provider(settings)
    if isinstance(provider, NullSearchProvider):
        return {"suite": "web", "skipped": "no web-search provider is configured"}

    client = WebSearchClient(
        provider=provider,
        settings=settings.model_copy(update={"web_search_max_calls_per_run": 50}),
    )
    client.begin_run()

    queries = [
        ("NFL injury report this week", 7),
        ("NFL trade news", 7),
        ("NFL depth chart changes", 14),
        ("NFL coach press conference quotes", 7),
        ("zzzqqq nonexistent nfl player xyzzy plugh 1823", None),
    ]

    rows: list[dict[str, Any]] = []
    for query, recency in queries:
        started = time.perf_counter()
        try:
            results, stats = client.search(query, recency_days=recency)
            error = None
        except RetrievalError as exc:
            results, stats, error = [], None, str(exc)

        dated = [item for item in results if item.published_at is not None]
        rows.append(
            {
                "query": query,
                "recency_days": recency,
                "error": error,
                "result_count": len(results),
                "provenance_complete": all(
                    item.url and item.title and item.domain and item.retrieved_at
                    for item in results
                ),
                "with_publication_date": len(dated),
                "quality_domain_share": (
                    round(sum(item.domain in QUALITY_DOMAINS for item in results) / len(results), 3)
                    if results
                    else None
                ),
                "domains": sorted({item.domain for item in results}),
                "dropped": stats.results_dropped if stats else 0,
                "injection_flags": stats.injection_flags if stats else 0,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )

    answered = [row for row in rows if row["result_count"] > 0]
    return {
        "suite": "web",
        "summary": {
            "provider": provider.name,
            "queries": len(rows),
            "queries_with_results": len(answered),
            "provenance_complete_rate": round(
                sum(row["provenance_complete"] for row in rows) / len(rows), 4
            ),
            "publication_date_coverage": round(
                sum(row["with_publication_date"] for row in rows)
                / max(sum(row["result_count"] for row in rows), 1),
                4,
            ),
            "median_latency_ms": round(statistics.median(row["duration_ms"] for row in rows), 1),
            "estimated_credits": len(rows),
        },
        "results": rows,
    }


def run_routing_suite(settings: Settings) -> dict[str, Any]:
    """End-to-end routing, groundedness and over-retrieval. Spends tokens and credits."""
    if not llm_is_configured(settings):
        return {"suite": "routing", "skipped": "no LLM provider is configured"}

    agent = build_agent(container=ServiceContainer(settings=settings), settings=settings)
    rows: list[dict[str, Any]] = []

    for case in ROUTING_CASES:
        try:
            result = agent.ask(case.question)
        except Exception as exc:
            rows.append({"question": case.question, "error": f"{type(exc).__name__}: {exc}"})
            continue

        trace = result.trace
        assert trace is not None
        kinds = {call.source_kind for call in trace.tool_calls if call.ok}
        lowered = result.answer.lower()

        rows.append(
            {
                "question": case.question,
                "expected_sources": sorted(case.expected),
                "sources_used": sorted(kinds),
                "routing_ok": (bool(kinds & case.expected) if case.expected else not kinds),
                "routing_scored": case.routing_scored,
                "over_retrieved": case.web_is_wasteful and "web" in kinds,
                "grounded": trace.is_grounded,
                "ungrounded_numbers": trace.ungrounded_numbers,
                "numbers_from_unstructured": trace.numbers_from_unstructured_sources,
                "citations_valid": trace.citations_are_valid,
                "fabricated_sources": trace.fabricated_sources,
                "abstained": any(marker in lowered for marker in _ABSTENTION_MARKERS),
                "expects_abstention": case.expects_abstention,
                "web_searches": trace.retrieval.web_searches,
                "knowledge_searches": trace.retrieval.knowledge_searches,
                "tools_used": trace.tools_used,
                "duration_ms": round(trace.duration_ms, 1),
                "total_tokens": trace.usage.total_tokens,
                "estimated_cost_usd": trace.estimated_cost_usd,
                "answer": result.answer,
            }
        )

    scored = [row for row in rows if "error" not in row]
    routed = [row for row in scored if row["routing_scored"]]
    abstention_cases = [row for row in scored if row["expects_abstention"]]
    return {
        "suite": "routing",
        "summary": {
            "model": settings.llm_model,
            "prompt_version": agent.prompt_version,
            "questions": len(rows),
            "errors": len(rows) - len(scored),
            "routing_accuracy": _rate(row["routing_ok"] for row in routed),
            "routing_questions_scored": len(routed),
            "over_retrieval_rate": _rate(row["over_retrieved"] for row in scored),
            "grounding_rate": _rate(row["grounded"] for row in scored),
            "citation_validity_rate": _rate(row["citations_valid"] for row in scored),
            "abstention_rate": (
                _rate(row["abstained"] for row in abstention_cases) if abstention_cases else None
            ),
            "total_web_searches": sum(row["web_searches"] for row in scored),
            "total_tokens": sum(row["total_tokens"] for row in scored),
            "total_cost_usd": round(sum(row["estimated_cost_usd"] or 0.0 for row in scored), 6),
            "median_latency_ms": (
                round(statistics.median(row["duration_ms"] for row in scored), 1)
                if scored
                else None
            ),
        },
        "results": rows,
    }


def _rate(values: Iterable[object]) -> float:
    items = [bool(value) for value in values]
    return round(sum(items) / len(items), 4) if items else 0.0


def run_single_query(settings: Settings, query: str) -> int:
    """One-off knowledge lookup, for tuning the relevance floor without a model."""
    retriever = build_retriever(settings)
    hits, stats = retriever.search(query, top_k=settings.rag_top_k * 2, min_score=0.0)
    print(f"Query:     {query}")
    print(f"Embedding: {retriever.embedder.model} @ {retriever.embedder.dimensions}d")
    print(f"Floor:     {settings.rag_min_score} (not applied here; showing all)")
    print(f"Latency:   {stats.duration_ms:.1f} ms\n")
    for hit in hits:
        marker = " " if hit.score >= settings.rag_min_score else "x"
        section = f" / {hit.chunk.section}" if hit.chunk.section else ""
        print(f"[{marker}] {hit.score:.4f}  {hit.chunk.document_id}{section}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("rag", "web", "routing", "all"),
        default="rag",
        help="Which suite to run. 'rag' is free; the others spend money.",
    )
    parser.add_argument("--query", help="Run one knowledge-base lookup and exit.")
    parser.add_argument("--output", type=Path, help="Write the report as JSON.")
    parser.add_argument("--model", help="Override the chat model for the routing suite.")
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    if args.model:
        overrides["llm_model"] = args.model
    settings = get_settings().model_copy(update=overrides) if overrides else get_settings()

    if args.query:
        return run_single_query(settings, args.query)

    suites = ("rag", "web", "routing") if args.suite == "all" else (args.suite,)
    report: dict[str, Any] = {}
    for suite in suites:
        runner = {"rag": run_rag_suite, "web": run_web_suite, "routing": run_routing_suite}[suite]
        report[suite] = runner(settings)
        summary = report[suite].get("summary") or {"skipped": report[suite].get("skipped")}
        print(f"\n=== {suite} ===")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
