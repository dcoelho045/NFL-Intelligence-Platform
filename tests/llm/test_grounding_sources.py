"""Three-bucket grounding and citation validation.

The load-bearing test here is
``TestMilestoneThreeBehaviourIsUnchanged``: partitioning numbers by source is a
change to existing Milestone 3 code, and it is only safe if a run that uses no
retrieval produces exactly the report it produced before.
"""

from __future__ import annotations

import pytest

from nfl_platform.llm.grounding import (
    available_numbers,
    check_citations,
    check_grounding,
    collect_sources,
    extract_numbers,
    partition_by_source,
)
from nfl_platform.llm.results import ToolResult
from nfl_platform.schemas.evidence import SourceKind


def structured(data: dict[str, object]) -> ToolResult:
    return ToolResult.success("get_player_stats", data, source_kind=SourceKind.STRUCTURED)


def web(results: list[dict[str, object]], **extra: object) -> ToolResult:
    return ToolResult.success(
        "search_web",
        {
            "query": "q",
            "provider": "stub",
            "result_count": len(results),
            "results": results,
            **extra,
        },
        source_kind=SourceKind.WEB,
    )


def knowledge(results: list[dict[str, object]]) -> ToolResult:
    return ToolResult.success(
        "search_knowledge_base",
        {"query": "q", "result_count": len(results), "min_score": 0.25, "results": results},
        source_kind=SourceKind.KNOWLEDGE_BASE,
    )


WEB_ARTICLE = {
    "url": "https://www.espn.com/nfl/story/_/id/1/x",
    "title": "Practice report",
    "domain": "espn.com",
    "snippet": "He gained 312 yards in a game against Denver back in 2019.",
    "retrieved_at": "2026-09-27T12:00:00+00:00",
}

KB_CHUNK = {
    "document_id": "fantasy-scoring-formats",
    "chunk_id": "fantasy-scoring-formats#000",
    "title": "Fantasy Scoring Formats",
    "text": "PPR means points per reception.",
    "document_type": "reference",
    "source": "authored",
    "score": 0.8,
}


class TestMilestoneThreeBehaviourIsUnchanged:
    """With no retrieval results, the report must be exactly what it always was."""

    def test_a_grounded_answer_is_still_grounded(self):
        report = check_grounding("He threw for 289 yards.", [structured({"passing_yards": 289})])
        assert report.is_grounded
        assert report.ungrounded == ()
        assert report.from_unstructured == ()

    def test_a_fabricated_number_is_still_caught(self):
        report = check_grounding("He threw for 312 yards.", [structured({"passing_yards": 289})])
        assert not report.is_grounded
        assert "312" in report.ungrounded

    def test_numbers_from_the_question_still_count(self):
        report = check_grounding(
            "In 2024 he threw for 289 yards.",
            [structured({"passing_yards": 289})],
            question="How did he do in 2024?",
        )
        assert report.is_grounded

    def test_rounding_tolerance_is_unchanged(self):
        report = check_grounding("He averaged 129.3 yards.", [structured({"avg": 129.25})])
        assert report.is_grounded

    def test_fraction_spoken_as_a_percentage_is_unchanged(self):
        report = check_grounding("A 25% target share.", [structured({"target_share": 0.25})])
        assert report.is_grounded

    def test_an_answer_with_no_numbers_is_grounded(self):
        assert check_grounding("I could not find that.", []).is_grounded


class TestThreeBuckets:
    def test_a_statistic_from_analytics_is_grounded(self):
        report = check_grounding(
            "He threw for 289 yards.",
            [structured({"passing_yards": 289}), web([WEB_ARTICLE])],
        )
        assert report.is_grounded
        assert report.from_unstructured == ()

    def test_a_number_only_in_web_text_is_bucketed_separately(self):
        report = check_grounding("Reports mention 312 yards.", [web([WEB_ARTICLE])])
        assert report.from_unstructured == ("312",)
        assert report.ungrounded == ()

    def test_web_sourced_numbers_do_not_break_grounding(self):
        """They are attributable, so they are recorded rather than treated as invention."""
        assert check_grounding("Reports mention 312 yards.", [web([WEB_ARTICLE])]).is_grounded

    def test_a_number_in_no_source_at_all_is_ungrounded(self):
        report = check_grounding("He threw for 999 yards.", [web([WEB_ARTICLE])])
        assert report.ungrounded == ("999",)
        assert not report.is_grounded

    def test_web_prose_cannot_launder_a_fabricated_statistic(self):
        """The reason this partitioning exists.

        Analytics says 289. The model says 312. An unrelated article happens to
        contain 312. Pooling the two would certify the fabrication.
        """
        results = [structured({"passing_yards": 289}), web([WEB_ARTICLE])]
        report = check_grounding("He threw for 312 yards.", results)
        assert "312" in report.from_unstructured
        assert "312" not in report.ungrounded
        # The key property: it is not reported as analytics-grounded.
        structured_numbers, _ = partition_by_source(results)
        assert 312.0 not in structured_numbers

    def test_knowledge_base_text_is_also_unstructured(self):
        chunk = {
            "document_id": "fantasy-scoring-formats",
            "chunk_id": "fantasy-scoring-formats#000",
            "title": "Fantasy Scoring Formats",
            "text": "A passing touchdown is worth 4 points.",
            "document_type": "reference",
            "source": "authored",
            "score": 0.8,
        }
        report = check_grounding("A passing touchdown is 4 points.", [knowledge([chunk])])
        assert report.from_unstructured == ("4",)

    def test_partitioning_separates_the_pools(self):
        structured_numbers, unstructured_numbers = partition_by_source(
            [structured({"passing_yards": 289}), web([WEB_ARTICLE])]
        )
        assert 289.0 in structured_numbers
        assert 312.0 in unstructured_numbers
        assert 312.0 not in structured_numbers

    def test_available_numbers_still_pools_everything_when_asked(self):
        pooled = available_numbers([structured({"passing_yards": 289}), web([WEB_ARTICLE])])
        assert {289.0, 312.0} <= pooled


class TestNumberExtraction:
    def test_finds_plain_numbers(self):
        assert [literal for literal, _, _ in extract_numbers("312 yards")] == ["312"]

    def test_ignores_gsis_identifiers(self):
        assert extract_numbers("player 00-0033873") == []

    def test_ignores_iso_dates(self):
        assert extract_numbers("on 2024-09-15") == []

    def test_handles_thousands_separators(self):
        assert extract_numbers("4,183 yards")[0][1] == 4183.0

    def test_marks_percentages(self):
        assert extract_numbers("25%")[0][2] is True

    def test_a_trailing_comma_is_punctuation_not_a_separator(self):
        """'September 23, 2026' is a date, not the number 23,000-something."""
        literals = [literal for literal, _, _ in extract_numbers("September 23, 2026")]
        assert literals == ["23", "2026"]

    def test_a_comma_only_separates_when_three_digits_follow(self):
        assert [literal for literal, _, _ in extract_numbers("1,234 and 5, 6")] == [
            "1,234",
            "5",
            "6",
        ]


class TestQuotedPublicationDates:
    """A date restated from retrieval metadata is provenance, not a statistic."""

    def test_a_publication_date_counts_as_attributable(self):
        result = web(
            [
                {
                    **WEB_ARTICLE,
                    "published_at": "2026-09-23T22:00:00+00:00",
                }
            ]
        )
        report = check_grounding("Reported on September 23, 2026.", [result])
        assert report.ungrounded == ()
        assert set(report.from_unstructured) == {"23", "2026"}

    def test_dates_do_not_leak_into_the_structured_pool(self):
        """A date must never be able to ground a statistic."""
        structured_numbers, unstructured_numbers = partition_by_source(
            [
                structured({"passing_yards": 289}),
                web([{**WEB_ARTICLE, "published_at": "2026-09-23T22:00:00+00:00"}]),
            ]
        )
        assert 23.0 not in structured_numbers
        assert 23.0 in unstructured_numbers

    def test_structured_results_do_not_contribute_date_components(self):
        report = check_grounding(
            "On September 23 he played.", [structured({"gameday": "2026-09-23"})]
        )
        assert "23" in report.ungrounded


class TestCitationChecking:
    def test_a_retrieved_url_is_valid(self):
        report = check_citations(f"See {WEB_ARTICLE['url']} for details.", [web([WEB_ARTICLE])])
        assert report.is_valid
        assert report.fabricated == ()

    def test_an_invented_url_is_flagged(self):
        report = check_citations(
            "See https://www.espn.com/nfl/story/_/id/99999/invented for details.",
            [web([WEB_ARTICLE])],
        )
        assert not report.is_valid
        assert report.fabricated

    def test_an_invented_domain_is_flagged(self):
        report = check_citations("According to rotoworld.com he is fine.", [web([WEB_ARTICLE])])
        assert "rotoworld.com" in report.fabricated

    def test_a_retrieved_domain_named_without_a_scheme_is_valid(self):
        report = check_citations("According to espn.com he is fine.", [web([WEB_ARTICLE])])
        assert report.is_valid

    def test_an_answer_with_no_citations_is_valid(self):
        assert check_citations("He threw for 289 yards.", [structured({"a": 1})]).is_valid

    def test_citing_anything_with_no_retrieval_at_all_is_flagged(self):
        report = check_citations("See https://made-up.test/x", [structured({"a": 1})])
        assert not report.is_valid

    def test_trailing_punctuation_is_not_part_of_the_url(self):
        report = check_citations(f"See {WEB_ARTICLE['url']}.", [web([WEB_ARTICLE])])
        assert report.is_valid


class TestFabricatedAttribution:
    """Crediting a channel that was never called is worse than answering openly."""

    @pytest.mark.parametrize(
        "answer",
        [
            "According to the knowledge base, PPR means points per reception.",
            "Per the knowledge base, a reception is worth one point.",
            "The knowledge base says a reception is worth one point.",
            "Based on the knowledge base, half-PPR awards half a point.",
        ],
    )
    def test_claiming_the_knowledge_base_without_calling_it_is_flagged(self, answer: str):
        report = check_citations(answer, [structured({"passing_yards": 289})])
        assert not report.is_valid
        assert "claimed knowledge base source" in report.fabricated

    def test_claiming_the_knowledge_base_after_calling_it_is_valid(self):
        report = check_citations(
            "According to the knowledge base, PPR means points per reception.",
            [knowledge([KB_CHUNK])],
        )
        assert report.is_valid

    def test_claiming_web_reporting_without_searching_is_flagged(self):
        report = check_citations(
            "According to recent reports he is expected to play.",
            [structured({"passing_yards": 289})],
        )
        assert "claimed web source" in report.fabricated

    def test_claiming_web_reporting_after_searching_is_valid(self):
        report = check_citations(
            "According to recent reports he is expected to play.", [web([WEB_ARTICLE])]
        )
        assert report.is_valid

    def test_a_failed_retrieval_does_not_count_as_consulted(self):
        from nfl_platform.llm.errors import ToolErrorDetail

        failure = ToolResult.failure(
            "search_knowledge_base",
            ToolErrorDetail("knowledge_base_unavailable", "missing", recovery="x"),
        )
        report = check_citations("According to the knowledge base, PPR is...", [failure])
        assert not report.is_valid

    def test_ordinary_prose_is_not_treated_as_an_attribution(self):
        report = check_citations(
            "He is questionable, per the team's official injury report.",
            [structured({"passing_yards": 289})],
        )
        assert report.is_valid


class TestSourceCollection:
    def test_structured_tools_are_recorded_by_name(self):
        sources = collect_sources([structured({"passing_yards": 289})])
        assert sources == [{"kind": "structured", "label": "get_player_stats"}]

    def test_web_sources_carry_url_domain_and_timestamps(self):
        source = collect_sources([web([WEB_ARTICLE])])[0]
        assert source["kind"] == "web"
        assert source["url"] == WEB_ARTICLE["url"]
        assert source["domain"] == "espn.com"
        assert source["retrieved_at"]

    def test_knowledge_sources_carry_the_document_and_its_date(self):
        chunk = {
            "document_id": "fantasy-scoring-formats",
            "chunk_id": "fantasy-scoring-formats#000",
            "title": "Fantasy Scoring Formats",
            "text": "PPR means points per reception.",
            "document_type": "reference",
            "source": "authored",
            "updated": "2026-09-27",
            "score": 0.8,
        }
        source = collect_sources([knowledge([chunk])])[0]
        assert source["kind"] == "knowledge_base"
        assert source["document_id"] == "fantasy-scoring-formats"
        assert source["updated"] == "2026-09-27"

    def test_duplicate_sources_are_collapsed(self):
        sources = collect_sources([web([WEB_ARTICLE]), web([WEB_ARTICLE])])
        assert len(sources) == 1

    def test_failed_tool_calls_contribute_no_sources(self):
        from nfl_platform.llm.errors import ToolErrorDetail

        failure = ToolResult.failure(
            "search_web", ToolErrorDetail("web_search_error", "down", recovery="x")
        )
        assert collect_sources([failure]) == []

    def test_sources_mix_all_three_kinds(self):
        chunk = {
            "document_id": "d",
            "chunk_id": "d#000",
            "title": "T",
            "text": "x",
            "document_type": "reference",
            "source": "authored",
            "score": 0.5,
        }
        sources = collect_sources([structured({"a": 1}), web([WEB_ARTICLE]), knowledge([chunk])])
        assert {source["kind"] for source in sources} == {
            "structured",
            "web",
            "knowledge_base",
        }
