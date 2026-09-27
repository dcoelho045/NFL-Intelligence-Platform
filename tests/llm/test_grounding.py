"""The numeric grounding detector."""

from __future__ import annotations

import pytest

from nfl_platform.llm.grounding import available_numbers, check_grounding, extract_numbers
from nfl_platform.llm.results import ToolResult


def result(data: dict) -> ToolResult:
    return ToolResult.success("get_player_stats", data)


class TestNumberExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("He threw for 312 yards.", [312.0]),
            ("An average of 24.5 points.", [24.5]),
            ("Exactly 1,234 yards.", [1234.0]),
            ("A 45% target share.", [45.0]),
            ("No numbers here at all.", []),
        ],
    )
    def test_numbers_are_found(self, text: str, expected: list[float]):
        assert [value for _, value, _ in extract_numbers(text)] == expected

    def test_player_ids_are_not_treated_as_claims(self):
        assert extract_numbers("Player 00-0033873 played.") == []

    def test_iso_dates_are_not_treated_as_claims(self):
        assert extract_numbers("Played on 2024-09-15.") == []

    def test_percent_sign_is_detected(self):
        assert extract_numbers("A 45% share.")[0][2] is True


class TestAvailableNumbers:
    def test_nested_values_are_collected(self):
        numbers = available_numbers([result({"totals": {"passing_yards": 312}})])
        assert 312.0 in numbers

    def test_values_inside_lists_are_collected(self):
        numbers = available_numbers([result({"games": [{"week": 3}, {"week": 4}]})])
        assert {3.0, 4.0} <= numbers

    def test_booleans_are_not_treated_as_numbers(self):
        assert available_numbers([result({"flag": True})]) == set()

    def test_numbers_inside_notes_are_collected(self):
        """Notes state real totals, so the model may legitimately quote them."""
        tool_result = ToolResult.success("t", {}, ["showing 10 of 17 entries"])
        assert {10.0, 17.0} <= available_numbers([tool_result])


class TestGroundingChecks:
    def test_quoted_number_is_grounded(self):
        report = check_grounding("He threw for 312 yards.", [result({"passing_yards": 312})])
        assert report.is_grounded

    def test_invented_number_is_flagged(self):
        report = check_grounding("He threw for 999 yards.", [result({"passing_yards": 312})])
        assert report.ungrounded == ("999",)

    def test_rounding_a_tool_value_is_accepted(self):
        report = check_grounding("About 24.5 points.", [result({"avg": 24.5137})])
        assert report.is_grounded

    def test_integer_rounding_is_accepted(self):
        report = check_grounding("Around 312 yards.", [result({"yards": 312.4})])
        assert report.is_grounded

    def test_share_expressed_as_a_percentage_is_accepted(self):
        """Shares are stored as 0-1 fractions but spoken as percentages."""
        report = check_grounding("A 28% target share.", [result({"target_share": 0.28})])
        assert report.is_grounded

    def test_conventional_rounding_of_a_half_is_accepted(self):
        """Python rounds ties to even, so round(129.25, 1) is 129.2, not 129.3."""
        report = check_grounding("129.3 rushing yards.", [result({"avg": 129.25})])
        assert report.is_grounded

    def test_either_direction_of_a_tie_is_accepted(self):
        report = check_grounding("129.2 rushing yards.", [result({"avg": 129.25})])
        assert report.is_grounded

    def test_a_large_number_is_not_read_as_a_percentage_of_a_counting_stat(self):
        """Only 0-1 fractions may be read as percentages; 5 targets is not 500%."""
        report = check_grounding("500 yards.", [result({"targets": 5})])
        assert report.ungrounded == ("500",)

    def test_numbers_from_the_question_are_available(self):
        """Echoing the season the user asked about is not a fabricated statistic."""
        report = check_grounding(
            "I found no player by that name in 2024.", [], "Who rushed for most yards in 2024?"
        )
        assert report.is_grounded

    def test_an_answer_with_no_tool_results_flags_every_number(self):
        report = check_grounding("He threw for 312 yards.", [])
        assert report.ungrounded == ("312",)

    def test_an_answer_with_no_numbers_is_grounded(self):
        report = check_grounding("I do not have that data.", [])
        assert report.is_grounded
        assert report.checked == ()

    def test_repeated_invented_number_is_reported_once(self):
        report = check_grounding("999 and again 999.", [result({})])
        assert report.ungrounded == ("999",)

    def test_derived_arithmetic_is_flagged(self):
        """The prompt forbids the model computing values, so a sum is a finding."""
        report = check_grounding("300 plus 200 makes 500 yards.", [result({"a": 300, "b": 200})])
        assert "500" in report.ungrounded
