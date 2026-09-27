"""Every domain exception maps to a payload a model can act on."""

from __future__ import annotations

import pytest

from nfl_platform.exceptions import (
    AmbiguousPlayerError,
    DatasetNotAvailableError,
    InvalidGameCountError,
    InvalidSeasonError,
    InvalidWeekError,
    NflPlatformError,
    PlayerNotFoundError,
    SchemaValidationError,
    TeamNotFoundError,
    UnknownDatasetError,
    UnknownToolError,
    UnresolvedPlayersError,
)
from nfl_platform.llm.errors import MAX_CANDIDATES, describe_exception

CASES: list[tuple[Exception, str]] = [
    (AmbiguousPlayerError("Mike Williams", (("1", "WR"), ("2", "WR"))), "ambiguous_player"),
    (
        UnresolvedPlayersError((("Josh Allen", (("1", "QB"), ("2", "C"))),), ("Nobody",)),
        "unresolved_players",
    ),
    (PlayerNotFoundError("00-9999999", "player stats"), "player_not_found"),
    (TeamNotFoundError("XYZ", "schedules", ("KC", "BUF")), "team_not_found"),
    (InvalidSeasonError(1850, 1999, 2024), "invalid_season"),
    (InvalidWeekError(99), "invalid_week"),
    (InvalidGameCountError(-1), "invalid_game_count"),
    (UnknownToolError("nope", ("a", "b")), "unknown_tool"),
    (DatasetNotAvailableError("player_stats", "offline"), "data_source_unavailable"),
    (UnknownDatasetError("bogus", frozenset({"a"})), "unknown_dataset"),
    (SchemaValidationError("player_stats", frozenset({"col"})), "schema_validation_failed"),
    (NflPlatformError("generic"), "internal_error"),
    (ValueError("bad argument"), "invalid_request"),
]


class TestExceptionMapping:
    @pytest.mark.parametrize(("exc", "code"), CASES, ids=[code for _, code in CASES])
    def test_exception_maps_to_expected_code(self, exc: Exception, code: str):
        assert describe_exception(exc).code == code

    @pytest.mark.parametrize(("exc", "code"), CASES, ids=[code for _, code in CASES])
    def test_every_error_carries_a_recovery_hint(self, exc: Exception, code: str):
        """The hint is what turns a dead end into a corrected second call."""
        assert describe_exception(exc).recovery.strip()

    @pytest.mark.parametrize(("exc", "code"), CASES, ids=[code for _, code in CASES])
    def test_payload_is_a_flat_dict(self, exc: Exception, code: str):
        payload = describe_exception(exc).as_dict()
        assert payload["code"] == code
        assert isinstance(payload["retryable"], bool)


class TestUnexpectedExceptions:
    def test_unknown_exception_becomes_a_generic_error(self):
        assert describe_exception(RuntimeError("boom")).code == "tool_error"

    def test_internals_are_not_leaked_into_the_prompt(self):
        """Tracebacks and paths must never reach a model context."""
        detail = describe_exception(RuntimeError("/secret/path/module.py line 42"))
        assert "secret" not in detail.message
        assert "secret" not in str(detail.context)


class TestDisambiguationPayload:
    def test_candidates_are_structured_for_selection(self):
        exc = AmbiguousPlayerError("Mike Williams", (("00-1", "WR, LAC"), ("00-2", "WR, NYJ")))
        context = describe_exception(exc).context
        assert context["candidates"][0] == {"player_id": "00-1", "description": "WR, LAC"}

    def test_candidate_list_is_capped_but_the_total_is_reported(self):
        candidates = tuple((f"00-{i}", f"WR {i}") for i in range(30))
        context = describe_exception(AmbiguousPlayerError("Many", candidates)).context
        assert len(context["candidates"]) == MAX_CANDIDATES
        assert context["total_candidates"] == 30

    def test_batch_failure_reports_every_name_together(self):
        """One retry must be able to fix the whole call, not one name per turn."""
        exc = UnresolvedPlayersError(
            (
                ("Josh Allen", (("00-1", "QB, BUF"), ("00-2", "C, TB"))),
                ("Lamar Jackson", (("00-3", "QB, BAL"), ("00-4", "CB, ATL"))),
            ),
            ("Nobody At All",),
        )
        context = describe_exception(exc).context
        assert [item["query"] for item in context["ambiguous"]] == [
            "Josh Allen",
            "Lamar Jackson",
        ]
        assert context["ambiguous"][0]["candidates"][0]["player_id"] == "00-1"
        assert context["not_found"] == ["Nobody At All"]

    def test_batch_failure_tells_the_model_to_fix_everything_at_once(self):
        exc = UnresolvedPlayersError((("Josh Allen", (("00-1", "QB"),)),), ())
        assert "one retry" in describe_exception(exc).recovery


class TestRetryability:
    def test_outage_is_marked_not_retryable(self):
        detail = describe_exception(DatasetNotAvailableError("player_stats", "offline"))
        assert detail.retryable is False

    def test_bad_arguments_are_marked_retryable(self):
        assert describe_exception(InvalidWeekError(99)).retryable is True

    def test_outage_recovery_forbids_estimating(self):
        detail = describe_exception(DatasetNotAvailableError("player_stats", "offline"))
        assert "not estimate" in detail.recovery
