"""Tool result serialisation: truncation, null suppression, envelope shape."""

from __future__ import annotations

import json

from nfl_platform.llm.errors import ToolErrorDetail
from nfl_platform.llm.results import MAX_LIST_ITEMS, ToolResult, serialise_dto
from nfl_platform.schemas.common import PlayerRef, SeasonType, StatWindow
from nfl_platform.schemas.player import PlayerGameLog, PlayerGameStats, PlayerStatLine


def _game(week: int) -> PlayerGameStats:
    return PlayerGameStats(
        season=2024,
        week=week,
        team="KC",
        opponent="BUF",
        stats=PlayerStatLine(passing_yards=250, passing_tds=2),
    )


def _game_log(count: int) -> PlayerGameLog:
    return PlayerGameLog(
        player=PlayerRef(player_id="00-0033873", name="Patrick Mahomes", position="QB", team="KC"),
        window=StatWindow(seasons=[2024], season_type=SeasonType.ALL, games=count),
        games=[_game(week) for week in range(1, count + 1)],
    )


class TestSerialisation:
    def test_nulls_are_dropped(self):
        payload, _ = serialise_dto(_game_log(1))
        assert "carries" not in payload["games"][0]["stats"]

    def test_populated_values_survive(self):
        payload, _ = serialise_dto(_game_log(1))
        assert payload["games"][0]["stats"]["passing_yards"] == 250

    def test_short_lists_are_not_truncated(self):
        payload, notes = serialise_dto(_game_log(3))
        assert len(payload["games"]) == 3
        assert notes == []

    def test_long_lists_are_capped(self):
        payload, _ = serialise_dto(_game_log(MAX_LIST_ITEMS + 5))
        assert len(payload["games"]) == MAX_LIST_ITEMS

    def test_truncation_is_announced_with_the_true_total(self):
        """Silent truncation would let a subset be described as the whole window."""
        total = MAX_LIST_ITEMS + 5
        _, notes = serialise_dto(_game_log(total))
        assert len(notes) == 1
        assert str(total) in notes[0]
        assert "games" in notes[0]

    def test_output_is_json_serialisable(self):
        payload, _ = serialise_dto(_game_log(2))
        assert json.loads(json.dumps(payload))


class TestEnvelope:
    def test_success_envelope(self):
        result = ToolResult.from_dto("get_player_game_log", _game_log(2))
        body = json.loads(result.to_json())
        assert body["ok"] is True
        assert body["tool"] == "get_player_game_log"
        assert "error" not in body

    def test_failure_envelope(self):
        detail = ToolErrorDetail("boom", "It broke.", recovery="Try again.")
        result = ToolResult.failure("some_tool", detail)
        body = json.loads(result.to_json())
        assert body["ok"] is False
        assert body["error"]["code"] == "boom"
        assert body["error"]["recovery"] == "Try again."
        assert "data" not in body

    def test_error_code_accessor(self):
        detail = ToolErrorDetail("boom", "It broke.", recovery="Try again.")
        assert ToolResult.failure("t", detail).error_code == "boom"
        assert ToolResult.success("t", {}).error_code is None

    def test_notes_appear_only_when_present(self):
        assert "notes" not in json.loads(ToolResult.success("t", {"a": 1}).to_json())
        assert "notes" in json.loads(ToolResult.success("t", {}, ["note"]).to_json())

    def test_retryable_flag_is_exposed_to_the_model(self):
        detail = ToolErrorDetail("boom", "It broke.", recovery="Stop.", retryable=False)
        body = json.loads(ToolResult.failure("t", detail).to_json())
        assert body["error"]["retryable"] is False
