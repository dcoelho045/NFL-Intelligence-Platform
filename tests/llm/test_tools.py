"""Every tool executes against the fixture data and returns a usable payload.

These run the real services, so they double as a check that the tool layer has
not drifted from the Milestone 2 interfaces it wraps.
"""

from __future__ import annotations

import pytest

from nfl_platform.llm.registry import ToolRegistry
from nfl_platform.services import ServiceContainer
from support import ALLEN_ID, MAHOMES_ID, UNKNOWN_PLAYER_ID

TEAM = "KC"
OPPONENT = "BUF"

# One representative call per registered tool.
TOOL_CALLS: list[tuple[str, dict[str, object]]] = [
    ("search_players", {"name": "Mahomes"}),
    ("get_player_profile", {"player": MAHOMES_ID}),
    ("get_player_stats", {"player": MAHOMES_ID, "last_n_games": 3}),
    ("get_player_game_log", {"player": MAHOMES_ID, "last_n_games": 3}),
    ("get_player_usage", {"player": MAHOMES_ID, "last_n_games": 3}),
    ("get_player_injuries", {"player": MAHOMES_ID}),
    ("compare_players", {"players": [MAHOMES_ID, ALLEN_ID], "last_n_games": 3}),
    ("get_team_stats", {"team": TEAM, "side": "offense"}),
    ("get_team_stats", {"team": TEAM, "side": "defense"}),
    ("get_team_form", {"team": TEAM}),
    ("get_team_form", {"team": TEAM, "split": "home"}),
    ("get_recent_games", {"team": TEAM}),
    ("get_team_injuries", {"team": TEAM}),
    ("get_matchup", {"team_a": TEAM, "team_b": OPPONENT}),
    ("get_fantasy_production", {"player": MAHOMES_ID, "last_n_games": 3}),
    ("get_fantasy_usage_trend", {"player": MAHOMES_ID}),
    ("get_fantasy_matchup_context", {"player": MAHOMES_ID, "opponent": OPPONENT}),
    ("compare_fantasy_players", {"players": [MAHOMES_ID, ALLEN_ID]}),
    ("list_reference_data", {}),
]


class TestToolExecution:
    @pytest.mark.parametrize(
        ("name", "arguments"), TOOL_CALLS, ids=[f"{n}-{i}" for i, (n, _) in enumerate(TOOL_CALLS)]
    )
    def test_tool_returns_a_successful_result(
        self,
        name: str,
        arguments: dict[str, object],
        registry: ToolRegistry,
        services: ServiceContainer,
    ):
        result = registry.execute(name, arguments, services)
        assert result.ok, result.error
        assert result.data is not None

    def test_every_registered_tool_is_covered_by_these_tests(self, registry: ToolRegistry):
        """A new tool must not be able to ship untested."""
        assert set(registry.names) == {name for name, _ in TOOL_CALLS}


class TestToolBehaviour:
    def test_player_stats_resolves_a_name_not_just_an_id(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("get_player_stats", {"player": "Patrick Mahomes"}, services)
        assert result.ok
        assert result.data["player"]["player_id"] == MAHOMES_ID

    def test_season_window_ignores_last_n_games(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        recent = registry.execute(
            "get_player_stats",
            {"player": MAHOMES_ID, "window": "recent", "last_n_games": 2},
            services,
        )
        season = registry.execute(
            "get_player_stats", {"player": MAHOMES_ID, "window": "season"}, services
        )
        assert season.data["window"]["games"] > recent.data["window"]["games"]

    def test_defense_side_returns_allowed_statistics(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("get_team_stats", {"team": TEAM, "side": "defense"}, services)
        assert any(key.startswith("total_allowed") for key in result.data)

    def test_home_split_returns_only_home_games(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("get_team_form", {"team": TEAM, "split": "home"}, services)
        assert result.data["split"] == "home"

    def test_reference_data_lists_teams_without_truncation(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        """A partial list of valid teams would read as a list of the only valid teams."""
        result = registry.execute("list_reference_data", {"resource": "teams"}, services)
        assert len(result.data["teams"]) > 10
        assert not result.notes

    def test_reference_data_reports_the_current_season(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("list_reference_data", {"resource": "seasons"}, services)
        assert result.data["current_season"] == 2024

    def test_long_lists_are_truncated_with_an_explicit_note(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        """Truncation must be announced, or a subset reads as the whole window."""
        result = registry.execute("get_team_injuries", {"team": TEAM}, services)
        assert result.data["player_count"] > 10
        assert len(result.data["entries"]) == 10
        assert any("entries" in note for note in result.notes)


class TestToolErrors:
    def test_unknown_player_is_a_structured_error(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("get_player_stats", {"player": UNKNOWN_PLAYER_ID}, services)
        assert not result.ok
        assert result.error_code == "player_not_found"
        assert result.error["recovery"]

    def test_ambiguous_name_returns_candidates_to_choose_from(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("get_player_stats", {"player": "Mike Williams"}, services)
        assert not result.ok
        assert result.error_code == "ambiguous_player"
        candidates = result.error["context"]["candidates"]
        assert len(candidates) >= 2
        assert all("player_id" in candidate for candidate in candidates)

    def test_unknown_team_lists_valid_teams(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("get_team_stats", {"team": "XYZ"}, services)
        assert not result.ok
        assert result.error_code == "team_not_found"
        assert result.error["context"]["valid_teams"]

    def test_invalid_season_is_rejected(self, registry: ToolRegistry, services: ServiceContainer):
        result = registry.execute(
            "get_player_stats", {"player": MAHOMES_ID, "seasons": [1850]}, services
        )
        assert not result.ok
        assert result.error_code == "invalid_season"

    def test_comparing_a_team_with_itself_is_rejected(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("get_matchup", {"team_a": TEAM, "team_b": TEAM}, services)
        assert not result.ok
        assert result.error_code == "invalid_request"

    def test_healthy_player_returns_success_not_failure(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        """An empty injury report is a fact about the player, not an error."""
        result = registry.execute("get_player_injuries", {"player": MAHOMES_ID}, services)
        assert result.ok
