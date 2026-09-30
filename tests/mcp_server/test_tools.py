"""Tool execution against the committed fixtures.

Every call here runs the real service stack over the 2024 Kansas City / Buffalo
Parquet slice. The assertions check two different things: that the protocol
plumbing works, and that the numbers are the deterministic ones -- a tool that
returned a plausible wrong figure would pass the first check and fail the second.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp import Client
from mcp.types import TextContent

from nfl_platform.schemas.fantasy import FantasyProduction
from nfl_platform.schemas.matchup import MatchupAnalysis
from nfl_platform.schemas.player import PlayerSeasonStats
from nfl_platform.schemas.team import TeamDefenseSummary
from nfl_platform.services import ServiceContainer
from support import ALLEN_ID, FIXTURE_SEASON, MAHOMES_ID

pytestmark = pytest.mark.anyio


async def call(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool and return its structured content, failing loudly on an error."""
    result = await client.call_tool(name, arguments)
    assert not result.is_error, _error_text(result.content)
    assert result.structured_content is not None
    return result.structured_content


def _error_text(content: list[Any]) -> str:
    return " ".join(block.text for block in content if isinstance(block, TextContent))


class TestPlayerTools:
    async def test_search_players_finds_a_gsis_id(self, client: Client):
        payload = await call(client, "search_players", {"name": "Patrick Mahomes"})
        assert any(candidate["player_id"] == MAHOMES_ID for candidate in payload["candidates"])

    async def test_player_profile_is_biographical_only(self, client: Client):
        payload = await call(client, "get_player_profile", {"player": MAHOMES_ID})
        assert payload["player_id"] == MAHOMES_ID
        assert payload["position"] == "QB"
        assert "passing_yards" not in payload

    async def test_player_stats_match_the_service(self, client: Client, services: ServiceContainer):
        """The MCP answer must be the analytics answer, not a re-derivation of it."""
        payload = await call(
            client,
            "get_player_stats",
            {"player": MAHOMES_ID, "seasons": [FIXTURE_SEASON], "last_n_games": 4},
        )
        expected = services.players.recent_performance(
            MAHOMES_ID, seasons=[FIXTURE_SEASON], last_n_games=4
        )
        assert payload == expected.model_dump(mode="json")

    async def test_player_stats_are_validated_against_the_output_schema(self, client: Client):
        payload = await call(
            client, "get_player_stats", {"player": MAHOMES_ID, "seasons": [FIXTURE_SEASON]}
        )
        assert PlayerSeasonStats.model_validate(payload)

    async def test_every_result_states_the_window_it_covers(self, client: Client):
        payload = await call(
            client,
            "get_player_stats",
            {"player": MAHOMES_ID, "seasons": [FIXTURE_SEASON], "last_n_games": 3},
        )
        window = payload["window"]
        assert window["seasons"] == [FIXTURE_SEASON]
        assert window["requested_last_n_games"] == 3
        assert window["games"] <= 3

    async def test_season_window_ignores_last_n_games(self, client: Client):
        payload = await call(
            client,
            "get_player_stats",
            {"player": MAHOMES_ID, "seasons": [FIXTURE_SEASON], "window": "season"},
        )
        assert payload["window"]["requested_last_n_games"] is None

    async def test_game_log_is_newest_first(self, client: Client):
        payload = await call(
            client,
            "get_player_game_log",
            {"player": ALLEN_ID, "seasons": [FIXTURE_SEASON], "last_n_games": 4},
        )
        weeks = [game["week"] for game in payload["games"]]
        assert weeks == sorted(weeks, reverse=True)

    async def test_usage_reports_opportunity_metrics(self, client: Client):
        payload = await call(
            client, "get_player_usage", {"player": ALLEN_ID, "seasons": [FIXTURE_SEASON]}
        )
        assert "games" in payload
        assert payload["player"]["player_id"] == ALLEN_ID

    async def test_compare_players_uses_one_shared_window(self, client: Client):
        payload = await call(
            client,
            "compare_players",
            {
                "players": [MAHOMES_ID, ALLEN_ID],
                "seasons": [FIXTURE_SEASON],
                "last_n_games": 4,
            },
        )
        assert len(payload["players"]) == 2
        assert len({player["window"]["games"] for player in payload["players"]}) == 1


class TestAbsenceIsNotAnError:
    """The most important behaviour on this server.

    "We looked and found nothing" and "we could not look" must never arrive in
    the same shape, because a client cannot tell them apart afterwards.
    """

    async def test_a_player_with_no_reported_injury_succeeds_with_an_empty_list(
        self, client: Client
    ):
        result = await client.call_tool(
            "get_player_injuries",
            {"player": MAHOMES_ID, "seasons": [FIXTURE_SEASON], "weeks": [1]},
        )
        assert not result.is_error
        assert result.structured_content is not None
        assert isinstance(result.structured_content["entries"], list)

    async def test_an_off_corpus_knowledge_query_succeeds_with_no_results(self, client: Client):
        payload = await call(
            client,
            "search_knowledge_base",
            {"query": "what is the melting point of tungsten"},
        )
        assert payload["result_count"] == 0
        assert payload["results"] == []


class TestTeamTools:
    async def test_team_offense_matches_the_service(
        self, client: Client, services: ServiceContainer
    ):
        result = await client.call_tool(
            "get_team_stats", {"team": "KC", "seasons": [FIXTURE_SEASON], "last_n_games": 4}
        )
        assert not result.is_error
        expected = services.teams.offense("KC", seasons=[FIXTURE_SEASON], last_n_games=4)
        assert result.structured_content == {"result": expected.model_dump(mode="json")}

    async def test_defense_side_returns_what_the_team_allowed(self, client: Client):
        """The union output schema must not let a defence payload validate as offence."""
        result = await client.call_tool(
            "get_team_stats",
            {"team": "BUF", "side": "defense", "seasons": [FIXTURE_SEASON]},
        )
        assert not result.is_error
        assert result.structured_content is not None
        payload = result.structured_content["result"]
        assert TeamDefenseSummary.model_validate(payload)
        assert any("allowed" in key for key in payload)
        assert not any(key.startswith("total_passing") for key in payload)

    async def test_team_form_reports_a_record(self, client: Client):
        payload = await call(client, "get_team_form", {"team": "KC", "seasons": [FIXTURE_SEASON]})
        assert payload["team"] == "KC"
        assert payload["wins"] + payload["losses"] + payload["ties"] == payload["window"]["games"]

    async def test_home_split_is_a_subset_of_all_games(self, client: Client):
        every = await call(client, "get_team_form", {"team": "KC", "seasons": [FIXTURE_SEASON]})
        home = await call(
            client,
            "get_team_form",
            {"team": "KC", "split": "home", "seasons": [FIXTURE_SEASON]},
        )
        assert home["window"]["games"] <= every["window"]["games"]

    async def test_team_injuries_identify_players(self, client: Client):
        payload = await call(
            client, "get_team_injuries", {"team": "KC", "seasons": [FIXTURE_SEASON], "week": 2}
        )
        assert payload["team"] == "KC"
        for entry in payload["entries"]:
            assert entry["player"]["name"]

    async def test_matchup_contains_no_prediction(self, client: Client):
        payload = await call(
            client,
            "get_matchup",
            {"team_a": "KC", "team_b": "BUF", "seasons": [FIXTURE_SEASON]},
        )
        assert MatchupAnalysis.model_validate(payload)
        serialised = json.dumps(payload).lower()
        assert "win_probability" not in serialised
        assert "prediction" not in serialised


class TestFantasyTools:
    async def test_production_is_scored_by_format(self, client: Client):
        ppr = await call(
            client,
            "get_fantasy_production",
            {"player": ALLEN_ID, "seasons": [FIXTURE_SEASON], "scoring": "ppr"},
        )
        standard = await call(
            client,
            "get_fantasy_production",
            {"player": ALLEN_ID, "seasons": [FIXTURE_SEASON], "scoring": "standard"},
        )
        assert FantasyProduction.model_validate(ppr)
        assert ppr["scoring"] == "ppr"
        assert standard["scoring"] == "standard"

    async def test_compare_fantasy_players_shares_one_window(self, client: Client):
        payload = await call(
            client,
            "compare_fantasy_players",
            {"players": [MAHOMES_ID, ALLEN_ID], "seasons": [FIXTURE_SEASON]},
        )
        assert len(payload["players"]) == 2


class TestRetrievalTools:
    async def test_web_search_returns_attributed_results(self, client: Client):
        payload = await call(
            client, "search_web", {"query": "Patrick Mahomes injury status", "recency_days": 7}
        )
        assert payload["result_count"] >= 1
        for item in payload["results"]:
            assert item["url"].startswith("http")
            assert item["domain"]
            assert item["retrieved_at"]

    async def test_web_results_are_labelled_untrusted(self, client: Client):
        payload = await call(client, "search_web", {"query": "Chiefs roster news"})
        assert any("untrusted" in note.lower() for note in payload["notes"])

    async def test_knowledge_search_returns_cited_passages(self, client: Client):
        payload = await call(client, "search_knowledge_base", {"query": "PPR scoring"})
        assert payload["result_count"] >= 1
        for item in payload["results"]:
            assert item["document_id"]
            assert item["chunk_id"]
            assert 0.0 <= item["score"] <= 1.0

    async def test_knowledge_results_are_labelled_stable_not_current(self, client: Client):
        payload = await call(client, "search_knowledge_base", {"query": "PPR scoring"})
        assert any("not current" in note.lower() for note in payload["notes"])


class TestBothChannelsAreFilled:
    async def test_content_carries_the_same_payload_as_structured_content(self, client: Client):
        """``content`` is for a model, ``structured_content`` is for code."""
        result = await client.call_tool("get_player_profile", {"player": MAHOMES_ID})
        text = _error_text(result.content)
        assert json.loads(text) == result.structured_content
