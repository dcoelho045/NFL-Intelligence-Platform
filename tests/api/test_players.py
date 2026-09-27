"""Tests for the player endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from nfl_platform.api.app import API_PREFIX
from support import ALLEN_ID, MAHOMES_ID

from .conftest import AMBIGUOUS_NAME


class TestPlayerEndpoints:
    def test_profile(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/players/{MAHOMES_ID}")
        assert response.status_code == 200
        assert response.json()["name"] == "Patrick Mahomes"

    def test_search_returns_candidates(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/players/search", params={"name": "Patrick Mahomes"})
        assert response.status_code == 200
        assert response.json()["match_count"] >= 1

    def test_search_route_is_not_shadowed_by_the_id_route(self, client: TestClient):
        """'/players/search' must not be captured as a player id."""
        response = client.get(f"{API_PREFIX}/players/search", params={"name": "Mahomes"})
        assert response.status_code == 200
        assert "candidates" in response.json()

    def test_recent_stats(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/players/{MAHOMES_ID}/stats/recent",
            params={"seasons": 2024, "last_n_games": 3},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["window"]["games"] == 3
        assert body["totals"]["passing_yards"] > 0

    def test_season_stats(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/players/{MAHOMES_ID}/stats/season", params={"seasons": 2024}
        )
        assert response.status_code == 200
        assert response.json()["window"]["requested_last_n_games"] is None

    def test_game_log(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/players/{MAHOMES_ID}/games",
            params={"seasons": 2024, "last_n_games": 4},
        )
        assert response.status_code == 200
        assert len(response.json()["games"]) == 4

    def test_usage(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/players/{MAHOMES_ID}/usage", params={"seasons": 2024})
        assert response.status_code == 200
        assert response.json()["snap_data_available"] is True

    def test_injuries(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/players/{MAHOMES_ID}/injuries", params={"seasons": 2024}
        )
        assert response.status_code == 200
        assert "entries" in response.json()

    def test_player_name_in_the_path_resolves(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/players/Patrick Mahomes")
        assert response.status_code == 200
        assert response.json()["player_id"] == MAHOMES_ID

    def test_compare(self, client: TestClient):
        response = client.post(
            f"{API_PREFIX}/players/compare",
            json={
                "player_ids": [MAHOMES_ID, ALLEN_ID],
                "seasons": [2024],
                "last_n_games": 4,
            },
        )
        assert response.status_code == 200
        assert len(response.json()["players"]) == 2

    def test_compare_reports_every_unresolvable_name_at_once(self, client: TestClient):
        response = client.post(
            f"{API_PREFIX}/players/compare",
            json={"player_ids": [AMBIGUOUS_NAME, "Josh Allen"], "seasons": [2024]},
        )
        assert response.status_code == 409
        body = response.json()
        assert body["error"] == "unresolved_players"
        assert {item["query"] for item in body["context"]["ambiguous"]} == {
            AMBIGUOUS_NAME,
            "Josh Allen",
        }
