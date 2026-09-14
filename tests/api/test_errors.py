"""Tests for domain exception to HTTP status mapping and request validation."""

from __future__ import annotations

from fastapi.testclient import TestClient

from nfl_platform.api.app import API_PREFIX
from support import ALLEN_ID, MAHOMES_ID, UNKNOWN_PLAYER_ID

AMBIGUOUS_NAME = "Mike Williams"


class TestNotFound:
    def test_unknown_player_is_404(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/players/{UNKNOWN_PLAYER_ID}")
        assert response.status_code == 404
        assert response.json()["error"] == "player_not_found"

    def test_unknown_team_is_404_and_lists_valid_teams(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/teams/ZZZ/offense", params={"seasons": 2024})
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "team_not_found"
        assert "KC" in body["context"]["valid_teams"]


class TestAmbiguity:
    def test_ambiguous_name_is_409_with_candidates(self, client: TestClient):
        """The caller gets the choices; a silent guess would look like a right answer."""
        response = client.get(f"{API_PREFIX}/players/{AMBIGUOUS_NAME}")
        assert response.status_code == 409
        body = response.json()
        assert body["error"] == "ambiguous_player"
        assert len(body["context"]["candidates"]) > 1
        assert all("player_id" in item for item in body["context"]["candidates"])

    def test_candidates_carry_a_human_readable_description(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/players/{AMBIGUOUS_NAME}")
        assert all(item["description"] for item in response.json()["context"]["candidates"])


class TestInvalidInput:
    def test_invalid_season_is_422(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/players/{MAHOMES_ID}/stats/recent", params={"seasons": 1850}
        )
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_season"

    def test_invalid_week_is_422(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/players/{MAHOMES_ID}/games",
            params={"seasons": 2024, "weeks": 99},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_week"

    def test_identical_matchup_teams_is_422(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/matchups",
            params={"team_a": "KC", "team_b": "KC", "seasons": 2024},
        )
        assert response.status_code == 422

    def test_last_n_games_below_range_is_rejected(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/players/{MAHOMES_ID}/games",
            params={"seasons": 2024, "last_n_games": 0},
        )
        assert response.status_code == 422

    def test_last_n_games_above_cap_is_rejected(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/players/{MAHOMES_ID}/games",
            params={"seasons": 2024, "last_n_games": 500},
        )
        assert response.status_code == 422

    def test_invalid_season_type_is_rejected(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/players/{MAHOMES_ID}/games",
            params={"seasons": 2024, "season_type": "PRESEASON"},
        )
        assert response.status_code == 422

    def test_missing_required_query_parameter_is_rejected(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/matchups", params={"team_a": "KC"})
        assert response.status_code == 422


class TestRequestBodyValidation:
    def test_compare_with_one_player_is_rejected(self, client: TestClient):
        response = client.post(f"{API_PREFIX}/players/compare", json={"player_ids": [MAHOMES_ID]})
        assert response.status_code == 422

    def test_compare_with_blank_id_is_rejected(self, client: TestClient):
        response = client.post(
            f"{API_PREFIX}/players/compare", json={"player_ids": [MAHOMES_ID, "  "]}
        )
        assert response.status_code == 422

    def test_unknown_body_field_is_rejected(self, client: TestClient):
        """extra='forbid' means a malformed tool call fails loudly, not silently."""
        response = client.post(
            f"{API_PREFIX}/players/compare",
            json={"player_ids": [MAHOMES_ID, ALLEN_ID], "bogus_field": 1},
        )
        assert response.status_code == 422
