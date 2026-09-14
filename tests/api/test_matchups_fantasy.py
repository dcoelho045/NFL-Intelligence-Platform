"""Tests for the matchup and fantasy endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from nfl_platform.api.app import API_PREFIX
from support import ALLEN_ID, MAHOMES_ID


class TestMatchup:
    def test_returns_both_sides(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/matchups",
            params={"team_a": "KC", "team_b": "BUF", "seasons": 2024},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["team_a"]["team"] == "KC"
        assert body["team_b"]["team"] == "BUF"

    def test_exposes_no_prediction(self, client: TestClient):
        """Descriptive endpoint; prediction arrives in a later milestone."""
        response = client.get(
            f"{API_PREFIX}/matchups",
            params={"team_a": "KC", "team_b": "BUF", "seasons": 2024},
        )
        assert not {"win_probability", "predicted_score"} & set(response.json())


class TestFantasy:
    def test_production(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/fantasy/players/{MAHOMES_ID}/production",
            params={"seasons": 2024, "last_n_games": 4},
        )
        assert response.status_code == 200
        assert response.json()["scoring"] == "ppr"

    def test_standard_scoring_is_selectable(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/fantasy/players/{MAHOMES_ID}/production",
            params={"seasons": 2024, "scoring": "standard"},
        )
        assert response.status_code == 200
        assert response.json()["scoring"] == "standard"

    def test_trends(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/fantasy/players/{MAHOMES_ID}/trends", params={"seasons": 2024}
        )
        assert response.status_code == 200
        assert "target_share_direction" in response.json()

    def test_matchup_context(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/fantasy/players/{MAHOMES_ID}/matchup",
            params={"seasons": 2024, "opponent": "BUF"},
        )
        assert response.status_code == 200
        assert response.json()["opponent"] == "BUF"

    def test_compare(self, client: TestClient):
        response = client.post(
            f"{API_PREFIX}/fantasy/compare",
            json={"player_ids": [MAHOMES_ID, ALLEN_ID], "seasons": [2024]},
        )
        assert response.status_code == 200
        assert len(response.json()["players"]) == 2
