"""Tests for the team endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from nfl_platform.api.app import API_PREFIX


class TestTeamEndpoints:
    def test_offense(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/teams/KC/offense", params={"seasons": 2024})
        assert response.status_code == 200
        assert response.json()["team"] == "KC"

    def test_defense_uses_allowed_prefix(self, client: TestClient):
        """The prefix is what stops 'gave up' being read as 'gained'."""
        response = client.get(f"{API_PREFIX}/teams/KC/defense", params={"seasons": 2024})
        assert response.status_code == 200
        body = response.json()
        assert "avg_allowed_passing_yards" in body
        assert "avg_passing_yards" not in body

    def test_form(self, client: TestClient):
        response = client.get(
            f"{API_PREFIX}/teams/KC/form", params={"seasons": 2024, "last_n_games": 5}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["wins"] + body["losses"] + body["ties"] == body["window"]["games"]

    def test_splits(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/teams/KC/splits", params={"seasons": 2024})
        assert response.status_code == 200
        body = response.json()
        assert body["home"]["split"] == "home"
        assert body["away"]["split"] == "away"

    def test_games(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/teams/KC/games", params={"seasons": 2024})
        assert response.status_code == 200
        assert response.json()["games"]

    def test_injuries(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/teams/KC/injuries", params={"seasons": 2024})
        assert response.status_code == 200
        assert "player_count" in response.json()

    def test_lowercase_team_is_normalised(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/teams/kc/offense", params={"seasons": 2024})
        assert response.status_code == 200
        assert response.json()["team"] == "KC"
