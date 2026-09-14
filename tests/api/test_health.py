"""Tests for the health probes and the generated OpenAPI schema."""

from __future__ import annotations

from fastapi.testclient import TestClient

from nfl_platform.api.app import API_PREFIX


class TestHealth:
    def test_liveness(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["version"]

    def test_readiness_reports_the_data_layer(self, client: TestClient):
        response = client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["data_layer_available"] is True


class TestOpenAPI:
    def test_schema_generates(self, client: TestClient):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.json()["info"]["title"] == "NFL Intelligence Platform API"

    def test_docs_are_served(self, client: TestClient):
        assert client.get("/docs").status_code == 200

    def test_every_expected_path_is_documented(self, client: TestClient):
        """Pins the public surface so an accidental route change fails a test."""
        paths = set(client.get("/openapi.json").json()["paths"])
        expected = {
            "/health",
            "/health/ready",
            f"{API_PREFIX}/players/search",
            f"{API_PREFIX}/players/compare",
            f"{API_PREFIX}/players/{{player_id}}",
            f"{API_PREFIX}/players/{{player_id}}/stats/recent",
            f"{API_PREFIX}/players/{{player_id}}/stats/season",
            f"{API_PREFIX}/players/{{player_id}}/usage",
            f"{API_PREFIX}/players/{{player_id}}/games",
            f"{API_PREFIX}/players/{{player_id}}/injuries",
            f"{API_PREFIX}/teams/{{team}}/offense",
            f"{API_PREFIX}/teams/{{team}}/defense",
            f"{API_PREFIX}/teams/{{team}}/form",
            f"{API_PREFIX}/teams/{{team}}/splits",
            f"{API_PREFIX}/teams/{{team}}/games",
            f"{API_PREFIX}/teams/{{team}}/injuries",
            f"{API_PREFIX}/matchups",
            f"{API_PREFIX}/fantasy/compare",
            f"{API_PREFIX}/fantasy/players/{{player_id}}/production",
            f"{API_PREFIX}/fantasy/players/{{player_id}}/trends",
            f"{API_PREFIX}/fantasy/players/{{player_id}}/matchup",
            f"{API_PREFIX}/meta/teams",
            f"{API_PREFIX}/meta/seasons",
        }
        assert expected <= paths


class TestMeta:
    def test_teams(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/meta/teams", params={"seasons": 2024})
        assert response.status_code == 200
        assert "KC" in response.json()

    def test_seasons(self, client: TestClient):
        response = client.get(f"{API_PREFIX}/meta/seasons")
        assert response.status_code == 200
        assert 2024 in response.json()
