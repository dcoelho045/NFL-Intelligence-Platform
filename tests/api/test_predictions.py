"""Prediction endpoints.

Built over the synthetic-league container so routing, validation, serialisation
and error mapping are exercised against real artifacts with no network.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from nfl_platform.api.app import create_app
from nfl_platform.config.settings import Settings
from nfl_platform.ml.registry.store import ModelRegistry
from nfl_platform.services import ServiceContainer
from synthetic import SYNTHETIC_TEAMS, SyntheticLeague, synthetic_player_id

PREFIX = "/api/v1/predictions"
HOME, AWAY = SYNTHETIC_TEAMS[0], SYNTHETIC_TEAMS[1]


@pytest.fixture
def prediction_client(prediction_services: ServiceContainer) -> Iterator[TestClient]:
    with TestClient(create_app(container=prediction_services)) as client:
        yield client


@pytest.fixture
def target(synthetic_league: SyntheticLeague) -> dict[str, int]:
    return {"season": synthetic_league.seasons[-1], "week": 8}


class TestModelsEndpoint:
    def test_lists_installed_models(self, prediction_client: TestClient):
        response = prediction_client.get(f"{PREFIX}/models")
        assert response.status_code == 200
        body = response.json()
        assert body["available"] is True
        assert {entry["problem"] for entry in body["models"]} == {
            "game_win",
            "fantasy_points",
        }

    def test_succeeds_even_with_no_models_installed(
        self, prediction_loader, settings: Settings, tmp_path
    ):
        """A status endpoint that fails when its subject is missing cannot do its job."""
        container = ServiceContainer(
            loader=prediction_loader,
            settings=settings,
            model_registry=ModelRegistry(root=tmp_path / "empty", settings=settings),
        )
        with TestClient(create_app(container=container)) as client:
            response = client.get(f"{PREFIX}/models")
        assert response.status_code == 200
        assert response.json()["available"] is False


class TestGameEndpoint:
    def test_returns_a_prediction(self, prediction_client: TestClient, target):
        response = prediction_client.get(
            f"{PREFIX}/game", params={"home_team": HOME, "away_team": AWAY, **target}
        )
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["home_win_probability"] <= 1.0
        assert body["predicted_winner"] in {HOME, AWAY}

    def test_the_response_carries_model_provenance(self, prediction_client: TestClient, target):
        response = prediction_client.get(
            f"{PREFIX}/game", params={"home_team": HOME, "away_team": AWAY, **target}
        )
        info = response.json()["model_info"]
        assert info["model_id"].startswith("game_win/")
        assert info["training_seasons"]

    def test_the_response_carries_caveats(self, prediction_client: TestClient, target):
        response = prediction_client.get(
            f"{PREFIX}/game", params={"home_team": HOME, "away_team": AWAY, **target}
        )
        assert response.json()["caveats"]

    def test_an_unknown_team_is_a_404(self, prediction_client: TestClient, target):
        response = prediction_client.get(
            f"{PREFIX}/game", params={"home_team": "ZZZ", "away_team": AWAY, **target}
        )
        assert response.status_code == 404
        assert response.json()["error"] == "team_not_found"

    def test_too_little_history_is_a_422(
        self, prediction_client: TestClient, synthetic_league: SyntheticLeague
    ):
        """A valid request the model cannot answer yet is not a server fault."""
        response = prediction_client.get(
            f"{PREFIX}/game",
            params={
                "home_team": HOME,
                "away_team": AWAY,
                "season": synthetic_league.seasons[0],
                "week": 1,
            },
        )
        assert response.status_code == 422
        assert response.json()["error"] == "insufficient_history"

    def test_an_invalid_week_is_rejected(self, prediction_client: TestClient, target):
        response = prediction_client.get(
            f"{PREFIX}/game",
            params={"home_team": HOME, "away_team": AWAY, "season": target["season"], "week": 99},
        )
        assert response.status_code == 422

    def test_no_model_installed_is_a_503(
        self, prediction_loader, settings: Settings, tmp_path, target
    ):
        container = ServiceContainer(
            loader=prediction_loader,
            settings=settings,
            model_registry=ModelRegistry(root=tmp_path / "empty", settings=settings),
        )
        with TestClient(create_app(container=container)) as client:
            response = client.get(
                f"{PREFIX}/game", params={"home_team": HOME, "away_team": AWAY, **target}
            )
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "model_not_available"
        assert "train_models.py" in body["context"]["hint"]


class TestSlateEndpoint:
    def test_returns_the_week(self, prediction_client: TestClient, target):
        response = prediction_client.get(f"{PREFIX}/slate/{target['season']}/{target['week']}")
        assert response.status_code == 200
        body = response.json()
        assert body["game_count"] == len(body["predictions"])

    def test_skipped_games_are_reported(
        self, prediction_client: TestClient, synthetic_league: SyntheticLeague
    ):
        response = prediction_client.get(f"{PREFIX}/slate/{synthetic_league.seasons[0]}/1")
        assert response.status_code == 200
        assert response.json()["skipped"]


class TestFantasyEndpoints:
    @pytest.fixture
    def player_id(self) -> str:
        return synthetic_player_id(SYNTHETIC_TEAMS[0], 1)

    def test_returns_a_projection(self, prediction_client: TestClient, player_id, target):
        response = prediction_client.get(f"{PREFIX}/fantasy/{player_id}", params=target)
        assert response.status_code == 200
        body = response.json()
        assert body["projected_points"] >= 0
        assert body["conditional_on_playing"] is True

    def test_the_playing_condition_is_in_the_caveats(
        self, prediction_client: TestClient, player_id, target
    ):
        response = prediction_client.get(f"{PREFIX}/fantasy/{player_id}", params=target)
        assert any("Conditional on" in c for c in response.json()["caveats"])

    def test_an_unknown_player_is_not_a_server_error(self, prediction_client: TestClient, target):
        response = prediction_client.get(f"{PREFIX}/fantasy/00-9999999", params=target)
        assert response.status_code in {404, 422}

    def test_compare_ranks_players(self, prediction_client: TestClient, target):
        players = [
            synthetic_player_id(SYNTHETIC_TEAMS[0], 1),
            synthetic_player_id(SYNTHETIC_TEAMS[1], 1),
        ]
        response = prediction_client.post(
            f"{PREFIX}/fantasy/compare", json={"players": players, **target}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["recommended_player_id"] in players
        assert body["conditional_on_playing"] is True

    def test_compare_rejects_a_single_player(self, prediction_client: TestClient, target):
        response = prediction_client.post(
            f"{PREFIX}/fantasy/compare",
            json={"players": [synthetic_player_id(SYNTHETIC_TEAMS[0], 1)], **target},
        )
        assert response.status_code == 422


class TestOpenApi:
    def test_the_prediction_routes_are_documented(self, prediction_client: TestClient):
        paths = prediction_client.get("/openapi.json").json()["paths"]
        assert f"{PREFIX}/game" in paths
        assert f"{PREFIX}/models" in paths
        assert f"{PREFIX}/fantasy/compare" in paths

    def test_predictions_have_their_own_tag(self, prediction_client: TestClient):
        """Visible in the URL and in the docs: these are estimates, not measurements."""
        schema = prediction_client.get("/openapi.json").json()
        names = (
            {tag["name"] for tag in schema["openapi_tags"]}
            if "openapi_tags" in schema
            else {tag["name"] for tag in schema.get("tags", [])}
        )
        assert "predictions" in names


class TestExistingEndpointsStillWork:
    def test_matchups_remain_descriptive(self, client: TestClient):
        """Milestone 2's promise: no prediction leaked into the analytics surface."""
        response = client.get(
            "/api/v1/matchups", params={"team_a": "KC", "team_b": "BUF", "seasons": [2024]}
        )
        assert response.status_code == 200
        body = response.json()
        assert "home_win_probability" not in body
        assert "model_info" not in body
