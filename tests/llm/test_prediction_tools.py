"""Prediction tools in the LLM catalogue.

The source-kind assertions carry the weight here. ``SourceKind.MODEL`` is what
lets the grounding check treat a win probability as traceable while keeping it
distinguishable from a measured statistic, and what makes the citation list
credit the model artifact rather than the tool name.
"""

from __future__ import annotations

import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.llm.grounding import check_grounding, collect_sources, partition_by_source
from nfl_platform.llm.registry import build_default_registry
from nfl_platform.llm.results import ToolResult
from nfl_platform.ml.registry.store import ModelRegistry
from nfl_platform.schemas.evidence import GROUNDING_KINDS, RETRIEVAL_KINDS, SourceKind
from nfl_platform.services import ServiceContainer
from synthetic import SYNTHETIC_TEAMS, SyntheticLeague, synthetic_player_id

HOME, AWAY = SYNTHETIC_TEAMS[0], SYNTHETIC_TEAMS[1]
PREDICTION_TOOLS = (
    "predict_game",
    "predict_slate",
    "project_fantasy_points",
    "compare_fantasy_projections",
)


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def target(synthetic_league: SyntheticLeague) -> dict[str, int]:
    return {"season": synthetic_league.seasons[-1], "week": 8}


class TestCatalogue:
    @pytest.mark.parametrize("name", PREDICTION_TOOLS)
    def test_the_tool_is_registered(self, registry, name: str):
        assert name in {s["function"]["name"] for s in registry.schemas()}

    @pytest.mark.parametrize("name", PREDICTION_TOOLS)
    def test_the_tool_speaks_for_the_model_channel(self, registry, name: str):
        assert registry.source_kind(name) is SourceKind.MODEL

    def test_the_analytics_tools_still_speak_for_the_structured_channel(self, registry):
        assert registry.source_kind("get_player_stats") is SourceKind.STRUCTURED

    def test_the_descriptions_forbid_the_model_inventing_a_number(self, registry):
        schemas = {s["function"]["name"]: s["function"] for s in registry.schemas()}
        assert (
            "Never state a win probability you worked out yourself"
            in (schemas["predict_game"]["description"])
        )
        assert (
            "Never estimate a projection yourself"
            in (schemas["project_fantasy_points"]["description"])
        )

    def test_the_fantasy_description_states_the_playing_condition(self, registry):
        schemas = {s["function"]["name"]: s["function"] for s in registry.schemas()}
        assert (
            "CONDITIONAL ON THE PLAYER PLAYING"
            in (schemas["project_fantasy_points"]["description"])
        )

    def test_the_schemas_reject_unknown_arguments(self, registry):
        schemas = {s["function"]["name"]: s["function"] for s in registry.schemas()}
        assert schemas["predict_game"]["parameters"]["additionalProperties"] is False


class TestExecution:
    def test_predict_game_returns_a_model_sourced_result(
        self, registry, prediction_services: ServiceContainer, target
    ):
        result = registry.execute(
            "predict_game",
            {"home_team": HOME, "away_team": AWAY, **target},
            prediction_services,
        )
        assert result.ok, result.error
        assert result.source_kind is SourceKind.MODEL
        assert 0.0 <= result.data["home_win_probability"] <= 1.0

    def test_project_fantasy_points_returns_a_projection(
        self, registry, prediction_services: ServiceContainer, target
    ):
        result = registry.execute(
            "project_fantasy_points",
            {"player": synthetic_player_id(SYNTHETIC_TEAMS[0], 1), **target},
            prediction_services,
        )
        assert result.ok, result.error
        assert result.data["conditional_on_playing"] is True

    def test_compare_projections_ranks(
        self, registry, prediction_services: ServiceContainer, target
    ):
        result = registry.execute(
            "compare_fantasy_projections",
            {
                "players": [
                    synthetic_player_id(SYNTHETIC_TEAMS[0], 1),
                    synthetic_player_id(SYNTHETIC_TEAMS[1], 1),
                ],
                **target,
            },
            prediction_services,
        )
        assert result.ok, result.error
        assert result.data["recommended_player_id"]

    def test_an_unknown_argument_is_rejected_rather_than_dropped(
        self, registry, prediction_services: ServiceContainer, target
    ):
        result = registry.execute(
            "predict_game",
            {"home_team": HOME, "away_team": AWAY, "quarter": 3, **target},
            prediction_services,
        )
        assert not result.ok

    def test_an_unavailable_model_becomes_a_recoverable_error(
        self, registry, prediction_loader, settings: Settings, tmp_path, target
    ):
        """A failed tool must come back as data the model can act on, not an exception."""
        container = ServiceContainer(
            loader=prediction_loader,
            settings=settings,
            model_registry=ModelRegistry(root=tmp_path / "empty", settings=settings),
        )
        result = registry.execute(
            "predict_game", {"home_team": HOME, "away_team": AWAY, **target}, container
        )
        assert not result.ok
        assert result.error_code == "model_not_available"
        assert "do not estimate" in result.error["recovery"].lower()

    def test_insufficient_history_tells_the_model_not_to_guess(
        self, registry, prediction_services: ServiceContainer, synthetic_league: SyntheticLeague
    ):
        result = registry.execute(
            "predict_game",
            {
                "home_team": HOME,
                "away_team": AWAY,
                "season": synthetic_league.seasons[0],
                "week": 1,
            },
            prediction_services,
        )
        assert result.error_code == "insufficient_history"
        assert "Do not guess" in result.error["recovery"]


class TestGrounding:
    def test_model_output_can_ground_a_number(self):
        """A probability this system computed is traceable, unlike retrieved prose."""
        result = ToolResult.success(
            "predict_game", {"home_win_probability": 0.63}, source_kind=SourceKind.MODEL
        )
        report = check_grounding("The model gives the home team a 0.63 chance.", [result])
        assert report.is_grounded
        assert not report.from_unstructured

    def test_a_number_the_model_invented_is_still_flagged(self):
        result = ToolResult.success(
            "predict_game", {"home_win_probability": 0.63}, source_kind=SourceKind.MODEL
        )
        report = check_grounding("The home team wins 81 percent of the time.", [result])
        assert not report.is_grounded

    def test_model_and_structured_results_share_the_grounding_bucket(self):
        model = ToolResult.success("predict_game", {"p": 0.63}, source_kind=SourceKind.MODEL)
        structured = ToolResult.success("get_team_stats", {"yards": 412.0})
        computed, retrieved = partition_by_source([model, structured])
        assert {0.63, 412.0} <= computed
        assert not retrieved

    def test_retrieved_text_stays_out_of_the_grounding_bucket(self):
        web = ToolResult.success("search_web", {"snippet": "312 yards"}, source_kind=SourceKind.WEB)
        computed, retrieved = partition_by_source([web])
        assert not computed
        assert 312.0 in retrieved

    def test_sources_credit_the_model_artifact_not_the_tool(self):
        result = ToolResult.success(
            "predict_game",
            {"model_info": {"model_id": "game_win/v1_logistic"}},
            source_kind=SourceKind.MODEL,
        )
        sources = collect_sources([result])
        assert sources == [{"kind": "model", "label": "game_win/v1_logistic"}]

    def test_the_grounding_and_retrieval_kinds_are_disjoint(self):
        assert not GROUNDING_KINDS & RETRIEVAL_KINDS
        assert SourceKind.MODEL in GROUNDING_KINDS
        assert SourceKind.MODEL not in RETRIEVAL_KINDS
