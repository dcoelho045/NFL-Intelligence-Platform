"""DTO contract tests.

These schemas are the stable interface Milestones 3 and 5 will expose as LLM and
MCP tool definitions, so JSON Schema generation and round-tripping are contract
guarantees rather than incidental behaviour.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

import nfl_platform.schemas as schemas
from nfl_platform.schemas.common import PlayerRef, SeasonType, SeasonWindow, StatWindow
from nfl_platform.schemas.player import PlayerStatAverages, PlayerStatLine
from nfl_platform.schemas.requests import FantasyComparisonRequest, PlayerComparisonRequest

DTO_CLASSES = [
    getattr(schemas, name)
    for name in schemas.__all__
    if isinstance(getattr(schemas, name), type) and issubclass(getattr(schemas, name), BaseModel)
]


class TestSchemaGeneration:
    def test_every_dto_is_exported(self):
        assert len(DTO_CLASSES) >= 20

    @pytest.mark.parametrize("model", DTO_CLASSES, ids=lambda m: m.__name__)
    def test_json_schema_generates(self, model):
        """Tool-calling and MCP both require a JSON Schema for every payload."""
        schema = model.model_json_schema()
        assert schema["type"] == "object"
        assert "properties" in schema or "$ref" in schema

    @pytest.mark.parametrize("model", DTO_CLASSES, ids=lambda m: m.__name__)
    def test_schema_is_json_serialisable(self, model):
        assert json.loads(json.dumps(model.model_json_schema()))

    @pytest.mark.parametrize("model", DTO_CLASSES, ids=lambda m: m.__name__)
    def test_every_field_is_documented_or_self_explanatory(self, model):
        """Models an LLM reads should not contain fields with no type information."""
        properties = model.model_json_schema().get("properties", {})
        for name, spec in properties.items():
            assert spec, f"{model.__name__}.{name} has an empty schema"


class TestStrictness:
    def test_unknown_fields_are_rejected(self):
        """A malformed tool call must fail loudly rather than be silently ignored."""
        with pytest.raises(ValidationError):
            PlayerRef(player_id="00-0033873", name="X", bogus=1)

    def test_dtos_are_immutable(self):
        ref = PlayerRef(player_id="00-0033873", name="Patrick Mahomes")
        with pytest.raises(ValidationError):
            ref.name = "Someone Else"

    def test_required_fields_are_enforced(self):
        with pytest.raises(ValidationError):
            PlayerRef(name="No id")


class TestRoundTrip:
    def test_player_ref_round_trips(self):
        original = PlayerRef(
            player_id="00-0033873", name="Patrick Mahomes", position="QB", team="KC"
        )
        assert PlayerRef.model_validate(json.loads(original.model_dump_json())) == original

    def test_stat_window_round_trips(self):
        original = StatWindow(
            seasons=[2024],
            season_type=SeasonType.REGULAR,
            weeks=None,
            games=4,
            requested_last_n_games=4,
        )
        assert StatWindow.model_validate(json.loads(original.model_dump_json())) == original

    def test_season_type_serialises_as_its_value(self):
        window = SeasonWindow(seasons=[2024], season_type=SeasonType.POST)
        assert json.loads(window.model_dump_json())["season_type"] == "POST"


class TestStatTypes:
    def test_totals_are_integers(self):
        line = PlayerStatLine(passing_yards=793, carries=17)
        assert isinstance(line.passing_yards, int)

    def test_totals_reject_fractional_counts(self):
        """Catching this is why totals and averages are separate models."""
        with pytest.raises(ValidationError):
            PlayerStatLine(passing_yards=264.33)

    def test_averages_accept_fractional_counts(self):
        averages = PlayerStatAverages(passing_yards=264.33, carries=5.67)
        assert averages.passing_yards == pytest.approx(264.33)

    def test_both_models_cover_the_same_statistics(self):
        assert set(PlayerStatLine.model_fields) == set(PlayerStatAverages.model_fields)


class TestRequestModels:
    def test_minimum_two_players(self):
        with pytest.raises(ValidationError):
            PlayerComparisonRequest(player_ids=["00-0033873"])

    def test_maximum_player_count_enforced(self):
        with pytest.raises(ValidationError):
            PlayerComparisonRequest(player_ids=[f"00-000000{n}" for n in range(9)])

    def test_blank_identifiers_rejected(self):
        with pytest.raises(ValidationError):
            PlayerComparisonRequest(player_ids=["00-0033873", "   "])

    def test_identifiers_are_trimmed(self):
        request = PlayerComparisonRequest(player_ids=[" 00-0033873 ", "00-0034857"])
        assert request.player_ids[0] == "00-0033873"

    def test_last_n_games_is_capped(self):
        with pytest.raises(ValidationError):
            PlayerComparisonRequest(player_ids=["a", "b"], last_n_games=500)

    def test_fantasy_request_defaults_to_ppr(self):
        request = FantasyComparisonRequest(player_ids=["a", "b"])
        assert request.scoring.value == "ppr"
