"""Registry invariants, argument validation, and dispatch."""

from __future__ import annotations

import json

import pytest

from nfl_platform.llm.registry import DEFAULT_TOOL_TYPES, ToolRegistry
from nfl_platform.llm.tools.base import Tool, ToolArgs
from nfl_platform.services import ServiceContainer
from support import MAHOMES_ID


class TestRegistryInvariants:
    def test_every_default_tool_is_registered(self, registry: ToolRegistry):
        assert len(registry) == len(DEFAULT_TOOL_TYPES)

    def test_tool_names_are_unique(self, registry: ToolRegistry):
        assert len(set(registry.names)) == len(registry.names)

    def test_duplicate_registration_is_rejected(self):
        tool_type = DEFAULT_TOOL_TYPES[0]
        with pytest.raises(ValueError, match="Duplicate tool name"):
            ToolRegistry([tool_type(), tool_type()])

    @pytest.mark.parametrize("tool_type", DEFAULT_TOOL_TYPES, ids=lambda t: t.name)
    def test_tool_declares_a_useful_description(self, tool_type: type[Tool]):
        """A thin description is the most common cause of wrong tool selection."""
        assert len(tool_type.description) > 40

    @pytest.mark.parametrize("tool_type", DEFAULT_TOOL_TYPES, ids=lambda t: t.name)
    def test_tool_name_is_snake_case(self, tool_type: type[Tool]):
        assert tool_type.name.islower()
        assert " " not in tool_type.name

    @pytest.mark.parametrize("tool_type", DEFAULT_TOOL_TYPES, ids=lambda t: t.name)
    def test_args_model_forbids_unknown_fields(self, tool_type: type[Tool]):
        """Hallucinated argument names must be rejected, not silently dropped."""
        assert issubclass(tool_type.args_model, ToolArgs)
        assert tool_type.args_model.model_config["extra"] == "forbid"

    @pytest.mark.parametrize("tool_type", DEFAULT_TOOL_TYPES, ids=lambda t: t.name)
    def test_schema_is_well_formed(self, tool_type: type[Tool]):
        schema = tool_type().json_schema()
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] == tool_type.name
        assert function["parameters"]["type"] == "object"
        assert "properties" in function["parameters"]

    def test_schemas_are_json_serialisable(self, registry: ToolRegistry):
        """The provider receives these as JSON; an unserialisable default is fatal."""
        assert json.loads(json.dumps(registry.schemas()))

    def test_schema_count_matches_tool_count(self, registry: ToolRegistry):
        assert len(registry.schemas()) == len(registry)


class TestDispatch:
    def test_known_tool_executes(self, registry: ToolRegistry, services: ServiceContainer):
        result = registry.execute(
            "get_player_stats", {"player": MAHOMES_ID, "last_n_games": 3}, services
        )
        assert result.ok
        assert result.data is not None

    def test_unknown_tool_returns_an_error_not_an_exception(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("get_vibes", "{}", services)
        assert not result.ok
        assert result.error_code == "unknown_tool"
        assert "get_player_stats" in result.error["context"]["available_tools"]

    def test_get_raises_for_unknown_tool(self, registry: ToolRegistry):
        from nfl_platform.exceptions import UnknownToolError

        with pytest.raises(UnknownToolError):
            registry.get("nope")


class TestArgumentValidation:
    def test_malformed_json_is_reported(self, registry: ToolRegistry, services: ServiceContainer):
        result = registry.execute("get_player_stats", "{not json", services)
        assert not result.ok
        assert result.error_code == "malformed_arguments"

    def test_non_object_arguments_are_reported(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("get_player_stats", "[1, 2]", services)
        assert not result.ok
        assert result.error_code == "malformed_arguments"

    def test_unknown_argument_is_rejected(self, registry: ToolRegistry, services: ServiceContainer):
        result = registry.execute(
            "get_player_stats", {"player": MAHOMES_ID, "vibes": "high"}, services
        )
        assert not result.ok
        assert result.error_code == "invalid_arguments"
        fields = [problem["field"] for problem in result.error["context"]["problems"]]
        assert "vibes" in fields

    def test_missing_required_argument_is_rejected(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("get_player_stats", {}, services)
        assert not result.ok
        assert result.error_code == "invalid_arguments"

    def test_out_of_range_argument_is_rejected(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute(
            "get_player_stats", {"player": MAHOMES_ID, "last_n_games": 0}, services
        )
        assert not result.ok
        assert result.error_code == "invalid_arguments"

    def test_too_few_players_to_compare_is_rejected(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("compare_players", {"players": [MAHOMES_ID]}, services)
        assert not result.ok
        assert result.error_code == "invalid_arguments"

    def test_invalid_argument_error_includes_the_real_schema(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        """The model needs the schema to correct itself, not just a complaint."""
        result = registry.execute("get_player_stats", {}, services)
        parameters = result.error["context"]["expected_parameters"]
        assert "player" in parameters["properties"]

    def test_empty_arguments_string_is_treated_as_no_arguments(
        self, registry: ToolRegistry, services: ServiceContainer
    ):
        result = registry.execute("list_reference_data", "", services)
        assert result.ok

    def test_wrong_type_is_rejected(self, registry: ToolRegistry, services: ServiceContainer):
        result = registry.execute(
            "get_player_stats", {"player": MAHOMES_ID, "seasons": "twenty-24"}, services
        )
        assert not result.ok
        assert result.error_code == "invalid_arguments"
