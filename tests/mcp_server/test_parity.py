"""The MCP catalogue against the LLM catalogue.

Both adapters describe the same capabilities to a model. They are deliberately
separate code -- generating one from the other would couple two adapters that
must stay independent -- so this is the guard that keeps them from drifting into
describing the same argument two different ways.

It is not a demand for identical catalogues. MCP exposes a curated subset, and
the assertion is about the tools that appear in both.
"""

from __future__ import annotations

from typing import Any

import pytest
from mcp import Client

from nfl_platform.llm.registry import build_default_registry
from nfl_platform.schemas.fields import (
    LastNGamesField,
    PlayerField,
    SeasonsField,
    SeasonTypeField,
)

pytestmark = pytest.mark.anyio

# Arguments both adapters take, which must therefore mean the same thing in both.
SHARED_ARGUMENTS = ("player", "seasons", "last_n_games", "season_type", "team", "players")


def llm_schemas() -> dict[str, dict[str, Any]]:
    return {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in build_default_registry().schemas()
    }


def properties(schema: dict[str, Any], name: str) -> dict[str, Any] | None:
    return schema.get("properties", {}).get(name)


def description(entry: dict[str, Any] | None) -> str | None:
    """A field's description, wherever Pydantic put it for an optional type."""
    if entry is None:
        return None
    if "description" in entry:
        return str(entry["description"])
    for branch in entry.get("anyOf", []):
        if "description" in branch:
            return str(branch["description"])
    return None


class TestTheTwoCataloguesAgree:
    async def test_every_mcp_tool_name_exists_in_the_llm_registry(self, client: Client):
        """A capability should not be named two different things in two adapters."""
        listed = await client.list_tools()
        assert {tool.name for tool in listed.tools} <= set(llm_schemas())

    async def test_the_curated_subset_is_a_deliberate_subset(self, client: Client):
        listed = await client.list_tools()
        omitted = set(llm_schemas()) - {tool.name for tool in listed.tools}
        assert omitted == {
            "get_recent_games",
            "get_fantasy_usage_trend",
            "get_fantasy_matchup_context",
            "list_reference_data",
        }

    async def test_shared_arguments_are_described_identically(self, client: Client):
        """Both adapters read from schemas/fields.py; this proves they still do."""
        listed = await client.list_tools()
        llm = llm_schemas()
        checked = 0
        for tool in listed.tools:
            for argument in SHARED_ARGUMENTS:
                mcp_entry = properties(tool.input_schema, argument)
                llm_entry = properties(llm[tool.name], argument)
                if mcp_entry is None or llm_entry is None:
                    continue
                assert description(mcp_entry) == description(llm_entry), f"{tool.name}.{argument}"
                checked += 1
        assert checked > 10, "the parity check matched too little to be meaningful"

    async def test_shared_numeric_bounds_agree(self, client: Client):
        listed = await client.list_tools()
        llm = llm_schemas()
        for tool in listed.tools:
            mcp_entry = properties(tool.input_schema, "last_n_games")
            llm_entry = properties(llm[tool.name], "last_n_games")
            if mcp_entry is None or llm_entry is None:
                continue
            assert _bounds(mcp_entry) == _bounds(llm_entry), tool.name


def _bounds(entry: dict[str, Any]) -> tuple[Any, Any]:
    for branch in [entry, *entry.get("anyOf", [])]:
        if "minimum" in branch or "maximum" in branch:
            return branch.get("minimum"), branch.get("maximum")
    return None, None


class TestTheSharedFieldsAreActuallyShared:
    """If these aliases stopped being the single source, parity would pass vacuously."""

    def test_the_llm_tools_import_them_from_the_schema_layer(self):
        from nfl_platform.llm.tools import common

        assert common.PlayerField is PlayerField
        assert common.SeasonsField is SeasonsField
        assert common.LastNGamesField is LastNGamesField
        assert common.SeasonTypeField is SeasonTypeField
