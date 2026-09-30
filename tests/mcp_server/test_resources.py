"""Resources: what an application loads, and what it must not be able to load.

The knowledge resources carry the security weight of this milestone. They are
the only place a caller-supplied string names a document, so they are the only
place a path could be constructed -- and the tests below pin the fact that it
cannot be.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp import Client
from mcp.types import TextResourceContents

from nfl_platform import __version__
from support import FIXTURE_SEASON

pytestmark = pytest.mark.anyio

FIXTURE_DOCUMENT_IDS = {"test-scoring", "test-usage", "test-conventions"}


async def read_json(client: Client, uri: str) -> Any:
    result = await client.read_resource(uri)
    contents = result.contents[0]
    assert isinstance(contents, TextResourceContents)
    return json.loads(contents.text)


class TestStatusResource:
    async def test_it_reports_what_this_deployment_can_do(self, client: Client):
        status = await read_json(client, "nfl://status")
        assert status["server_name"] == "nfl-intelligence"
        assert status["version"] == __version__
        assert status["data_layer_available"] is True
        assert status["knowledge_base_available"] is True

    async def test_it_distinguishes_enabled_from_configured(self, client: Client):
        """Two different reasons web search might not run; a client needs both."""
        status = await read_json(client, "nfl://status")
        assert "web_search_enabled" in status
        assert "web_search_available" in status

    async def test_it_carries_no_secret(self, client: Client):
        status = await read_json(client, "nfl://status")
        serialised = json.dumps(status).lower()
        assert "api_key" not in serialised
        assert "sk-" not in serialised
        assert "tvly" not in serialised


class TestReferenceResources:
    async def test_seasons_resource_names_the_current_season(self, client: Client):
        payload = await read_json(client, "nfl://reference/seasons")
        assert payload["current_season"] == FIXTURE_SEASON
        assert payload["teams"] is None

    async def test_teams_template_is_resolved_per_season(self, client: Client):
        payload = await read_json(client, f"nfl://reference/teams/{FIXTURE_SEASON}")
        assert payload["seasons"] == [FIXTURE_SEASON]
        assert "KC" in (payload["teams"] or [])

    async def test_a_non_numeric_season_is_rejected(self, client: Client):
        with pytest.raises(Exception) as caught:
            await client.read_resource("nfl://reference/teams/last-year")
        assert "season" in str(caught.value).lower()


class TestKnowledgeResources:
    async def test_the_corpus_lists_every_document_without_bodies(self, client: Client):
        payload = await read_json(client, "nfl://knowledge/documents")
        assert payload["document_count"] == len(FIXTURE_DOCUMENT_IDS)
        assert {item["document_id"] for item in payload["documents"]} == FIXTURE_DOCUMENT_IDS
        for item in payload["documents"]:
            assert "text" not in item
            assert item["characters"] > 0

    async def test_a_document_is_returned_verbatim_with_its_provenance(self, client: Client):
        payload = await read_json(client, "nfl://knowledge/documents/test-scoring")
        assert payload["document_id"] == "test-scoring"
        assert payload["updated"] == "2024-01-01"
        assert payload["source"] == "authored"
        assert "PPR means points per reception" in payload["text"]

    async def test_an_unknown_document_is_an_error_with_the_valid_ids(self, client: Client):
        with pytest.raises(Exception) as caught:
            await client.read_resource("nfl://knowledge/documents/does-not-exist")
        message = str(caught.value)
        assert "knowledge_document_not_found" in message
        assert "test-scoring" in message
