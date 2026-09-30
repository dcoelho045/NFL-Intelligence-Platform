"""The adapter's own seams, tested directly rather than through the protocol.

Everything else in this directory drives a client. These are the paths a client
cannot easily reach: an unexpected exception, a degraded data layer, the process
entry point.
"""

from __future__ import annotations

import json

import pytest
from mcp.server.mcpserver.exceptions import ResourceError, ToolError

from nfl_platform.exceptions import (
    DatasetNotAvailableError,
    NflPlatformError,
    PlayerNotFoundError,
    WebSearchNotConfiguredError,
)
from nfl_platform.mcp_server import errors
from nfl_platform.tool_errors import ToolErrorDetail


def payload(text: str) -> dict[str, object]:
    return json.loads(text[text.index("{") :])


class TestRendering:
    def test_a_rendered_error_is_greppable_and_parseable(self):
        text = errors.render_detail(
            ToolErrorDetail("some_code", "Something went wrong.", recovery="Try X.")
        )
        assert text.startswith("some_code: Something went wrong.")
        assert payload(text)["recovery"] == "Try X."

    def test_context_survives_the_round_trip(self):
        text = errors.render_detail(
            ToolErrorDetail(
                "team_not_found",
                "No such team.",
                recovery="Use a valid one.",
                context={"valid_teams": ["KC", "BUF"]},
            )
        )
        assert payload(text)["context"] == {"valid_teams": ["KC", "BUF"]}

    def test_only_unavailable_codes_carry_the_notice(self):
        available = errors.render_detail(
            ToolErrorDetail("player_not_found", "No player.", recovery="Search.")
        )
        unavailable = errors.render_detail(
            ToolErrorDetail("web_search_not_configured", "Off.", recovery="Do not retry.")
        )
        assert "notice" not in payload(available)
        assert "not evidence" in str(payload(unavailable)["notice"])

    @pytest.mark.parametrize("code", sorted(errors.UNAVAILABLE_CODES))
    def test_every_unavailable_code_says_it_is_not_a_finding(self, code: str):
        """A client must never read "could not check" as "nothing to find"."""
        text = errors.render_detail(ToolErrorDetail(code, "message", recovery="recovery"))
        assert "not evidence that no such information exists" in str(payload(text)["notice"])


class TestTranslation:
    def test_a_domain_error_keeps_its_taxonomy(self):
        error = errors.to_tool_error(PlayerNotFoundError("00-9999999", "the players directory"))
        assert isinstance(error, ToolError)
        assert payload(str(error))["code"] == "player_not_found"

    def test_an_unknown_exception_becomes_a_generic_error(self):
        """A bug must not leak its internals into a model's context."""
        error = errors.to_tool_error(RuntimeError("connection string: user:password@host"))
        text = str(error)
        assert "password" not in text
        assert payload(text)["code"] == "tool_error"

    def test_an_unavailable_data_source_is_not_retryable(self):
        error = errors.to_tool_error(DatasetNotAvailableError("schedules", "network down"))
        detail = payload(str(error))
        assert detail["code"] == "data_source_unavailable"
        assert detail["retryable"] is False
        assert "notice" in detail


class TestContextManagers:
    def test_tool_errors_translates_a_domain_failure(self):
        with pytest.raises(ToolError) as caught, errors.tool_errors():
            raise WebSearchNotConfiguredError("no key")
        assert "web_search_not_configured" in str(caught.value)

    def test_tool_errors_passes_an_existing_tool_error_through_unchanged(self):
        original = ToolError("already translated")
        with pytest.raises(ToolError) as caught, errors.tool_errors():
            raise original
        assert caught.value is original

    def test_tool_errors_is_transparent_on_success(self):
        with errors.tool_errors():
            value = 1 + 1
        assert value == 2

    def test_resource_errors_translates_a_domain_failure(self):
        with pytest.raises(ResourceError) as caught, errors.resource_errors():
            raise NflPlatformError("something broke")
        assert "internal_error" in str(caught.value)

    def test_resource_errors_passes_an_existing_resource_error_through(self):
        original = ResourceError("already translated")
        with pytest.raises(ResourceError) as caught, errors.resource_errors():
            raise original
        assert caught.value is original

    def test_resource_errors_hides_internals_from_an_unexpected_failure(self):
        with pytest.raises(ResourceError) as caught, errors.resource_errors():
            raise RuntimeError("/secret/path/on/disk")
        assert "/secret/path" not in str(caught.value)


class TestUnavailableHelper:
    def test_it_builds_a_non_retryable_error_with_the_notice(self):
        error = errors.unavailable(
            "web_search_not_configured", "Switched off.", recovery="Do not retry."
        )
        detail = payload(str(error))
        assert detail["retryable"] is False
        assert "notice" in detail
