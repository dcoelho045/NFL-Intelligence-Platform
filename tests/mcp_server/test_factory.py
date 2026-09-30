"""The server factory and the process entry point.

The other tests here assume a built server. These cover how it gets built: the
container injection that makes offline testing possible, the hardening applied
after registration, and the stdio entry point that a host actually launches.
"""

from __future__ import annotations

import json
from typing import Never

import pytest
from mcp import Client
from mcp.server import MCPServer
from mcp.types import TextResourceContents

from nfl_platform.config.settings import Settings
from nfl_platform.exceptions import DatasetNotAvailableError
from nfl_platform.mcp_server import build_mcp_server
from nfl_platform.mcp_server.__main__ import main
from nfl_platform.services import ServiceContainer
from nfl_platform.services.meta import MetaService


class TestFactory:
    def test_it_builds_without_touching_the_network_or_a_key(
        self, services: ServiceContainer, mcp_settings: Settings
    ):
        assert isinstance(build_mcp_server(container=services, settings=mcp_settings), MCPServer)

    def test_the_injected_container_is_the_one_used(self, mcp_settings: Settings):
        """Injection is what makes every other test in this directory offline."""
        calls: list[str] = []

        class RecordingContainer(ServiceContainer):
            @property
            def meta(self) -> MetaService:
                calls.append("meta")
                return super().meta

        container = RecordingContainer(settings=mcp_settings)
        build_mcp_server(container=container, settings=mcp_settings)
        assert calls == [], "building the server must not execute any service call"

    def test_the_name_comes_from_settings(self, services: ServiceContainer, mcp_settings: Settings):
        renamed = mcp_settings.model_copy(update={"mcp_server_name": "scouting-desk"})
        server = build_mcp_server(container=services, settings=renamed)
        assert server.name == "scouting-desk"


class TestArgumentHardening:
    def test_unknown_arguments_are_rejected(
        self, services: ServiceContainer, mcp_settings: Settings
    ):
        """Pins a guarantee built on SDK internals, so an upgrade cannot drop it.

        If this fails after an SDK bump, ``reject_unknown_arguments`` needs
        rewriting against the new structure -- not deleting.
        """
        server = build_mcp_server(container=services, settings=mcp_settings)
        for tool in server._tool_manager.list_tools():
            assert tool.fn_metadata.arg_model.model_config.get("extra") == "forbid", tool.name
            assert tool.parameters.get("additionalProperties") is False, tool.name


class TestDegradedDataLayer:
    """A broken data source must be reported, never silently answered around."""

    @pytest.fixture
    def broken_services(self, mcp_settings: Settings) -> ServiceContainer:
        class BrokenContainer(ServiceContainer):
            @property
            def loader(self) -> Never:
                raise DatasetNotAvailableError("schedules", "network unreachable")

        return BrokenContainer(settings=mcp_settings)

    @pytest.mark.anyio
    async def test_status_reports_the_data_layer_as_unavailable(
        self, broken_services: ServiceContainer, mcp_settings: Settings
    ):
        server = build_mcp_server(container=broken_services, settings=mcp_settings)
        async with Client(server) as client:
            result = await client.read_resource("nfl://status")
            contents = result.contents[0]
            assert isinstance(contents, TextResourceContents)
            status = json.loads(contents.text)

        assert status["data_layer_available"] is False
        assert status["current_season"] is None


class TestEntryPoint:
    def test_main_builds_a_server_and_serves_it(self, monkeypatch: pytest.MonkeyPatch):
        """The console script must wire settings, logging and stdio together."""
        transports: list[str] = []

        def fake_run(self: MCPServer, transport: str = "stdio", **kwargs: object) -> None:
            transports.append(transport)

        monkeypatch.setattr(MCPServer, "run", fake_run)
        assert main() == 0
        assert transports == ["stdio"]

    def test_main_writes_nothing_to_stdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """Under stdio the protocol owns stdout; a stray print corrupts the stream."""

        def fake_run(self: MCPServer, transport: str = "stdio", **kwargs: object) -> None:
            return None

        monkeypatch.setattr(MCPServer, "run", fake_run)
        main()
        assert capsys.readouterr().out == ""
