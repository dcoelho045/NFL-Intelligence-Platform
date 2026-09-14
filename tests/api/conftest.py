"""Fixtures for the API tests.

The application is built with the offline fixture container, so these exercise
the real routing, validation, serialisation, and error-mapping stack without
touching the network.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from nfl_platform.api.app import create_app
from nfl_platform.services import ServiceContainer

# Nine players share this name in the fixture slice.
AMBIGUOUS_NAME = "Mike Williams"


@pytest.fixture
def client(services: ServiceContainer) -> Iterator[TestClient]:
    with TestClient(create_app(container=services)) as test_client:
        yield test_client
