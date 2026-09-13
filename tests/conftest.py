"""Shared pytest fixtures.

The default suite is fully offline: every loader is backed by the committed
Parquet fixtures in ``tests/fixtures/nflverse``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.data.ingestion import NflDataLoader, ParquetDataSource
from support import FIXTURE_SEASON, CountingDataSource

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "nflverse"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture
def settings() -> Settings:
    """Settings isolated from the developer's local .env file."""
    return Settings(_env_file=None, default_season=FIXTURE_SEASON)


@pytest.fixture
def fixture_source() -> ParquetDataSource:
    return ParquetDataSource(FIXTURE_DIR)


@pytest.fixture
def loader(fixture_source: ParquetDataSource, settings: Settings) -> NflDataLoader:
    """Offline loader pinned to the fixture season."""
    return NflDataLoader(
        source=fixture_source,
        settings=settings,
        current_season=FIXTURE_SEASON,
    )


@pytest.fixture
def counting_source(fixture_source: ParquetDataSource) -> CountingDataSource:
    return CountingDataSource(fixture_source)
