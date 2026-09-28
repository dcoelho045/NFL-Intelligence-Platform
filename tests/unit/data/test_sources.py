"""Tests for data-source error handling.

The live source is exercised here without network access by substituting a fake
nflreadpy module, so failure modes are verified in the default offline suite.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.data.ingestion import Dataset, NflreadpyDataSource, ParquetDataSource
from nfl_platform.exceptions import DatasetNotAvailableError


class FakeNflreadpy:
    """Minimal stand-in for the nflreadpy module."""

    def __init__(self, **functions: Any) -> None:
        for name, function in functions.items():
            setattr(self, name, function)


def _configured_source(fake_module: object) -> NflreadpyDataSource:
    source = NflreadpyDataSource(Settings(_env_file=None))
    # Bypass network configuration; the fake module replaces nflreadpy entirely.
    source._configured = True
    source._nflreadpy = fake_module  # type: ignore[attr-defined]
    return source


class TestNflreadpySource:
    def test_missing_loader_function_raises(self):
        source = _configured_source(FakeNflreadpy())
        with pytest.raises(DatasetNotAvailableError) as excinfo:
            source.fetch(Dataset.SCHEDULES, [2024])
        assert "load_schedules" in str(excinfo.value)

    def test_upstream_exception_is_wrapped(self):
        def explode(**_: Any) -> pl.DataFrame:
            raise ConnectionError("nflverse unreachable")

        source = _configured_source(FakeNflreadpy(load_schedules=explode))
        with pytest.raises(DatasetNotAvailableError) as excinfo:
            source.fetch(Dataset.SCHEDULES, [2024])
        assert "nflverse unreachable" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ConnectionError)

    def test_non_dataframe_result_raises(self):
        source = _configured_source(FakeNflreadpy(load_schedules=lambda **_: {"not": "a frame"}))
        with pytest.raises(DatasetNotAvailableError) as excinfo:
            source.fetch(Dataset.SCHEDULES, [2024])
        assert "DataFrame" in str(excinfo.value)

    def test_season_aware_dataset_requires_seasons(self):
        source = _configured_source(FakeNflreadpy(load_schedules=lambda **_: pl.DataFrame()))
        with pytest.raises(DatasetNotAvailableError) as excinfo:
            source.fetch(Dataset.SCHEDULES, None)
        assert "seasons must be supplied" in str(excinfo.value)

    def test_seasons_are_forwarded_to_nflreadpy(self):
        captured: dict[str, Any] = {}

        def capture(**kwargs: Any) -> pl.DataFrame:
            captured.update(kwargs)
            return pl.DataFrame()

        source = _configured_source(FakeNflreadpy(load_schedules=capture))
        source.fetch(Dataset.SCHEDULES, [2023, 2024])
        assert captured["seasons"] == [2023, 2024]

    def test_season_free_dataset_omits_seasons(self):
        captured: dict[str, Any] = {}

        def capture(**kwargs: Any) -> pl.DataFrame:
            captured.update(kwargs)
            return pl.DataFrame()

        source = _configured_source(FakeNflreadpy(load_players=capture))
        source.fetch(Dataset.PLAYERS, None)
        assert "seasons" not in captured


class TestParquetSource:
    def test_missing_file_raises(self, tmp_path):
        source = ParquetDataSource(tmp_path)
        with pytest.raises(DatasetNotAvailableError) as excinfo:
            source.fetch(Dataset.SCHEDULES, [2024])
        assert "no local Parquet file" in str(excinfo.value)

    def test_corrupt_file_raises(self, tmp_path):
        (tmp_path / "schedules.parquet").write_bytes(b"this is not parquet")
        source = ParquetDataSource(tmp_path)
        with pytest.raises(DatasetNotAvailableError) as excinfo:
            source.fetch(Dataset.SCHEDULES, [2024])
        assert "could not read" in str(excinfo.value)

    def test_directory_is_exposed(self, tmp_path):
        assert ParquetDataSource(tmp_path).directory == tmp_path
