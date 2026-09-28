"""Tests for the Parquet snapshot store."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.data.ingestion import Dataset, NflDataLoader, ParquetDataSource
from nfl_platform.data.storage import SnapshotStore
from nfl_platform.exceptions import DatasetNotAvailableError


class TestSnapshotStore:
    def test_round_trip(self, tmp_path, loader: NflDataLoader):
        store = SnapshotStore(tmp_path)
        original = loader.load_schedules(2024)
        store.write("week06", Dataset.SCHEDULES, original)
        restored = store.read("week06", Dataset.SCHEDULES)
        assert restored.equals(original)

    def test_manifest_records_metadata(self, tmp_path, loader: NflDataLoader):
        store = SnapshotStore(tmp_path)
        frame = loader.load_schedules(2024)
        store.write("week06", Dataset.SCHEDULES, frame)
        entry = store.manifest("week06")["schedules"]
        assert entry.rows == frame.height
        assert entry.columns == frame.width
        assert entry.seasons == [2024]
        assert entry.captured_at

    def test_manifest_accumulates_datasets(self, tmp_path, loader: NflDataLoader):
        store = SnapshotStore(tmp_path)
        store.write("s", Dataset.SCHEDULES, loader.load_schedules(2024))
        store.write("s", Dataset.INJURIES, loader.load_injuries(2024))
        assert set(store.manifest("s")) == {"schedules", "injuries"}

    def test_exists_reports_presence(self, tmp_path, loader: NflDataLoader):
        store = SnapshotStore(tmp_path)
        assert store.exists("s", Dataset.SCHEDULES) is False
        store.write("s", Dataset.SCHEDULES, loader.load_schedules(2024))
        assert store.exists("s", Dataset.SCHEDULES) is True

    def test_reading_a_missing_snapshot_raises(self, tmp_path):
        store = SnapshotStore(tmp_path)
        with pytest.raises(DatasetNotAvailableError):
            store.read("nope", Dataset.SCHEDULES)

    def test_manifest_of_missing_snapshot_is_empty(self, tmp_path):
        assert SnapshotStore(tmp_path).manifest("nope") == {}

    def test_snapshot_is_replayable_as_a_data_source(self, tmp_path, loader: NflDataLoader):
        """The point of snapshots: re-run an analysis against exactly the data it saw."""
        store = SnapshotStore(tmp_path)
        store.write("replay", Dataset.SCHEDULES, loader.load_schedules(2024))
        replay_loader = NflDataLoader(
            source=ParquetDataSource(tmp_path / "replay"),
            settings=Settings(_env_file=None),
            current_season=2024,
        )
        assert replay_loader.load_schedules(2024).height == loader.load_schedules(2024).height

    def test_frame_without_season_column_records_empty_seasons(self, tmp_path):
        store = SnapshotStore(tmp_path)
        store.write("s", Dataset.PLAYERS, pl.DataFrame({"gsis_id": ["00-0000001"]}))
        assert store.manifest("s")["players"].seasons == []
