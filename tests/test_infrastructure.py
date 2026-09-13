"""Tests for injury analytics, configuration, and snapshot storage."""

from __future__ import annotations

import json
import logging

import polars as pl
import pytest

from nfl_platform.analytics import get_injury_information, get_team_injuries
from nfl_platform.config.logging import JsonFormatter, configure_logging, get_logger
from nfl_platform.config.settings import Settings
from nfl_platform.data.ingestion import Dataset, NflDataLoader
from nfl_platform.data.storage import SnapshotStore
from nfl_platform.exceptions import DatasetNotAvailableError, InvalidWeekError
from support import MAHOMES_ID, UNKNOWN_PLAYER_ID


class TestInjuries:
    @staticmethod
    def _most_reported_player(loader: NflDataLoader) -> str:
        injuries = loader.load_injuries(2024)
        counts = injuries.group_by("gsis_id").len().sort("len", descending=True)
        return str(counts.item(0, "gsis_id"))

    def test_healthy_player_returns_empty_frame_not_an_error(self, loader: NflDataLoader):
        frame = get_injury_information(UNKNOWN_PLAYER_ID, loader=loader)
        assert frame.is_empty()
        assert "report_status" in frame.columns

    def test_known_injured_player_has_rows(self, loader: NflDataLoader):
        injuries = loader.load_injuries(2024)
        some_id = injuries.get_column("gsis_id").drop_nulls().to_list()[0]
        frame = get_injury_information(some_id, loader=loader)
        assert frame.height > 0
        assert frame.get_column("gsis_id").unique().to_list() == [some_id]

    def test_ordered_most_recent_first(self, loader: NflDataLoader):
        busiest = self._most_reported_player(loader)
        weeks = get_injury_information(busiest, loader=loader).get_column("week").to_list()
        assert weeks == sorted(weeks, reverse=True)

    def test_latest_only_returns_single_row(self, loader: NflDataLoader):
        busiest = self._most_reported_player(loader)
        assert get_injury_information(busiest, loader=loader, latest_only=True).height == 1

    def test_week_filter(self, loader: NflDataLoader):
        frame = get_team_injuries("KC", loader=loader, week=2)
        assert frame.get_column("week").unique().to_list() == [2]

    def test_invalid_week_raises(self, loader: NflDataLoader):
        with pytest.raises(InvalidWeekError):
            get_injury_information(MAHOMES_ID, loader=loader, weeks=0)

    def test_team_injuries_scoped_to_team(self, loader: NflDataLoader):
        frame = get_team_injuries("BUF", loader=loader)
        assert frame.get_column("team").unique().to_list() == ["BUF"]

    def test_unknown_team_returns_empty(self, loader: NflDataLoader):
        assert get_team_injuries("ZZZ", loader=loader).is_empty()


class TestSettings:
    def test_defaults_do_not_require_env(self):
        settings = Settings(_env_file=None)
        assert settings.cache_mode == "filesystem"
        assert settings.default_season is None
        assert settings.log_level == "INFO"

    def test_env_prefix_is_honoured(self, monkeypatch):
        monkeypatch.setenv("NFL_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("NFL_DEFAULT_SEASON", "2022")
        settings = Settings(_env_file=None)
        assert settings.log_level == "DEBUG"
        assert settings.default_season == 2022

    def test_blank_default_season_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("NFL_DEFAULT_SEASON", "  ")
        assert Settings(_env_file=None).default_season is None

    def test_invalid_log_level_rejected(self, monkeypatch):
        monkeypatch.setenv("NFL_LOG_LEVEL", "CHATTY")
        with pytest.raises(ValueError):
            Settings(_env_file=None)

    def test_paths_are_resolved_absolute(self):
        settings = Settings(_env_file=None)
        assert settings.resolved_cache_dir().is_absolute()
        assert settings.resolved_snapshot_dir().is_absolute()

    def test_loader_default_season_falls_back_to_current(self, fixture_source):
        loader = NflDataLoader(
            source=fixture_source,
            settings=Settings(_env_file=None),
            current_season=2024,
        )
        assert loader.default_season == 2024


class TestLogging:
    def test_json_formatter_includes_extra_fields(self):
        record = logging.LogRecord(
            name="nfl_platform.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Loaded dataset",
            args=None,
            exc_info=None,
        )
        record.dataset = "schedules"
        payload = json.loads(JsonFormatter().format(record))
        assert payload["message"] == "Loaded dataset"
        assert payload["level"] == "INFO"
        assert payload["dataset"] == "schedules"

    def test_configure_logging_is_idempotent(self):
        settings = Settings(_env_file=None)
        configure_logging(settings)
        configure_logging(settings)
        assert len(logging.getLogger("nfl_platform").handlers) == 1

    def test_package_logger_does_not_propagate(self):
        configure_logging(Settings(_env_file=None))
        assert logging.getLogger("nfl_platform").propagate is False

    def test_get_logger_is_namespaced(self):
        assert get_logger("nfl_platform.data.x").name == "nfl_platform.data.x"


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
        from nfl_platform.data.ingestion import ParquetDataSource

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
