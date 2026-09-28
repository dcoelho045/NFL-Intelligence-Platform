"""Tests for the dataset registry, loader, schema validation, and caching."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_platform.data.ingestion import (
    DATASETS,
    Dataset,
    NflDataLoader,
    ParquetDataSource,
    get_dataset_spec,
)
from nfl_platform.data.ingestion.validation import select_columns, validate_schema
from nfl_platform.exceptions import (
    DatasetNotAvailableError,
    InvalidSeasonError,
    SchemaValidationError,
    UnknownDatasetError,
)
from support import CountingDataSource, StubDataSource


class TestRegistry:
    def test_every_enum_member_has_a_spec(self):
        assert set(DATASETS) == set(Dataset)

    def test_lookup_by_string(self):
        spec = get_dataset_spec("schedules")
        assert spec.name is Dataset.SCHEDULES
        assert spec.nflreadpy_function == "load_schedules"

    def test_unknown_dataset_raises(self):
        with pytest.raises(UnknownDatasetError) as excinfo:
            get_dataset_spec("touchdowns")
        assert "touchdowns" in str(excinfo.value)

    def test_players_is_not_season_aware(self):
        assert get_dataset_spec(Dataset.PLAYERS).season_aware is False


class TestSuccessfulLoading:
    @pytest.mark.parametrize("dataset", list(Dataset))
    def test_every_dataset_loads_and_validates(self, loader: NflDataLoader, dataset: Dataset):
        frame = loader.load(dataset)
        assert isinstance(frame, pl.DataFrame)
        assert frame.height > 0
        assert get_dataset_spec(dataset).required_columns <= set(frame.columns)

    def test_typed_accessors_match_generic_load(self, loader: NflDataLoader):
        assert loader.load_schedules().height == loader.load(Dataset.SCHEDULES).height
        assert loader.load_players().height == loader.load(Dataset.PLAYERS).height

    def test_season_filtering_is_applied(self, loader: NflDataLoader):
        frame = loader.load_player_stats(2024)
        assert frame.get_column("season").unique().to_list() == [2024]

    def test_unavailable_season_yields_empty_frame_not_an_error(self, loader: NflDataLoader):
        # 2023 is a valid season but absent from the fixture slice.
        frame = loader.load_player_stats(2023)
        assert frame.is_empty()
        assert "player_id" in frame.columns

    def test_available_seasons_reports_only_populated_seasons(self, loader: NflDataLoader):
        assert loader.available_seasons(Dataset.PLAYER_STATS, [2023, 2024]) == [2024]


class TestInvalidInput:
    def test_season_below_range_rejected(self, loader: NflDataLoader):
        with pytest.raises(InvalidSeasonError):
            loader.load_player_stats(1990)

    def test_future_season_rejected(self, loader: NflDataLoader):
        with pytest.raises(InvalidSeasonError):
            loader.load_player_stats(2099)

    def test_missing_parquet_file_raises(self, tmp_path):
        empty_loader = NflDataLoader(
            source=ParquetDataSource(tmp_path),
            current_season=2024,
        )
        with pytest.raises(DatasetNotAvailableError) as excinfo:
            empty_loader.load_schedules(2024)
        assert "schedules" in str(excinfo.value)


class TestSchemaValidation:
    def test_missing_required_column_raises(self):
        spec = get_dataset_spec(Dataset.SCHEDULES)
        incomplete = pl.DataFrame({"game_id": ["2024_01_BAL_KC"], "season": [2024]})
        with pytest.raises(SchemaValidationError) as excinfo:
            validate_schema(incomplete, spec)
        assert excinfo.value.dataset == "schedules"
        assert "home_team" in excinfo.value.missing_columns

    def test_extra_upstream_columns_are_tolerated(self, loader: NflDataLoader):
        frame = loader.load_schedules()
        spec = get_dataset_spec(Dataset.SCHEDULES)
        assert len(frame.columns) > len(spec.required_columns)

    def test_loader_surfaces_schema_errors(self):
        stub = StubDataSource({Dataset.INJURIES: pl.DataFrame({"season": [2024]})})
        broken_loader = NflDataLoader(source=stub, current_season=2024)
        with pytest.raises(SchemaValidationError):
            broken_loader.load_injuries(2024)

    def test_column_projection_keeps_required_columns(self, loader: NflDataLoader):
        frame = loader.load_play_by_play(2024, columns=["yards_gained"])
        spec = get_dataset_spec(Dataset.PLAY_BY_PLAY)
        assert "yards_gained" in frame.columns
        assert spec.required_columns <= set(frame.columns)
        assert frame.width < 20

    def test_projection_of_unknown_column_raises(self, loader: NflDataLoader):
        with pytest.raises(SchemaValidationError):
            loader.load_play_by_play(2024, columns=["not_a_real_column"])

    def test_select_columns_passthrough_when_none(self):
        spec = get_dataset_spec(Dataset.SCHEDULES)
        frame = pl.DataFrame({column: [1] for column in spec.required_columns})
        assert select_columns(frame, None, spec) is frame


class TestCaching:
    def test_repeated_loads_hit_the_source_once(self, counting_source: CountingDataSource):
        cached_loader = NflDataLoader(source=counting_source, current_season=2024)
        cached_loader.load_schedules(2024)
        cached_loader.load_schedules(2024)
        assert counting_source.calls.count(Dataset.SCHEDULES) == 1

    def test_different_seasons_are_cached_separately(self, counting_source: CountingDataSource):
        cached_loader = NflDataLoader(source=counting_source, current_season=2024)
        cached_loader.load_schedules(2024)
        cached_loader.load_schedules(2023)
        assert counting_source.calls.count(Dataset.SCHEDULES) == 2

    def test_clear_cache_forces_refetch(self, counting_source: CountingDataSource):
        cached_loader = NflDataLoader(source=counting_source, current_season=2024)
        cached_loader.load_schedules(2024)
        cached_loader.clear_cache()
        cached_loader.load_schedules(2024)
        assert counting_source.calls.count(Dataset.SCHEDULES) == 2

    def test_cache_can_be_disabled(self, counting_source: CountingDataSource):
        cached_loader = NflDataLoader(
            source=counting_source, current_season=2024, enable_cache=False
        )
        cached_loader.load_schedules(2024)
        cached_loader.load_schedules(2024)
        assert counting_source.calls.count(Dataset.SCHEDULES) == 2
