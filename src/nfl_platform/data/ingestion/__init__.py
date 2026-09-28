"""Ingestion of nflverse datasets."""

from nfl_platform.data.ingestion.datasets import DATASETS, Dataset, DatasetSpec, get_dataset_spec
from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.data.ingestion.sources import (
    DataSource,
    NflreadpyDataSource,
    ParquetDataSource,
)

__all__ = [
    "DATASETS",
    "DataSource",
    "Dataset",
    "DatasetSpec",
    "NflDataLoader",
    "NflreadpyDataSource",
    "ParquetDataSource",
    "get_dataset_spec",
]
