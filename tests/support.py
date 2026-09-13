"""Test doubles shared across the suite.

Kept out of ``conftest.py`` so test modules can import these directly without
relying on the conftest module being importable as a package.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from nfl_platform.data.ingestion import Dataset, ParquetDataSource

FIXTURE_SEASON = 2024

# Identifiers that exist in the committed fixture slice.
MAHOMES_ID = "00-0033873"
ALLEN_ID = "00-0034857"
UNKNOWN_PLAYER_ID = "00-9999999"


class CountingDataSource:
    """Wraps a data source and records how many times each dataset was fetched."""

    def __init__(self, inner: ParquetDataSource) -> None:
        self._inner = inner
        self.calls: list[Dataset] = []

    def fetch(self, dataset: Dataset, seasons: list[int] | None, **kwargs: Any) -> pl.DataFrame:
        self.calls.append(dataset)
        return self._inner.fetch(dataset, seasons, **kwargs)


class StubDataSource:
    """Returns caller-supplied frames, for exercising validation and error paths."""

    def __init__(self, frames: dict[Dataset, pl.DataFrame]) -> None:
        self._frames = frames

    def fetch(
        self,
        dataset: Dataset,
        seasons: list[int] | None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        return self._frames[dataset]
