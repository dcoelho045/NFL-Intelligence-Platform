"""Data sources: the seam between this project and nflverse.

``DataSource`` is the only place that talks to the network. Analytics and tests
depend on the protocol, not on ``nflreadpy``, which is what allows the entire
default test suite to run offline against fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import polars as pl

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.data.ingestion.datasets import Dataset, get_dataset_spec
from nfl_platform.exceptions import DatasetNotAvailableError

logger = get_logger(__name__)


@runtime_checkable
class DataSource(Protocol):
    """Fetches a raw dataset frame. Implementations must not filter or reshape."""

    def fetch(self, dataset: Dataset, seasons: list[int] | None, **kwargs: Any) -> pl.DataFrame:
        """Return the raw frame for ``dataset``, covering ``seasons`` when applicable."""
        ...


class NflreadpyDataSource:
    """Live nflverse source backed by ``nflreadpy``.

    Download caching is delegated to nflreadpy rather than reimplemented here;
    it already provides filesystem caching with a configurable TTL.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._configured = False

    def _ensure_configured(self) -> None:
        if self._configured:
            return
        import nflreadpy
        from nflreadpy.config import update_config

        cache_dir = self._settings.resolved_cache_dir()
        if self._settings.cache_mode == "filesystem":
            cache_dir.mkdir(parents=True, exist_ok=True)

        update_config(
            cache_mode=self._settings.cache_mode,
            cache_dir=cache_dir,
            cache_duration=self._settings.cache_duration_seconds,
            timeout=self._settings.request_timeout_seconds,
            verbose=False,
        )
        self._nflreadpy = nflreadpy
        self._configured = True

    def fetch(self, dataset: Dataset, seasons: list[int] | None, **kwargs: Any) -> pl.DataFrame:
        self._ensure_configured()
        spec = get_dataset_spec(dataset)
        loader = getattr(self._nflreadpy, spec.nflreadpy_function, None)
        if loader is None:
            raise DatasetNotAvailableError(
                spec.name.value,
                f"nflreadpy has no function {spec.nflreadpy_function!r}; "
                "the installed nflreadpy version may be incompatible.",
            )

        call_kwargs = dict(kwargs)
        if spec.season_aware:
            if seasons is None:
                raise DatasetNotAvailableError(
                    spec.name.value, "seasons must be supplied for a season-aware dataset"
                )
            call_kwargs["seasons"] = seasons

        logger.debug(
            "Fetching dataset from nflverse",
            extra={"dataset": spec.name.value, "seasons": seasons},
        )
        try:
            frame = loader(**call_kwargs)
        except Exception as error:  # nflreadpy raises transport/parse errors of many types
            raise DatasetNotAvailableError(spec.name.value, str(error)) from error

        if not isinstance(frame, pl.DataFrame):
            raise DatasetNotAvailableError(
                spec.name.value, f"expected a Polars DataFrame, got {type(frame).__name__}"
            )
        return frame


class ParquetDataSource:
    """Reads datasets from local Parquet files named ``<dataset>.parquet``.

    Backs both the test fixtures and reproducible replays of a snapshot
    directory, so an analysis can be re-run against frozen data.
    """

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)

    @property
    def directory(self) -> Path:
        return self._directory

    def fetch(
        self,
        dataset: Dataset,
        seasons: list[int] | None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        spec = get_dataset_spec(dataset)
        path = self._directory / f"{spec.name.value}.parquet"
        if not path.is_file():
            raise DatasetNotAvailableError(spec.name.value, f"no local Parquet file at {path}")
        try:
            frame = pl.read_parquet(path)
        except Exception as error:
            raise DatasetNotAvailableError(
                spec.name.value, f"could not read {path}: {error}"
            ) from error

        if spec.season_aware and seasons is not None and "season" in frame.columns:
            # Dtypes vary upstream (ff_opportunity stores season as a string).
            frame = frame.filter(pl.col("season").cast(pl.Int32, strict=False).is_in(seasons))
        return frame
