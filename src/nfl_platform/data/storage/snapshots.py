"""Reproducible local Parquet snapshots.

nflverse data is mutable: stat corrections change historical rows. A snapshot
freezes the exact frames an analysis used so results can be reproduced later.
This is deliberately file-based; a database arrives when the API milestone needs
one, not before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nfl_platform.config.logging import get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.data.ingestion.datasets import Dataset, get_dataset_spec
from nfl_platform.exceptions import DatasetNotAvailableError

logger = get_logger(__name__)

_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    """Metadata describing one snapshotted dataset."""

    dataset: str
    rows: int
    columns: int
    seasons: list[int]
    captured_at: str


class SnapshotStore:
    """Writes and reads Parquet snapshots under a named directory."""

    def __init__(self, root: Path | str | None = None, settings: Settings | None = None) -> None:
        resolved_settings = settings or get_settings()
        self._root = Path(root) if root is not None else resolved_settings.resolved_snapshot_dir()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, name: str, dataset: Dataset | str) -> Path:
        spec = get_dataset_spec(dataset)
        return self._root / name / f"{spec.name.value}.parquet"

    def write(self, name: str, dataset: Dataset | str, frame: pl.DataFrame) -> Path:
        """Persist ``frame`` and record it in the snapshot manifest."""
        spec = get_dataset_spec(dataset)
        path = self.path_for(name, spec.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)

        seasons: list[int] = []
        if "season" in frame.columns:
            seasons = sorted(frame.get_column("season").unique().drop_nulls().to_list())

        self._record(
            name,
            SnapshotEntry(
                dataset=spec.name.value,
                rows=frame.height,
                columns=frame.width,
                seasons=seasons,
                captured_at=datetime.now(UTC).isoformat(),
            ),
        )
        logger.info(
            "Wrote snapshot",
            extra={"snapshot": name, "dataset": spec.name.value, "rows": frame.height},
        )
        return path

    def read(self, name: str, dataset: Dataset | str) -> pl.DataFrame:
        """Read a previously written snapshot."""
        spec = get_dataset_spec(dataset)
        path = self.path_for(name, spec.name)
        if not path.is_file():
            raise DatasetNotAvailableError(
                spec.name.value, f"no snapshot {name!r} at {path}"
            )
        return pl.read_parquet(path)

    def exists(self, name: str, dataset: Dataset | str) -> bool:
        return self.path_for(name, dataset).is_file()

    def manifest(self, name: str) -> dict[str, SnapshotEntry]:
        """Return the manifest for snapshot ``name``, keyed by dataset."""
        raw = self._read_manifest(name)
        return {
            key: SnapshotEntry(
                dataset=str(value["dataset"]),
                rows=int(value["rows"]),
                columns=int(value["columns"]),
                seasons=[int(season) for season in value["seasons"]],
                captured_at=str(value["captured_at"]),
            )
            for key, value in raw.items()
        }

    def _manifest_path(self, name: str) -> Path:
        return self._root / name / _MANIFEST_NAME

    def _read_manifest(self, name: str) -> dict[str, dict[str, Any]]:
        path = self._manifest_path(name)
        if not path.is_file():
            return {}
        content: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return content

    def _record(self, name: str, entry: SnapshotEntry) -> None:
        manifest = self._read_manifest(name)
        manifest[entry.dataset] = {
            "dataset": entry.dataset,
            "rows": entry.rows,
            "columns": entry.columns,
            "seasons": entry.seasons,
            "captured_at": entry.captured_at,
        }
        path = self._manifest_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
