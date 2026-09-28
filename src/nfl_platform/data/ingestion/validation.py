"""Schema validation for loaded datasets."""

from __future__ import annotations

import polars as pl

from nfl_platform.data.ingestion.datasets import DatasetSpec
from nfl_platform.exceptions import SchemaValidationError

# nflverse is not internally consistent: ff_opportunity ships ``season`` as a
# string and ``week`` as a float, while every other dataset uses Int32. Casting
# here means downstream joins and filters never have to care.
_KEY_COLUMN_DTYPES: dict[str, pl.DataType] = {
    "season": pl.Int32(),
    "week": pl.Int32(),
}


def normalize_key_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Coerce ``season`` and ``week`` to Int32 wherever they appear.

    Uses a non-strict cast so an unparseable value becomes null rather than
    aborting the whole load.
    """
    casts = [
        pl.col(column).cast(dtype, strict=False).alias(column)
        for column, dtype in _KEY_COLUMN_DTYPES.items()
        if column in frame.columns and frame.schema[column] != dtype
    ]
    return frame.with_columns(casts) if casts else frame


def validate_schema(frame: pl.DataFrame, spec: DatasetSpec) -> pl.DataFrame:
    """Assert that ``frame`` carries every column this project reads.

    Extra upstream columns are allowed and passed through untouched.
    """
    missing = spec.required_columns - set(frame.columns)
    if missing:
        raise SchemaValidationError(spec.name.value, frozenset(missing))
    return frame


def select_columns(
    frame: pl.DataFrame, columns: list[str] | None, spec: DatasetSpec
) -> pl.DataFrame:
    """Project ``frame`` down to ``columns``, keeping the dataset's required columns.

    Used to keep play-by-play memory bounded without losing the columns needed to
    identify a row.
    """
    if columns is None:
        return frame
    requested = set(columns) | spec.required_columns
    missing = requested - set(frame.columns)
    if missing:
        raise SchemaValidationError(spec.name.value, frozenset(missing))
    ordered = [column for column in frame.columns if column in requested]
    return frame.select(ordered)
