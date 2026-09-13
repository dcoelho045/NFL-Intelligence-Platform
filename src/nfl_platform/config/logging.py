"""Structured logging configuration.

Text output is used for local development; JSON output (``NFL_LOG_JSON=true``)
produces machine-parseable records for the observability milestone later on.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from nfl_platform.config.settings import Settings, get_settings

_ROOT_LOGGER_NAME = "nfl_platform"

# Attributes LogRecord always carries; anything else was attached via ``extra``.
_STANDARD_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON, including ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_KEYS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings | None = None) -> None:
    """Attach a single stderr handler to the package logger. Safe to call repeatedly."""
    resolved = settings or get_settings()
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(resolved.log_level)
    # Library code must not hijack the application's root logger configuration.
    logger.propagate = False

    for existing in list(logger.handlers):
        logger.removeHandler(existing)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if resolved.log_json
        else logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
    )
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger."""
    suffix = name.removeprefix(f"{_ROOT_LOGGER_NAME}.")
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{suffix}")
