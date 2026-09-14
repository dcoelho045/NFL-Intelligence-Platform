"""Tests for structured logging configuration."""

from __future__ import annotations

import json
import logging

from nfl_platform.config.logging import JsonFormatter, configure_logging, get_logger
from nfl_platform.config.settings import Settings


class TestJsonFormatter:
    def test_includes_extra_fields(self):
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


class TestConfiguration:
    def test_configure_logging_is_idempotent(self):
        settings = Settings(_env_file=None)
        configure_logging(settings)
        configure_logging(settings)
        assert len(logging.getLogger("nfl_platform").handlers) == 1

    def test_package_logger_does_not_propagate(self):
        """Library code must not hijack the application's root logger."""
        configure_logging(Settings(_env_file=None))
        assert logging.getLogger("nfl_platform").propagate is False

    def test_get_logger_is_namespaced(self):
        assert get_logger("nfl_platform.data.x").name == "nfl_platform.data.x"
