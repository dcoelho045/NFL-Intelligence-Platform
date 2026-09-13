"""Configuration and logging setup."""

from nfl_platform.config.logging import configure_logging, get_logger
from nfl_platform.config.settings import Settings, get_settings

__all__ = ["Settings", "configure_logging", "get_logger", "get_settings"]
