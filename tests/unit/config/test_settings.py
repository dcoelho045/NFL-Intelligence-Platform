"""Tests for environment-driven settings."""

from __future__ import annotations

import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.data.ingestion import NflDataLoader


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
