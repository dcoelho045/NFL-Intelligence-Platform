"""Tests for injury analytics."""

from __future__ import annotations

import pytest

from nfl_platform.analytics import get_injury_information, get_team_injuries
from nfl_platform.data.ingestion import NflDataLoader
from nfl_platform.exceptions import InvalidWeekError
from support import MAHOMES_ID, UNKNOWN_PLAYER_ID


class TestPlayerInjuries:
    @staticmethod
    def _most_reported_player(loader: NflDataLoader) -> str:
        injuries = loader.load_injuries(2024)
        counts = injuries.group_by("gsis_id").len().sort("len", descending=True)
        return str(counts.item(0, "gsis_id"))

    def test_healthy_player_returns_empty_frame_not_an_error(self, loader: NflDataLoader):
        frame = get_injury_information(UNKNOWN_PLAYER_ID, loader=loader)
        assert frame.is_empty()
        assert "report_status" in frame.columns

    def test_known_injured_player_has_rows(self, loader: NflDataLoader):
        injuries = loader.load_injuries(2024)
        some_id = injuries.get_column("gsis_id").drop_nulls().to_list()[0]
        frame = get_injury_information(some_id, loader=loader)
        assert frame.height > 0
        assert frame.get_column("gsis_id").unique().to_list() == [some_id]

    def test_ordered_most_recent_first(self, loader: NflDataLoader):
        busiest = self._most_reported_player(loader)
        weeks = get_injury_information(busiest, loader=loader).get_column("week").to_list()
        assert weeks == sorted(weeks, reverse=True)

    def test_latest_only_returns_single_row(self, loader: NflDataLoader):
        busiest = self._most_reported_player(loader)
        assert get_injury_information(busiest, loader=loader, latest_only=True).height == 1

    def test_invalid_week_raises(self, loader: NflDataLoader):
        with pytest.raises(InvalidWeekError):
            get_injury_information(MAHOMES_ID, loader=loader, weeks=0)


class TestTeamInjuries:
    def test_week_filter(self, loader: NflDataLoader):
        frame = get_team_injuries("KC", loader=loader, week=2)
        assert frame.get_column("week").unique().to_list() == [2]

    def test_scoped_to_team(self, loader: NflDataLoader):
        frame = get_team_injuries("BUF", loader=loader)
        assert frame.get_column("team").unique().to_list() == ["BUF"]

    def test_unknown_team_returns_empty(self, loader: NflDataLoader):
        assert get_team_injuries("ZZZ", loader=loader).is_empty()
