"""Tests for domain validation primitives."""

from __future__ import annotations

import pytest

from nfl_platform.domain import (
    EARLIEST_SEASON,
    format_season_range,
    looks_like_gsis_id,
    normalize_player_id,
    normalize_seasons,
    normalize_team,
    normalize_weeks,
    validate_game_count,
    validate_season,
    validate_week,
)
from nfl_platform.exceptions import (
    InvalidGameCountError,
    InvalidSeasonError,
    InvalidWeekError,
)

LATEST = 2024


class TestSeasons:
    def test_none_resolves_to_default(self):
        assert normalize_seasons(None, default_season=2023, latest_season=LATEST) == [2023]

    def test_single_int(self):
        assert normalize_seasons(2022, default_season=2023, latest_season=LATEST) == [2022]

    def test_iterable_is_sorted_and_deduplicated(self):
        result = normalize_seasons([2023, 2021, 2023], default_season=2024, latest_season=LATEST)
        assert result == [2021, 2023]

    @pytest.mark.parametrize("season", [1998, 2030, "2023", 20.23])
    def test_invalid_season_rejected(self, season):
        with pytest.raises(InvalidSeasonError):
            validate_season(season, latest_season=LATEST)

    def test_bool_is_not_a_season(self):
        # bool is a subclass of int; nflreadpy uses True to mean "all seasons",
        # which this project deliberately does not accept implicitly.
        with pytest.raises(InvalidSeasonError):
            validate_season(True, latest_season=LATEST)

    def test_empty_iterable_rejected(self):
        with pytest.raises(InvalidSeasonError):
            normalize_seasons([], default_season=2023, latest_season=LATEST)

    def test_earliest_season_boundary(self):
        assert validate_season(EARLIEST_SEASON, latest_season=LATEST) == EARLIEST_SEASON
        with pytest.raises(InvalidSeasonError):
            validate_season(EARLIEST_SEASON - 1, latest_season=LATEST)


class TestWeeks:
    def test_none_means_all_weeks(self):
        assert normalize_weeks(None) is None

    def test_single_week(self):
        assert normalize_weeks(5) == [5]

    def test_iterable_sorted_and_deduplicated(self):
        assert normalize_weeks([4, 2, 4]) == [2, 4]

    @pytest.mark.parametrize("week", [0, 23, -1, "3", 2.5, True])
    def test_invalid_week_rejected(self, week):
        with pytest.raises(InvalidWeekError):
            validate_week(week)

    def test_empty_iterable_rejected(self):
        with pytest.raises(InvalidWeekError):
            normalize_weeks([])


class TestGameCount:
    def test_valid(self):
        assert validate_game_count(4) == 4

    @pytest.mark.parametrize("value", [0, -3, "4", 4.0, True])
    def test_invalid(self, value):
        with pytest.raises(InvalidGameCountError):
            validate_game_count(value)


class TestIdentifiers:
    def test_team_is_uppercased_and_trimmed(self):
        assert normalize_team("  kc ") == "KC"

    def test_empty_team_rejected(self):
        with pytest.raises(ValueError):
            normalize_team("   ")

    def test_non_string_team_rejected(self):
        with pytest.raises(TypeError):
            normalize_team(12)  # type: ignore[arg-type]

    def test_historical_abbreviation_is_preserved(self):
        # Rewriting OAK to LV would corrupt queries about the 2019 Raiders.
        assert normalize_team("OAK") == "OAK"

    def test_player_id_trimmed(self):
        assert normalize_player_id(" 00-0033873 ") == "00-0033873"

    def test_empty_player_id_rejected(self):
        with pytest.raises(ValueError):
            normalize_player_id("")

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("00-0033873", True), ("00-33873", False), ("QB-0033873", False), ("", False)],
    )
    def test_gsis_format_detection(self, value, expected):
        assert looks_like_gsis_id(value) is expected


class TestFormatting:
    @pytest.mark.parametrize(
        ("seasons", "expected"),
        [([], "<none>"), ([2023], "2023"), ([2021, 2022, 2023], "2021-2023")],
    )
    def test_format_season_range(self, seasons, expected):
        assert format_season_range(seasons) == expected
