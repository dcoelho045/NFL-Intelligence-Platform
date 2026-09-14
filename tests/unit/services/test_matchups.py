"""Tests for MatchupAnalyticsService."""

from __future__ import annotations

import pytest

from nfl_platform.exceptions import InvalidWeekError, TeamNotFoundError
from nfl_platform.services import ServiceContainer


class TestMatchup:
    def test_returns_both_sides(self, services: ServiceContainer):
        analysis = services.matchups.analyze("KC", "BUF", seasons=2024)
        assert analysis.team_a.team == "KC"
        assert analysis.team_b.team == "BUF"

    def test_each_side_has_offense_defense_and_form(self, services: ServiceContainer):
        analysis = services.matchups.analyze("KC", "BUF", seasons=2024)
        assert analysis.team_a.offense.team == "KC"
        assert analysis.team_a.defense.team == "KC"
        assert analysis.team_a.form.team == "KC"

    def test_identical_teams_rejected(self, services: ServiceContainer):
        with pytest.raises(ValueError):
            services.matchups.analyze("KC", "KC", seasons=2024)

    def test_unknown_team_raises(self, services: ServiceContainer):
        with pytest.raises(TeamNotFoundError):
            services.matchups.analyze("KC", "ZZZ", seasons=2024)

    def test_note_flags_a_missing_fixture(self, services: ServiceContainer):
        """KC and BUF do not meet in the fixture's week 1-6 slice."""
        analysis = services.matchups.analyze("KC", "BUF", seasons=2024)
        assert analysis.scheduled_game is None
        assert analysis.note is not None
        assert "No scheduled meeting" in analysis.note

    def test_contains_no_prediction_fields(self, services: ServiceContainer):
        """Matchup analysis is descriptive; prediction belongs to a later milestone."""
        fields = set(services.matchups.analyze("KC", "BUF", seasons=2024).model_dump())
        assert not {"win_probability", "predicted_score", "projection"} & fields

    def test_invalid_week_raises(self, services: ServiceContainer):
        with pytest.raises(InvalidWeekError):
            services.matchups.analyze("KC", "BUF", seasons=2024, week=99)
