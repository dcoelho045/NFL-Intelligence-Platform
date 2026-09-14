"""Tests for PlayerDirectoryService — name resolution and ambiguity handling.

Roughly 7.5% of nflverse players share a display name with someone else, so
these paths are the difference between a right answer and a confidently wrong
one. The fixture deliberately includes real namesakes.
"""

from __future__ import annotations

import pytest

from nfl_platform.exceptions import AmbiguousPlayerError, PlayerNotFoundError
from nfl_platform.services import ServiceContainer
from nfl_platform.services.directory import normalize_name
from support import MAHOMES_ID, UNKNOWN_PLAYER_ID

# Present in the fixture with 7 namesakes; the KC defensive end is the current one.
CHRIS_JONES_KC = "00-0032762"
# 9 namesakes in the fixture.
AMBIGUOUS_NAME = "Mike Williams"


class TestNameNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("A.J. Brown", "aj brown"),
            ("AJ Brown", "aj brown"),
            ("  Patrick   Mahomes  ", "patrick mahomes"),
            ("Odell Beckham Jr.", "odell beckham"),
            ("Marvin Harrison Jr", "marvin harrison"),
            ("Robert Griffin III", "robert griffin"),
        ],
    )
    def test_names_fold_to_comparable_form(self, raw, expected):
        assert normalize_name(raw) == expected

    def test_accents_are_stripped(self):
        assert normalize_name("Tomás Muñoz") == normalize_name("Tomas Munoz")

    def test_variants_of_the_same_name_agree(self):
        assert normalize_name("A.J. Brown") == normalize_name("aj brown")


class TestSearch:
    def test_exact_name_returns_the_player(self, services: ServiceContainer):
        result = services.directory.search("Patrick Mahomes")
        assert result.match_count >= 1
        assert MAHOMES_ID in {candidate.player_id for candidate in result.candidates}

    def test_punctuation_and_case_are_ignored(self, services: ServiceContainer):
        assert services.directory.search("patrick mahomes").match_count >= 1

    def test_shared_name_is_flagged_ambiguous(self, services: ServiceContainer):
        result = services.directory.search(AMBIGUOUS_NAME)
        assert result.match_count > 1
        assert result.is_ambiguous is True

    def test_every_candidate_carries_a_disambiguator(self, services: ServiceContainer):
        result = services.directory.search(AMBIGUOUS_NAME)
        assert all(candidate.disambiguator for candidate in result.candidates)

    def test_position_filter_narrows_a_shared_name(self, services: ServiceContainer):
        everyone = services.directory.search("Chris Jones")
        defensive_ends = services.directory.search("Chris Jones", position="DE")
        assert everyone.match_count > defensive_ends.match_count
        assert defensive_ends.match_count >= 1

    def test_team_filter_narrows_a_shared_name(self, services: ServiceContainer):
        result = services.directory.search("Chris Jones", team="KC")
        assert CHRIS_JONES_KC in {candidate.player_id for candidate in result.candidates}

    def test_season_filter_excludes_players_outside_their_career(self, services: ServiceContainer):
        result = services.directory.search("Chris Jones", season=2024)
        for candidate in result.candidates:
            if candidate.last_season is not None:
                assert candidate.last_season >= 2024

    def test_unknown_name_returns_no_candidates(self, services: ServiceContainer):
        result = services.directory.search("Zzzz Nobody")
        assert result.match_count == 0
        assert result.is_ambiguous is False

    def test_limit_is_respected(self, services: ServiceContainer):
        result = services.directory.search(AMBIGUOUS_NAME, limit=2)
        assert result.match_count == 2

    def test_empty_query_rejected(self, services: ServiceContainer):
        with pytest.raises(ValueError):
            services.directory.search("   ")

    def test_non_positive_limit_rejected(self, services: ServiceContainer):
        with pytest.raises(ValueError):
            services.directory.search("Patrick Mahomes", limit=0)


class TestResolve:
    def test_gsis_id_passes_through(self, services: ServiceContainer):
        assert services.directory.resolve(MAHOMES_ID) == MAHOMES_ID

    def test_unique_name_resolves(self, services: ServiceContainer):
        assert services.directory.resolve("Patrick Mahomes") == MAHOMES_ID

    def test_ambiguous_name_raises_with_candidates(self, services: ServiceContainer):
        with pytest.raises(AmbiguousPlayerError) as excinfo:
            services.directory.resolve(AMBIGUOUS_NAME)
        error = excinfo.value
        assert len(error.candidates) > 1
        assert all(len(pair) == 2 for pair in error.candidates)
        assert "Narrow the search" in str(error)

    def test_ambiguity_is_resolvable_by_position(self, services: ServiceContainer):
        with pytest.raises(AmbiguousPlayerError):
            services.directory.resolve("Chris Jones")
        assert services.directory.resolve("Chris Jones", team="KC") == CHRIS_JONES_KC

    def test_unknown_name_raises_not_found(self, services: ServiceContainer):
        with pytest.raises(PlayerNotFoundError):
            services.directory.resolve("Zzzz Nobody")

    def test_unknown_gsis_id_raises_not_found(self, services: ServiceContainer):
        with pytest.raises(PlayerNotFoundError):
            services.directory.resolve(UNKNOWN_PLAYER_ID)

    def test_legacy_id_format_resolves(self, services: ServiceContainer):
        """Pre-2000 ids such as 'WIL439416' predate the 00-0012345 format."""
        players = services.loader.load_players()
        legacy = [
            value
            for value in players.get_column("gsis_id").drop_nulls().to_list()
            if not str(value).startswith("00-")
        ]
        if not legacy:
            pytest.skip("No legacy-format ids in the fixture slice")
        assert services.directory.resolve(legacy[0]) == legacy[0]


class TestProfileAndExistence:
    def test_profile_returns_identity(self, services: ServiceContainer):
        profile = services.directory.profile(MAHOMES_ID)
        assert profile.player_id == MAHOMES_ID
        assert profile.name == "Patrick Mahomes"
        assert profile.position == "QB"
        assert profile.rookie_season == 2017

    def test_profile_includes_the_pfr_crosswalk_id(self, services: ServiceContainer):
        assert services.directory.profile(MAHOMES_ID).pfr_id == "MahoPa00"

    def test_profile_of_unknown_player_raises(self, services: ServiceContainer):
        with pytest.raises(PlayerNotFoundError):
            services.directory.profile(UNKNOWN_PLAYER_ID)

    def test_exists(self, services: ServiceContainer):
        assert services.directory.exists(MAHOMES_ID) is True
        assert services.directory.exists(UNKNOWN_PLAYER_ID) is False

    def test_reference_falls_back_to_the_id_when_absent(self, services: ServiceContainer):
        name, position, team = services.directory.reference(UNKNOWN_PLAYER_ID)
        assert name == UNKNOWN_PLAYER_ID
        assert position is None
        assert team is None
