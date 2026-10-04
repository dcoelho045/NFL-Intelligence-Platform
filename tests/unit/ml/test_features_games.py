"""Game feature construction.

Split into three concerns: the values are right, nothing from the future reaches
them, and the declared contract matches what is actually built.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_platform.ml.features.games import (
    GAME_FEATURE_SCHEMA_VERSION,
    GAME_FEATURES,
    LABEL,
    build_game_features,
    build_team_form,
    build_team_game_frame,
    training_rows,
)
from nfl_platform.ml.features.schema import Availability
from synthetic import SyntheticLeague


@pytest.fixture(scope="module")
def league() -> SyntheticLeague:
    from synthetic import build_league

    return build_league()


@pytest.fixture(scope="module")
def features(league: SyntheticLeague) -> pl.DataFrame:
    return build_game_features(league.schedules, league.team_stats)


def _schedule(rows: list[dict[str, object]]) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "game_type": "REG",
        "gameday": "2023-09-10",
        "home_rest": 7.0,
        "away_rest": 7.0,
        "div_game": 0,
        "roof": "outdoors",
        "surface": "grass",
        "location": "Home",
        "spread_line": 0.0,
        "total_line": 44.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


class TestTeamGameFrame:
    def test_each_game_becomes_two_rows(self):
        frame = build_team_game_frame(
            _schedule(
                [
                    {
                        "game_id": "g1",
                        "season": 2023,
                        "week": 1,
                        "home_team": "AAA",
                        "away_team": "BBB",
                        "home_score": 24.0,
                        "away_score": 17.0,
                    }
                ]
            )
        )
        assert frame.height == 2
        assert set(frame.get_column("team").to_list()) == {"AAA", "BBB"}

    def test_points_are_mirrored_between_the_two_sides(self):
        frame = build_team_game_frame(
            _schedule(
                [
                    {
                        "game_id": "g1",
                        "season": 2023,
                        "week": 1,
                        "home_team": "AAA",
                        "away_team": "BBB",
                        "home_score": 24.0,
                        "away_score": 17.0,
                    }
                ]
            )
        )
        rows = {row["team"]: row for row in frame.to_dicts()}
        assert rows["AAA"]["points_for"] == 24.0
        assert rows["BBB"]["points_against"] == 24.0

    def test_a_tie_counts_as_half_a_win_in_the_form_record(self):
        """Ties leave the label but must not leave a team's record."""
        frame = build_team_game_frame(
            _schedule(
                [
                    {
                        "game_id": "g1",
                        "season": 2023,
                        "week": 1,
                        "home_team": "AAA",
                        "away_team": "BBB",
                        "home_score": 20.0,
                        "away_score": 20.0,
                    }
                ]
            )
        )
        assert frame.get_column("wins").to_list() == [0.5, 0.5]

    def test_unplayed_games_survive_and_are_marked_incomplete(self):
        frame = build_team_game_frame(
            _schedule(
                [
                    {
                        "game_id": "g1",
                        "season": 2023,
                        "week": 1,
                        "home_team": "AAA",
                        "away_team": "BBB",
                        "home_score": None,
                        "away_score": None,
                    }
                ]
            )
        )
        assert frame.height == 2
        assert frame.get_column("completed").to_list() == [False, False]

    def test_venue_flags_are_case_insensitive(self):
        frame = build_team_game_frame(
            _schedule(
                [
                    {
                        "game_id": "g1",
                        "season": 2023,
                        "week": 1,
                        "home_team": "AAA",
                        "away_team": "BBB",
                        "home_score": 1.0,
                        "away_score": 0.0,
                        "roof": "DOME",
                        "surface": "FieldTurf",
                        "location": "Neutral",
                    }
                ]
            )
        )
        row = frame.to_dicts()[0]
        assert (row["is_dome"], row["is_turf"], row["neutral_site"]) == (1, 1, 1)

    def test_box_scores_are_optional(self):
        """Without team_stats the efficiency columns are null, not missing."""
        frame = build_team_game_frame(
            _schedule(
                [
                    {
                        "game_id": "g1",
                        "season": 2023,
                        "week": 1,
                        "home_team": "AAA",
                        "away_team": "BBB",
                        "home_score": 7.0,
                        "away_score": 3.0,
                    }
                ]
            ),
            None,
        )
        assert frame.item(0, "epa_off") is None


class TestTeamForm:
    def test_only_completed_games_enter_the_history(self, league: SyntheticLeague):
        schedules = league.schedules.with_columns(
            pl.when(pl.col("season") == league.seasons[-1])
            .then(None)
            .otherwise(pl.col("home_score"))
            .alias("home_score")
        )
        form = build_team_form(build_team_game_frame(schedules, league.team_stats))
        assert league.seasons[-1] not in form.get_column("season").unique().to_list()

    def test_form_is_empty_when_nothing_has_been_played(self):
        frame = build_team_game_frame(
            _schedule(
                [
                    {
                        "game_id": "g1",
                        "season": 2023,
                        "week": 1,
                        "home_team": "AAA",
                        "away_team": "BBB",
                        "home_score": None,
                        "away_score": None,
                    }
                ]
            )
        )
        assert build_team_form(frame).is_empty()


class TestGameFeatures:
    def test_one_row_per_game(self, features: pl.DataFrame, league: SyntheticLeague):
        """Two rows per game would let one game straddle a train/test split."""
        played = league.schedules.filter(pl.col("home_score").is_not_null())
        assert features.height == played.height
        assert features.get_column("game_id").n_unique() == features.height

    def test_every_declared_feature_is_present(self, features: pl.DataFrame):
        missing = [name for name in GAME_FEATURES.names if name not in features.columns]
        assert not missing

    def test_no_undeclared_numeric_feature_sneaks_in(self, features: pl.DataFrame):
        """A feature with no FeatureSpec cannot be audited for availability."""
        identity = {
            "game_id",
            "season",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "completed",
            "spread_line",
            "total_line",
            "home_history_games",
            "away_history_games",
            "time_index",
            LABEL,
        }
        surplus = set(features.columns) - identity - set(GAME_FEATURES.names)
        assert not surplus

    def test_the_label_is_the_home_result(self, features: pl.DataFrame):
        sample = features.filter(pl.col(LABEL).is_not_null()).head(50)
        for row in sample.to_dicts():
            assert row[LABEL] == int(row["home_score"] > row["away_score"])

    def test_no_feature_set_member_is_market_derived(self):
        """The primary game model must not restate the betting line."""
        assert GAME_FEATURES.market_features == ()
        assert all(spec.availability is not Availability.MARKET_CLOSE for spec in GAME_FEATURES)

    def test_week_one_of_the_first_season_has_no_history(self, features: pl.DataFrame):
        first = features.sort("time_index").head(1).to_dicts()[0]
        assert first["history_games_min"] == 0
        assert first["off_epa_per_play_diff"] is None

    def test_history_accumulates_across_a_season_boundary(
        self, league: SyntheticLeague, features: pl.DataFrame
    ):
        """Week 1 of a later season must inherit the previous season's form."""
        later = features.filter((pl.col("season") == league.seasons[3]) & (pl.col("week") == 1))
        assert later.height > 0
        assert later.get_column("history_games_min").min() > 0

    def test_unplayed_games_are_scorable_when_requested(self, league: SyntheticLeague):
        upcoming = build_game_features(league.schedules, league.team_stats, completed_only=False)
        assert upcoming.height >= league.schedules.height


class TestTrainingRows:
    def test_ties_are_excluded(self, features: pl.DataFrame):
        tied = features.with_columns(pl.lit(None, dtype=pl.Int8).alias(LABEL))
        assert training_rows(tied).is_empty()

    def test_thin_history_rows_are_excluded(self, features: pl.DataFrame):
        kept = training_rows(features, min_history=4)
        assert kept.get_column("history_games_min").min() >= 4

    def test_incomplete_games_are_excluded(self, league: SyntheticLeague):
        upcoming = build_game_features(league.schedules, league.team_stats, completed_only=False)
        assert training_rows(upcoming).get_column("completed").all()


class TestTemporalLeakage:
    """Truncation invariance: deleting the future must not change the past."""

    @pytest.mark.parametrize("cutoff_season", [2016, 2019])
    def test_features_are_unchanged_when_later_seasons_are_deleted(
        self, league: SyntheticLeague, features: pl.DataFrame, cutoff_season: int
    ):
        truncated = build_game_features(
            league.schedules.filter(pl.col("season") <= cutoff_season),
            league.team_stats.filter(pl.col("season") <= cutoff_season),
        )
        expected = features.filter(pl.col("season") <= cutoff_season)
        assert_frame_equal(truncated, expected, check_row_order=True)

    def test_features_are_unchanged_when_later_weeks_are_deleted(
        self, league: SyntheticLeague, features: pl.DataFrame
    ):
        season = league.seasons[5]
        cutoff = 6
        keep = (pl.col("season") < season) | (
            (pl.col("season") == season) & (pl.col("week") <= cutoff)
        )
        truncated = build_game_features(
            league.schedules.filter(keep), league.team_stats.filter(keep)
        )
        assert_frame_equal(truncated, features.filter(keep), check_row_order=True)

    def test_altering_a_future_result_cannot_change_an_earlier_feature_row(
        self, league: SyntheticLeague, features: pl.DataFrame
    ):
        """The sharpest form of the check: rewrite the future, not just remove it."""
        season = league.seasons[6]
        tampered = league.schedules.with_columns(
            pl.when((pl.col("season") == season) & (pl.col("week") > 7))
            .then(pl.lit(99.0))
            .otherwise(pl.col("home_score"))
            .alias("home_score")
        )
        rebuilt = build_game_features(tampered, league.team_stats)
        mask = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") <= 7))
        assert_frame_equal(
            rebuilt.filter(mask).select(GAME_FEATURES.names),
            features.filter(mask).select(GAME_FEATURES.names),
        )


class TestFeatureSchema:
    def test_the_version_is_declared(self):
        assert GAME_FEATURES.version == GAME_FEATURE_SCHEMA_VERSION

    def test_every_feature_declares_when_it_becomes_known(self):
        for spec in GAME_FEATURES:
            assert isinstance(spec.availability, Availability)
            assert spec.description.strip()
            assert spec.dataset.strip()

    def test_history_based_features_declare_their_window(self):
        for spec in GAME_FEATURES:
            if spec.availability is Availability.PRIOR_GAMES:
                assert spec.window, f"{spec.name} has no declared lookback"
