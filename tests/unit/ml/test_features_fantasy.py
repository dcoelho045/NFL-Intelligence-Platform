"""Fantasy feature construction, population definition, and leakage guards."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_platform.ml.features.fantasy import (
    FANTASY_FEATURE_SCHEMA_VERSION,
    FANTASY_FEATURES,
    LABEL,
    MODELLED_POSITIONS,
    STANDARD_LABEL,
    build_fantasy_features,
    build_player_week_frame,
    training_rows,
)
from nfl_platform.ml.features.schema import Availability
from synthetic import SyntheticLeague, build_league


@pytest.fixture(scope="module")
def league() -> SyntheticLeague:
    return build_league()


def _build(league: SyntheticLeague, **overrides: object) -> pl.DataFrame:
    kwargs: dict[str, object] = {
        "team_stats": league.team_stats,
        "snap_counts": league.snap_counts,
        "ff_opportunity": league.ff_opportunity,
        "players": league.players,
    }
    kwargs.update(overrides)
    return build_fantasy_features(league.player_stats, league.schedules, **kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def features(league: SyntheticLeague) -> pl.DataFrame:
    return _build(league)


class TestPopulation:
    def test_only_modelled_positions_are_included(self, features: pl.DataFrame):
        positions = set(features.get_column("position").unique().to_list())
        assert positions <= set(MODELLED_POSITIONS)

    def test_other_positions_are_dropped(self, league: SyntheticLeague):
        with_kicker = pl.concat(
            [
                league.player_stats,
                league.player_stats.head(1).with_columns(pl.lit("K").alias("position")),
            ],
            how="vertical",
        )
        built = build_fantasy_features(with_kicker, league.schedules, players=league.players)
        assert "K" not in built.get_column("position").unique().to_list()

    def test_training_rows_are_restricted_to_games_the_player_played(self, features: pl.DataFrame):
        """This is what makes the projection conditional on playing."""
        assert training_rows(features).get_column("played").all()

    def test_a_player_who_did_not_play_is_excluded_rather_than_labelled_zero(
        self, league: SyntheticLeague
    ):
        benched = league.player_stats.with_columns(
            pl.when(pl.col("week") == 3).then(0.0).otherwise(pl.col("targets")).alias("targets"),
        )
        frame = build_player_week_frame(
            benched,
            league.schedules,
            snap_counts=None,
            players=league.players,
        )
        inactive = frame.filter(
            (pl.col("week") == 3)
            & (pl.col("attempts") == 0)
            & (pl.col("carries") == 0)
            & (pl.col("targets") == 0)
        )
        assert not inactive.get_column("played").any()

    def test_box_score_usage_stands_in_when_snap_data_is_missing(self, league: SyntheticLeague):
        """Snap counts start in 2012; earlier rows must not all read as inactive."""
        frame = build_player_week_frame(
            league.player_stats, league.schedules, snap_counts=None, players=league.players
        )
        assert frame.get_column("played").any()


class TestFeatureValues:
    def test_every_declared_feature_is_present(self, features: pl.DataFrame):
        missing = [name for name in FANTASY_FEATURES.names if name not in features.columns]
        assert not missing

    def test_both_scoring_labels_are_carried(self, features: pl.DataFrame):
        assert LABEL in features.columns
        assert STANDARD_LABEL in features.columns

    def test_position_indicators_are_one_hot(self, features: pl.DataFrame):
        flags = features.select(
            [pl.col(f"is_{position.lower()}") for position in MODELLED_POSITIONS]
        )
        assert flags.sum_horizontal().unique().to_list() == [1]

    def test_implied_team_total_flips_sign_with_home_and_away(self):
        """nflverse states the spread from the home side; getting that backwards
        would invert the most informative environment feature."""
        schedules = pl.DataFrame(
            [
                {
                    "game_id": "g1",
                    "season": 2023,
                    "game_type": "REG",
                    "week": 1,
                    "gameday": "2023-09-10",
                    "home_team": "AAA",
                    "away_team": "BBB",
                    "home_score": 24.0,
                    "away_score": 17.0,
                    "home_rest": 7.0,
                    "away_rest": 7.0,
                    "div_game": 0,
                    "roof": "outdoors",
                    "surface": "grass",
                    "location": "Home",
                    "spread_line": 6.0,
                    "total_line": 44.0,
                }
            ]
        )
        stats = pl.DataFrame(
            [
                {
                    "player_id": player_id,
                    "player_display_name": f"{team} Player",
                    "position": "WR",
                    "season": 2023,
                    "week": 1,
                    "game_id": "g1",
                    "team": team,
                    "opponent_team": opponent,
                    "targets": 5.0,
                    "fantasy_points_ppr": 10.0,
                }
                for player_id, team, opponent in (
                    ("00-0000001", "AAA", "BBB"),
                    ("00-0000002", "BBB", "AAA"),
                )
            ]
        )
        frame = build_player_week_frame(stats, schedules)
        rows = {row["team"]: row["implied_team_total"] for row in frame.to_dicts()}
        # total 44, spread 6: home 25, away 19.
        assert rows["AAA"] == pytest.approx(25.0)
        assert rows["BBB"] == pytest.approx(19.0)

    def test_optional_datasets_degrade_to_nulls_rather_than_failing(self, league: SyntheticLeague):
        frame = build_fantasy_features(
            league.player_stats,
            league.schedules,
            team_stats=None,
            snap_counts=None,
            ff_opportunity=None,
            players=None,
        )
        assert frame.height > 0
        assert frame.get_column("snap_share_recent").null_count() == frame.height

    def test_history_games_counts_only_prior_games(self, features: pl.DataFrame):
        first = features.sort("time_index").head(1).to_dicts()[0]
        assert first["history_games"] == 0
        assert first["ppr_per_game_recent"] is None

    def test_opponent_allowed_is_null_before_the_opponent_has_played(self, features: pl.DataFrame):
        first_week = features.sort("time_index").head(4)
        assert first_week.get_column("opponent_ppr_allowed_to_position").null_count() > 0


class TestTrainingRows:
    def test_thin_history_is_excluded(self, features: pl.DataFrame):
        kept = training_rows(features, min_history=3)
        assert kept.get_column("history_games").min() >= 3

    def test_unlabelled_rows_are_excluded(self, features: pl.DataFrame):
        blanked = features.with_columns(pl.lit(None, dtype=pl.Float64).alias(LABEL))
        assert training_rows(blanked).is_empty()


class TestTemporalLeakage:
    def test_features_are_unchanged_when_later_seasons_are_deleted(
        self, league: SyntheticLeague, features: pl.DataFrame
    ):
        cutoff = league.seasons[6]
        truncated = build_fantasy_features(
            league.player_stats.filter(pl.col("season") <= cutoff),
            league.schedules.filter(pl.col("season") <= cutoff),
            team_stats=league.team_stats.filter(pl.col("season") <= cutoff),
            snap_counts=league.snap_counts.filter(pl.col("season") <= cutoff),
            ff_opportunity=league.ff_opportunity.filter(pl.col("season") <= cutoff),
            players=league.players,
        )
        expected = features.filter(pl.col("season") <= cutoff)
        assert_frame_equal(
            truncated.select(FANTASY_FEATURES.names),
            expected.select(FANTASY_FEATURES.names),
        )

    def test_rewriting_future_production_cannot_change_an_earlier_row(
        self, league: SyntheticLeague, features: pl.DataFrame
    ):
        season = league.seasons[5]
        tampered = league.player_stats.with_columns(
            pl.when((pl.col("season") == season) & (pl.col("week") > 6))
            .then(pl.lit(99.0))
            .otherwise(pl.col("fantasy_points_ppr"))
            .alias("fantasy_points_ppr")
        )
        rebuilt = build_fantasy_features(
            tampered,
            league.schedules,
            team_stats=league.team_stats,
            snap_counts=league.snap_counts,
            ff_opportunity=league.ff_opportunity,
            players=league.players,
        )
        mask = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") <= 6))
        assert_frame_equal(
            rebuilt.filter(mask).select(FANTASY_FEATURES.names),
            features.filter(mask).select(FANTASY_FEATURES.names),
        )


class TestFeatureSchema:
    def test_the_version_is_declared(self):
        assert FANTASY_FEATURES.version == FANTASY_FEATURE_SCHEMA_VERSION

    def test_every_feature_declares_when_it_becomes_known(self):
        for spec in FANTASY_FEATURES:
            assert isinstance(spec.availability, Availability)
            assert spec.description.strip()

    def test_the_market_feature_is_declared_as_market_derived(self):
        """Market inputs must be labelled so the no-market comparison is possible."""
        assert FANTASY_FEATURES.market_features == ("implied_team_total",)

    def test_the_market_free_variant_drops_exactly_that_feature(self):
        without = FANTASY_FEATURES.without_market()
        assert "implied_team_total" not in without.names
        assert len(without) == len(FANTASY_FEATURES) - 1
