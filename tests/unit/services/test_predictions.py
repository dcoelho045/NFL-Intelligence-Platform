"""The prediction service: DTO contract, provenance, and failure behaviour."""

from __future__ import annotations

import pytest

from nfl_platform.config.settings import Settings
from nfl_platform.exceptions import (
    InsufficientHistoryError,
    ModelNotAvailableError,
    TeamNotFoundError,
)
from nfl_platform.ml.registry.store import ModelRegistry
from nfl_platform.schemas.fantasy import FantasyScoring
from nfl_platform.schemas.prediction import (
    FantasyProjection,
    GamePrediction,
    ModelRole,
    PredictionProblem,
)
from nfl_platform.services import ServiceContainer
from synthetic import SYNTHETIC_TEAMS, SyntheticLeague, synthetic_player_id

HOME, AWAY = SYNTHETIC_TEAMS[0], SYNTHETIC_TEAMS[1]


@pytest.fixture
def predictions(prediction_services: ServiceContainer):
    return prediction_services.predictions


@pytest.fixture
def target_week(synthetic_league: SyntheticLeague) -> tuple[int, int]:
    """A mid-season week late enough for both teams to have real history."""
    return synthetic_league.seasons[-1], 8


class TestModelStatus:
    def test_reports_installed_models(self, predictions):
        status = predictions.model_status()
        assert status.available
        assert status.enabled
        problems = {entry.problem for entry in status.models}
        assert problems == {PredictionProblem.GAME_WIN, PredictionProblem.FANTASY_POINTS}

    def test_each_entry_carries_its_provenance(self, predictions):
        for entry in predictions.model_status().models:
            assert entry.model_id
            assert entry.training_seasons
            assert entry.test_seasons
            assert entry.feature_schema_version
            assert entry.primary_metric

    def test_training_and_test_seasons_never_overlap(self, predictions):
        """Surfaced to every caller, so the claim is checkable without the report."""
        for entry in predictions.model_status().models:
            assert not set(entry.training_seasons) & set(entry.test_seasons)

    def test_an_empty_registry_reports_unavailable_rather_than_failing(
        self, prediction_loader, settings: Settings, tmp_path
    ):
        container = ServiceContainer(
            loader=prediction_loader,
            settings=settings,
            model_registry=ModelRegistry(root=tmp_path / "empty", settings=settings),
        )
        status = container.predictions.model_status()
        assert not status.available
        assert any("train_models.py" in note for note in status.notes)

    def test_disabled_predictions_are_reported_as_configuration(
        self, prediction_loader, trained_model_dir
    ):
        off = Settings(_env_file=None, model_dir=trained_model_dir, ml_enable_predictions=False)
        container = ServiceContainer(
            loader=prediction_loader,
            settings=off,
            model_registry=ModelRegistry(root=trained_model_dir, settings=off),
        )
        status = container.predictions.model_status()
        assert not status.enabled
        assert any("switched off" in note for note in status.notes)


class TestPredictGame:
    def test_returns_a_game_prediction(self, predictions, target_week):
        season, week = target_week
        result = predictions.predict_game(HOME, AWAY, season=season, week=week)
        assert isinstance(result, GamePrediction)

    def test_probabilities_are_complementary(self, predictions, target_week):
        season, week = target_week
        result = predictions.predict_game(HOME, AWAY, season=season, week=week)
        assert result.home_win_probability + result.away_win_probability == pytest.approx(1.0)

    def test_the_predicted_winner_matches_the_probability(self, predictions, target_week):
        season, week = target_week
        result = predictions.predict_game(HOME, AWAY, season=season, week=week)
        expected = HOME if result.home_win_probability >= 0.5 else AWAY
        assert result.predicted_winner == expected

    def test_the_baseline_is_reported_alongside_the_model(self, predictions, target_week):
        """So a reader can see what the model adds over 'home teams usually win'."""
        season, week = target_week
        result = predictions.predict_game(HOME, AWAY, season=season, week=week)
        assert result.baseline_home_win_probability is not None

    def test_model_provenance_travels_with_the_prediction(self, predictions, target_week):
        season, week = target_week
        result = predictions.predict_game(HOME, AWAY, season=season, week=week)
        assert result.model_info.problem is PredictionProblem.GAME_WIN
        assert result.model_info.role is not ModelRole.BENCHMARK
        assert result.model_info.model_id.startswith("game_win/")

    def test_the_basis_reports_how_much_history_each_team_had(self, predictions, target_week):
        season, week = target_week
        result = predictions.predict_game(HOME, AWAY, season=season, week=week)
        assert set(result.basis.history_games) == {HOME, AWAY}
        assert result.basis.as_of_week == week

    def test_caveats_say_it_is_an_estimate(self, predictions, target_week):
        season, week = target_week
        result = predictions.predict_game(HOME, AWAY, season=season, week=week)
        assert any("not a measured statistic" in caveat for caveat in result.caveats)

    def test_top_factors_are_derived_from_the_model(self, predictions, target_week):
        season, week = target_week
        result = predictions.predict_game(HOME, AWAY, season=season, week=week)
        assert result.top_factors
        for factor in result.top_factors:
            assert factor.direction in {"increases", "decreases"}

    def test_the_basis_states_that_the_market_is_not_an_input(self, predictions, target_week):
        season, week = target_week
        result = predictions.predict_game(HOME, AWAY, season=season, week=week)
        assert any("not model inputs" in note for note in result.basis.notes)

    def test_week_one_of_the_first_season_has_too_little_history(
        self, predictions, synthetic_league: SyntheticLeague
    ):
        with pytest.raises(InsufficientHistoryError):
            predictions.predict_game(HOME, AWAY, season=synthetic_league.seasons[0], week=1)

    def test_an_unknown_team_is_rejected(self, predictions, target_week):
        season, week = target_week
        with pytest.raises(TeamNotFoundError):
            predictions.predict_game("ZZZ", AWAY, season=season, week=week)

    def test_a_hypothetical_matchup_is_still_answerable(self, predictions, target_week):
        """Two teams not scheduled against each other is a legitimate question."""
        season, week = target_week
        result = predictions.predict_game(
            SYNTHETIC_TEAMS[0], SYNTHETIC_TEAMS[5], season=season, week=week
        )
        assert 0.0 <= result.home_win_probability <= 1.0

    def test_an_unavailable_model_raises_rather_than_guessing(
        self, prediction_loader, settings: Settings, tmp_path
    ):
        container = ServiceContainer(
            loader=prediction_loader,
            settings=settings,
            model_registry=ModelRegistry(root=tmp_path / "empty", settings=settings),
        )
        with pytest.raises(ModelNotAvailableError):
            container.predictions.predict_game(HOME, AWAY, season=2020, week=8)


class TestPredictSlate:
    def test_covers_the_week(self, predictions, target_week):
        season, week = target_week
        slate = predictions.predict_slate(season, week)
        assert slate.game_count == len(slate.predictions)
        assert slate.game_count > 0

    def test_every_prediction_is_for_the_requested_week(self, predictions, target_week):
        season, week = target_week
        slate = predictions.predict_slate(season, week)
        assert {(p.season, p.week) for p in slate.predictions} == {(season, week)}

    def test_unpredictable_games_are_listed_rather_than_dropped(
        self, predictions, synthetic_league: SyntheticLeague
    ):
        """A shorter slate must never read as 'these games are not being played'."""
        slate = predictions.predict_slate(synthetic_league.seasons[0], 1)
        assert slate.game_count == 0
        assert slate.skipped


class TestProjectFantasyPoints:
    @pytest.fixture
    def player_id(self) -> str:
        return synthetic_player_id(SYNTHETIC_TEAMS[0], 1)

    def test_returns_a_projection(self, predictions, player_id, target_week):
        season, week = target_week
        result = predictions.project_fantasy_points(player_id, season=season, week=week)
        assert isinstance(result, FantasyProjection)
        assert result.projected_points >= 0.0

    def test_the_projection_is_conditional_on_playing(self, predictions, player_id, target_week):
        """The load-bearing semantic of this milestone's fantasy model."""
        season, week = target_week
        result = predictions.project_fantasy_points(player_id, season=season, week=week)
        assert result.conditional_on_playing is True
        assert any("Conditional on the player" in caveat for caveat in result.caveats)

    def test_the_scoring_format_is_ppr(self, predictions, player_id, target_week):
        season, week = target_week
        result = predictions.project_fantasy_points(player_id, season=season, week=week)
        assert result.scoring is FantasyScoring.PPR

    def test_standard_scoring_is_refused_rather_than_mislabelled(
        self, predictions, player_id, target_week
    ):
        season, week = target_week
        with pytest.raises(ValueError, match="Only ppr"):
            predictions.project_fantasy_points(
                player_id, season=season, week=week, scoring=FantasyScoring.STANDARD
            )

    def test_the_baseline_and_typical_error_are_reported(self, predictions, player_id, target_week):
        season, week = target_week
        result = predictions.project_fantasy_points(player_id, season=season, week=week)
        assert result.baseline_projected_points is not None
        assert result.typical_error is not None

    def test_model_provenance_travels_with_the_projection(
        self, predictions, player_id, target_week
    ):
        season, week = target_week
        result = predictions.project_fantasy_points(player_id, season=season, week=week)
        assert result.model_info.problem is PredictionProblem.FANTASY_POINTS

    def test_no_quantile_interval_is_claimed_in_this_version(
        self, predictions, player_id, target_week
    ):
        """Absent rather than fabricated: a Gaussian interval here would be a lie."""
        season, week = target_week
        result = predictions.project_fantasy_points(player_id, season=season, week=week)
        assert result.floor_points is None
        assert result.ceiling_points is None

    def test_a_player_with_no_history_is_refused(
        self, predictions, player_id, synthetic_league: SyntheticLeague
    ):
        with pytest.raises(InsufficientHistoryError):
            predictions.project_fantasy_points(
                player_id, season=synthetic_league.seasons[0], week=1
            )


class TestCompareProjections:
    @pytest.fixture
    def players(self) -> list[str]:
        return [
            synthetic_player_id(SYNTHETIC_TEAMS[0], 1),
            synthetic_player_id(SYNTHETIC_TEAMS[1], 1),
        ]

    def test_ranks_highest_first(self, predictions, players, target_week):
        season, week = target_week
        result = predictions.compare_projections(players, season=season, week=week)
        points = [p.projected_points for p in result.projections]
        assert points == sorted(points, reverse=True)

    def test_recommends_the_top_projection(self, predictions, players, target_week):
        season, week = target_week
        result = predictions.compare_projections(players, season=season, week=week)
        assert result.recommended_player_id == result.projections[0].player.player_id

    def test_reports_the_margin(self, predictions, players, target_week):
        season, week = target_week
        result = predictions.compare_projections(players, season=season, week=week)
        assert result.margin is not None

    def test_always_states_the_playing_condition(self, predictions, players, target_week):
        season, week = target_week
        result = predictions.compare_projections(players, season=season, week=week)
        assert any("conditional on the player playing" in c for c in result.caveats)

    def test_an_unprojectable_player_becomes_a_caveat_not_a_failure(
        self, predictions, players, target_week
    ):
        season, week = target_week
        result = predictions.compare_projections([*players, "00-9999999"], season=season, week=week)
        assert len(result.projections) == 2
        assert any("00-9999999" in c for c in result.caveats)

    def test_fewer_than_two_players_is_rejected(self, predictions, players, target_week):
        season, week = target_week
        with pytest.raises(ValueError, match="at least two"):
            predictions.compare_projections(players[:1], season=season, week=week)

    def test_too_many_players_is_rejected(self, predictions, target_week):
        season, week = target_week
        many = [synthetic_player_id(team, 1) for team in SYNTHETIC_TEAMS[:7]]
        with pytest.raises(ValueError, match="At most"):
            predictions.compare_projections(many, season=season, week=week)


class TestServiceBoundary:
    def test_no_dataframe_escapes_the_service(self, predictions, target_week):
        """The milestone-2 boundary still holds for the new surface."""
        season, week = target_week
        result = predictions.predict_game(HOME, AWAY, season=season, week=week)
        assert not hasattr(result, "to_pandas")
        assert result.model_dump()

    def test_predictions_are_deterministic(self, predictions, target_week):
        """Same inputs, same frozen artifact, same number. Every time."""
        season, week = target_week
        first = predictions.predict_game(HOME, AWAY, season=season, week=week)
        second = predictions.predict_game(HOME, AWAY, season=season, week=week)
        assert first.home_win_probability == second.home_win_probability
