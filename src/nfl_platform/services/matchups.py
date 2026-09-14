"""Matchup analysis service.

Composes both teams' offence, defence, and form with the scheduling context.
Strictly descriptive: no win probability, no projected score. Prediction is
Milestone 6, and blending the two would let a descriptive endpoint be read as a
predictive one.
"""

from __future__ import annotations

import polars as pl

from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.domain import SeasonInput, normalize_team, validate_week
from nfl_platform.schemas.common import GameContext, SeasonType
from nfl_platform.schemas.matchup import HeadToHeadGame, MatchupAnalysis, MatchupSide
from nfl_platform.services.base import AnalyticsService
from nfl_platform.services.teams import TeamAnalyticsService

_MIN_RELIABLE_GAMES = 3


class MatchupAnalyticsService(AnalyticsService):
    """Side-by-side descriptive comparison of two teams."""

    def __init__(self, loader: NflDataLoader, teams: TeamAnalyticsService | None = None) -> None:
        super().__init__(loader)
        self._teams = teams or TeamAnalyticsService(loader)

    def analyze(
        self,
        team_a: str,
        team_b: str,
        *,
        seasons: SeasonInput = None,
        week: int | None = None,
        last_n_games: int | None = 4,
        season_type: SeasonType = SeasonType.ALL,
    ) -> MatchupAnalysis:
        """Compare two teams, optionally anchored to a scheduled week."""
        first = normalize_team(team_a)
        second = normalize_team(team_b)
        if first == second:
            raise ValueError("A matchup requires two different teams.")
        if week is not None:
            validate_week(week)

        schedules = self._loader.load_schedules(seasons)
        scheduled = self._find_scheduled_game(schedules, first, second, week)

        side_a = self._build_side(
            first, scheduled, seasons=seasons, last_n_games=last_n_games, season_type=season_type
        )
        side_b = self._build_side(
            second, scheduled, seasons=seasons, last_n_games=last_n_games, season_type=season_type
        )

        return MatchupAnalysis(
            window=self.build_season_window(seasons, season_type, week),
            team_a=side_a,
            team_b=side_b,
            scheduled_game=self._to_game_context(scheduled),
            head_to_head=self._head_to_head(schedules, first, second),
            note=self._note(side_a, side_b, scheduled, week),
        )

    def _find_scheduled_game(
        self,
        schedules: pl.DataFrame,
        team_a: str,
        team_b: str,
        week: int | None,
    ) -> dict[str, object] | None:
        """Locate the fixture between the two teams, optionally in a given week."""
        pairing = ((pl.col("home_team") == team_a) & (pl.col("away_team") == team_b)) | (
            (pl.col("home_team") == team_b) & (pl.col("away_team") == team_a)
        )
        matches = schedules.filter(pairing)
        if week is not None:
            matches = matches.filter(pl.col("week") == week)
        if matches.is_empty():
            return None
        matches = matches.sort(["season", "week"], descending=True, nulls_last=True)
        return matches.head(1).to_dicts()[0]

    def _build_side(
        self,
        team: str,
        scheduled: dict[str, object] | None,
        *,
        seasons: SeasonInput,
        last_n_games: int | None,
        season_type: SeasonType,
    ) -> MatchupSide:
        is_home: bool | None = None
        rest: int | None = None
        if scheduled is not None:
            is_home = self.row_str(scheduled, "home_team") == team
            rest = self.row_int(scheduled, "home_rest" if is_home else "away_rest")

        return MatchupSide(
            team=team,
            is_home=is_home,
            offense=self._teams.offense(
                team, seasons=seasons, last_n_games=last_n_games, season_type=season_type
            ),
            defense=self._teams.defense(
                team, seasons=seasons, last_n_games=last_n_games, season_type=season_type
            ),
            form=self._teams.recent_form(
                team, seasons=seasons, last_n_games=last_n_games, season_type=season_type
            ),
            rest_days=rest,
        )

    def _to_game_context(self, scheduled: dict[str, object] | None) -> GameContext | None:
        if scheduled is None:
            return None
        return GameContext(
            game_id=self.row_str(scheduled, "game_id") or "",
            season=self.row_int(scheduled, "season") or 0,
            week=self.row_int(scheduled, "week") or 1,
            game_type=self.row_str(scheduled, "game_type"),
            gameday=self.row_str(scheduled, "gameday"),
            home_team=self.row_str(scheduled, "home_team"),
            away_team=self.row_str(scheduled, "away_team"),
            roof=self.row_str(scheduled, "roof"),
            surface=self.row_str(scheduled, "surface"),
            spread_line=self.row_float(scheduled, "spread_line"),
            total_line=self.row_float(scheduled, "total_line"),
        )

    def _head_to_head(
        self, schedules: pl.DataFrame, team_a: str, team_b: str
    ) -> list[HeadToHeadGame]:
        """Completed previous meetings within the requested seasons, newest first."""
        pairing = ((pl.col("home_team") == team_a) & (pl.col("away_team") == team_b)) | (
            (pl.col("home_team") == team_b) & (pl.col("away_team") == team_a)
        )
        played = schedules.filter(
            pairing & pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
        ).sort(["season", "week"], descending=True, nulls_last=True)

        games: list[HeadToHeadGame] = []
        for row in played.iter_rows(named=True):
            home = self.row_str(row, "home_team") or ""
            away = self.row_str(row, "away_team") or ""
            home_score = self.row_int(row, "home_score")
            away_score = self.row_int(row, "away_score")
            winner: str | None = None
            if home_score is not None and away_score is not None and home_score != away_score:
                winner = home if home_score > away_score else away
            games.append(
                HeadToHeadGame(
                    game_id=self.row_str(row, "game_id") or "",
                    season=self.row_int(row, "season") or 0,
                    week=self.row_int(row, "week") or 0,
                    home_team=home,
                    away_team=away,
                    home_score=home_score,
                    away_score=away_score,
                    winner=winner,
                )
            )
        return games

    def _note(
        self,
        side_a: MatchupSide,
        side_b: MatchupSide,
        scheduled: dict[str, object] | None,
        week: int | None,
    ) -> str | None:
        """Surface caveats a reader should see before quoting these numbers."""
        notes: list[str] = []
        if scheduled is None:
            notes.append(
                f"No scheduled meeting found{f' in week {week}' if week else ''}; "
                "the comparison is not anchored to a specific game."
            )
        sample = min(side_a.form.window.games, side_b.form.window.games)
        if sample < _MIN_RELIABLE_GAMES:
            notes.append(f"Small sample: the narrower team profile covers only {sample} game(s).")
        return " ".join(notes) if notes else None
