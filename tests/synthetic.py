"""Synthetic NFL data and tiny trained models for the offline test suite.

The committed nflverse fixture is two teams over six weeks -- ideal for
analytics assertions, useless for a time-aware model that needs ten complete
seasons before a split can even be planned. Downloading real data in CI is off
the table, so the ML tests run on a generated league instead.

The generator is deliberately *signal-bearing*, not random noise. Each team gets
a latent strength, scores are drawn around it, and home teams get a bump. That
matters for two reasons:

* a model trained on pure noise cannot be distinguished from a broken one, so
  "the classifier beats the baseline" becomes a real assertion;
* the leakage tests need features that would change if the future leaked. On
  noise they would pass vacuously.

Everything is seeded, so a failure reproduces exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

SYNTHETIC_TEAMS: tuple[str, ...] = (
    "AAA",
    "BBB",
    "CCC",
    "DDD",
    "EEE",
    "FFF",
    "GGG",
    "HHH",
)
SYNTHETIC_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

#: Enough for plan_splits: 4 train + 4 validation + 2 test.
SYNTHETIC_SEASONS: tuple[int, ...] = tuple(range(2012, 2024))
SYNTHETIC_WEEKS: int = 14

HOME_ADVANTAGE = 2.0


def synthetic_player_id(team: str, index: int) -> str:
    """A GSIS-shaped identifier, so domain validation treats it as an id not a name."""
    return f"00-{SYNTHETIC_TEAMS.index(team) * 100 + index:07d}"


def synthetic_player_name(team: str, index: int) -> str:
    return f"{team} Player {index}"


@dataclass(frozen=True, slots=True)
class SyntheticLeague:
    """One generated league, with every frame the feature builders consume."""

    schedules: pl.DataFrame
    team_stats: pl.DataFrame
    player_stats: pl.DataFrame
    snap_counts: pl.DataFrame
    ff_opportunity: pl.DataFrame
    players: pl.DataFrame

    @property
    def seasons(self) -> tuple[int, ...]:
        values = self.schedules.get_column("season").unique().to_list()
        return tuple(sorted(int(value) for value in values))


def _round_robin(teams: tuple[str, ...], week: int) -> list[tuple[str, str]]:
    """A rotating pairing so every team plays once a week and opponents vary."""
    offset = (week - 1) % (len(teams) - 1)
    rotated = [teams[0], *teams[1:][offset:], *teams[1:][:offset]]
    half = len(rotated) // 2
    pairs = []
    for index in range(half):
        home, away = rotated[index], rotated[len(rotated) - 1 - index]
        if week % 2 == 0:
            home, away = away, home
        pairs.append((home, away))
    return pairs


def build_league(
    *,
    seasons: tuple[int, ...] = SYNTHETIC_SEASONS,
    weeks: int = SYNTHETIC_WEEKS,
    teams: tuple[str, ...] = SYNTHETIC_TEAMS,
    seed: int = 7,
    players_per_team: int = 4,
    incomplete_final_season: bool = False,
) -> SyntheticLeague:
    """Generate a complete, internally consistent synthetic league.

    Args:
        seasons: Seasons to generate.
        weeks: Regular-season weeks per season.
        teams: Team abbreviations.
        seed: Seeds every draw, so results are reproducible.
        players_per_team: Modelled players per team per game.
        incomplete_final_season: Leave the last season's later weeks scheduled
            but unplayed, which is what the season-completeness rule and the
            upcoming-week inference path need to be tested against.

    Returns:
        Frames shaped like the nflverse datasets the builders read.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 4, len(teams))
    strength = {team: float(value) for team, value in zip(teams, latent, strict=True)}

    schedule_rows: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []
    player_rows: list[dict[str, object]] = []
    snap_rows: list[dict[str, object]] = []
    opportunity_rows: list[dict[str, object]] = []

    for season in seasons:
        # Strength drifts a little between seasons so the model has something
        # that genuinely changes rather than a fixed ranking to memorise.
        strength = {team: value + float(rng.normal(0, 1.5)) for team, value in strength.items()}
        for week in range(1, weeks + 1):
            played = not (incomplete_final_season and season == seasons[-1] and week > weeks // 2)
            for home, away in _round_robin(teams, week):
                game_id = f"{season}_{week:02d}_{away}_{home}"
                home_points = max(
                    0.0, round(float(rng.normal(21 + strength[home] + HOME_ADVANTAGE, 7)))
                )
                away_points = max(0.0, round(float(rng.normal(21 + strength[away], 7))))
                if home_points == away_points:
                    home_points += 3.0

                schedule_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "game_type": "SB" if week == weeks else "REG",
                        "week": week,
                        "gameday": f"{season}-09-{min(28, week):02d}",
                        "home_team": home,
                        "away_team": away,
                        "home_score": home_points if played else None,
                        "away_score": away_points if played else None,
                        "result": (home_points - away_points) if played else None,
                        "home_rest": 7.0 + float(week % 3),
                        "away_rest": 7.0,
                        "div_game": int(week % 4 == 0),
                        "roof": "dome" if week % 5 == 0 else "outdoors",
                        "surface": "fieldturf" if week % 3 == 0 else "grass",
                        "location": "Home",
                        "spread_line": round(strength[home] - strength[away] + HOME_ADVANTAGE, 1),
                        "total_line": round(42.0 + float(rng.normal(0, 3)), 1),
                    }
                )
                if not played:
                    continue

                for team, opponent, points in (
                    (home, away, home_points),
                    (away, home, away_points),
                ):
                    team_rows.append(
                        _team_stat_row(game_id, season, week, team, opponent, points, rng)
                    )
                    for index in range(players_per_team):
                        rows = _player_rows(
                            game_id, season, week, team, opponent, index, points, rng
                        )
                        player_rows.append(rows[0])
                        snap_rows.append(rows[1])
                        opportunity_rows.append(rows[2])

    players = pl.DataFrame(
        [
            {
                "gsis_id": synthetic_player_id(team, index),
                "pfr_id": f"PFR{team}{index}",
                "display_name": synthetic_player_name(team, index),
                "football_name": synthetic_player_name(team, index),
                "short_name": synthetic_player_name(team, index),
                "position": SYNTHETIC_POSITIONS[index % len(SYNTHETIC_POSITIONS)],
                "position_group": SYNTHETIC_POSITIONS[index % len(SYNTHETIC_POSITIONS)],
                "latest_team": team,
                "status": "ACT",
                "rookie_season": seasons[0],
                "last_season": seasons[-1],
                "draft_year": seasons[0] - 1,
                "draft_round": 1,
                "draft_pick": index + 1,
                "draft_team": team,
                "college_name": "Test University",
                "height": 74,
                "weight": 220,
                "jersey_number": index + 1,
                "years_of_experience": 3,
                "birth_date": "1995-01-01",
            }
            for team in teams
            for index in range(players_per_team)
        ]
    )

    return SyntheticLeague(
        schedules=pl.DataFrame(schedule_rows),
        team_stats=pl.DataFrame(team_rows),
        player_stats=pl.DataFrame(player_rows),
        snap_counts=pl.DataFrame(snap_rows),
        ff_opportunity=pl.DataFrame(opportunity_rows),
        players=players,
    )


def _team_stat_row(
    game_id: str,
    season: int,
    week: int,
    team: str,
    opponent: str,
    points: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    attempts = float(rng.integers(25, 45))
    carries = float(rng.integers(18, 34))
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "season_type": "REG",
        "team": team,
        "opponent_team": opponent,
        "attempts": attempts,
        "sacks_suffered": float(rng.integers(0, 5)),
        "carries": carries,
        # Tied to points so efficiency features carry real signal rather than
        # decorating the frame.
        "passing_epa": float(points - 21) * 0.4 + float(rng.normal(0, 2)),
        "rushing_epa": float(points - 21) * 0.2 + float(rng.normal(0, 2)),
        "passing_yards": 120.0 + points * 5 + float(rng.normal(0, 30)),
        "sack_yards_lost": float(rng.integers(0, 30)),
        "rushing_yards": 60.0 + points * 2 + float(rng.normal(0, 25)),
        "passing_first_downs": float(rng.integers(6, 18)),
        "rushing_first_downs": float(rng.integers(3, 12)),
        "passing_interceptions": float(rng.integers(0, 3)),
        "rushing_fumbles_lost": float(rng.integers(0, 2)),
        "sack_fumbles_lost": 0.0,
        "receiving_fumbles_lost": 0.0,
        "penalty_yards": float(rng.integers(20, 90)),
    }


def _player_rows(
    game_id: str,
    season: int,
    week: int,
    team: str,
    opponent: str,
    index: int,
    team_points: float,
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    position = SYNTHETIC_POSITIONS[index % len(SYNTHETIC_POSITIONS)]
    player_id = synthetic_player_id(team, index)
    name = synthetic_player_name(team, index)
    share = 0.35 - 0.06 * index
    points = max(0.0, float(team_points) * share + float(rng.normal(0, 3)))
    targets = float(rng.integers(0, 12)) if position != "QB" else 0.0
    carries = float(rng.integers(0, 18)) if position == "RB" else 0.0
    attempts = float(rng.integers(25, 42)) if position == "QB" else 0.0

    player = {
        "player_id": player_id,
        "player_display_name": name,
        "position": position,
        "position_group": position,
        "season": season,
        "week": week,
        "season_type": "REG",
        "game_id": game_id,
        "team": team,
        "opponent_team": opponent,
        "attempts": attempts,
        "carries": carries,
        "targets": targets,
        "receptions": max(0.0, targets - float(rng.integers(0, 4))),
        "target_share": round(share, 3),
        "air_yards_share": round(share * 0.9, 3),
        "wopr": round(share * 1.5, 3),
        "fantasy_points": round(points * 0.8, 2),
        "fantasy_points_ppr": round(points, 2),
    }
    snaps = {
        "game_id": game_id,
        "season": season,
        "week": week,
        "game_type": "REG",
        "team": team,
        "opponent": opponent,
        "pfr_player_id": f"PFR{team}{index}",
        "player": name,
        "position": position,
        "offense_snaps": float(rng.integers(20, 70)),
        "offense_pct": round(float(rng.uniform(0.3, 1.0)), 2),
    }
    opportunity = {
        "game_id": game_id,
        "season": season,
        "week": week,
        "posteam": team,
        "player_id": player_id,
        "full_name": name,
        "position": position,
        "total_fantasy_points": round(points, 2),
        "total_fantasy_points_exp": round(points + float(rng.normal(0, 1.5)), 2),
    }
    return player, snaps, opportunity
