"""Regenerate the committed test fixtures from live nflverse data.

Fixtures are small, real slices of nflverse data (real column names, real
dtypes, real values) so schema-validation tests catch genuine upstream changes.
They are regenerated deliberately, never during a test run.

Usage:
    python scripts/build_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_platform.data.ingestion import Dataset, NflDataLoader  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "nflverse"

# A narrow, stable slice: two high-profile teams over the first six weeks of a
# completed season. Small enough to commit, rich enough to exercise joins.
SEASON = 2024
TEAMS = ("KC", "BUF")
WEEKS = tuple(range(1, 7))
PBP_WEEKS = (1, 2)


def _team_filter(team_column: str, opponent_column: str | None = None) -> pl.Expr:
    expression = pl.col(team_column).is_in(TEAMS)
    if opponent_column is not None:
        expression = expression | pl.col(opponent_column).is_in(TEAMS)
    return expression


def build() -> None:
    loader = NflDataLoader()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    schedules = loader.load_schedules(SEASON).filter(
        pl.col("week").is_in(WEEKS) & _team_filter("home_team", "away_team")
    )

    player_stats = loader.load_player_stats(SEASON).filter(
        pl.col("week").is_in(WEEKS) & _team_filter("team", "opponent_team")
    )

    team_stats = loader.load_team_stats(SEASON).filter(
        pl.col("week").is_in(WEEKS) & _team_filter("team", "opponent_team")
    )

    rosters_weekly = loader.load_rosters_weekly(SEASON).filter(
        pl.col("week").is_in(WEEKS) & _team_filter("team")
    )

    injuries = loader.load_injuries(SEASON).filter(
        pl.col("week").is_in(WEEKS) & _team_filter("team")
    )

    snap_counts = loader.load_snap_counts(SEASON).filter(
        pl.col("week").is_in(WEEKS) & _team_filter("team", "opponent")
    )

    ff_opportunity = loader.load_fantasy_opportunity(SEASON).filter(
        pl.col("week").is_in(WEEKS) & _team_filter("posteam")
    )

    pbp = loader.load_play_by_play(SEASON).filter(
        pl.col("week").is_in(PBP_WEEKS) & _team_filter("posteam", "defteam")
    )

    # Keep only the players referenced by the slices above, so the crosswalk stays small.
    referenced_ids = set(player_stats.get_column("player_id").drop_nulls().to_list())
    referenced_ids |= set(rosters_weekly.get_column("gsis_id").drop_nulls().to_list())
    all_players = loader.load_players()
    players = all_players.filter(pl.col("gsis_id").is_in(list(referenced_ids)))

    # Also pull in anyone sharing a display name with a referenced player. Roughly
    # 7.5% of nflverse players have a name collision, and the directory service
    # must be testable against real ambiguity rather than a synthetic stand-in.
    referenced_names = set(players.get_column("display_name").drop_nulls().to_list())
    namesakes = all_players.filter(
        pl.col("display_name").is_in(list(referenced_names))
        & ~pl.col("gsis_id").is_in(list(referenced_ids))
    )
    players = pl.concat([players, namesakes], how="vertical")
    print(f"  (+{namesakes.height} namesakes added for ambiguity tests)")

    frames: dict[Dataset, pl.DataFrame] = {
        Dataset.SCHEDULES: schedules,
        Dataset.PLAYER_STATS: player_stats,
        Dataset.TEAM_STATS: team_stats,
        Dataset.ROSTERS_WEEKLY: rosters_weekly,
        Dataset.PLAYERS: players,
        Dataset.INJURIES: injuries,
        Dataset.SNAP_COUNTS: snap_counts,
        Dataset.FF_OPPORTUNITY: ff_opportunity,
        Dataset.PLAY_BY_PLAY: pbp,
    }

    for dataset, frame in frames.items():
        path = FIXTURE_DIR / f"{dataset.value}.parquet"
        frame.write_parquet(path)
        size_kb = path.stat().st_size / 1024
        print(
            f"{dataset.value:<16} {frame.height:>6} rows  {frame.width:>4} cols  {size_kb:>8.1f} KB"
        )


if __name__ == "__main__":
    build()
