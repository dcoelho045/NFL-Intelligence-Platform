"""Registry of the nflverse datasets this project depends on.

Keeping dataset metadata in one table means column contracts, season awareness,
and documentation are declared once instead of being duplicated across loaders.
Only datasets Milestone 1 actually needs are registered; adding an entry is a
deliberate act, not a side effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from nfl_platform.exceptions import UnknownDatasetError


class Dataset(StrEnum):
    """Canonical names for the registered nflverse datasets."""

    SCHEDULES = "schedules"
    PLAYER_STATS = "player_stats"
    TEAM_STATS = "team_stats"
    PLAY_BY_PLAY = "play_by_play"
    ROSTERS_WEEKLY = "rosters_weekly"
    PLAYERS = "players"
    INJURIES = "injuries"
    SNAP_COUNTS = "snap_counts"
    FF_OPPORTUNITY = "ff_opportunity"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Declarative contract for one nflverse dataset.

    ``required_columns`` is intentionally a minimal subset: the columns this
    project reads. Validating the full upstream schema would make the project
    brittle to harmless nflverse additions.
    """

    name: Dataset
    nflreadpy_function: str
    season_aware: bool
    grain: str
    required_columns: frozenset[str]
    description: str


_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name=Dataset.SCHEDULES,
        nflreadpy_function="load_schedules",
        season_aware=True,
        grain="one row per game",
        required_columns=frozenset(
            {
                "game_id",
                "season",
                "game_type",
                "week",
                "gameday",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "result",
            }
        ),
        description="Game schedules and final results, including betting lines and venue.",
    ),
    DatasetSpec(
        name=Dataset.PLAYER_STATS,
        nflreadpy_function="load_player_stats",
        season_aware=True,
        grain="one row per player per game",
        required_columns=frozenset(
            {
                "player_id",
                "player_display_name",
                "position",
                "season",
                "week",
                "season_type",
                "team",
                "opponent_team",
                "fantasy_points",
                "fantasy_points_ppr",
            }
        ),
        description="Weekly player box-score statistics including fantasy points.",
    ),
    DatasetSpec(
        name=Dataset.TEAM_STATS,
        nflreadpy_function="load_team_stats",
        season_aware=True,
        grain="one row per team per game",
        required_columns=frozenset(
            {
                "season",
                "week",
                "team",
                "season_type",
                "opponent_team",
                "passing_yards",
                "rushing_yards",
            }
        ),
        description="Weekly team statistics; a team's defence is derived from opponent rows.",
    ),
    DatasetSpec(
        name=Dataset.PLAY_BY_PLAY,
        nflreadpy_function="load_pbp",
        season_aware=True,
        grain="one row per play",
        required_columns=frozenset(
            {
                "game_id",
                "season",
                "week",
                "posteam",
                "defteam",
                "play_type",
                "epa",
            }
        ),
        description="Play-by-play detail. Large; always request a column projection.",
    ),
    DatasetSpec(
        name=Dataset.ROSTERS_WEEKLY,
        nflreadpy_function="load_rosters_weekly",
        season_aware=True,
        grain="one row per player per team per week",
        required_columns=frozenset(
            {"season", "week", "team", "gsis_id", "full_name", "position", "status"}
        ),
        description="Weekly team rosters, used to resolve a player's team in a given week.",
    ),
    DatasetSpec(
        name=Dataset.PLAYERS,
        nflreadpy_function="load_players",
        season_aware=False,
        grain="one row per player",
        required_columns=frozenset(
            {"gsis_id", "display_name", "position", "pfr_id", "latest_team", "status"}
        ),
        description="Player master table and the identifier crosswalk (GSIS, PFR, ESPN, PFF).",
    ),
    DatasetSpec(
        name=Dataset.INJURIES,
        nflreadpy_function="load_injuries",
        season_aware=True,
        grain="one row per player per team per week",
        required_columns=frozenset(
            {
                "season",
                "week",
                "team",
                "gsis_id",
                "full_name",
                "position",
                "report_status",
                "practice_status",
                "report_primary_injury",
            }
        ),
        description="Official weekly injury reports and practice participation.",
    ),
    DatasetSpec(
        name=Dataset.SNAP_COUNTS,
        nflreadpy_function="load_snap_counts",
        season_aware=True,
        grain="one row per player per game",
        required_columns=frozenset(
            {
                "game_id",
                "season",
                "week",
                "team",
                "player",
                "pfr_player_id",
                "position",
                "offense_snaps",
                "offense_pct",
            }
        ),
        description="Per-game snap counts and snap share, keyed by PFR id (not GSIS id).",
    ),
    DatasetSpec(
        name=Dataset.FF_OPPORTUNITY,
        nflreadpy_function="load_ff_opportunity",
        season_aware=True,
        grain="one row per player per game",
        required_columns=frozenset({"season", "week", "player_id", "posteam"}),
        description="Expected fantasy points and opportunity modelling from ffopportunity.",
    ),
)

DATASETS: Mapping[Dataset, DatasetSpec] = MappingProxyType({spec.name: spec for spec in _SPECS})


def get_dataset_spec(dataset: Dataset | str) -> DatasetSpec:
    """Look up a dataset spec by name, raising :class:`UnknownDatasetError` if absent."""
    try:
        key = Dataset(dataset)
    except ValueError:
        raise UnknownDatasetError(str(dataset), frozenset(item.value for item in Dataset)) from None
    return DATASETS[key]
