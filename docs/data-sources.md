# Data Sources

Every dataset this project consumes, why it is included, and what it costs.

There are three, and they do not overlap:

| Source                                    | Answers                | Authoritative for   |
| ----------------------------------------- | ---------------------- | ------------------- |
| nflverse (below)                          | what happened in games | every NFL statistic |
| Web search (Tavily)                       | what is true now       | current information |
| Local knowledge corpus (`data/knowledge`) | what the words mean    | definitions, rules  |

The latter two arrived in Milestone 4 and are documented in
[rag-design.md](rag-design.md). Statistics are never retrieved; retrieved text
is never treated as a statistic.

## Upstream provider

All data comes from [nflverse](https://github.com/nflverse) via the
[`nflreadpy`](https://nflreadpy.nflverse.com/) Python package, which is the
official port of the R package `nflreadr`.

`nflreadpy` was chosen because it returns Polars frames natively, handles
download caching, and is maintained by the nflverse team. The alternative —
scraping or hand-rolling HTTP against the nflverse-data release assets — would
duplicate maintained work for no benefit.

Underlying repositories:

| Repository                                                 | Provides                                                       |
| ---------------------------------------------------------- | -------------------------------------------------------------- |
| [nflverse-data](https://github.com/nflverse/nflverse-data) | play-by-play, stats, rosters, schedules, injuries, snap counts |
| [dynastyprocess](https://github.com/dynastyprocess/data)   | fantasy identifier crosswalks                                  |
| [ffopportunity](https://github.com/ffverse/ffopportunity)  | expected fantasy points                                        |

Update cadence is published on the
[nflverse automation status page](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).
In season, most feeds refresh nightly; play-by-play refreshes within hours of
each game's conclusion.

## Registered datasets

Only datasets with a concrete purpose are registered. Adding one is a deliberate
edit to the registry in
[src/nfl_platform/data/ingestion/datasets.py](../src/nfl_platform/data/ingestion/datasets.py).

| Dataset          | Loader                | Why it is here                                                                                        |
| ---------------- | --------------------- | ----------------------------------------------------------------------------------------------------- |
| `schedules`      | `load_schedules`      | Game results, opponents, rest days, venue, betting lines. Backs `get_recent_games`.                   |
| `player_stats`   | `load_player_stats`   | Weekly box scores and fantasy points. Backs `get_player_recent_stats`.                                |
| `team_stats`     | `load_team_stats`     | Team production per game; opponent rows yield defense. Backs `get_team_stats` and `get_team_defense`. |
| `play_by_play`   | `load_pbp`            | Situational detail (EPA, personnel, down/distance) for later milestones.                              |
| `rosters_weekly` | `load_rosters_weekly` | Which players were on which roster in a given week; handles mid-season moves.                         |
| `players`        | `load_players`        | Player master table and the GSIS ↔ PFR identifier crosswalk.                                          |
| `injuries`       | `load_injuries`       | Official injury designations and practice participation. Backs `get_injury_information`.              |
| `snap_counts`    | `load_snap_counts`    | Snap share, the cleanest available usage signal. Backs `get_player_usage`.                            |
| `ff_opportunity` | `load_ff_opportunity` | Expected fantasy points, useful for separating usage from finishing luck.                             |

### Deliberately excluded

`load_nextgen_stats`, `load_ftn_charting`, `load_participation`,
`load_contracts`, `load_draft_picks`, `load_combine`, `load_depth_charts`,
`load_trades`, `load_officials`, `load_ff_rankings`.

Each may earn a place later. None has a Milestone 1 consumer, and the master
specification is explicit that datasets are not downloaded without a reason.

## Cost and size

| Dataset          | Approximate size (one season) | Columns |
| ---------------- | ----------------------------- | ------- |
| `schedules`      | ~100 KB                       | 46      |
| `injuries`       | ~1 MB                         | 16      |
| `snap_counts`    | ~2 MB                         | 16      |
| `rosters_weekly` | ~5 MB                         | 36      |
| `players`        | ~5 MB (all-time)              | 39      |
| `team_stats`     | ~2 MB                         | 138     |
| `player_stats`   | ~20 MB                        | 150     |
| `ff_opportunity` | ~15 MB                        | 159     |
| `play_by_play`   | ~150 MB                       | 372     |

Play-by-play dominates. `load_play_by_play` therefore accepts a `columns`
projection, and the required-column contract is kept small so a projection stays
useful.

## Caching

Configured through settings and applied to `nflreadpy` on first use:

| Setting         | Env var                       | Default        |
| --------------- | ----------------------------- | -------------- |
| Cache mode      | `NFL_CACHE_MODE`              | `filesystem`   |
| Cache directory | `NFL_CACHE_DIR`               | `./data/cache` |
| Cache TTL       | `NFL_CACHE_DURATION_SECONDS`  | 86 400 (24 h)  |
| HTTP timeout    | `NFL_REQUEST_TIMEOUT_SECONDS` | 60             |

`data/` is git-ignored. Cached downloads are never committed.

## Test fixtures

The default test suite never touches the network. It runs against committed
Parquet fixtures in `tests/fixtures/nflverse/` — a real slice of 2024 data
(Kansas City and Buffalo, weeks 1–6; play-by-play limited to weeks 1–2).

Fixtures are real nflverse output, not synthetic frames, so schema-validation
tests exercise genuine column names and dtypes. Total committed size is under
1 MB.

Regenerate deliberately with:

```powershell
python scripts/build_fixtures.py
```

Aggregation-correctness tests use small synthetic frames with hand-computed
expected values, so a bug in the aggregation cannot hide behind a bug in the
fixture.

Network-dependent tests are marked `integration` and deselected by default:

```powershell
pytest -m integration
```

Their purpose is to detect upstream schema drift, which frozen fixtures cannot.

## Licensing and attribution

- Most nflverse data is licensed **CC-BY 4.0**.
- FTN charting data is **CC-BY-SA 4.0** (not currently used by this project).
- Attribution to nflverse is required in anything published from this data.
- This project is not affiliated with or endorsed by the NFL.

## Reliability notes

- **Data is mutable.** Stat corrections change historical rows days after a game.
  Use `SnapshotStore` when a result must be reproducible.
- **In-season timing matters.** A Sunday-evening query may see partially updated
  feeds.
- **`get_current_season()` rolls over in the offseason**, so the "current" season
  may have no rows yet. Pass seasons explicitly for anything important; that is
  why `NflDataLoader` resolves `None` to a configured default rather than hiding
  an implicit current-season assumption.
- **Availability varies by era.** Snap counts begin in 2012, injury reports in
  2009, and play-by-play in 1999.
