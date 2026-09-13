# NFL Intelligence Platform

A production-style AI engineering platform that answers NFL questions with
**evidence-backed** responses.


## Requirements

- Python 3.12 or newer
- Git

## Setup

```powershell
# 1. Clone and enter the repository
git clone https://github.com/dcoelho045/NFL-Intelligence-Platform.git
cd NFL-Intelligence-Platform

# 2. Create and activate a virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install the project with development dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 4. Create a local environment file
Copy-Item .env.example .env
```

On macOS/Linux, substitute `python3.12 -m venv .venv` and
`source .venv/bin/activate`.

## Configuration

All configuration is environment-driven. Copy `.env.example` to `.env` and edit
as needed; `.env` is git-ignored and must never be committed.

| Variable                      | Purpose                                      | Default            |
| ----------------------------- | -------------------------------------------- | ------------------ |
| `NFL_CACHE_DIR`               | Directory for cached nflverse downloads      | `./data/cache`     |
| `NFL_CACHE_MODE`              | `filesystem`, `memory`, or `off`             | `filesystem`       |
| `NFL_CACHE_DURATION_SECONDS`  | How long a cached download may be reused     | `86400`            |
| `NFL_REQUEST_TIMEOUT_SECONDS` | HTTP timeout for downloads                   | `60`               |
| `NFL_SNAPSHOT_DIR`            | Directory for reproducible Parquet snapshots | `./data/snapshots` |
| `NFL_DEFAULT_SEASON`          | Season used when a caller omits one          | current season     |
| `NFL_LOG_LEVEL`               | Logging level                                | `INFO`             |
| `NFL_LOG_JSON`                | Emit JSON logs instead of text               | `false`            |



Every analytics function takes an explicit `loader`, which is what allows the
default test suite to run entirely offline against fixtures.

Two conventions worth knowing:

- `last_n_games` counts **games with data**, not calendar weeks, so bye weeks and
  inactive weeks do not silently shrink the sample.
- `get_team_defense` prefixes every statistic with `allowed_`, so "what KC gave
  up" can never be confused with "what KC gained".

## Testing

```powershell
# Offline unit tests (default) - no network access required
pytest

# Include the network-dependent integration tests
pytest -m integration

# With coverage
pytest --cov=nfl_platform --cov-report=term-missing
```

Integration tests are marked `integration` and deselected by default so the
default suite is fast, deterministic, and usable in CI without network access.

## Code quality

```powershell
ruff check .
ruff format --check .
mypy src
```

## Project layout

```text
src/nfl_platform/
  config/      settings (pydantic-settings) and structured logging
  data/
    ingestion/ dataset registry, typed loaders, schema validation, data sources
    storage/   Parquet snapshots for reproducibility
  analytics/   deterministic, LLM-free analytics functions
  domain.py    identifier and season/week validation
  exceptions.py
scripts/       fixture generation
tests/
  fixtures/    small committed Parquet fixtures with real nflverse schemas
docs/          specification, milestone plans, data documentation
```

## Documentation

- [Data model](docs/data-model.md) — entities, identifiers, grain, relationships
- [Data sources](docs/data-sources.md) — nflverse datasets, cadence, licensing, limitations


## Acknowledgement

NFL data is provided by [nflverse](https://github.com/nflverse). Most nflverse
data is licensed CC-BY 4.0; FTN charting data is CC-BY-SA 4.0. This project is
not affiliated with or endorsed by the NFL.
