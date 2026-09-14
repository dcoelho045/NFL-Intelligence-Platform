# NFL Intelligence Platform

A production-style AI engineering platform that answers NFL questions with
**evidence-backed** responses.

The governing principle: **an LLM is never the source of truth for a number.**
Statistics come from deterministic functions over
[nflverse](https://github.com/nflverse) data. The LLM's eventual job is to
orchestrate tools and synthesise an answer, never to recall a statistic.

## Status

| Milestone | Scope                                                       | Status      |
| --------- | ----------------------------------------------------------- | ----------- |
| 1         | NFL data foundation (`nflreadpy` + Polars)                  | Complete    |
| 2         | Analytics service layer + FastAPI                           | Complete    |
| 3         | LLM tool calling                                            | Not started |
| 4-11      | RAG, MCP, ML, agents, evaluation, observability, A2A, CI/CD | Not started |

There is intentionally **no LLM functionality in the repository yet** — enforced
by [tests/test_architecture.py](tests/test_architecture.py).

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

## Running the API

```powershell
uvicorn nfl_platform.api.app:app --reload
```

| URL                            | Purpose        |
| ------------------------------ | -------------- |
| <http://127.0.0.1:8000/docs>   | Swagger UI     |
| <http://127.0.0.1:8000/redoc>  | ReDoc          |
| <http://127.0.0.1:8000/health> | Liveness probe |

```bash
# Recent form
curl "http://127.0.0.1:8000/api/v1/players/00-0033873/stats/recent?seasons=2024&last_n_games=3"

# What a defence allowed
curl "http://127.0.0.1:8000/api/v1/teams/KC/defense?seasons=2024&last_n_games=4"

# Descriptive matchup comparison
curl "http://127.0.0.1:8000/api/v1/matchups?team_a=KC&team_b=BUF&seasons=2024"
```

See [docs/api.md](docs/api.md) for the full reference.

## Using it from Python

The HTTP API is one adapter. The services behind it are importable directly,
which is how the LLM and MCP layers will call them:

```python
from nfl_platform.services import ServiceContainer

container = ServiceContainer()

stats = container.players.recent_performance("00-0033873", seasons=2024, last_n_games=4)
print(stats.player.name, stats.per_game.passing_yards)

defense = container.teams.defense("KC", seasons=2024, last_n_games=4)
print(defense.avg_allowed_passing_yards)
```

Services return typed Pydantic DTOs, not DataFrames.

### Conventions worth knowing

- `last_n_games` counts **games with data**, not calendar weeks, so bye weeks and
  inactive weeks never silently shrink a sample.
- Defensive statistics are prefixed `allowed_`, so "what KC gave up" can never be
  confused with "what KC gained".
- Players may be addressed by GSIS id **or name**. Around 7.5% of nflverse
  players share a display name, so an ambiguous name returns **409** with the
  candidates rather than guessing.
- Matchup analysis is **descriptive**. It contains no prediction.

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
  api/         FastAPI adapter: routers, DI, error mapping
  services/    OOP analytics services - the stable interface
  schemas/     Pydantic DTOs - the typed contract
  analytics/   deterministic, LLM-free computation
  data/
    ingestion/ dataset registry, typed loaders, schema validation, sources
    storage/   Parquet snapshots for reproducibility
  config/      settings (pydantic-settings) and structured logging
  domain.py    identifier and season/week validation
  exceptions.py
scripts/       fixture generation
tests/
  fixtures/    small committed Parquet fixtures with real nflverse schemas
docs/          architecture, API reference, data documentation
```

Dependencies flow strictly downward: `api → services → analytics → data`.
[tests/test_architecture.py](tests/test_architecture.py) enforces this.

## Documentation

- [Architecture](docs/architecture.md) — layering, boundaries, design decisions
- [API reference](docs/api.md) — endpoints, conventions, error semantics
- [Data model](docs/data-model.md) — entities, identifiers, grain, relationships
- [Data sources](docs/data-sources.md) — nflverse datasets, cadence, licensing

## Acknowledgement

NFL data is provided by [nflverse](https://github.com/nflverse). Most nflverse
data is licensed CC-BY 4.0; FTN charting data is CC-BY-SA 4.0. This project is
not affiliated with or endorsed by the NFL.
