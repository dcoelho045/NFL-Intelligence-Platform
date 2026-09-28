# NFL Intelligence Platform

A production-style AI engineering platform that answers NFL questions with
**evidence-backed** responses.

The governing principle: **an LLM is never the source of truth for a number.**
Statistics come from deterministic functions over
[nflverse](https://github.com/nflverse) data. The LLM's job is to orchestrate
tools and synthesise an answer, never to recall a statistic.

## Status

| Milestone | Scope                                                  | Status      |
| --------- | ------------------------------------------------------ | ----------- |
| 1         | NFL data foundation (`nflreadpy` + Polars)             | Complete    |
| 2         | Analytics service layer + FastAPI                      | Complete    |
| 3         | LLM tool calling                                       | Complete    |
| 4         | Web retrieval + lightweight local RAG                  | Complete    |
| 5-11      | MCP, ML, agents, evaluation, observability, A2A, CI/CD | Not started |

The assistant added in Milestone 3 cannot compute a statistic: the `openai` SDK
is importable only in `llm/providers/` and `retrieval/providers/`, the `llm/`
package may not import the analytics, data or retrieval layers, and every number
in an answer is checked against the tool results that produced it — all enforced
by [tests/architecture/test_layering.py](tests/architecture/test_layering.py).

Milestone 4 adds two more evidence sources without weakening that. Web search
covers anything current (injuries, transactions, news); a small curated corpus
covers stable knowledge (scoring formats, metric definitions, this project's own
methodology). Statistics still come only from the deterministic analytics layer,
and the grounding check now distinguishes numbers traceable to it from numbers
found only in retrieved prose. Retrieved web content is treated as untrusted
input, and there is deliberately no tool that fetches a URL the model chose. See
[docs/rag-design.md](docs/rag-design.md).

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
# Add OPENAI_API_KEY (assistant + embeddings) and TAVILY_API_KEY (web search).
# Both are optional: without them the analytics API still works, and the
# assistant reports the missing capability instead of failing.

# 5. Build the local knowledge index (costs about $0.0002)
python scripts/build_knowledge_index.py
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

The assistant needs one more. Everything above works without it:

| Variable                 | Purpose                                      | Default   |
| ------------------------ | -------------------------------------------- | --------- |
| `NFL_LLM_API_KEY`        | OpenAI API key (`OPENAI_API_KEY` also works) | none      |
| `NFL_LLM_MODEL`          | Chat model                                   | `gpt-4.1` |
| `NFL_LLM_TEMPERATURE`    | Sampling temperature; blank omits it         | `0.0`     |
| `NFL_LLM_MAX_ITERATIONS` | Model round-trips per question               | `5`       |
| `NFL_LLM_MAX_TOOL_CALLS` | Tool executions per question                 | `12`      |

Leave `NFL_LLM_TEMPERATURE` blank for reasoning models such as `o4-mini` and
`gpt-5-mini`, which reject an explicit temperature.

Without a key the API still starts and every analytics endpoint works; only
`/api/v1/assistant/*` returns 503.

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

## Asking the assistant

```powershell
# Interactive
python -m nfl_platform.llm.cli

# One question, showing the tool calls and cost behind the answer
python -m nfl_platform.llm.cli -q "How has Patrick Mahomes played recently?" --trace
```

Or over HTTP:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/assistant/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Should I start Mahomes or Allen this week?"}'
```

The response carries the answer, the `evidence` list of tool calls that produced
it, and run statistics including tokens, estimated cost, and any number that
could not be traced back to a tool result.

```python
from nfl_platform.llm.factory import build_agent

agent = build_agent()
result = agent.ask("How has the Kansas City defense performed recently?")
print(result.answer)
print(result.trace.tools_used, result.trace.estimated_cost_usd)
```

The assistant selects and calls the same services shown below — it has no other
source of numbers. See [docs/llm-tools.md](docs/llm-tools.md) for the boundary,
the 17 tools, and how errors and grounding are handled.

## Using it from Python

The HTTP API is one adapter. The services behind it are importable directly,
which is how the assistant — and the future MCP layer — call them:

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
# Offline unit tests (default) - no network access, no API key required
pytest

# Include the network-dependent integration tests
pytest -m integration

# Live LLM tests - costs money, needs an API key
pytest -m llm

# With coverage
pytest --cov=nfl_platform --cov-report=term-missing
```

Integration and `llm` tests are deselected by default so the default suite is
fast, deterministic, and usable in CI without network access or a key.

The assistant tests fake **only the model**. Tools execute for real against the
committed fixtures, so they exercise genuine argument validation, service calls,
and error mapping rather than mocks of them.

```powershell
# Measure tool selection, grounding, latency and cost on a fixed question set
python scripts/llm_benchmark.py --output docs/llm-evaluation.json
```

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
  llm/         tool-calling assistant: agent loop, tools, prompts, tracing
    providers/ the chat SDK adapter
  retrieval/   web search and local RAG - a peer of analytics/
    web/       provider Protocol, Tavily and Brave adapters, cache, policy
    knowledge/ corpus loading, chunking, embeddings, Parquet vector store
    providers/ the embeddings SDK adapter
  services/    OOP analytics and retrieval services - the stable interface
  schemas/     Pydantic DTOs - the typed contract
  analytics/   deterministic, LLM-free computation
  data/
    ingestion/ dataset registry, typed loaders, schema validation, sources
    storage/   Parquet snapshots for reproducibility
  config/      settings (pydantic-settings) and structured logging
  domain.py    identifier and season/week validation
  exceptions.py
data/knowledge/ the curated RAG corpus (15 markdown documents, committed)
scripts/       fixture generation, LLM benchmark, index build, retrieval benchmark
tests/
  fixtures/    committed Parquet fixtures, recorded web payloads, a test corpus
docs/          architecture, API reference, LLM boundary, RAG design, data docs
```

Dependencies flow strictly downward: `api → services → {analytics, retrieval} →
data`, with `llm` a sibling of `api` that also stops at `services`.
[tests/architecture/test_layering.py](tests/architecture/test_layering.py)
enforces this.

## Documentation

- [Architecture](docs/architecture.md) — layering, boundaries, design decisions
- [API reference](docs/api.md) — endpoints, conventions, error semantics
- [LLM tool calling](docs/llm-tools.md) — the model/analytics boundary, tools,
  validation, error handling, and numeric grounding
- [Data model](docs/data-model.md) — entities, identifiers, grain, relationships
- [Data sources](docs/data-sources.md) — nflverse datasets, cadence, licensing

## Acknowledgement

NFL data is provided by [nflverse](https://github.com/nflverse). Most nflverse
data is licensed CC-BY 4.0; FTN charting data is CC-BY-SA 4.0. This project is
not affiliated with or endorsed by the NFL.
