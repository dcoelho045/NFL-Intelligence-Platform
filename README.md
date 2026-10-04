# NFL Intelligence Platform

A production-style AI engineering platform that answers NFL questions with
**evidence-backed** responses.

The governing principle: **an LLM is never the source of truth for a number.**
Statistics come from deterministic functions over
[nflverse](https://github.com/nflverse) data, and predictions come from trained
model artifacts. The LLM's job is to orchestrate tools and synthesise an answer,
never to recall a statistic or estimate a probability.

## Status

| Milestone | Scope                                         | Status      |
| --------- | --------------------------------------------- | ----------- |
| 1         | NFL data foundation (`nflreadpy` + Polars)    | Complete    |
| 2         | Analytics service layer + FastAPI             | Complete    |
| 3         | LLM tool calling                              | Complete    |
| 4         | Web retrieval + lightweight local RAG         | Complete    |
| 5         | MCP server (tools, resources, prompts)        | Complete    |
| 6         | ML prediction models (game + fantasy)         | Complete    |
| 7-11      | Agents, evaluation, observability, A2A, CI/CD | Not started |

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

Milestone 5 exposes those same capabilities over the Model Context Protocol, so
any MCP host (VS Code, Claude Desktop, a future agent) can use them without
importing this package. The MCP server is a **sibling** of the HTTP API and the
in-process assistant, not a layer above them: all three call one
`ServiceContainer`, and the MCP tools contain no business logic of their own. See
[docs/mcp-design.md](docs/mcp-design.md).

Milestone 6 adds the only numbers in the platform that describe games which have
not been played: a calibrated home-win probability and a weekly fantasy
projection. Both come from versioned model artifacts trained offline on a pinned
data snapshot — never from the LLM, which may explain a prediction but cannot
produce one. Predictions carry the model that made them, the history they rest
on, and their caveats, and are tagged `SourceKind.MODEL` so they stay
distinguishable from measured statistics end to end.

On the frozen 2024–2025 test period the game model reaches a Brier score of
0.222 and 63.4% accuracy against a 0.249 / 54.1% home-field baseline, and the
fantasy model reaches 4.62 MAE against a 4.96 rolling-average baseline. The
game model that ships is **logistic regression, not LightGBM**: the boosted
model did not beat it by more than fold-to-fold noise on validation, so the
promotion rule kept the simpler one. See
[docs/ml-predictions.md](docs/ml-predictions.md).

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

The MCP server adds two optional settings, both with working defaults:

| Variable                    | Purpose                                         | Default            |
| --------------------------- | ----------------------------------------------- | ------------------ |
| `NFL_MCP_SERVER_NAME`       | Name advertised to MCP clients                  | `nfl-intelligence` |
| `NFL_MCP_ENABLE_WEB_SEARCH` | Whether the server may spend web-search credits | `true`             |

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

## Running the MCP server

The same capabilities, exposed over the Model Context Protocol for AI hosts:

```powershell
# As a host launches it (stdio; no port, no web server needed)
python -m nfl_platform.mcp_server

# Or, after `pip install -e .`
nfl-mcp

# Inspect it interactively in a browser (needs Node for npx)
npx @modelcontextprotocol/inspector python -m nfl_platform.mcp_server
```

Register it with VS Code (`.vscode/mcp.json`), Claude Desktop, or any other MCP
client:

```json
{
  "servers": {
    "nfl-intelligence": {
      "command": "${workspaceFolder}/.venv/Scripts/python.exe",
      "args": ["-m", "nfl_platform.mcp_server"],
      "env": {
        "OPENAI_API_KEY": "${env:OPENAI_API_KEY}",
        "TAVILY_API_KEY": "${env:TAVILY_API_KEY}"
      }
    }
  }
}
```

It offers 15 read-only tools, 5 resources (`nfl://status`,
`nfl://reference/...`, `nfl://knowledge/...`) and 3 prompts. Both keys are
optional: without them the deterministic analytics tools still work and the
retrieval tools report themselves unavailable rather than inventing an answer.
See [docs/mcp-design.md](docs/mcp-design.md).

## Using it from Python

The HTTP API is one adapter. The services behind it are importable directly,
which is how both the assistant and the MCP server call them:

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
- Matchup analysis is **descriptive**. It contains no prediction. Predictions
  live under `/api/v1/predictions` and carry the model that produced them.
- Fantasy projections are **conditional on the player playing**. They contain no
  estimate of the chance he is inactive, so check injury status separately.

## Predictions

```powershell
# Pin the training data once, then train (no API key, no GPU, under a minute)
python scripts/train_models.py snapshot --from-season 2002
python scripts/train_models.py train

# Inspect the recorded train / validation / frozen-test metrics
python scripts/report_models.py
python scripts/report_models.py --features   # the feature dictionary
```

Model artifacts are JSON and LightGBM text -- **no pickle anywhere** -- and are
not committed: they are derived output, verified by checksum on every load, and
rebuildable from the pinned snapshot. The API reports predictions as unavailable
rather than guessing when none are installed. See
[docs/ml-predictions.md](docs/ml-predictions.md).

## Testing

```powershell
# Offline unit tests (default) - no network access, no API key required
pytest

# Include the network-dependent integration tests
pytest -m integration

# Live LLM tests - costs money, needs an API key
pytest -m llm

# Training over the full pinned snapshot - slow, needs a downloaded snapshot
pytest -m training

# With coverage
pytest --cov=nfl_platform --cov-report=term-missing
```

Integration, `llm`, `web` and `training` tests are deselected by default so the
default suite is fast, deterministic, and usable in CI without network access or
a key. The ML tests run against a generated league in `tests/synthetic.py`, so
CI never downloads nflverse data and never trains on real data.

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
  mcp_server/  MCP adapter: tools, resources, prompts, error translation
    prompts/   markdown prompt templates
  llm/         tool-calling assistant: agent loop, tools, prompts, tracing
    providers/ the chat SDK adapter
  retrieval/   web search and local RAG - a peer of analytics/
    web/       provider Protocol, Tavily and Brave adapters, cache, policy
    knowledge/ corpus loading, chunking, embeddings, Parquet vector store
    providers/ the embeddings SDK adapter
  ml/          prediction models - a peer of analytics/ and retrieval/
    features/  as-of feature builders, shared by training and inference
    datasets/  matrix assembly and time-aware splitting (offline)
    models/    estimators; the only place sklearn and lightgbm are imported
    evaluation/ metrics and the walk-forward backtest (offline)
    training/  the training pipeline and promotion rule (offline)
    registry/  versioned, checksummed, pickle-free artifacts
    predict/   load an artifact and score it; no fit() anywhere
  services/    OOP analytics, retrieval and prediction services
  schemas/     Pydantic DTOs - the typed contract, plus shared argument fields
  analytics/   deterministic, LLM-free computation
  data/
    ingestion/ dataset registry, typed loaders, schema validation, sources
    storage/   Parquet snapshots for reproducibility
  config/      settings (pydantic-settings) and structured logging
  domain.py    identifier and season/week validation
  exceptions.py
  tool_errors.py  adapter-neutral exception to tool-error mapping
data/knowledge/ the curated RAG corpus (15 markdown documents, committed)
data/models/   trained artifacts (generated, not committed)
scripts/       fixtures, benchmarks, index build, model training and reporting
tests/
  fixtures/    committed Parquet fixtures, recorded web payloads, a test corpus
  synthetic.py generated league used by the ML tests
docs/          architecture, API, LLM boundary, RAG, MCP and ML design
```

Dependencies flow strictly downward: `api → services → {analytics, retrieval,
ml} → data`, with `llm` and `mcp_server` siblings of `api` that also stop at
`services`. Inside `ml`, training is unreachable from serving.
[tests/architecture/test_layering.py](tests/architecture/test_layering.py)
enforces both.

## Documentation

- [Architecture](docs/architecture.md) — layering, boundaries, design decisions
- [API reference](docs/api.md) — endpoints, conventions, error semantics
- [LLM tool calling](docs/llm-tools.md) — the model/analytics boundary, tools,
  validation, error handling, and numeric grounding
- [MCP design](docs/mcp-design.md) — why MCP sits at this boundary, how it
  differs from REST, and the tool/resource/prompt catalogue
- [RAG design](docs/rag-design.md) — web retrieval and the local knowledge base
- [ML predictions](docs/ml-predictions.md) — features and when each becomes
  known, leakage prevention, time-aware validation, baselines and results,
  artifacts and reproducibility
- [Data model](docs/data-model.md) — entities, identifiers, grain, relationships
- [Data sources](docs/data-sources.md) — nflverse datasets, cadence, licensing

## Acknowledgement

NFL data is provided by [nflverse](https://github.com/nflverse). Most nflverse
data is licensed CC-BY 4.0; FTN charting data is CC-BY-SA 4.0. This project is
not affiliated with or endorsed by the NFL.
