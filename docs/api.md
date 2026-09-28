# API Reference

Interactive documentation is generated from the code. With the server running:

- Swagger UI — <http://127.0.0.1:8000/docs>
- ReDoc — <http://127.0.0.1:8000/redoc>
- OpenAPI schema — <http://127.0.0.1:8000/openapi.json>

This page covers the conventions and error semantics that the generated docs
do not explain.

## Running

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn nfl_platform.api.app:app --reload
```

The first request for a season downloads from nflverse and is slow; subsequent
requests are served from cache.

## Conventions

### `last_n_games` counts games, not weeks

A bye week, an inactive week, or a mid-season trade means "the last 4 weeks" can
contain fewer than 4 games. Counting games keeps sample sizes honest.

Every analytical response carries a `window` object reporting what was actually
covered:

```json
"window": {
  "seasons": [2024],
  "season_type": "ALL",
  "weeks": null,
  "games": 3,
  "requested_last_n_games": 3
}
```

When `games` is lower than `requested_last_n_games`, the player or team did not
play that many games in scope.

Use `weeks` instead when you want specific week numbers. It is applied before
`last_n_games`.

### `season_type` defaults to `ALL`

Defaulting to `REG` would make "the last 4 games" silently stop at week 18 and
exclude a team's most recent — and most relevant — playoff games. Pass
`season_type=REG` to exclude the postseason.

### Defensive statistics are prefixed `allowed_`

`/teams/KC/defense` returns `avg_allowed_passing_yards`, never
`avg_passing_yards`. The prefix makes it impossible to mistake what a team gave
up for what it gained — an error that would be invisible once an LLM is quoting
these numbers.

Defence is derived from opponent rows: the row where `opponent_team == 'KC'`
describes what an offence produced against Kansas City.

### Totals and averages are different types

`totals` uses integers; `per_game` uses floats. 5.67 carries per game is not an
integer, and forcing it into one would silently round.

EPA appears only in `per_game`. It is a rate statistic, so a season total is
meaningless and is returned as `null`.

### Every total ships with its average

Publishing `total_allowed_passing_tds` without `avg_allowed_passing_tds` invites
the caller to divide one by the other. A human doing that is fine; an LLM doing
it produces a number that looks retrieved but was invented. Team summaries
therefore expose a per-game counterpart for every total, and a test enforces it
on the defensive summary.

### Players can be addressed by id or name

```text
GET /api/v1/players/00-0033873
GET /api/v1/players/Patrick%20Mahomes
```

Names are matched after stripping accents, punctuation, and generational
suffixes, so `A.J. Brown`, `AJ Brown`, and `aj brown` are equivalent, and
`Odell Beckham Jr.` matches `Odell Beckham`.

**Ambiguous names return 409, never a guess.** About 7.5% of nflverse players
share a display name.

## Endpoints

### Health

| Method | Path            | Purpose                                     |
| ------ | --------------- | ------------------------------------------- |
| GET    | `/health`       | Liveness. No data access.                   |
| GET    | `/health/ready` | Readiness. Confirms the data layer answers. |

`/health/ready` returns 200 with `"status": "degraded"` rather than an error
status, because an orchestrator wants a signal, not a stack trace.

### Players — `/api/v1/players`

| Method | Path                        | Purpose                         |
| ------ | --------------------------- | ------------------------------- |
| GET    | `/search`                   | Resolve a name to candidates    |
| GET    | `/{player_id}`              | Profile and career detail       |
| GET    | `/{player_id}/stats/recent` | Recent totals and averages      |
| GET    | `/{player_id}/stats/season` | Full-season totals and averages |
| GET    | `/{player_id}/usage`        | Snaps, targets, carries, shares |
| GET    | `/{player_id}/games`        | Game-by-game log                |
| GET    | `/{player_id}/injuries`     | Injury-report history           |
| POST   | `/compare`                  | Compare players over one window |

### Teams — `/api/v1/teams`

| Method | Path               | Purpose                              |
| ------ | ------------------ | ------------------------------------ |
| GET    | `/{team}/offense`  | What the offence produced            |
| GET    | `/{team}/defense`  | What the defence allowed             |
| GET    | `/{team}/form`     | Recent record and scoring            |
| GET    | `/{team}/splits`   | Home versus away                     |
| GET    | `/{team}/games`    | Game log from the team's perspective |
| GET    | `/{team}/injuries` | Team injury report                   |

For `/splits`, `last_n_games` applies _within_ each split, so it means "last N
home games" and "last N away games" rather than N games divided unevenly.

### Matchups — `/api/v1/matchups`

| Method | Path                    | Purpose                         |
| ------ | ----------------------- | ------------------------------- |
| GET    | `?team_a=KC&team_b=BUF` | Descriptive two-team comparison |

Returns both sides' offence, defence, and form, plus the scheduled fixture and
head-to-head history.

**Contains no prediction.** No win probability, no projected score. A predictive
endpoint arrives in a later milestone and will be explicitly labelled as such.

### Fantasy — `/api/v1/fantasy`

| Method | Path                              | Purpose                             |
| ------ | --------------------------------- | ----------------------------------- |
| GET    | `/players/{player_id}/production` | Weekly scoring, totals, consistency |
| GET    | `/players/{player_id}/trends`     | Direction of opportunity            |
| GET    | `/players/{player_id}/matchup`    | What the opponent's defence allowed |
| POST   | `/compare`                        | Compare fantasy production          |

`scoring` accepts `ppr` (default) or `standard`.

`trends` describes what already happened by comparing the two halves of the
window. It is not a forecast, and the direction is withheld entirely when fewer
than four games are available.

### Meta — `/api/v1/meta`

| Method | Path       | Purpose                                       |
| ------ | ---------- | --------------------------------------------- |
| GET    | `/teams`   | Valid abbreviations for the requested seasons |
| GET    | `/seasons` | Seasons that currently return data            |

Team lists are derived from the data, so historical abbreviations such as `OAK`
stay correct for the seasons in which they were used.

### Assistant — `/api/v1/assistant`

| Method | Path     | Purpose                                  |
| ------ | -------- | ---------------------------------------- |
| POST   | `/ask`   | Answer a natural-language NFL question   |
| GET    | `/tools` | The tool catalogue with argument schemas |

This is the only part of the API that involves a language model, and the only
part that costs money per request. Everything above works without one.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/assistant/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How has Patrick Mahomes played recently?"}'
```

```json
{
  "answer": "Over his last 3 games Mahomes threw for 812 yards ...",
  "evidence": [
    {
      "tool": "get_player_stats",
      "arguments": { "player": "00-0033873", "last_n_games": 3 }
    }
  ],
  "sources": [{ "kind": "structured", "label": "get_player_stats" }],
  "stats": {
    "model": "gpt-4.1",
    "prompt_version": "analyst_v2",
    "iterations": 2,
    "stop_reason": "completed",
    "duration_ms": 4120.5,
    "total_tokens": 3908,
    "estimated_cost_usd": 0.0421,
    "tools_used": ["get_player_stats"],
    "ungrounded_numbers": [],
    "numbers_from_unstructured_sources": [],
    "fabricated_sources": [],
    "retrieval": {
      "web_searches": 0,
      "web_cache_hits": 0,
      "web_results": 0,
      "web_results_dropped": 0,
      "knowledge_searches": 0,
      "knowledge_hits": 0
    }
  }
}
```

A question that uses retrieval returns richer sources:

```json
{
  "sources": [
    { "kind": "structured", "label": "get_player_stats" },
    {
      "kind": "web",
      "label": "Chiefs QB listed as full participant",
      "url": "https://www.espn.com/nfl/story/_/id/.../practice-report",
      "domain": "espn.com",
      "published_at": "2026-09-25T14:00:00+00:00",
      "retrieved_at": "2026-09-27T09:12:44+00:00"
    },
    {
      "kind": "knowledge_base",
      "label": "Injury Report Designations",
      "document_id": "injury-report-designations",
      "updated": "2026-09-27"
    }
  ]
}
```

Null fields are omitted. `sources` is built from the run trace, not written by
the model, so the assistant cannot cite a source it did not retrieve.
`numbers_from_unstructured_sources` lists figures traceable only to retrieved
text rather than to the analytics layer — see
[rag-design.md](rag-design.md#grounding-three-buckets).

Three fields make the answer checkable rather than merely plausible:

- **`evidence`** is built from the tool calls that actually executed, not written
  by the model. It cannot cite a tool that never ran.
- **`stats.ungrounded_numbers`** lists any number in the answer that could not be
  traced to a tool result. Empty is the expected case; non-empty warrants
  scrutiny.
- **`stats.stop_reason`** is `completed`, or names the budget that was hit.

`history` carries earlier turns; the assistant holds no server-side session
state. See [llm-tools.md](llm-tools.md) for the full boundary.

## Errors

Every failure returns the same shape:

```json
{
  "error": "ambiguous_player",
  "message": "Player name 'Josh Allen' matches 2 players: ...",
  "context": { "candidates": [ ... ] }
}
```

| Status | `error`                    | Meaning                                               |
| ------ | -------------------------- | ----------------------------------------------------- |
| 400    | `unknown_dataset`          | Dataset name not in the registry                      |
| 404    | `player_not_found`         | No such player                                        |
| 404    | `team_not_found`           | No such team; `context.valid_teams` lists real ones   |
| 409    | `ambiguous_player`         | Name matches several; `context.candidates` lists them |
| 422    | `invalid_season`           | Outside 1999–current                                  |
| 422    | `invalid_week`             | Outside 1–22                                          |
| 422    | `invalid_game_count`       | `last_n_games` not a positive integer                 |
| 422    | `invalid_request`          | Other argument errors                                 |
| 500    | `schema_validation_failed` | Upstream nflverse schema drift — our problem          |
| 502    | `llm_provider_error`       | The model provider failed; `context.retryable`        |
| 503    | `data_source_unavailable`  | nflverse unreachable — transient, retry               |
| 503    | `llm_not_configured`       | No provider key; assistant only, analytics unaffected |

### 409 is the important one

```bash
curl "http://127.0.0.1:8000/api/v1/players/Josh%20Allen"
```

```json
{
  "error": "ambiguous_player",
  "message": "Player name 'Josh Allen' matches 2 players: 00-0034857 (QB, BUF, 2018-2026); 00-0030833 (C, TB, 2014-2016). Narrow the search by position, team, or season.",
  "context": {
    "query": "Josh Allen",
    "candidates": [
      { "player_id": "00-0034857", "description": "QB, BUF, 2018-2026" },
      { "player_id": "00-0030833", "description": "C, TB, 2014-2016" }
    ]
  }
}
```

Silently returning the first match would give real statistics for the wrong
person — a failure that looks exactly like a correct answer. Narrow the request:

```bash
curl "http://127.0.0.1:8000/api/v1/players/search?name=Josh%20Allen&position=QB"
```

### Unknown request fields are rejected

Request bodies use `extra="forbid"`. A misspelled field returns 422 rather than
being silently ignored, which matters most when the caller is a language model
generating arguments.

## Examples

```bash
# Recent form
curl "http://127.0.0.1:8000/api/v1/players/00-0033873/stats/recent?seasons=2024&last_n_games=3"

# What a defence allowed
curl "http://127.0.0.1:8000/api/v1/teams/KC/defense?seasons=2024&last_n_games=4"

# Home/away splits
curl "http://127.0.0.1:8000/api/v1/teams/KC/splits?seasons=2024"

# Matchup
curl "http://127.0.0.1:8000/api/v1/matchups?team_a=KC&team_b=BUF&seasons=2024"

# Compare players
curl -X POST "http://127.0.0.1:8000/api/v1/players/compare" \
  -H "Content-Type: application/json" \
  -d '{"player_ids":["00-0033873","00-0034857"],"seasons":[2024],"last_n_games":4}'
```

## Using the services without HTTP

The API is one adapter. The services it calls are importable directly, which is
how Milestones 3 and 5 will use them:

```python
from nfl_platform.services import ServiceContainer

container = ServiceContainer()
stats = container.players.recent_performance("00-0033873", seasons=2024, last_n_games=4)
print(stats.per_game.passing_yards)
```

Same types, same validation, same errors — no HTTP involved.

## Not yet present

No authentication, rate limiting, or pagination. The API binds to localhost and
serves public read-only data. Authentication is required before anything
internet-facing and is scheduled for the CI/CD and polish milestone.
