# Milestone 1 — NFL Data Foundation

## Objective

Build a reliable Python data foundation around `nflreadpy`.

Do NOT implement LLMs, RAG, MCP, agents, or ML prediction models in this milestone.

## Requirements

Create a clean Python project structure similar to:

```text
src/
  data/
    ingestion/
    storage/
  analytics/
  config/
tests/
docs/
```

Use:

- Python
- nflreadpy
- Polars
- pytest
- type hints
- structured logging where appropriate

## Data Layer

Implement reusable functions/classes for loading relevant NFL datasets.

Prioritize:

- schedules
- player statistics
- team statistics
- play-by-play
- rosters
- injuries
- snap counts
- fantasy-related data where available

Do not download or process unnecessary datasets.

## Data Engineering Requirements

The pipeline should:

- identify seasons/weeks explicitly
- cache data where practical
- handle missing data
- validate expected columns
- provide useful errors
- avoid hard-coded player/team assumptions
- make data loading reproducible

## Initial Analytics Functions

Create deterministic functions such as:

```python
get_player_recent_stats(player_id, weeks)
get_team_stats(team_id, weeks)
get_team_defense(team_id, weeks)
get_player_usage(player_id, weeks)
get_recent_games(team_id, weeks)
get_injury_information(player_id)
```

Exact interfaces may differ if Copilot determines a better design.

## Testing

Create unit tests covering:

- successful data loading
- missing data
- invalid identifiers
- week/season filtering
- aggregation correctness
- empty results
- schema validation

Tests must use fixtures/mocks where practical instead of repeatedly downloading large datasets.

## Documentation

Create:

- `docs/data-model.md`
- `docs/data-sources.md`

Document:

- datasets used
- important columns
- identifiers
- relationships
- known limitations

## Acceptance Criteria

- [ ] Python project runs locally
- [ ] nflreadpy data can be loaded reproducibly
- [ ] Core analytics functions work
- [ ] Unit tests pass
- [ ] Data model is documented
- [ ] No LLM functionality exists yet
- [ ] No secrets are committed
- [ ] README contains setup instructions

## Copilot Instructions

Implement this milestone only.

Before coding, inspect the repository and propose the implementation plan.

After implementation:

1. Run tests.
2. Fix failures.
3. Review the architecture for unnecessary complexity.
4. Update documentation.
5. Summarize files changed and commands used.

Do not move on to Milestone 2 automatically.
