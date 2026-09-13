# Milestone 2 — NFL Analytics Service

## Objective

Turn the data foundation into a clean analytics service that future LLM tools can call.

Do NOT implement RAG, MCP, agents, or A2A yet.

## Requirements

Create a service layer separating:

```text
Raw NFL data
     |
     v
Data access
     |
     v
Analytics service
     |
     v
Stable typed interfaces
```

The LLM should eventually interact with the stable interfaces, not raw Polars DataFrames.

## Capabilities

Implement and test:

### Player

- recent performance
- season performance
- usage
- snap share
- target/carry share where applicable
- game-by-game performance

### Team

- offensive performance
- defensive performance
- recent form
- home/away performance

### Matchup

Create a matchup analysis function combining relevant team/player information.

Example conceptual interface:

```python
analyze_matchup(team_a, team_b, season, week)
```

### Fantasy

Create deterministic functions for:

- recent fantasy production
- usage trends
- matchup information
- player comparisons

## API

Expose the analytics service through FastAPI if appropriate.

The API should have:

- typed request/response models
- validation
- error handling
- health endpoint
- OpenAPI documentation

## Testing

Add:

- unit tests
- service-level tests
- API tests

Test edge cases and invalid inputs.

## Important Constraint

No LLM should be required to use this service.

The service must be useful as a normal programmatic API.

## Acceptance Criteria

- [ ] Stable analytics interfaces exist
- [ ] API works locally
- [ ] Request/response schemas are typed
- [ ] Tests pass
- [ ] Documentation exists
- [ ] Raw data access is separated from business logic
- [ ] No LLM/RAG/MCP functionality yet

## Copilot Instructions

Inspect Milestone 1 before implementation.

Do not rewrite working Milestone 1 components unless necessary.

Prioritize clean interfaces because later milestones will expose these capabilities as LLM tools and MCP tools.
