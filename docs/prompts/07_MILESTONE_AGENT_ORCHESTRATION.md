# Milestone 7 — Agent Orchestration

## Objective

Turn the assistant into a multi-agent system with clear responsibilities.

Use an agent orchestration framework such as LangGraph if appropriate.

## Agents

Implement at least:

### Orchestrator

Determines what kind of request the user made and coordinates work.

### Fantasy Agent

Handles:

- start/sit
- player comparisons
- fantasy projections

Uses analytics, prediction models, and RAG when appropriate.

### Prediction Agent

Handles:

- game winner
- win probability
- score/projection questions

Uses ML models and analytics.

### Research Agent

Handles:

- explanations
- trends
- contextual questions
- RAG-heavy research

## Architecture

```text
User
 |
 v
Orchestrator
 |
 +--> Fantasy Agent
 |
 +--> Prediction Agent
 |
 +--> Research Agent
 |
 v
Evidence collection
 |
 v
Final response
```

## Agent State

Define explicit structured state.

Avoid hidden global state.

Track:

- user request
- intent
- selected agent
- tool results
- retrieved documents
- model predictions
- errors
- final response metadata

## Tool Selection

Agents should use tools rather than hallucinating information.

## Observability

Create an execution trace such as:

```text
Intent: fantasy_start_sit

Agent: FantasyAgent

Tools:
  get_player_stats
  get_matchup
  get_injury
  project_fantasy_points

RAG:
  3 documents retrieved

Model:
  fantasy-model-v2

Final response generated
```

Do not expose private chain-of-thought.

Expose only useful execution metadata.

## Acceptance Criteria

- [ ] Multiple agents exist
- [ ] Orchestrator routes requests
- [ ] Agents use tools
- [ ] Agents can use RAG
- [ ] Prediction agent uses ML model
- [ ] Execution traces exist
- [ ] Tests cover routing and agent failures
- [ ] Architecture is documented

## Copilot Instructions

Understand the existing tool-calling system before introducing the orchestration framework.

Do not duplicate tools or analytics logic.

Keep agents modular enough to test independently.
