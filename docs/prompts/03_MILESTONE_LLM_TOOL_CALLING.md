# Milestone 3 — LLM Tool Calling

## Objective

Add an LLM assistant that can dynamically call NFL analytics tools.

This is the first LLM milestone.

## Architecture

```text
User question
     |
     v
LLM
     |
     +--> get_player_stats
     +--> get_team_stats
     +--> get_matchup
     +--> get_injury
     +--> get_recent_games
     |
     v
LLM synthesis
     |
     v
Answer
```

## Core Principle

The LLM must not invent numerical NFL statistics.

If a number is needed, the model must obtain it from a tool or clearly state that the information is unavailable.

## Tools

Wrap existing analytics capabilities as typed tools.

Suggested tools:

```text
get_player_stats
get_team_stats
get_matchup
get_recent_games
get_player_usage
get_injury_information
compare_players
```

Keep tool implementations thin. Business logic belongs in the analytics service.

## Prompt Engineering

Create versioned prompts:

```text
prompts/
  orchestrator_v1.txt
  analyst_v1.txt
```

Prompts should define:

- role
- tool-use rules
- evidence requirements
- uncertainty behavior
- response format

## Structured Responses

Where practical, use structured outputs for:

- selected tool
- player comparison
- prediction request
- evidence list
- final recommendation

## Model Trade-Off Experiment

Test at least two model configurations/models if practical.

Record:

- response quality
- latency
- token usage
- approximate cost
- tool selection accuracy

Do not optimize for one metric only.

## Testing

Create tests for:

- correct tool selection
- malformed tool input
- unavailable data
- tool failure
- numerical grounding
- multi-tool questions

## Acceptance Criteria

- [ ] User can ask NFL questions
- [ ] LLM dynamically calls tools
- [ ] Tool arguments are validated
- [ ] Numerical claims come from tools
- [ ] Tool failures are handled
- [ ] Prompt is versioned
- [ ] Tests exist
- [ ] Basic latency/cost information is recorded

## Copilot Instructions

Do not implement RAG, MCP, A2A, or multi-agent orchestration yet.

Use the existing analytics service rather than duplicating calculations.

Document exactly how the LLM/tool boundary works.
