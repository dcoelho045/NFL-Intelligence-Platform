# Milestone 10 — Agent2Agent Communication

## Objective

Experiment with A2A for communication between independent specialized agents.

## Important Concept

MCP answers:

> How does an agent access tools/resources?

A2A answers:

> How can independent agents communicate and collaborate?

Use A2A only where agent-to-agent communication adds value.

## Architecture

```text
              Orchestrator
                   |
          +--------+--------+
          |                 |
          v                 v
   Prediction Agent <----> Research Agent
          |
          v
    Fantasy Agent
```

The exact topology may change based on the implementation.

## A2A Use Case

Example:

User:

> "Should I start Player A or Player B, considering their matchups?"

Fantasy Agent may request matchup context from Research Agent.

Research Agent can return structured evidence.

Fantasy Agent then combines it with:

- projection model
- player usage
- matchup data

## Requirements

Document:

- why A2A is useful
- why ordinary function calls are insufficient for this use case
- how A2A differs from MCP
- agent capabilities
- message/task contracts
- failure handling

## Testing

Test:

- successful agent request
- invalid request
- timeout
- unavailable agent
- malformed response
- partial failure

## Acceptance Criteria

- [ ] At least two independent agents communicate through A2A
- [ ] Contracts are documented
- [ ] Failure handling exists
- [ ] Tests exist
- [ ] MCP vs A2A distinction is documented

## Copilot Instructions

Use current official A2A documentation/specification.

Do not invent protocol behavior.

Do not replace internal function calls with A2A unless there is a demonstrated architectural reason.
