# NFL Intelligence Platform — Master Project Specification

## Project Goal

Build a production-style, portfolio-ready AI Engineering platform that combines:

- NFL data engineering using `nflreadpy`
- Python analytics services
- LLMs and prompt engineering
- Tool calling
- Retrieval-Augmented Generation (RAG)
- Model Context Protocol (MCP)
- Agent orchestration
- Multi-agent communication with A2A
- Machine-learning prediction models
- Automated testing and evaluation of AI outputs
- Observability, cost/latency measurement, and CI/CD
- GitHub Copilot Agent as an AI-assisted development tool

The finished system should answer NFL questions such as:

- "Who is more likely to win this week's game?"
- "Should I start Player A or Player B in fantasy?"
- "Why is Player A projected poorly this week?"
- "How has Team X defended against mobile quarterbacks?"
- "What factors are driving this matchup prediction?"

## Core Engineering Principle

The LLM must NOT be the source of truth for numerical NFL statistics.

Numerical facts and predictions should come from deterministic analytics functions, databases, or ML models. The LLM should orchestrate tools, retrieve evidence, and synthesize an answer.

## Target Architecture

```text
User
 |
 v
Web/API Interface
 |
 v
Agent Orchestrator
 |
 +--> Fantasy Agent
 +--> Prediction Agent
 +--> Research Agent
 |
 +--> MCP tools
 |
 +--> RAG service
 |
 +--> ML prediction services
 |
 v
Evidence-backed response

Supporting systems:
- NFL data pipeline
- PostgreSQL / analytical storage
- Vector store / pgvector
- Evaluation framework
- Observability
- Automated tests
- CI/CD
```

## Technology Direction

Primary language: Python

Expected technologies to evaluate/use:

- nflreadpy
- Polars
- FastAPI
- PostgreSQL
- pgvector or Qdrant
- OpenAI API or another modern LLM provider
- MCP Python SDK
- LangGraph or another agent orchestration framework
- scikit-learn
- XGBoost or LightGBM
- PyTorch
- A2A
- pytest
- GitHub Actions and/or Azure DevOps
- OpenTelemetry or an equivalent observability solution

Do not add technologies merely to make the stack look impressive. Every dependency must have a reason.

## Non-Goals

Do not:

- build everything at once
- put structured NFL statistics into a vector database just because it is RAG
- allow the LLM to fabricate statistics
- train a huge neural network without a useful baseline
- add TensorFlow solely because a job posting mentions it
- create unnecessary microservices before the architecture requires them

## Definition of Done

The finished project should have:

1. A working NFL data/analytics layer
2. A tool-calling LLM assistant
3. A RAG pipeline
4. An MCP server
5. At least one ML prediction system
6. An orchestrated multi-agent system
7. Automated AI evaluation
8. Observability for model/tool execution
9. A2A experimentation between agents
10. Automated tests and CI/CD
11. Clear architecture documentation
12. A polished README with architecture diagrams and examples

## Development Rules

Every milestone must:

- preserve previous functionality
- include tests for new functionality
- update documentation
- avoid unnecessary rewrites
- use type hints
- handle errors explicitly
- keep business logic separate from interfaces
- keep secrets out of source control
- use object oriented programming 
- modern coding standards and best practices

Do not begin a later milestone until the current milestone's acceptance criteria are satisfied.
