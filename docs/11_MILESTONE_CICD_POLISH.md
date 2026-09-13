# Milestone 11 — Production Engineering, CI/CD, and Portfolio Polish

## Objective

Turn the project into a polished AI Engineering portfolio project.

## Testing Layers

Ensure the repository has:

```text
unit tests
integration tests
API tests
tool tests
MCP tests
agent tests
RAG retrieval tests
ML evaluation
LLM evaluation
```

Do not make every test depend on external APIs.

Use mocks/fixtures for deterministic tests.

## CI/CD

Create automated pipeline stages:

```text
Lint
 |
Unit tests
 |
Integration tests
 |
Build
 |
Evaluation
 |
Package/deploy
```

Use GitHub Actions, Azure DevOps, or both if useful.

## Containerization

Containerize appropriate components.

Keep architecture simple.

If Kubernetes is added, it should solve an actual deployment/scaling problem rather than being included just for resume keywords.

## Security

Review:

- secrets
- environment variables
- dependency vulnerabilities
- prompt injection risks
- tool authorization
- arbitrary query execution
- untrusted documents
- logging of sensitive data

## README

Rewrite the README as a portfolio-quality engineering document.

Include:

1. Project overview
2. Problem statement
3. Architecture diagram
4. Technologies
5. Example user questions
6. Tool-calling example
7. RAG example
8. MCP architecture
9. Agent architecture
10. A2A architecture
11. ML results
12. AI evaluation results
13. Model cost/latency comparison
14. Testing strategy
15. CI/CD
16. Design trade-offs
17. Known limitations
18. Future work

## Demo

Create a short demo flow:

```text
Question
  |
Agent routing
  |
Tools/RAG/model
  |
Execution trace
  |
Evidence
  |
Final answer
```

## Interview Documentation

Create:

`docs/interview-guide.md`

Include:

- architecture explanation
- why RAG was used
- why structured data was not put into RAG
- MCP vs REST
- MCP vs A2A
- agent orchestration decisions
- model selection trade-offs
- evaluation strategy
- temporal leakage
- hallucination mitigation
- cost optimization
- testing strategy
- biggest engineering challenges
- lessons learned

## Acceptance Criteria

- [ ] CI passes
- [ ] Tests are organized
- [ ] Evaluation runs automatically
- [ ] README is portfolio quality
- [ ] Architecture diagram exists
- [ ] Demo works
- [ ] Interview guide exists
- [ ] Security review completed
- [ ] Project can be explained without relying on buzzwords

## Copilot Instructions

Perform a final engineering review.

Look for:

- duplicated logic
- dead code
- fragile tests
- unnecessary dependencies
- security problems
- poor abstractions
- undocumented assumptions
- missing error handling

Do not add technologies merely for resume keywords.
