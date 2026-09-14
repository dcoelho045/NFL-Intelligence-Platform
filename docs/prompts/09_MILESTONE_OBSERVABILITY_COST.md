# Milestone 9 — Observability and Model Trade-Offs

## Objective

Make the system observable enough to diagnose AI failures and compare LLM configurations.

## Trace

For each request, capture appropriate non-sensitive metadata:

```text
request id
model
prompt version
agent
tools called
retrieval count
latency
input tokens
output tokens
estimated cost
success/failure
evaluation score where available
```

Do not store unnecessary sensitive user data.

## Model Comparison

Compare at least two model configurations/providers when practical.

Evaluate:

- capability
- tool-calling reliability
- factual accuracy
- latency
- token usage
- cost

Create a table such as:

| Model | Accuracy | Tool Accuracy | Latency | Cost |
|---|---:|---:|---:|---:|

Use actual measured data.

## Failure Analysis

Build a way to identify common failures:

```text
Tool failure
Retrieval failure
Hallucination
Bad routing
Bad prediction
Timeout
Rate limit
```

## Cost Optimization

Experiment with strategies such as:

- cheaper models for simple routing
- stronger models for difficult synthesis
- caching
- reducing unnecessary context
- limiting retrieval
- structured outputs

Measure whether optimizations hurt quality.

## Acceptance Criteria

- [ ] Requests are traceable
- [ ] Latency and token usage are measurable
- [ ] Model comparisons are documented
- [ ] Cost optimization experiment exists
- [ ] Failures can be diagnosed

## Copilot Instructions

Do not collect sensitive information unnecessarily.

Do not claim cost savings without measurements.
