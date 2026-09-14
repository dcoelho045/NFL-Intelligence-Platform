# Milestone 8 — Automated AI Evaluation

## Objective

Build a serious evaluation system for the AI application.

This milestone is essential for demonstrating AI Engineering maturity.

## Evaluation Dataset

Create a version-controlled benchmark containing at least 100 questions over time.

Categories:

- factual NFL questions
- player comparisons
- fantasy start/sit
- matchup analysis
- game predictions
- RAG questions
- adversarial/ambiguous questions
- questions where information is unavailable

Each benchmark case should define expected behavior.

## Evaluate

### Tool Selection

Was the correct tool selected?

### Retrieval

Was useful evidence retrieved?

### Groundedness

Are factual claims supported by available evidence?

### Citation/Source Accuracy

Are sources actually relevant to the answer?

### Numerical Correctness

Do reported statistics match tool results?

### Hallucination

Does the model invent facts?

### Response Quality

Is the response useful, clear, and appropriately uncertain?

### Performance

Track:

- latency
- token usage
- approximate cost

## ML Evaluation

Keep ML metrics separate from LLM metrics.

## Regression Testing

Every important change to prompts, models, retrieval, or agents should be able to run the evaluation suite.

Create thresholds where practical.

Example:

```text
tool_selection >= 95%
groundedness >= 90%
numerical_accuracy >= 98%
```

Do not blindly use these exact thresholds if the benchmark shows they are inappropriate.

## Prompt Experiments

Version prompts and compare:

```text
prompt_v1
prompt_v2
prompt_v3
```

Record evaluation results.

## Acceptance Criteria

- [ ] Benchmark dataset exists
- [ ] Automated evaluation runs
- [ ] Multiple AI quality dimensions are measured
- [ ] Regression failures are detectable
- [ ] Prompt versions can be compared
- [ ] Cost/latency are tracked
- [ ] Evaluation results are documented

## Copilot Instructions

Treat evaluations as tests, not demos.

Do not modify benchmark answers simply to make the model look better.

If an evaluation fails, investigate whether the problem is:
- retrieval
- tool selection
- prompt
- model
- data
- orchestration
- evaluator error
