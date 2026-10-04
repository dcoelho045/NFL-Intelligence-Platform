# Copilot Agent Workflow

## Purpose

Use GitHub Copilot Agent as an AI-assisted software engineering tool while retaining human architectural control and code review.

## How To Start Each Milestone

Give Copilot the corresponding milestone file and say:

> Complete Milestone X from `path/to/MILESTONE_X.md`.

Then instruct it to:

1. Inspect the repository.
2. Inspect the master project specification.
3. Inspect all completed milestone documentation.
4. Identify existing relevant code.
5. Propose a short implementation plan.
6. Implement the milestone incrementally.
7. Write tests alongside implementation.
8. Run tests.
9. Fix failures.
10. Update documentation.
11. Summarize changes.
12. Stop.

## Important Rule

Copilot must NOT automatically begin the next milestone.

## Before Accepting a Milestone

Review:

- architecture
- generated code
- tests
- dependencies
- error handling
- security
- documentation

Run the full applicable test suite yourself.

## Git Workflow

If on main branch, create a new branch
Use one branch per milestone if practical:

```text
feature/milestone-01-data-foundation
feature/milestone-02-analytics-api
feature/milestone-03-llm-tools
...
```

Create a PR after each meaningful milestone.

## Prompting Copilot

Prefer precise engineering instructions.

Good:

> Implement the player analytics service described in Milestone 2. Preserve existing public interfaces. Add unit and API tests. Do not introduce LLM functionality.

Bad:

> Build the whole NFL AI platform.

## Ask Copilot To Explain

For important architectural changes ask:

- Why did you choose this abstraction?
- What alternatives did you consider?
- What are the failure modes?
- How is this tested?
- What assumptions are being made?
- What would need to change at scale?

## Review AI-Generated Code

Never blindly accept generated code.

Check:

- correctness
- maintainability
- security
- performance
- test quality
- dependency necessity

## Useful Copilot Tasks

Use Copilot for:

- scaffolding
- test generation
- refactoring
- documentation
- debugging
- code review
- implementation
- architecture exploration

Use your own judgment for:

- system boundaries
- security
- model selection
- evaluation methodology
- production trade-offs

## Interview Value

Keep a record of meaningful examples where Copilot:

1. Generated an initial implementation.
2. Failed a test or produced an incorrect design.
3. You identified the problem.
4. You refined the prompt/requirements.
5. Copilot produced a better implementation.
6. You validated it with tests.

These are excellent examples for explaining AI-assisted development.
