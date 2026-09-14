# Milestone 4 — Retrieval-Augmented Generation

## Objective

Add RAG for unstructured NFL knowledge while keeping structured statistics in the analytics system.

## Critical Design Rule

RAG is for information that benefits from semantic retrieval.

Do NOT embed every NFL statistical table and ask the vector database to perform numerical analysis.

Structured statistics should continue coming from the analytics/data layer.

## RAG Architecture

```text
Documents
   |
   v
Ingestion
   |
   v
Chunking + metadata
   |
   v
Embeddings
   |
   v
Vector storage
   |
   v
Retriever
   |
   v
LLM
```

## Knowledge Base

Use legally/appropriately available sources.

Potential content:

- NFL terminology
- team/player descriptions
- historical explanations
- scheme/tendency documentation
- user-provided research
- permitted public articles/documents

Track source metadata.

Every retrieved document should have identifiable provenance.

## Retrieval

Implement:

- chunking
- metadata
- embeddings
- similarity search
- metadata filtering

If useful, experiment with:

- hybrid search
- reranking

Do not add these unless evaluation shows benefit.

## Grounded Answers

The assistant should distinguish:

```text
Structured evidence
Knowledge-base evidence
Model prediction
LLM interpretation
```

Responses should cite or identify sources where appropriate.

## Evaluation

Create a retrieval benchmark.

Measure:

- hit rate
- precision
- recall
- ranking quality where practical
- source correctness

Create a small golden dataset before optimizing retrieval.

## Testing

Test:

- ingestion
- duplicate handling
- chunk metadata
- retrieval
- missing documents
- irrelevant questions
- source attribution

## Acceptance Criteria

- [ ] RAG pipeline works end-to-end
- [ ] Structured and unstructured data have clear boundaries
- [ ] Retrieved evidence has provenance
- [ ] Retrieval benchmark exists
- [ ] Retrieval tests pass
- [ ] Assistant can combine tool results with RAG results

## Copilot Instructions

Do not replace the existing analytics service with RAG.

Explain architectural decisions in `docs/rag-design.md`.

Prefer simple infrastructure first.
