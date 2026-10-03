# Milestone 4 --- Web Retrieval + Lightweight RAG

## Objective

Add retrieval capabilities for unstructured NFL knowledge while keeping
the system current, inexpensive, and lightweight.

The **primary retrieval mechanism is web retrieval/search** for
information that changes frequently. A **small local RAG corpus** should
also be maintained to demonstrate and evaluate traditional RAG without
requiring a large document store.

Structured NFL statistics remain in the deterministic analytics system.

## Architecture

```text
                         User Question
                              |
                              v
                             LLM
                         /    |    \
                        /     |     \
                       v      v      v
              Analytics   Web Search  Local RAG
                 Tools       |           |
                   |         |           |
                   v         v           v
              Structured   Current    Stable / curated
              NFL data     web info   domain knowledge
                        \      |       /
                         \     |      /
                          v    v    v
                         Evidence
                            |
                            v
                       LLM synthesis
                            |
                            v
                         Answer
```

The LLM may use one or more retrieval sources depending on the question.

## Critical Design Rules

### 1. Structured statistics are NOT RAG

Do **not** embed NFL statistical tables and ask a vector database to
perform numerical analysis.

Questions such as player receiving yards, rushing attempts, team points,
snap counts, game statistics, player usage, and fantasy statistics
should continue to use the deterministic analytics/data layer from
Milestones 1--3.

### 2. Web retrieval is the primary source for current information

Use web retrieval/search for frequently changing information such as:

- current injuries
- recent news
- roster and depth-chart changes
- trades
- coach/player statements
- current team/player context
- game previews
- current fantasy/news context

Do not build a large local archive of news articles merely to keep the
system current.

### 3. Keep a small local RAG corpus

Maintain a small, version-controlled knowledge base for stable or
curated information that is useful for demonstrating traditional RAG.

Potential content:

- NFL terminology
- rules and scoring explanations
- fantasy scoring definitions
- statistical definitions
- stable football concepts
- project methodology
- user-provided research
- appropriately licensed/public-domain or otherwise permitted
  documents

The local corpus should remain intentionally small. Do not download
thousands of articles simply to demonstrate RAG.

## Web Retrieval

Implement a web retrieval interface that can:

- accept a natural-language search query
- retrieve relevant current information
- preserve source URLs/titles and useful provenance
- return bounded, relevant context to the LLM
- handle no-result and failed-search cases
- avoid unnecessary searches when deterministic analytics already
  provide the answer

Prefer an existing supported web-search/API capability rather than
building and maintaining a custom web crawler unless there is a clear
engineering reason to do so.

Keep the provider behind a clean interface so it can be replaced later.

Example conceptual interface:

```python
search_web(
    query: str,
    recency_days: int | None = None,
    max_results: int = 5,
) -> list[WebSearchResult]
```

Exact interfaces may differ if the implementation has a better design.

## Local RAG Pipeline

Implement a lightweight local RAG pipeline:

```text
Local Documents
      |
      v
Ingestion
      |
      v
Chunking + Metadata
      |
      v
Embeddings
      |
      v
Small Vector Store
      |
      v
Retriever
      |
      v
Relevant Context
```

The vector store should be lightweight and inexpensive.

Reasonable options include PostgreSQL + pgvector if PostgreSQL is
already used, a local vector database, or another simple
embedded/vector-storage option.

Do not introduce a hosted vector database solely for resume keywords.

## Embeddings

Use a configurable embedding model.

Document:

- model/provider
- embedding dimensions
- approximate cost
- why it was selected
- how it can be replaced

Avoid embedding duplicate content unnecessarily.

## Chunking and Metadata

Each local document chunk should preserve useful metadata such as:

```text
document_id
source
title
section
created/updated date when available
document type
chunk_id
```

Web results should preserve appropriate provenance such as:

```text
source URL
title
publisher/domain when available
retrieval timestamp
```

Do not store more web content than is necessary for the request.

## Retrieval Routing

The system should distinguish among:

```text
Structured evidence
    -> deterministic analytics tools

Current external information
    -> web retrieval

Stable/curated knowledge
    -> local RAG

Model prediction
    -> ML prediction service (later milestone)

LLM interpretation
    -> final synthesis
```

The LLM should be able to use multiple evidence sources when
appropriate.

Examples:

```text
"How many yards did Player X have last week?"
        -> Analytics tool

"What is Player X's current injury status?"
        -> Web retrieval

"What does PPR scoring mean?"
        -> Local RAG

"Should I start Player X or Player Y?"
        -> Analytics + web retrieval + later ML model
```

Do not require every request to go through web search or local RAG.

## Grounded Answers

The assistant should distinguish:

```text
Structured evidence
Web evidence
Local knowledge-base evidence
Model prediction
LLM interpretation
```

Where appropriate, responses should identify or cite the sources used.

The system must not treat an irrelevant or low-quality retrieved result
as authoritative.

If required evidence cannot be found, the assistant should clearly state
that limitation rather than inventing information.

## Freshness

Web retrieval should be preferred when freshness matters.

For local documents, record enough metadata to know when information was
created or last updated.

Do not imply that local knowledge is current when it may be stale.

## Evaluation

Create a small retrieval benchmark covering both retrieval paths.

Evaluate web retrieval for relevance, source quality, freshness,
provenance correctness, ability to find current information, and
no-result behavior.

Evaluate local RAG for hit rate, precision/recall where practical,
ranking quality where practical, chunk relevance, source correctness,
and metadata preservation.

Evaluate combined answers for correct retrieval-source selection,
continued use of analytics for structured questions, use of web
retrieval for current questions, use of local RAG for stable knowledge,
groundedness, source attribution, and hallucination behavior.

Create a small golden dataset before optimizing retrieval.

## Testing

Test:

- local document ingestion
- duplicate handling
- chunking
- metadata preservation
- embedding generation
- local retrieval
- web-search interface
- web-search failures
- empty results
- stale/irrelevant results
- source attribution
- routing between analytics, web retrieval, and local RAG
- combining tool results with retrieved evidence

Tests should use mocks/fixtures where appropriate so the test suite does
not depend on live web searches.

## Cost and Storage Constraints

This project is intentionally designed to remain inexpensive.

Prefer:

- small local document corpus
- limited retrieval results
- configurable result limits
- caching where useful
- avoiding unnecessary embeddings
- avoiding large-scale web crawling
- avoiding a hosted vector database unless justified
- avoiding unnecessary LLM calls
- mocked external services in automated tests

Track approximate retrieval/API costs so later milestones can measure
trade-offs.

## Security and Web Content

Treat web content as untrusted external input.

Do not allow retrieved text to override system/application instructions.

Consider prompt-injection risks in retrieved web content.

Do not execute code, tools, or instructions merely because they appear
in retrieved content.

## Acceptance Criteria

- [x] Web retrieval works end-to-end
- [x] Web results preserve useful provenance
- [x] Small local RAG corpus exists
- [x] Local RAG pipeline works end-to-end
- [x] Embeddings are configurable
- [x] Vector storage is lightweight
- [x] Structured NFL statistics remain in the analytics service
- [x] Retrieval source selection is documented
- [x] Retrieval benchmark exists
- [x] Retrieval tests pass
- [x] Assistant can combine analytics, web evidence, and local RAG
      evidence
- [x] Current information can be retrieved without maintaining a large
      document archive
- [x] Storage and API usage remain bounded and configurable
- [x] `docs/rag-design.md` documents the architecture and trade-offs

## Copilot Instructions

Inspect Milestones 1--3 before implementation.

Do not replace the existing analytics service with RAG.

Treat web retrieval as the primary mechanism for current NFL
information.

Keep the local RAG corpus intentionally small and focused on
stable/curated knowledge.

Prefer simple infrastructure first.

Do not add a hosted vector database, custom crawler, reranker,
query-rewriting agent, or additional LLM solely for complexity. Add such
components only if evaluation demonstrates a concrete benefit.

Do not implement MCP, multi-agent orchestration, A2A, or ML prediction
functionality in this milestone.

Explain architectural decisions and cost/storage trade-offs in
`docs/rag-design.md`.

Before implementation, propose the architecture and implementation plan
for approval.

After implementation:

1.  Run tests.
2.  Fix failures.
3.  Test both web retrieval and local RAG with representative questions.
4.  Review the routing between analytics, web retrieval, and local RAG.
5.  Review cost and storage implications.
6.  Update documentation.
7.  Summarize files changed and commands used.
8.  Do not automatically begin Milestone 5.
