---
document_id: corpus-readme
title: About This Knowledge Base
document_type: reference
source: authored
updated: 2026-09-27
---

## What this corpus is

A small, version-controlled set of markdown documents covering stable football
knowledge and this platform's own methodology. It is the retrieval source for
questions about what things mean, not about what happened or what is happening
now.

Every document is written for this project. Nothing is scraped, and no
third-party text is reproduced, which keeps the licensing position simple.

## What it is deliberately not

It is not a news archive. Keeping a local corpus current would mean ingesting
articles continuously, which is expensive, quickly stale, and unnecessary when
web retrieval can answer current questions on demand.

It is not a statistics store. Numerical facts come from the deterministic
analytics layer. Embedding statistical tables and asking a vector search to do
arithmetic produces confident wrong answers.

## Document metadata

Each document begins with a front-matter block:

- `document_id` — stable identifier used in citations, unique across the corpus
- `title` — human-readable name
- `document_type` — reference, glossary, rules, methodology, or research
- `source` — where the content came from; `authored` means written for this
  project
- `updated` — when the content was last reviewed

The `updated` date travels with every chunk to the assistant, so curated
knowledge is never presented as current information.

## Adding a document

Add a markdown file with the front-matter block above and rebuild the index
with `python scripts/build_knowledge_index.py`. Headings define chunk
boundaries, so a document organised into clear sections retrieves better than
one long block of prose.

Keep the corpus small. Its value is precision on a narrow set of stable
questions, not coverage.

## User research

Notes and research provided by a user can be added here with
`document_type: research` and a `source` describing their origin. They are
retrieved the same way as anything else and cited by document id.
