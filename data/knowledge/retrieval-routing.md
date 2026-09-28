---
document_id: retrieval-routing
title: Choosing an Evidence Source
document_type: methodology
source: authored
updated: 2026-09-27
---

## Three sources, three jobs

The assistant has three ways to obtain evidence, and they do not overlap.

Deterministic analytics tools answer questions about what happened in games
that have been played: yardage, touchdowns, targets, carries, snaps, scores,
records, fantasy points, shares, and rates. These are the only acceptable
source for a statistic.

Web retrieval answers questions about what is true now: injury status,
transactions, depth-chart changes, suspensions, coaching decisions, statements,
weather, and news. Anything that could have changed since yesterday belongs
here.

The local knowledge base answers questions about what things mean: scoring
formats, metric definitions, rules, roster designations, and how this platform
computes what it computes. This content is stable and is not current.

## Worked examples

- "How many receiving yards did this player have last week?" — analytics only.
- "Is this player expected to play on Sunday?" — web retrieval only.
- "What does PPR mean?" — knowledge base only.
- "What is target share and what was this player's last month?" — knowledge
  base for the definition, analytics for the figure.
- "Should I start player A or player B?" — analytics for production and usage,
  web retrieval for current availability, and clearly labelled judgement to
  combine them.
- "Why was this player's usage down after the trade?" — analytics for the usage
  change, web retrieval for the circumstances.

## Questions that need no retrieval at all

Many questions are answered entirely by the analytics tools. Searching the web
for a completed statistic is slower, costs money, and produces a worse answer
than the deterministic source. Searching the knowledge base for a statistic
returns nothing useful by design.

Retrieval is used when it adds evidence, not by default.

## When sources disagree

If a retrieved article reports a statistic that differs from the analytics
layer, the analytics layer is authoritative for the statistic. The discrepancy
may still be worth mentioning, attributed to its source.

If retrieved sources disagree with each other, report the disagreement rather
than silently picking one.
