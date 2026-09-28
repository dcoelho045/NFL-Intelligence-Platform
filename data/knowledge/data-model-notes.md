---
document_id: data-model-notes
title: Data Model and Identifiers
document_type: methodology
source: authored
updated: 2026-09-27
---

## Source

All structured statistics come from nflverse, accessed through the nflreadpy
package. nflverse aggregates publicly available NFL data and is the same source
used by much of the public football analytics community.

## Datasets and their grain

- Schedules — one row per game
- Player stats — one row per player per week
- Team stats — one row per team per week
- Play by play — one row per play
- Weekly rosters — one row per player per week
- Players — one row per player, not season-aware
- Injuries — one row per player per injury report
- Snap counts — one row per player per game
- Fantasy opportunity — one row per player per week

Knowing the grain matters: a question about a player's season total is an
aggregation over weekly rows, and a question about a single play is only
answerable from play-by-play data.

## Player identifiers

The GSIS identifier is the canonical key, formatted `00-` followed by seven
digits. Other identifier systems exist in the wider ecosystem, but this
platform resolves everything to GSIS.

Display names are not unique. Name-based lookup returns a disambiguation list
when more than one player matches.

## Team abbreviations

Teams are identified by abbreviation. Relocated franchises change abbreviation
by era, and historical rows keep the abbreviation in use at the time:

- Oakland Raiders (OAK) became Las Vegas Raiders (LV) in 2020
- San Diego Chargers (SD) became Los Angeles Chargers (LAC) in 2017
- St. Louis Rams (STL) became Los Angeles Rams (LAR) in 2016
- Washington used WAS throughout, under several team names

A query using the current abbreviation will not match older rows, and this
platform does not silently remap them. The valid list for a given season can be
requested directly.

## Perspective in team statistics

Team statistics are reported from two perspectives: what a team produced on
offence, and what it allowed on defence. A "points" figure is meaningless
without knowing which. Every team response states its perspective.

## Coverage

Data begins in 1999. Coverage varies by dataset: play-by-play and basic box
score data go back furthest, while advanced tracking-derived measures begin
later. Asking for a season a dataset does not cover returns an explicit gap
rather than an empty result.

## Caching

Downloads are cached on the filesystem with a configurable lifetime, defaulting
to 24 hours. This means structured data may lag live results by up to a day,
which is one of the reasons current information is retrieved from the web
instead.
