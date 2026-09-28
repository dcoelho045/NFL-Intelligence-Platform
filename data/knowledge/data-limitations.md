---
document_id: data-limitations
title: Known Data Limitations
document_type: methodology
source: authored
updated: 2026-09-27
---

## The structured data is not live

Statistics are downloaded and cached, with a default cache lifetime of 24
hours. Upstream itself updates on its own schedule. A game that finished a few
hours ago may not yet be present.

Any question about what is happening now — injuries, transactions, depth
charts, weather, inactives — should be answered from web retrieval, not from
the statistical record.

## In-progress weeks are partial

During a playing week, some games have been played and others have not. A team
statistic for the current week may cover an incomplete set of games. Windows
report the number of games that contributed, which is the check for this.

## Advanced metrics have shorter histories

Basic box score data extends to 1999. Metrics derived from richer tracking
data begin considerably later. A comparison across eras using an advanced
metric may silently compare different numbers of seasons.

## Snap counts and participation

Snap count data is not available for the entire history, and participation data
is incomplete in early seasons. A missing snap share is not a zero snap share.

## Injury data is a report, not a diagnosis

The injuries dataset records what teams published on official injury reports.
Teams have latitude in how they describe an injury, and the report reflects
practice participation and game status rather than medical fact.

## Trades and mid-season movement

A traded player's rows carry the team he played for at the time. A season total
therefore spans two teams, and a team-filtered query will return only part of
his season. Player-level queries are the reliable way to see a full season for
a traded player.

## Name changes

Players occasionally change their listed name. The GSIS identifier remains
stable across such changes, which is another reason the identifier rather than
the name is canonical.

## What this platform does not have

There is no projection system, no betting market feed beyond the spread and
total recorded on the schedule, no weather data, no contract or salary data,
and no coaching or scheme classification. Questions in those areas are either
web-retrieval questions or out of scope entirely.
