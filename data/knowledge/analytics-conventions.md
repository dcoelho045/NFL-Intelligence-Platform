---
document_id: analytics-conventions
title: Analytics Conventions in This Platform
document_type: methodology
source: authored
updated: 2026-09-27
---

## Last N games, not last N weeks

Every windowed statistic in this platform counts games, not weeks. Asking for a
player's last four games returns the four most recent games in which he
appeared, which may span five or six calendar weeks.

This is deliberate. A bye week, an inactive week, or a mid-season trade all
mean that "the last four weeks" contains fewer than four games. Averaging over
weeks silently divides by the wrong number, and the resulting per-game figure
is wrong in a way that looks plausible.

When specific week numbers are genuinely wanted, an explicit weeks filter is
available instead.

## Season type is always stated

Results carry a season type of REG, POST, or ALL. Mixing regular-season and
postseason games without saying so changes both sample size and opponent
quality. Every window returned by this platform records which it covers.

## Windows accompany every number

Every statistical response carries a window describing the seasons, season
type, weeks, and the number of games that actually contributed. A figure
without its window is not interpretable: "312 yards" could be one game or one
season.

The requested game count and the achieved game count are reported separately,
because asking for the last eight games and receiving five is itself
information.

## Totals and averages are both returned

Where a summary is produced, totals and per-game averages are returned
together. This removes the need for anyone downstream — including a language
model — to divide one by the other, which is the most common way a derived
number becomes wrong.

## Nulls mean not applicable

Fields that do not apply to a player are omitted rather than set to zero. A
receiver has no completion total. Reading an absent field as zero would turn
"this does not apply" into a false statistic.

## No estimation anywhere in the deterministic layer

The analytics functions are pure: the same inputs always produce the same
output, and nothing is inferred, smoothed, or projected. If a number is not in
the source data, no number is returned.

## Player identity

The canonical player key is the nflverse GSIS identifier, formatted like
`00-0033873`. Names are ambiguous — a meaningful share of players share a
display name with someone else — so a name that matches more than one player
produces a disambiguation response listing the candidates rather than a guess.
