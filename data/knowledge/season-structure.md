---
document_id: season-structure
title: Season Structure and Season Types
document_type: rules
source: authored
updated: 2026-09-27
---

## Preseason, regular season, postseason

An NFL season has three parts. The preseason consists of exhibition games whose
results do not count. The regular season determines playoff qualification. The
postseason is a single-elimination tournament ending in the Super Bowl.

In nflverse data these are distinguished by season type. This project uses two
values: `REG` for regular-season games and `POST` for postseason games. `ALL`
requests both. Preseason data is not used.

## Weeks and games

The regular season runs 18 weeks in which each team plays 17 games and has one
bye week. Before the 2021 season the regular season was 17 weeks and 16 games.

Because of the bye, a team's week number and its game number do not match after
the bye has passed. This is the reason analytics in this project count games
rather than weeks.

## Bye weeks

Every team has one scheduled week with no game. A player therefore has no
statistics for that week, and the absence is not an indication of injury or
benching.

## Postseason weeks

Postseason rounds continue the week numbering rather than restarting it. Week
numbers above the regular-season maximum correspond to the Wild Card, Divisional,
Conference Championship and Super Bowl rounds. Week numbers in this project are
therefore valid from 1 to 22.

## Season labelling

A season is labelled by the calendar year in which it begins. The 2024 season
starts in September 2024 and its Super Bowl is played in February 2025. A
question about "the 2025 Super Bowl" is usually a question about the 2024
season.

## Playoff qualification

Each conference sends seven teams: four division winners, seeded one to four by
record, and three wild-card teams seeded five to seven. The top seed in each
conference receives a first-round bye.
