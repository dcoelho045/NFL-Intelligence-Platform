Review how {player} has been performing{when}.

Work in this order:

1. Call `get_player_stats` for the totals and per-game averages over the window.
2. Call `get_player_game_log` to see the week-to-week shape behind those
   averages. An average hides both a steady player and a volatile one.
3. Call `get_player_usage` for snap share, target share and opportunity, which is
   what explains a change in production.
4. Call `get_player_injuries` if a drop or absence needs explaining.
5. Call `search_knowledge_base` if a metric needs defining (WOPR, air yards
   share, how "last N games" is counted). That corpus is stable reference
   material, not news.

Then write the review:

- Separate what happened (the numbers) from why it might have happened (your
  reading of usage, role and health).
- Every number must come from a tool result, with its window stated.
- An empty injury list means nothing was reported in that window. It is not a
  statement about the player's condition today.
- Do not project future performance. No tool here produces a projection, and an
  invented one would be indistinguishable from a real one to the reader.
