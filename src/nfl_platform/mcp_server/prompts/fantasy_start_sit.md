I need a start/sit decision between {player_a} and {player_b} in {scoring} scoring{when}.

Work in this order:

1. Call `compare_fantasy_players` with both players and the scoring format, so
   both are measured over an identical window. Do not make two separate calls:
   comparing different game counts is the most common way a start/sit answer
   misleads.
2. Call `get_player_usage` for each player if the question turns on role,
   workload or target share rather than on points already scored.
3. Call `search_web` for anything the statistical record cannot contain: this
   week's injury designation, a depth-chart change, an expected absence. If web
   search reports itself unavailable, say the current status could not be
   checked rather than assuming both players are healthy.

Then answer:

- Give a clear recommendation, and say which evidence drove it.
- Every number must come from a tool result, with the window it covers stated.
- Past fantasy points and their standard deviation describe consistency that has
  already happened. They are not a projection, and no tool here produces one.
  Any forward-looking statement is your reasoning, labelled as such.
- If the two players are close enough that the data does not separate them, say
  so instead of manufacturing a tiebreaker.
