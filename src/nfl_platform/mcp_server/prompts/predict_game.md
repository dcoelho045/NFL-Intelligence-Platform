Predict the outcome of {away_team} at {home_team}{when}, and explain what drives the
prediction.

Follow this order. It matters: the model supplies the number, the analytics
supply the explanation, and neither substitutes for the other.

1. Call `predict_game` with home_team={home_team} and away_team={away_team}.
   This is the only acceptable source for a win probability. If it returns an
   error, say that no model prediction is available and continue with steps 2-4
   as a descriptive comparison. Do not invent a probability, and do not convert
   a betting line into one yourself.

2. Read `top_factors` from the prediction. Those are the features that actually
   moved this specific prediction, computed by the model. Use them to decide
   what is worth explaining rather than guessing at what might matter.

3. Call `get_team_stats` and `get_team_form` for both teams to put real numbers
   behind those factors, and `get_team_injuries` for context the model does not
   use. Injury information is not a model input, so treat it as a reason the
   prediction might be wrong rather than as part of it.

4. Optionally call `search_knowledge_base` for any metric you need to define
   (EPA per play, success rate). Definitions come from there, not from memory.

Then write the answer:

- Lead with the probability and the model version from `model_info`.
- Say plainly that it is a model estimate for a game that has not been played,
  not a measured statistic.
- Quote `baseline_home_win_probability` alongside it so the reader can see what
  the model adds over the trivial "home teams usually win" answer.
- If `market_home_win_probability` is present, report it as a separate reference
  point and note that the market is not an input to the model. Where the two
  disagree, that disagreement is the interesting part -- say which factors the
  model is weighting that the line apparently is not.
- Pass on every entry in `caveats`. Thin history and uncalibrated probabilities
  change how the number should be read.
- Keep deterministic statistics and model output visibly separate throughout.
