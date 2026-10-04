# Milestone 6 — NFL Prediction Models

## Objective

Build actual machine-learning models for game prediction and fantasy projection.

The LLM should explain model outputs, not replace the predictive model.

## Game Prediction

Start with a simple baseline.

Possible baselines:

- home-team historical win rate
- logistic regression

Then implement a stronger model such as:

- XGBoost
- LightGBM

Potential features:

- offensive efficiency
- defensive efficiency
- EPA/play
- success rate
- turnover differential
- recent form
- home/away
- rest
- injuries where reliable

## Fantasy Projection

Build a model that predicts fantasy production.

Potential features:

- recent fantasy points
- snap share
- target/carry share
- routes
- red-zone usage
- opponent defense
- game environment
- injuries
- recent trends

## Temporal Leakage

This is a critical requirement.

Training/evaluation must only use information that would have been available at prediction time.

Do NOT randomly split time-series NFL data if that causes future information to leak into training.

Use time-aware validation.

## Evaluation

Game prediction:

- accuracy
- ROC-AUC
- Brier score
- calibration

Fantasy prediction:

- MAE
- RMSE
- rank correlation where useful

Compare models against a baseline.

## PyTorch

After a strong baseline exists, implement a small PyTorch experiment if it provides learning value.

Do not use PyTorch merely for resume keyword matching.

## Model Interface

Create a stable model service:

```text
predict_game(...)
project_fantasy_points(...)
```

Return:

- prediction
- probability/projection
- model version
- relevant metadata

## Acceptance Criteria

- [x] Baseline exists
- [x] Stronger ML model exists
- [x] Time-aware evaluation exists
- [x] Metrics are recorded
- [x] No temporal leakage
- [x] Model artifacts are versioned appropriately
- [x] Prediction service integrates with the existing application
- [x] PyTorch experiment is documented if implemented

## Copilot Instructions

Do not allow Copilot to optimize against the test set.

Require reproducible training/evaluation.

Document feature definitions and exactly when each feature becomes available.
