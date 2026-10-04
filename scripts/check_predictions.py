"""Ad-hoc end-to-end check of the prediction service against installed artifacts.

Not a test: it needs trained models and live or cached nflverse data, neither of
which the CI suite has. Use it to sanity-check a freshly trained model.

    python scripts/check_predictions.py 2025 1 KC BUF
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_platform.services.container import ServiceContainer


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2025
    week = int(argv[1]) if len(argv) > 1 else 1
    home = argv[2] if len(argv) > 2 else "KC"
    away = argv[3] if len(argv) > 3 else "BUF"

    container = ServiceContainer()
    status = container.predictions.model_status()
    print(f"available={status.available} enabled={status.enabled} dir={status.model_dir}")
    for entry in status.models:
        print(
            f"  {entry.model_id:<34} {entry.role:<9} {entry.primary_metric}="
            f"{entry.validation_score} calibration={entry.calibration_method}"
        )
    if not status.available:
        print("No models installed. Run: python scripts/train_models.py train")
        return 1

    prediction = container.predictions.predict_game(home, away, season=season, week=week)
    print(f"\n{prediction.away_team} at {prediction.home_team}, {season} week {prediction.week}")
    print(f"  model      : {prediction.model_info.model_id}")
    print(f"  home win   : {prediction.home_win_probability:.3f}")
    print(f"  baseline   : {prediction.baseline_home_win_probability}")
    print(f"  market ref : {prediction.market_home_win_probability}")
    print(f"  winner     : {prediction.predicted_winner}")
    for factor in prediction.top_factors:
        print(f"    {factor.feature:<32} {factor.contribution:+.4f} ({factor.direction})")
    for caveat in prediction.caveats:
        print(f"  ! {caveat}")

    slate = container.predictions.predict_slate(season, week)
    print(f"\nslate: {slate.game_count} predicted, {len(slate.skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
