"""Print the recorded train / validation / test metrics for every artifact.

Reads what training wrote; computes nothing. Use it to inspect results without
retraining, and to check that the frozen test period was scored exactly once.

    python scripts/report_models.py
    python scripts/report_models.py --features    # the feature dictionary

``--features`` renders the feature tables in docs/ml-predictions.md straight
from the FeatureSet objects, so the documented availability of each input cannot
drift away from the code that builds it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Imported after the path shim, which is what makes this runnable without an
# editable install. Same pattern as scripts/build_knowledge_index.py.
from nfl_platform.ml.features.fantasy import FANTASY_FEATURES  # noqa: E402
from nfl_platform.ml.features.games import GAME_FEATURES  # noqa: E402
from nfl_platform.ml.features.schema import FeatureSet  # noqa: E402

GAME_KEYS = (
    ("brier", "brier_score"),
    ("acc", "accuracy"),
    ("auc", "roc_auc"),
    ("logloss", "log_loss"),
    ("ece", "expected_calibration_error"),
    ("bss_vs_base", "brier_skill_vs_baseline"),
)
FANTASY_KEYS = (
    ("mae", "mae"),
    ("rmse", "rmse"),
    ("r2", "r_squared"),
    ("week_rho", "within_week_spearman"),
    ("vs_base", "mae_improvement_vs_baseline"),
)


def render(metrics: dict[str, object], keys: tuple[tuple[str, str], ...]) -> str:
    if not metrics:
        return "not evaluated"
    parts = []
    for label, key in keys:
        value = metrics.get(key)
        if isinstance(value, int | float):
            parts.append(f"{label}={value:.4f}")
    rows = metrics.get("rows")
    if rows:
        parts.append(f"n={rows}")
    return " ".join(parts) or "not evaluated"


def render_features(name: str, feature_set: FeatureSet) -> str:
    lines = [
        f"### {name} (`{feature_set.version}`, {len(feature_set)} features)",
        "",
        "| Feature | Source dataset | Available | Lookback | Meaning |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in feature_set:
        window = spec.window or "point-in-time"
        lines.append(
            f"| `{spec.name}` | `{spec.dataset}` | {spec.availability.value} "
            f"| {window} | {spec.description} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", action="store_true", help="Print the feature dictionary instead."
    )
    args = parser.parse_args()

    if args.features:
        print(render_features("Game prediction", GAME_FEATURES))
        print()
        print(render_features("Fantasy projection", FANTASY_FEATURES))
        return 0

    for problem in ("game_win", "fantasy_points"):
        path = ROOT / "data" / "models" / problem / "training_report.json"
        if not path.is_file():
            print(f"{problem}: no training report. Run scripts/train_models.py train.")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        keys = GAME_KEYS if problem == "game_win" else FANTASY_KEYS
        plan = report["plan"]
        print("=" * 78)
        print(f"{problem}  default={report['default_version']}  rows={report['rows']}")
        print(f"  fit seasons   : {plan['fit_seasons'][0]}-{plan['fit_seasons'][-1]}")
        print(f"  validation    : {plan['validation_seasons']}")
        print(f"  FROZEN TEST   : {plan['test_seasons']}")
        for artifact in report["artifacts"]:
            print(f"  --- {artifact['version']} ({artifact['role']}) ---")
            print(f"      train : {render(artifact['train_metrics'], keys)}")
            print(f"      valid : {render(artifact['validation_metrics'], keys)}")
            print(f"      test  : {render(artifact['test_metrics'], keys)}")
            print(f"      test evaluations: {artifact['test_evaluations']}")
            if artifact.get("promotion"):
                print(f"      {artifact['promotion']['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
