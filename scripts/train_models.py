"""Train the NFL prediction models.

Two steps. Pin the data once, then train from the pinned copy as often as you
like -- that is what makes a recorded metric reproducible, since nflverse
revises historical rows and "train on the latest" quietly means something
different every month.

    python scripts/train_models.py snapshot                 # download and pin
    python scripts/train_models.py train                    # train both problems
    python scripts/train_models.py train --problem game     # one problem
    python scripts/train_models.py train --dry-run          # evaluate, write nothing

Artifacts land under ``data/models/<problem>/<version>/`` and are not committed:
they are derived output, rebuildable from the snapshot in under a minute.

Nothing here calls a language model, and no step costs anything beyond the
initial download.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_platform.ml.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
