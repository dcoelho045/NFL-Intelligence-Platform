"""Training entry point.

Lives in ``ml/`` rather than only in ``scripts/`` so it is importable and
testable, and so ``nfl-train`` can be a console script. It is still offline-only
code: nothing in the serving path imports this module, and an architecture test
checks that.

Two steps, separable on purpose:

``snapshot``
    Download the datasets once and pin them as a named Parquet snapshot. Pinning
    is what makes a metric reproducible -- nflverse revises historical data, so
    "train on the latest" quietly means "train on something different each time".

``train``
    Build features from the snapshot, run the walk-forward evaluation, write
    versioned artifacts, and promote whichever model the validation folds
    support.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from nfl_platform.config.logging import configure_logging, get_logger
from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.data.ingestion.datasets import Dataset
from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.data.ingestion.sources import ParquetDataSource
from nfl_platform.data.storage.snapshots import SnapshotStore
from nfl_platform.ml.features.fantasy import build_fantasy_features
from nfl_platform.ml.features.games import build_game_features
from nfl_platform.ml.registry.store import ModelRegistry
from nfl_platform.ml.training.pipeline import (
    TrainingOutcome,
    summarise,
    train_fantasy_models,
    train_game_models,
)

logger = get_logger(__name__)

#: Datasets a full training run needs. Deliberately excludes play-by-play: it is
#: several gigabytes and this version's features do not use it.
TRAINING_DATASETS: tuple[Dataset, ...] = (
    Dataset.SCHEDULES,
    Dataset.TEAM_STATS,
    Dataset.PLAYER_STATS,
    Dataset.SNAP_COUNTS,
    Dataset.FF_OPPORTUNITY,
    Dataset.PLAYERS,
)

#: nflverse publishes different history per dataset, and asking for a season a
#: dataset does not cover is an error rather than an empty frame. Requesting the
#: intersection keeps one short dataset from failing the whole snapshot.
DATASET_EARLIEST_SEASON: dict[Dataset, int] = {
    Dataset.SNAP_COUNTS: 2012,
    Dataset.FF_OPPORTUNITY: 2006,
}

#: Game features work back to the start of nflverse coverage, but the league
#: before realignment is a different competition in ways the model cannot see.
DEFAULT_GAME_START = 2002
#: Snap counts begin in 2012, and the fantasy features lean on them heavily.
#: Starting a year later leaves a prior season behind every week-1 row.
DEFAULT_FANTASY_START = 2013


def build_snapshot(
    name: str,
    seasons: Sequence[int],
    *,
    settings: Settings | None = None,
    datasets: Sequence[Dataset] = TRAINING_DATASETS,
) -> Path:
    """Download the training datasets once and pin them to disk."""
    resolved = settings or get_settings()
    loader = NflDataLoader(settings=resolved)
    store = SnapshotStore(settings=resolved)
    season_list = list(seasons)

    for dataset in datasets:
        if dataset is Dataset.PLAYERS:
            requested: list[int] | None = None
        else:
            earliest = DATASET_EARLIEST_SEASON.get(dataset, 0)
            requested = [season for season in season_list if season >= earliest]
            if not requested:
                logger.warning(
                    "Snapshot skipped: no requested season is covered",
                    extra={"dataset": str(dataset), "earliest": earliest},
                )
                continue
        frame = loader.load(dataset, requested)
        path = store.write(name, dataset, frame)
        logger.info(
            "Snapshot written",
            extra={"dataset": str(dataset), "rows": frame.height, "path": str(path)},
        )
    return store.path_for(name, datasets[0]).parent


def load_snapshot(name: str, settings: Settings | None = None) -> dict[Dataset, pl.DataFrame]:
    """Read a pinned snapshot back, skipping datasets it does not contain."""
    resolved = settings or get_settings()
    store = SnapshotStore(settings=resolved)
    frames: dict[Dataset, pl.DataFrame] = {}
    for dataset in TRAINING_DATASETS:
        if store.exists(name, dataset):
            frames[dataset] = store.read(name, dataset)
    if Dataset.SCHEDULES not in frames:
        raise FileNotFoundError(
            f"Snapshot {name!r} has no schedules. Create it with: "
            f"python scripts/train_models.py snapshot --name {name}"
        )
    return frames


def _source_frames(
    snapshot: str | None, settings: Settings, fixtures: Path | None
) -> dict[Dataset, pl.DataFrame]:
    if fixtures is not None:
        source = ParquetDataSource(fixtures)
        loader = NflDataLoader(source=source, settings=settings, enable_cache=False)
        frames = {}
        for dataset in TRAINING_DATASETS:
            try:
                frames[dataset] = loader.load(dataset, None)
            except Exception:
                continue
        return frames
    if snapshot:
        return load_snapshot(snapshot, settings)
    raise ValueError("Specify --snapshot or --fixtures.")


def run_training(
    *,
    settings: Settings,
    snapshot: str | None,
    fixtures: Path | None,
    problems: Sequence[str],
    test_seasons: int,
    validation_seasons: int,
    game_start: int,
    fantasy_start: int,
    command: str,
    dry_run: bool,
) -> list[TrainingOutcome]:
    """Build features, evaluate, and write artifacts for the selected problems."""
    frames = _source_frames(snapshot, settings, fixtures)
    schedules = frames[Dataset.SCHEDULES]
    team_stats = frames.get(Dataset.TEAM_STATS)
    registry = ModelRegistry(settings=settings)
    outcomes: list[TrainingOutcome] = []

    if "game" in problems:
        features = build_game_features(schedules.filter(pl.col("season") >= game_start), team_stats)
        outcome = train_game_models(
            features,
            settings=settings,
            snapshot=snapshot,
            command=command,
            test_seasons=test_seasons,
            validation_seasons=validation_seasons,
        )
        outcomes.append(outcome)
        if not dry_run:
            _persist(registry, outcome)

    if "fantasy" in problems:
        player_stats = frames.get(Dataset.PLAYER_STATS)
        if player_stats is None:
            raise ValueError("Fantasy training needs player_stats in the snapshot.")
        features = build_fantasy_features(
            player_stats.filter(pl.col("season") >= fantasy_start),
            schedules.filter(pl.col("season") >= fantasy_start - 1),
            team_stats=team_stats,
            snap_counts=frames.get(Dataset.SNAP_COUNTS),
            ff_opportunity=frames.get(Dataset.FF_OPPORTUNITY),
            players=frames.get(Dataset.PLAYERS),
        )
        outcome = train_fantasy_models(
            features,
            settings=settings,
            snapshot=snapshot,
            command=command,
            test_seasons=test_seasons,
            validation_seasons=validation_seasons,
        )
        outcomes.append(outcome)
        if not dry_run:
            _persist(registry, outcome)

    return outcomes


def _persist(registry: ModelRegistry, outcome: TrainingOutcome) -> None:
    """Write every artifact, then promote the one the validation folds chose.

    All of them are written, not just the winner. A baseline that cannot be
    loaded and served is not a baseline the promotion rule can fall back to, and
    a benchmark nobody can reproduce is just a number in a document.
    """
    for artifact in outcome.artifacts:
        registry.save(artifact, make_default=artifact.metadata.version == outcome.default_version)


def _report_path(settings: Settings, problem: str) -> Path:
    return settings.resolved_model_dir() / problem / "training_report.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    snap = sub.add_parser("snapshot", help="Download and pin the training datasets.")
    snap.add_argument("--name", default=None, help="Snapshot name. Defaults to the setting.")
    snap.add_argument("--from-season", type=int, default=DEFAULT_GAME_START)
    snap.add_argument("--to-season", type=int, default=None, help="Defaults to current season.")

    train = sub.add_parser("train", help="Train, evaluate and version the models.")
    train.add_argument("--snapshot", default=None, help="Snapshot to train from.")
    train.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="Train from a Parquet fixture directory instead of a snapshot (tests only).",
    )
    train.add_argument(
        "--problem", choices=("game", "fantasy", "all"), default="all", help="What to train."
    )
    train.add_argument("--test-seasons", type=int, default=2)
    train.add_argument("--validation-seasons", type=int, default=4)
    train.add_argument("--game-start", type=int, default=DEFAULT_GAME_START)
    train.add_argument("--fantasy-start", type=int, default=DEFAULT_FANTASY_START)
    train.add_argument(
        "--dry-run", action="store_true", help="Evaluate and report without writing artifacts."
    )

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    command = "python scripts/train_models.py " + " ".join(argv or sys.argv[1:])

    if args.action == "snapshot":
        name = args.name or settings.ml_training_snapshot
        latest = args.to_season or NflDataLoader(settings=settings).current_season
        seasons = list(range(args.from_season, latest + 1))
        path = build_snapshot(name, seasons, settings=settings)
        print(f"Snapshot {name!r} written to {path} covering {seasons[0]}-{seasons[-1]}.")
        return 0

    problems = ("game", "fantasy") if args.problem == "all" else (args.problem,)
    snapshot = args.snapshot
    if snapshot is None and args.fixtures is None:
        snapshot = settings.ml_training_snapshot

    outcomes = run_training(
        settings=settings,
        snapshot=snapshot,
        fixtures=args.fixtures,
        problems=problems,
        test_seasons=args.test_seasons,
        validation_seasons=args.validation_seasons,
        game_start=args.game_start,
        fantasy_start=args.fantasy_start,
        command=command,
        dry_run=args.dry_run,
    )

    for outcome in outcomes:
        print(summarise(outcome))
        if not args.dry_run:
            report: dict[str, Any] = {
                **outcome.report,
                "artifacts": [a.metadata.as_dict() for a in outcome.artifacts],
            }
            path = _report_path(settings, outcome.problem)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
            print(f"  report: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
