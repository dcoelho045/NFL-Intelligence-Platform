"""Time-aware splitting.

Random K-fold is wrong for this data in two independent ways, and both matter.

*It leaks forward.* Rule changes, the 2020 crowdless season, the move to a
17-game schedule in 2021 and a decade of rising pass rates all mean a 2024 row
tells you about the 2024 scoring environment. Letting one into a fold that
scores 2010 inflates every metric without improving a single real prediction.

*It mismatches the use.* The model is always asked the same question -- "fit on
everything through last week, now predict next week". Expanding-window
walk-forward *is* that question, repeated. Any other split measures something
nobody will ever ask for.

The final test period is chosen from the data rather than hard-coded, so a
re-run against a newer snapshot moves it forward instead of quietly evaluating
on seasons that have since become training-era. :func:`complete_seasons` decides
what counts as finished; a season still in progress must never become a test
season, because its weeks are not a random sample of a season.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import polars as pl

#: Seasons held out as the frozen final test period.
DEFAULT_TEST_SEASONS: Final = 2
#: Seasons used for model selection, immediately before the test period.
DEFAULT_VALIDATION_SEASONS: Final = 4
#: Below this, walk-forward folds are fitted on too little to mean anything.
MIN_TRAIN_SEASONS: Final = 4
#: Distinct completed games a season needs before it counts as finished. A full
#: regular season is 256-272; anything near that has been played out.
MIN_COMPLETE_SEASON_GAMES: Final = 240


@dataclass(frozen=True, slots=True)
class TimeFold:
    """One expanding-window fold: fit on everything before, score one season."""

    name: str
    train_seasons: tuple[int, ...]
    eval_seasons: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.train_seasons
            and self.eval_seasons
            and (max(self.train_seasons) >= min(self.eval_seasons))
        ):
            raise ValueError(
                f"Fold {self.name!r} trains on {max(self.train_seasons)} and evaluates "
                f"{min(self.eval_seasons)}; training must end strictly before evaluation."
            )


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """The whole evaluation design, decided once and recorded in metadata."""

    train_seasons: tuple[int, ...]
    validation_seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    folds: tuple[TimeFold, ...]
    rationale: str

    @property
    def fit_seasons(self) -> tuple[int, ...]:
        """Seasons the shipped artifact is finally fitted on.

        Training plus validation: once selection is finished there is no reason
        to throw away the most recent four seasons, and the test period is still
        untouched.
        """
        return tuple(sorted({*self.train_seasons, *self.validation_seasons}))

    def as_dict(self) -> dict[str, object]:
        return {
            "train_seasons": list(self.train_seasons),
            "validation_seasons": list(self.validation_seasons),
            "test_seasons": list(self.test_seasons),
            "fit_seasons": list(self.fit_seasons),
            "folds": [
                {
                    "name": fold.name,
                    "train_seasons": list(fold.train_seasons),
                    "eval_seasons": list(fold.eval_seasons),
                }
                for fold in self.folds
            ],
            "rationale": self.rationale,
        }


def complete_seasons(frame: pl.DataFrame, *, season_column: str = "season") -> tuple[int, ...]:
    """Seasons that finished, judged from the data rather than the calendar.

    A season counts as complete when it contains a completed championship game,
    which is unambiguous and needs no clock. Where ``game_type`` is unavailable
    the fallback is a regular-season-sized volume of completed games, since the
    failure being guarded against -- treating a half-played season as a test
    period -- is a volume problem either way.
    """
    if frame.is_empty() or season_column not in frame.columns:
        return ()

    completed = frame
    if "completed" in frame.columns:
        completed = frame.filter(pl.col("completed"))
    if completed.is_empty():
        return ()

    if "game_type" in completed.columns:
        finals = completed.filter(pl.col("game_type").cast(pl.Utf8).str.to_uppercase() == "SB")
        if not finals.is_empty():
            found = finals.get_column(season_column).unique().to_list()
            return tuple(sorted(int(season) for season in found if season is not None))

    # Fallback: a regular-season-sized volume of distinct games. Counting rows
    # instead would be wrong at any grain finer than one row per game -- a
    # player-week frame reaches 200 rows in the first weekend of a season, which
    # would let an in-progress season be chosen as the frozen test period.
    counter = pl.col("game_id").n_unique() if "game_id" in completed.columns else pl.len()
    counts = completed.group_by(season_column).agg(counter.alias("games"))
    kept = counts.filter(pl.col("games") >= MIN_COMPLETE_SEASON_GAMES)
    found = kept.get_column(season_column).to_list()
    return tuple(sorted(int(season) for season in found if season is not None))


def plan_splits(
    seasons: Sequence[int],
    *,
    test_seasons: int = DEFAULT_TEST_SEASONS,
    validation_seasons: int = DEFAULT_VALIDATION_SEASONS,
    min_train_seasons: int = MIN_TRAIN_SEASONS,
) -> SplitPlan:
    """Lay out train / validation / test and the walk-forward folds.

    Args:
        seasons: Complete seasons present in the pinned snapshot, any order.
        test_seasons: Most recent seasons to freeze as the final test period.
        validation_seasons: Seasons immediately before the test period, used for
            every model-selection decision.
        min_train_seasons: Seasons the first fold must be able to fit on.

    Returns:
        A plan whose folds expand: fold *k* trains on everything before
        validation season *k* and scores that season alone.

    Raises:
        ValueError: When there are not enough complete seasons to leave a
            training window, a validation window and a test window. Shrinking
            one of them silently would produce a plan whose metrics cannot
            support the conclusions drawn from them.
    """
    ordered = tuple(sorted({int(season) for season in seasons}))
    required = min_train_seasons + validation_seasons + test_seasons
    if len(ordered) < required:
        raise ValueError(
            f"Need at least {required} complete seasons to plan a split "
            f"({min_train_seasons} train + {validation_seasons} validation + "
            f"{test_seasons} test); got {len(ordered)}."
        )

    test = ordered[-test_seasons:] if test_seasons else ()
    remaining = ordered[: len(ordered) - test_seasons] if test_seasons else ordered
    validation = remaining[-validation_seasons:] if validation_seasons else ()
    train = remaining[: len(remaining) - validation_seasons] if validation_seasons else remaining

    folds = tuple(
        TimeFold(
            name=f"validation_{season}",
            train_seasons=tuple(s for s in ordered if s < season),
            eval_seasons=(season,),
        )
        for season in validation
    )

    rationale = (
        f"Seasons {ordered[0]}-{ordered[-1]} are complete in this snapshot. The latest "
        f"{test_seasons} ({', '.join(str(s) for s in test)}) are frozen as the final test "
        f"period and scored once. The {validation_seasons} before them "
        f"({', '.join(str(s) for s in validation)}) are walk-forward validation folds, where "
        "every feature, hyperparameter and promotion decision is made. The test period is "
        "chosen from the snapshot rather than hard-coded, so re-running against newer data "
        "moves it forward instead of testing on seasons that have become training-era."
    )
    return SplitPlan(
        train_seasons=train,
        validation_seasons=validation,
        test_seasons=test,
        folds=folds,
        rationale=rationale,
    )
