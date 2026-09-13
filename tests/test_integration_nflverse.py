"""Network-dependent tests against live nflverse endpoints.

Deselected by default. Run explicitly with:

    pytest -m integration

These guard against upstream schema drift, which unit tests against frozen
fixtures cannot detect.
"""

from __future__ import annotations

import pytest

from nfl_platform.data.ingestion import Dataset, NflDataLoader, get_dataset_spec

pytestmark = pytest.mark.integration

LIVE_SEASON = 2024


@pytest.fixture(scope="module")
def live_loader() -> NflDataLoader:
    return NflDataLoader()


@pytest.mark.parametrize("dataset", [d for d in Dataset if d is not Dataset.PLAY_BY_PLAY])
def test_live_dataset_satisfies_its_schema_contract(live_loader: NflDataLoader, dataset: Dataset):
    spec = get_dataset_spec(dataset)
    frame = live_loader.load(dataset, LIVE_SEASON if spec.season_aware else None)
    assert frame.height > 0
    assert spec.required_columns <= set(frame.columns)


def test_live_play_by_play_projection(live_loader: NflDataLoader):
    frame = live_loader.load_play_by_play(LIVE_SEASON, columns=["yards_gained"])
    assert frame.height > 0
    assert "yards_gained" in frame.columns


def test_current_season_is_plausible(live_loader: NflDataLoader):
    assert 2024 <= live_loader.current_season <= 2100
