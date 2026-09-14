"""Reference-data endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from nfl_platform.api.dependencies import MetaDep, SeasonsQuery

router = APIRouter(prefix="/meta", tags=["meta"])


@router.get("/teams", response_model=list[str], summary="Valid team abbreviations")
def valid_teams(meta: MetaDep, seasons: SeasonsQuery = None) -> list[str]:
    """Teams present in the requested seasons.

    Derived from the data, so historical abbreviations such as `OAK` remain
    correct for the seasons in which they were used.
    """
    return meta.teams(seasons)


@router.get("/seasons", response_model=list[int], summary="Seasons with data")
def available_seasons(meta: MetaDep) -> list[int]:
    """Seasons that currently return schedule data, most recent first."""
    return meta.seasons()
