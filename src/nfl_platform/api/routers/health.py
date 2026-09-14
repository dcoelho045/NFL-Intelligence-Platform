"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from nfl_platform import __version__
from nfl_platform.api.dependencies import ContainerDep
from nfl_platform.exceptions import NflPlatformError
from nfl_platform.schemas.common import HealthStatus, ReadinessStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus, summary="Liveness probe")
def health() -> HealthStatus:
    """Confirm the process is running. Performs no data access, so it never fails slowly."""
    return HealthStatus(status="ok", version=__version__)


@router.get("/health/ready", response_model=ReadinessStatus, summary="Readiness probe")
def readiness(container: ContainerDep) -> ReadinessStatus:
    """Confirm the data layer answers.

    Reports ``degraded`` with HTTP 200 rather than raising: an orchestrator wants
    a readiness signal, not an exception trace.
    """
    try:
        seasons = container.meta.seasons()
    except NflPlatformError as error:
        return ReadinessStatus(
            status="degraded",
            version=__version__,
            data_layer_available=False,
            detail=str(error),
        )
    return ReadinessStatus(
        status="ready",
        version=__version__,
        data_layer_available=True,
        current_season=container.loader.current_season,
        detail=None if seasons else "No seasons currently return data.",
    )
