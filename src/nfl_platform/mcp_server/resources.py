"""Resources: data an application loads, as opposed to actions a model takes.

That split is the whole reason this file is short. A player's statistics over a
window is a parameterised query with identity resolution and season handling --
that is a tool. A curated document, the list of valid teams, or whether the
knowledge index has been built is something an application fetches to build
context, and that is a resource.

There is deliberately no ``player://`` or ``game://`` space. Those would be
high-cardinality, unlistable, and would duplicate tools that already answer the
same questions with validated arguments.
"""

from __future__ import annotations

from mcp.server import MCPServer

from nfl_platform import __version__
from nfl_platform.config.settings import Settings
from nfl_platform.exceptions import NflPlatformError
from nfl_platform.mcp_server.errors import resource_errors
from nfl_platform.schemas.evidence import KnowledgeCorpusIndex, KnowledgeDocument
from nfl_platform.schemas.mcp import ReferenceData, ServerStatus
from nfl_platform.schemas.prediction import ModelRegistryStatus
from nfl_platform.services.container import ServiceContainer

__all__ = ["register_resources"]


def register_resources(mcp: MCPServer, container: ServiceContainer, settings: Settings) -> None:
    """Register every resource this server exposes."""

    @mcp.resource(
        "nfl://status",
        title="Server status",
        mime_type="application/json",
    )
    def status() -> ServerStatus:
        """Which capabilities are usable on this deployment, and from what data."""
        documents, chunks, model, built_at = container.knowledge.status()
        try:
            current_season: int | None = container.loader.current_season
            data_layer_available = True
        except NflPlatformError:
            current_season, data_layer_available = None, False

        web_enabled = settings.mcp_enable_web_search
        models = container.predictions.model_status()
        return ServerStatus(
            server_name=settings.mcp_server_name,
            version=__version__,
            data_layer_available=data_layer_available,
            current_season=current_season,
            web_search_enabled=web_enabled,
            web_search_available=web_enabled and container.research.available,
            web_search_provider=container.research.provider_name,
            knowledge_base_available=container.knowledge.available,
            knowledge_base_documents=documents,
            knowledge_base_chunks=chunks,
            embedding_model=model,
            knowledge_base_built_at=built_at,
            predictions_enabled=models.enabled,
            prediction_models_available=models.available,
            prediction_models=[entry.model_id for entry in models.models],
        )

    @mcp.resource(
        "nfl://models",
        title="Installed prediction models",
        mime_type="application/json",
    )
    def models() -> ModelRegistryStatus:
        """Which prediction models are installed, and what each was trained on.

        Read this before quoting a prediction. It carries the training seasons,
        the frozen test period, the validation score and whether probabilities
        were calibrated -- the context that turns a probability into a claim a
        reader can judge.

        Never fails when nothing is installed: ``available`` reports that, which
        is the question this resource exists to answer.
        """
        return container.predictions.model_status()

    @mcp.resource(
        "nfl://reference/seasons",
        title="Seasons with data",
        mime_type="application/json",
    )
    def seasons() -> ReferenceData:
        """The seasons that currently return data, and which one 'this season' means."""
        with resource_errors():
            return ReferenceData(
                current_season=container.meta.current_season(),
                seasons=container.meta.seasons(),
                teams=None,
            )

    @mcp.resource(
        "nfl://reference/teams/{season}",
        title="Valid team abbreviations",
        mime_type="application/json",
    )
    def teams(season: str) -> ReferenceData:
        """Team abbreviations that appear in one season.

        Derived from the schedule rather than a hard-coded list, because relocated
        franchises change abbreviation by season (OAK before LV, SD before LAC).
        """
        with resource_errors():
            parsed = _season_number(season)
            return ReferenceData(
                current_season=container.meta.current_season(),
                seasons=[parsed],
                teams=container.meta.teams(parsed),
            )

    @mcp.resource(
        "nfl://knowledge/documents",
        title="Curated knowledge documents",
        mime_type="application/json",
    )
    def knowledge_documents() -> KnowledgeCorpusIndex:
        """Every curated reference document, listed without its body."""
        with resource_errors():
            return container.knowledge.documents()

    @mcp.resource(
        "nfl://knowledge/documents/{document_id}",
        title="Curated knowledge document",
        mime_type="application/json",
    )
    def knowledge_document(document_id: str) -> KnowledgeDocument:
        """Read one curated document in full, verbatim.

        Stable reference material, not current information. The 'updated' field says
        when it was last reviewed.
        """
        with resource_errors():
            return container.knowledge.document(document_id)


def _season_number(season: str) -> int:
    """Parse a season out of a URI segment.

    URI template values arrive as strings, so this is the boundary where the
    domain's season validation gets to run.
    """
    try:
        return int(season)
    except ValueError:
        raise ValueError(f"Season must be a four-digit year, not {season!r}.") from None
