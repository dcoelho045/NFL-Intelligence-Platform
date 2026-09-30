"""Run the NFL MCP server.

    python -m nfl_platform.mcp_server          # stdio, the transport hosts launch
    nfl-mcp                                    # the same thing, as a console script

stdio is the only implemented transport. It needs no port, no certificate and no
authentication story: the trust boundary is the local user who launched the
process, which is the model every editor-installed MCP server uses. An HTTP
transport would need real authentication before it deserved to exist, and that
is not this milestone.

One consequence is load-bearing: under stdio the protocol owns stdout. Logging
goes to stderr -- which ``configure_logging`` already guarantees -- and nothing
in this package may ``print``.
"""

from __future__ import annotations

import sys

from nfl_platform.config.logging import configure_logging, get_logger
from nfl_platform.config.settings import get_settings
from nfl_platform.mcp_server.server import build_mcp_server

logger = get_logger(__name__)


def main() -> int:
    """Serve the MCP server over stdio until the client disconnects."""
    settings = get_settings()
    configure_logging(settings)
    server = build_mcp_server(settings=settings)
    try:
        server.run(transport="stdio")
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logger.info("MCP server interrupted")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
