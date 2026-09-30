"""Domain exception to tool-error mapping.

The mapping moved to :mod:`nfl_platform.tool_errors` in Milestone 5 so the MCP
adapter could share one error taxonomy without importing a sibling adapter. It
is re-exported here because this is where the LLM layer expects to find it.
"""

from __future__ import annotations

from nfl_platform.tool_errors import MAX_CANDIDATES, ToolErrorDetail, describe_exception

__all__ = ["MAX_CANDIDATES", "ToolErrorDetail", "describe_exception"]
