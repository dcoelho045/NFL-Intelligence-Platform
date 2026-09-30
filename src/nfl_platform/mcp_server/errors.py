"""Domain exception to MCP error translation.

MCP has one channel for a failed tool call: ``is_error=True`` with text content.
There is no structured error object, so the structured detail this project
already produces -- code, message, recovery hint, context -- is rendered into
that text rather than thrown away. A model reading the result still learns which
player ids to choose from, and an application can still match on the code.

Two things are kept strictly apart here:

*A failure* is an exception. It becomes a ``ToolError``.

*An absence* is not. No reported injuries, no search hits, no matching passages
are all successful results carrying an empty list, and they never reach this
module. Collapsing the two is how "we could not check" becomes "there is
nothing", which is the most damaging mistake this server could make.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from mcp.server.mcpserver.exceptions import ResourceError, ToolError

from nfl_platform.config.logging import get_logger
from nfl_platform.tool_errors import ToolErrorDetail, describe_exception

logger = get_logger(__name__)

# Codes that mean "this capability is switched off", as distinct from "this
# capability ran and found nothing". Callers must never read them as evidence.
UNAVAILABLE_CODES = frozenset(
    {
        "web_search_not_configured",
        "web_search_error",
        "web_search_budget_exhausted",
        "knowledge_base_unavailable",
        "data_source_unavailable",
    }
)

_UNAVAILABLE_NOTICE = (
    "This is the state of the capability, not a finding about the world: "
    "it is not evidence that no such information exists."
)


def render_detail(detail: ToolErrorDetail) -> str:
    """Render a structured failure as the single string MCP allows.

    The leading ``code: message`` keeps it greppable; the JSON tail keeps the
    candidate lists and valid-value sets machine-readable.
    """
    payload: dict[str, Any] = {
        "code": detail.code,
        "message": detail.message,
        "recovery": detail.recovery,
        "retryable": detail.retryable,
    }
    if detail.code in UNAVAILABLE_CODES:
        payload["notice"] = _UNAVAILABLE_NOTICE
    if detail.context:
        payload["context"] = detail.context
    return f"{detail.code}: {detail.message} {json.dumps(payload, default=str)}"


def to_tool_error(exc: Exception) -> ToolError:
    """Translate any exception into the error an MCP client will receive."""
    detail = describe_exception(exc)
    if detail.code in ("tool_error", "internal_error"):
        logger.exception("MCP tool failed unexpectedly")
    return ToolError(render_detail(detail))


@contextmanager
def tool_errors() -> Iterator[None]:
    """Run a service call, translating domain failures into MCP tool errors."""
    try:
        yield
    except ToolError:
        raise
    except Exception as exc:
        raise to_tool_error(exc) from exc


@contextmanager
def resource_errors() -> Iterator[None]:
    """The resource-side equivalent; resources report through ``ResourceError``."""
    try:
        yield
    except ResourceError:
        raise
    except Exception as exc:
        detail = describe_exception(exc)
        if detail.code in ("tool_error", "internal_error"):
            logger.exception("MCP resource failed unexpectedly")
        raise ResourceError(render_detail(detail)) from exc


def unavailable(code: str, message: str, *, recovery: str) -> ToolError:
    """A capability that is switched off, reported without ambiguity."""
    return ToolError(
        render_detail(ToolErrorDetail(code, message, recovery=recovery, retryable=False))
    )
