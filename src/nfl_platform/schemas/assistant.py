"""DTOs for the assistant endpoint.

These live in ``schemas/`` with the rest of the contract so the HTTP layer keeps
its rule of owning no types of its own. The run statistics are included in the
response on purpose: an evidence-backed answer is only checkable if the caller
can see which tools produced it and whether any number went untraced.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from nfl_platform.schemas.common import BaseDTO
from nfl_platform.schemas.evidence import SourceRef

MAX_QUESTION_LENGTH = 1000
MAX_HISTORY_TURNS = 20


class AssistantTurn(BaseDTO):
    """One prior message in a multi-turn conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class AssistantAskRequest(BaseDTO):
    """A question for the analyst assistant."""

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="A natural-language NFL question.",
    )
    history: list[AssistantTurn] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_TURNS,
        description="Earlier turns. The assistant holds no server-side session state.",
    )


class EvidenceItem(BaseDTO):
    """One tool call that contributed to the answer.

    Derived from the run trace rather than written by the model, so it records
    what was actually executed.
    """

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class RetrievalStatsDTO(BaseDTO):
    """What retrieval cost this run, in units the token counters do not cover."""

    web_searches: int = 0
    web_cache_hits: int = 0
    web_results: int = 0
    web_results_dropped: int = Field(
        default=0, description="Results removed by domain policy or injection filtering."
    )
    knowledge_searches: int = 0
    knowledge_hits: int = 0


class AssistantRunStats(BaseDTO):
    """Latency, token, and cost information for one answered question."""

    model: str
    prompt_version: str
    iterations: int
    stop_reason: str
    duration_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None = None
    tools_used: list[str] = Field(default_factory=list)
    ungrounded_numbers: list[str] = Field(
        default_factory=list,
        description=(
            "Numbers in the answer that could not be traced to any source. "
            "Reported for inspection; a non-empty list warrants scrutiny."
        ),
    )
    numbers_from_unstructured_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Numbers traceable only to retrieved web or knowledge-base text, not to the "
            "deterministic analytics layer. These should be attributed to their source "
            "rather than stated as retrieved statistics."
        ),
    )
    fabricated_sources: list[str] = Field(
        default_factory=list,
        description="URLs or domains cited in the answer that no tool actually returned.",
    )
    retrieval: RetrievalStatsDTO = Field(default_factory=RetrievalStatsDTO)


class AssistantAnswerResponse(BaseDTO):
    """The assistant's answer and the record backing it."""

    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    sources: list[SourceRef] = Field(
        default_factory=list,
        description=(
            "Every source that contributed, labelled by kind. Derived from the run "
            "trace, so the assistant cannot cite something it did not retrieve."
        ),
    )
    stats: AssistantRunStats


class ToolDescription(BaseDTO):
    """One tool the assistant can call."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCatalogue(BaseDTO):
    """Every tool available to the assistant."""

    count: int
    tools: list[ToolDescription]
