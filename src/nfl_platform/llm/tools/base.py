"""The `Tool` contract.

A tool is deliberately thin: an argument model, a description, and one call into
a service. If a tool ever needs an `if` that changes a number, that logic belongs
in `services/` where it can be unit-tested without a model in the loop.

Argument schemas are generated from Pydantic rather than hand-written JSON. That
is the payoff of the Milestone 2 DTO discipline -- the schema the model sees and
the validation that runs cannot drift apart, because they are the same object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from nfl_platform.llm.results import ToolResult
from nfl_platform.services.container import ServiceContainer


class ToolArgs(BaseModel):
    """Base for tool argument models.

    ``extra="forbid"`` is the important setting: a hallucinated argument name is
    rejected loudly instead of being silently dropped, which is how a model
    learns it asked for something that does not exist.
    """

    model_config = ConfigDict(extra="forbid")


class Tool[ArgsT: ToolArgs](ABC):
    """One callable capability exposed to the model."""

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[ToolArgs]]

    @abstractmethod
    def invoke(self, container: ServiceContainer, args: ArgsT) -> ToolResult:
        """Execute against the deterministic services and wrap the result."""

    def json_schema(self) -> dict[str, Any]:
        """The provider-facing tool definition."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }
