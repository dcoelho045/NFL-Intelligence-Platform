"""Prompts: templates a person picks, not instructions a model receives unasked.

A prompt renders to text and nothing else. No model is called here and no data is
fetched; the rendered message tells the host's model which tools to call and in
what order, which is where this project's "numbers come from tools" rule reaches
a client we do not control.

The templates are Markdown files beside this module for the same reason the
analyst system prompts are: prompt text is content, it is reviewed as content,
and burying it in string literals makes both editing and diffing worse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field

from nfl_platform.schemas.fantasy import FantasyScoring

__all__ = ["load_template", "register_prompts"]

PROMPT_DIR = Path(__file__).parent

# Prompt arguments are free text from a person, and they are interpolated into a
# message. Bounding them keeps a pasted document out of the conversation.
MAX_ARGUMENT_CHARS = 80

NameArg = Annotated[str, Field(min_length=1, max_length=MAX_ARGUMENT_CHARS)]
TeamArg = Annotated[str, Field(min_length=2, max_length=MAX_ARGUMENT_CHARS)]
WeekArg = Annotated[
    str, Field(default="", max_length=16, description="Week number, or blank for the latest data.")
]


def load_template(name: str) -> str:
    """Read one prompt template by name."""
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def _when(week: str, season: str = "") -> str:
    """Render the optional time qualifier shared by every template."""
    parts = []
    if week.strip():
        parts.append(f"week {week.strip()}")
    if season.strip():
        parts.append(f"the {season.strip()} season")
    return f" in {' of '.join(parts)}" if parts else ""


def register_prompts(mcp: MCPServer) -> None:
    """Register every prompt this server exposes."""

    @mcp.prompt(title="Analyse a matchup")
    def analyze_matchup(
        team_a: TeamArg,
        team_b: TeamArg,
        week: WeekArg = "",
        season: Annotated[str, Field(default="", max_length=16)] = "",
    ) -> str:
        """Compare two teams using deterministic analytics, with sourced current context."""
        return load_template("analyze_matchup").format(
            team_a=team_a.strip(), team_b=team_b.strip(), when=_when(week, season)
        )

    @mcp.prompt(title="Fantasy start/sit")
    def fantasy_start_sit(
        player_a: NameArg,
        player_b: NameArg,
        scoring: Annotated[
            str, Field(default=FantasyScoring.PPR.value, max_length=16)
        ] = FantasyScoring.PPR.value,
        week: WeekArg = "",
    ) -> str:
        """Decide between two players over one identical, sourced window."""
        return load_template("fantasy_start_sit").format(
            player_a=player_a.strip(),
            player_b=player_b.strip(),
            scoring=scoring.strip() or FantasyScoring.PPR.value,
            when=_when(week),
        )

    @mcp.prompt(title="Player performance review")
    def player_performance_review(
        player: NameArg,
        season: Annotated[str, Field(default="", max_length=16)] = "",
    ) -> str:
        """Review what a player has produced, and what their usage explains about it."""
        return load_template("player_performance_review").format(
            player=player.strip(), when=_when("", season)
        )
