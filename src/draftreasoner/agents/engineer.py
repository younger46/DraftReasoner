"""EngineerAgent — the ReAct worker: the LLM decides which tool to call (or when to answer)."""
from __future__ import annotations

from typing import Any

from draftreasoner.agents.base import AgentContext, AgentResult, BaseAgent
from draftreasoner.agents.registry import register_agent
from draftreasoner.core.state import AgentState


@register_agent
class EngineerAgent(BaseAgent):
    name = "engineer"
    description = "ReAct loop: LLM decides whether to call a tool and which one."

    def can_handle(self, subcategory: str) -> bool:
        return True

    def run(self, state: AgentState, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        from draftreasoner.agents.react_engine import run_react

        return run_react(state, ctx)
