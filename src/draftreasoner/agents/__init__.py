"""Agent primitives: unit agents + registry.

The supervisor (draftreasoner.core.agent.MechAgent) runs a *plan* of unit agents.
Today that plan holds one `engineer` agent (single-agent behaviour); to go
multi-agent, register more unit agents here and extend the plan (see
`core.agent.PLANS`).
"""

from draftreasoner.agents.base import AgentContext, AgentResult, BaseAgent
from draftreasoner.agents.registry import agent_names, build_agents, get_agent, register_agent

__all__ = [
    "AgentContext",
    "AgentResult",
    "BaseAgent",
    "agent_names",
    "build_agents",
    "get_agent",
    "register_agent",
]
