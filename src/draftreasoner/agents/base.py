"""Unit-agent abstraction. A unit agent owns a slice of the work and returns an
`AgentResult`. The supervisor composes unit agents into a plan."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from draftreasoner.core.config import Settings
from draftreasoner.core.state import AgentHandoff, AgentState
from draftreasoner.providers.vlm import BaseProvider
from draftreasoner.tools.base import Tool


@dataclass
class AgentContext:
    """Shared runtime handed to every unit agent (settings + backend + tools)."""

    settings: Settings
    provider: BaseProvider
    tools: dict[str, Tool]


@dataclass
class AgentResult:
    ok: bool
    answer: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    handoffs: list[AgentHandoff] = field(default_factory=list)
    final_output: str = ""


class BaseAgent(ABC):
    """Implement `run(state, ctx) -> AgentResult`. Optionally gate with `can_handle`."""

    name: str = ""
    description: str = ""

    def can_handle(self, subcategory: str) -> bool:
        return True

    @abstractmethod
    def run(self, state: AgentState, ctx: AgentContext, **kwargs: Any) -> AgentResult: ...
