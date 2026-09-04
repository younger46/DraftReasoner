"""Runtime state objects for a single MechVQA query."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    """A grounded claim produced by a tool, with provenance + confidence."""

    source: str
    claim: str
    confidence: float
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "claim": self.claim,
            "confidence": self.confidence,
            "detail": self.detail,
        }


@dataclass
class AgentHandoff:
    """A message passed from one agent to another (or to the supervisor)."""

    from_agent: str
    to_agent: str
    instruction: str
    result: str = ""


@dataclass
class AgentState:
    """Everything the agent knows while answering one question."""

    question: str
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # route / plan decided by the planner
    route: str = ""
    evidence: list[Evidence] = field(default_factory=list)

    reasoning: str = ""
    answer: str = ""
    final_output: str = ""
    confidence: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)
    handoffs: list[AgentHandoff] = field(default_factory=list)
    subagent_results: list[dict[str, Any]] = field(default_factory=list)

    retries: int = 0
    max_retries: int = 2

    @property
    def subcategory(self) -> str:
        return str(self.metadata.get("subcategory") or "")

    @property
    def capability(self) -> str:
        return str(self.metadata.get("capability") or "")

    @property
    def language(self) -> str:
        lang = str(self.metadata.get("language") or "")
        return "en" if "en" in lang.lower() else "zh"

    def add_evidence(self, source: str, claim: str, confidence: float, detail: str = "") -> None:
        self.evidence.append(Evidence(source, claim, confidence, detail))

    def evidence_summary(self) -> str:
        return "\n".join(f"[{e.source}] {e.claim}" for e in self.evidence)
