"""MechAgent — a supervisor that runs a *plan* of unit agents.

Single-agent default: the plan is `["engineer"]` (one tool-augmented unit that does
perceive -> reason -> verify). To go multi-agent, register more unit agents in
`draftreasoner.agents.*` and extend `PLANS`/`DEFAULT_PLAN` below with more names;
the supervisor runs them in order, collects handoffs, and merges their answers.
"""

from __future__ import annotations

from typing import Any

from draftreasoner.agents.base import AgentContext
from draftreasoner.agents.registry import build_agents, get_agent
from draftreasoner.core.config import Settings, resolve_image_path
from draftreasoner.core.state import AgentHandoff, AgentState
from draftreasoner.prompts.agent import ANSWER_TEMPLATE
from draftreasoner.providers.vlm import NullProvider, create_provider
from draftreasoner.tools.registry import build_tools


# subcategory -> ordered list of unit-agent names. Default single-agent plan is ["engineer"].
# Example multi-agent plan (uncomment to route a capability through several specialists):
#   "Geometric Calculation": ["vision", "geometry", "engineer"],
PLANS: dict[str, list[str]] = {}
DEFAULT_PLAN: list[str] = ["engineer"]


class MechAgent:
    """Top-level orchestrator. `run(record)` / `answer(...)` mirror the old single-agent API."""

    def __init__(self, settings: Settings | None = None, plans: dict[str, list[str]] | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.plans = plans if plans is not None else PLANS
        self.provider = create_provider(self.settings)
        self.tools = {t.name: t for t in build_tools()}
        self.agents = build_agents()  # registered unit agents (plugins)

    # ---- planning / dispatch (the single -> multi seam) ----
    def plan_for(self, state: AgentState) -> list[str]:
        return list(self.plans.get(state.subcategory, DEFAULT_PLAN))

    def _context(self) -> AgentContext:
        return AgentContext(settings=self.settings, provider=self.provider, tools=self.tools)

    def answer(self, question: str, image_path: str | None = None, metadata: dict[str, Any] | None = None) -> str:
        state = AgentState(
            question=question,
            image_path=image_path,
            metadata=metadata or {},
            max_retries=self.settings.max_retries,
        )
        results = []
        for name in self.plan_for(state):
            agent = get_agent(name)
            res = agent.run(state, self._context())
            state.subagent_results.append({"agent": name, "answer": res.answer, "confidence": res.confidence})
            state.handoffs.extend(res.handoffs)
            if not res.ok:
                state.handoffs.append(AgentHandoff(name, "supervisor", "agent failed", res.answer))
            results.append(res)
        state.final_output = self._merge(results, state)
        return state.final_output

    def run(self, record: dict[str, Any]) -> str:
        image_rel = (record.get("images") or [None])[0]
        image_path = str(resolve_image_path(self.settings.images_dir, image_rel)) if image_rel else None
        question = record["messages"][0]["content"]
        metadata = record.get("metadata") or {}
        return self.answer(question, image_path, metadata)

    # ---- result merging (multi-agent synthesis) ----
    def _merge(self, results: list, state: AgentState) -> str:
        if not results:
            return ANSWER_TEMPLATE.format(reasoning="no agent produced output", answer="")
        if len(results) == 1:
            return results[0].final_output
        answers = [r.answer for r in results if r.answer]
        if isinstance(self.provider, NullProvider):
            merged_answer = " ; ".join(answers) or "(no answers)"
            reasoning = "merged from sub-agents"
        else:
            system = "You are an expert mechanical engineer. Merge the sub-agent answers into one final answer."
            user = (
                f"Question: {state.question}\n"
                "Sub-agent answers:\n" + "\n".join(f"- {a}" for a in answers) + "\nGive the final answer."
            )
            merged_answer = self.provider.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                image_path=state.image_path,
            )
            reasoning = "merged from sub-agents by supervisor"
        return ANSWER_TEMPLATE.format(reasoning=reasoning, answer=merged_answer)
