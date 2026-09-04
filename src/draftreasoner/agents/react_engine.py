# -*- coding: utf-8 -*-
"""ReAct loop built on LangGraph StateGraph: agent -> tools -> (back to agent | end).

The LLM decides whether to call a tool and which one; ToolNode executes the chosen
tool(s) and feeds the observation back, until the model produces a final answer.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from draftreasoner.agents.base import AgentContext, AgentResult
from draftreasoner.core.state import AgentState
from draftreasoner.core.media import image_data_uri
from draftreasoner.prompts.agent import ANSWER_TEMPLATE, REACT_SYSTEM
from draftreasoner.providers.vlm import NullProvider

MAX_TOOL_ROUNDS = 5


def _last_text(messages: list[BaseMessage]) -> str:
    """Most recent non-empty assistant text (avoids empty answer when force-ended)."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    return ""


class _ReactState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_graph(llm: Any, tools: list[Any], max_rounds: int = MAX_TOOL_ROUNDS):
    graph = StateGraph(_ReactState)
    graph.add_node("agent", lambda s: {"messages": [llm.invoke(s["messages"])]})
    graph.add_node("tools", ToolNode(tools))

    def _route(s: _ReactState) -> str:
        last = s["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        if not tool_calls:
            return END
        if len([m for m in s["messages"] if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)]) >= max_rounds:
            return END
        return "tools"

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_react(state: AgentState, ctx: AgentContext) -> AgentResult:
    try:
        from draftreasoner.agents.react_tools import wrap_react_tools
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ReAct backend requires `uv sync` (langchain/langgraph).") from exc

    if isinstance(ctx.provider, NullProvider):
        raise RuntimeError("ReAct backend requires a real LLM provider (DR_API_KEY/DR_MODEL).")

    llm = ctx.provider.llm
    tools = wrap_react_tools(ctx.tools, state.image_path, ctx.provider, state.question)
    graph = _build_graph(llm.bind_tools(tools), tools)

    content = [
        {"type": "text", "text": f"Question: {state.question}\nAnswer in the question's language, final with <answer>...</answer>."}
    ]
    if state.image_path:
        content.append({"type": "image_url", "image_url": {"url": image_data_uri(state.image_path)}})

    result = graph.invoke({"messages": [SystemMessage(content=REACT_SYSTEM), HumanMessage(content=content)]})
    answer = _last_text(result["messages"])

    state.answer = answer
    state.reasoning = "react-loop (langgraph)"
    state.final_output = ANSWER_TEMPLATE.format(reasoning=state.reasoning, answer=answer)
    return AgentResult(ok=True, answer=answer, reasoning=state.reasoning, confidence=0.0, final_output=state.final_output)
