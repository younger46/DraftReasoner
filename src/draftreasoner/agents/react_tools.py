"""Wrap MechAgent tools as LangChain tools so the ReAct loop can invoke them."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from draftreasoner.tools.base import Tool, ToolResult

# tool names exposed to the ReAct LLM
REACT_ALLOWED = ["FigureParse", "AnnotationExtract", "ViewAlign", "GeometrySolve", "StdKB"]


def _run_tool(t: Tool, name: str, image_path: str | None, provider: Any, question: str, focus: str) -> ToolResult:
    if name == "FigureParse":
        return t.run(image_path=image_path, sub_figure_index=1)
    if name == "StdKB":
        return t.run(rule=(focus or "unnoted_tolerance"))
    if name == "GeometrySolve":
        return t.run(question=question, image_path=image_path, bbox=None)
    # AnnotationExtract / ViewAlign
    return t.run(image_path=image_path, bbox=None, provider=provider, question=question)


def wrap_react_tools(tool_map: dict[str, Tool], image_path: str | None, provider: Any, question: str) -> list[Any]:
    result = []
    for name in REACT_ALLOWED:
        t = tool_map.get(name)
        if t is None:
            continue

        def make_fn(t: Tool = t, name: str = name):
            def run(_focus: str = "") -> str:
                try:
                    r = _run_tool(t, name, image_path, provider, question, _focus)
                except Exception as exc:  # pragma: no cover - defensive
                    r = ToolResult(ok=False, error=str(exc))
                return json.dumps(
                    {"ok": r.ok, "data": r.data, "evidence": r.evidence, "error": r.error},
                    ensure_ascii=False,
                )

            return run

        fn = make_fn()
        fn.__name__ = name
        fn.__qualname__ = name
        fn.__doc__ = f"{t.description} Returns JSON {{ok, data, evidence, error}}."
        result.append(tool(fn))
    return result
