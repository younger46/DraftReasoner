"""Smoke tests for the ReAct-only MechAgent scaffold. Run with pytest or `python tests/test_smoke.py`."""
from __future__ import annotations

from draftreasoner.core.agent import MechAgent
from draftreasoner.core.config import Settings
from draftreasoner.evaluation.judge import extract_answer
from draftreasoner.tools.geometry_solve import DimChainSolver
from draftreasoner.tools.registry import build_tools
from draftreasoner.tools.std_kb import StdKB


def test_geometry_solver_chain():
    res = DimChainSolver().run(
        target="L", chain=[{"label": "a", "value": 10, "sign": "+"}, {"label": "b", "value": 2, "sign": "-"}]
    )
    assert res.ok and res.data["value"] == 8.0


def test_stdkb():
    res = StdKB().run(rule="unnoted_tolerance")
    assert res.ok and "GB/T 1804" in res.data["std"]


def test_registry_has_react_tools():
    names = {t.name for t in build_tools()}
    assert {"FigureParse", "AnnotationExtract", "ViewAlign", "GeometrySolve", "StdKB"} <= names
    assert "EvidenceCheck" not in names


def test_judge_extract_answer():
    text = "<think>loop</think>\n<answer>5.0 mm</answer>"
    assert extract_answer(text) == "5.0 mm"
    assert extract_answer("plain answer") == "plain answer"


def test_react_tool_adapter():
    from draftreasoner.agents.react_tools import wrap_react_tools

    tools = {t.name: t for t in build_tools()}
    react = wrap_react_tools(tools, image_path=None, provider=None, question="q")
    names = {t.name for t in react}
    assert {"FigureParse", "AnnotationExtract", "ViewAlign", "GeometrySolve", "StdKB"} <= names
    assert "EvidenceCheck" not in names


def test_react_requires_provider():
    settings = Settings(api_key="", verbose=False, backend="react")
    agent = MechAgent(settings)
    try:
        agent.answer("q", image_path=None, metadata={"subcategory": "Geometric Calculation"})
        assert False, "expected RuntimeError when no LLM provider is configured"
    except RuntimeError:
        pass


if __name__ == "__main__":
    test_geometry_solver_chain()
    test_stdkb()
    test_registry_has_react_tools()
    test_judge_extract_answer()
    test_react_tool_adapter()
    test_react_requires_provider()
    print("all smoke tests passed")
