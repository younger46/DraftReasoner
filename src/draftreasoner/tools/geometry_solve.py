"""几何求解 — deterministic dimension-chain solver (targets GC)."""

from __future__ import annotations

import re
from typing import Any

from draftreasoner.tools.base import Tool, ToolResult
from draftreasoner.tools.registry import register


_NUM = re.compile(r"-?(\d+(?:\.\d+)?)")


def parse_number(text: str) -> float | None:
    m = _NUM.search(text)
    return float(m.group(1)) if m else None


def parse_gc_formula(question: str) -> str | None:
    q = question.lower()
    if ("之差的一半" in question or "的一半" in question and "之差" in question) or (
        "half" in q and ("difference" in q or "outer" in q or "inner" in q)
    ):
        return "difference_half"
    if "之差" in question or "difference" in q:
        return "difference"
    if "之和" in question or "sum" in q or "total" in q:
        return "sum"
    return None


@register
class DimChainSolver(Tool):
    name = "GeometrySolve"
    description = (
        "Solve a target dimension from a dimensional chain by adding/subtracting signed "
        "segments. Input: {'target': label, 'chain': [{'label','value','sign'}...]}."
    )

    def run(
        self,
        target: str = "",
        chain: list[dict[str, Any]] | None = None,
        question: str = "",
        image_path: str = "",
        bbox: tuple[int, int, int, int] | None = None,
        **_kwargs: Any,
    ) -> ToolResult:
        chain = chain or []
        if not chain and question and image_path:
            return self._solve_from_ocr(question, image_path, bbox)

        total = 0.0
        details: list[str] = []
        for seg in chain:
            value = float(seg.get("value", 0))
            sign = 1.0 if str(seg.get("sign", "+")).strip().startswith("+") else -1.0
            total += sign * value
            details.append(f"{seg.get('label','?')}:{' +' if sign>0 else ' -'} {value}")
        claim = f"{target or 'target'} = {total}"
        return self.ok(
            {"target": target, "value": total, "chain_terms": details},
            confidence=0.99,
            evidence=[{"source": self.name, "claim": claim, "confidence": 0.99}],
        )

    def _solve_from_ocr(
        self, question: str, image_path: str, bbox: tuple[int, int, int, int] | None
    ) -> ToolResult:
        formula = parse_gc_formula(question)
        if formula is None:
            return self.ok({"note": "unrecognized formula; supply a chain"}, confidence=0.0, evidence=[])
        try:
            from draftreasoner.tools.ocr import read_dimensions

            dims = read_dimensions(image_path, None)  # use the drawing-area crop, ignore coarse panel bbox
        except Exception as exc:  # pragma: no cover - OCR/engine missing
            return self.fail(f"dimension read failed: {exc}")
        values = [d["value"] for d in dims if d["value"] >= 10]
        if len(values) < 2:
            return self.ok({"note": f"found {len(values)} dim value(s); need 2"}, confidence=0.0, evidence=[])
        outer, inner = max(values), min(values)
        if formula == "difference_half":
            result = (outer - inner) / 2
            claim = f"({outer}-{inner})/2 = {result}"
        elif formula == "difference":
            result = outer - inner
            claim = f"{outer}-{inner} = {result}"
        else:  # sum
            result = outer + inner
            claim = f"{outer}+{inner} = {result}"
        return self.ok(
            {"value": result, "outer": outer, "inner": inner, "formula": formula},
            confidence=0.95,
            evidence=[{"source": self.name, "claim": claim, "confidence": 0.95}],
        )
