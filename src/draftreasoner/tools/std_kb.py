"""标准库 — GB/T-style standards lookup (targets CJ and part of DA)."""

from __future__ import annotations

from typing import Any

from draftreasoner.tools.base import Tool, ToolResult
from draftreasoner.tools.registry import register


# Minimal, extensible standards table. Replace/extend for real GB/T lookups.
_STANDARDS: dict[str, dict[str, str]] = {
    "unnoted_tolerance": {"std": "GB/T 1804-2000 m", "note": "未注线性尺寸公差 中等级(m)"},
    "unnoted_geometric": {"std": "GB/T 1184-1996 K", "note": "未注几何公差 精密级(K)"},
    "chamfer": {"std": "C0.5/C1", "note": "未注倒角按标准默认"},
    "roughness": {"std": "Ra", "note": "表面粗糙度由 Ra 值标注"},
}


@register
class StdKB(Tool):
    name = "StdKB"
    description = "Look up a drafting/standards convention by rule name to assist compliance checks."

    def run(self, rule: str = "", **_kwargs: Any) -> ToolResult:
        entry = _STANDARDS.get(rule)
        if not entry:
            return self.fail(f"unknown standard rule: {rule!r}")
        return self.ok(entry, confidence=0.9, evidence=[{"source": self.name, "claim": f"{rule}: {entry['std']}"}])
