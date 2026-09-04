"""视图对齐 — identify views and section/cross-view correspondence via a vision model (PM)."""

from __future__ import annotations

from typing import Any

from draftreasoner.prompts.agent import VIEW_PROMPT
from draftreasoner.tools.base import Tool, ToolResult
from draftreasoner.tools.registry import register
from draftreasoner.tools.vision import call_json, has_provider


@register
class ViewAlign(Tool):
    name = "ViewAlign"
    description = "Identify views (front/top/side/section/local), and map features or section lines across views."

    def run(
        self,
        image_path: str = "",
        bbox: tuple[int, int, int, int] | None = None,
        provider: Any = None,
        question: str = "",
        **_kwargs: Any,
    ) -> ToolResult:
        if not has_provider(provider) or not image_path:
            return self.ok({"views": [], "section_map": [], "note": "no provider"}, confidence=0.0, evidence=[])
        data = call_json(provider, VIEW_PROMPT, image_path, bbox)
        views = data.get("views") if isinstance(data, dict) else []
        section_map = data.get("section_map") if isinstance(data, dict) else []
        claims = []
        for v in views[:10]:
            name = v.get("name") if isinstance(v, dict) else str(v)
            claims.append({"source": "ViewAlign", "claim": f"view={name}", "confidence": 0.75})
        for s in section_map[:10]:
            sec = s.get("section") if isinstance(s, dict) else str(s)
            loc = s.get("location", "") if isinstance(s, dict) else ""
            claims.append({"source": "ViewAlign", "claim": f"section={sec} in {loc}", "confidence": 0.75})
        return self.ok({"views": views, "section_map": section_map}, confidence=0.75, evidence=claims)
