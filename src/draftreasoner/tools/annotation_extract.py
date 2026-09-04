"""标注提取 — extract dimensions/datums/GD&T/roughness via a vision model into structured annotations."""

from __future__ import annotations

from typing import Any

from draftreasoner.prompts.agent import ANNOTATION_PROMPT
from draftreasoner.tools.base import Tool, ToolResult
from draftreasoner.tools.registry import register
from draftreasoner.tools.vision import call_json, claims_from, has_provider


@register
class AnnotationExtract(Tool):
    name = "AnnotationExtract"
    description = (
        "Extract structured drafting annotations (dimensions, datums, geometric tolerance, "
        "roughness, chamfer, limit deviation) and bind them to referenced features."
    )

    def run(
        self,
        image_path: str = "",
        bbox: tuple[int, int, int, int] | None = None,
        provider: Any = None,
        question: str = "",
        **_kwargs: Any,
    ) -> ToolResult:
        if not has_provider(provider) or not image_path:
            return self.ok({"annotations": [], "note": "no provider"}, confidence=0.0, evidence=[])

        data = call_json(provider, ANNOTATION_PROMPT, image_path, bbox)
        annotations = data.get("annotations") if isinstance(data, dict) else []
        if not isinstance(annotations, list):
            annotations = []
        claims = claims_from(annotations)
        return self.ok({"annotations": annotations}, confidence=0.8, evidence=claims)
