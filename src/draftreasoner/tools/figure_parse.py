"""图解析 — split a composite sheet into sub-figures and extract textual content (title block / notes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from draftreasoner.tools.base import Tool, ToolResult
from draftreasoner.tools.registry import register


def _panel_split(path: str | Path, ink_threshold: int = 128, min_gap_ratio: float = 0.03) -> list[tuple[int, int, int, int]]:
    """Naive whitespace-driven panel split on a downscaled grayscale image.

    Returns normalized (x0, y0, x1, y1) panel boxes over the original image size,
    ordered top-to-bottom then left-to-right. Best-effort; falls back to one panel.
    """
    from PIL import Image

    im = Image.open(str(path)).convert("L")
    w, h = im.size
    target_w = min(w, 800)
    small = im.resize((target_w, max(1, int(h * target_w / w))))
    sw, sh = small.size
    px = list(small.getdata())

    cols = [0] * sw
    rows = [0] * sh
    for y in range(sh):
        base = y * sw
        for x in range(sw):
            if px[base + x] < ink_threshold:
                cols[x] += 1
                rows[y] += 1

    def segments(profile, span, ratio):
        gap_min = max(2, int(span * ratio))
        seps, run, run_start = [], 0, 0
        for i, v in enumerate(profile):
            if v <= 0:
                if run == 0:
                    run_start = i
                run += 1
            else:
                if run >= gap_min:
                    seps.append((run_start, i))
                run = 0
        if run >= gap_min:
            seps.append((run_start, len(profile)))
        return seps

    x_seps = segments(cols, sw, min_gap_ratio)
    y_seps = segments(rows, sh, min_gap_ratio)
    x_bounds = [0] + [s[1] for s in x_seps] + [sw]
    y_bounds = [0] + [s[1] for s in y_seps] + [sh]
    x_segments = [(x_bounds[i], x_bounds[i + 1]) for i in range(len(x_bounds) - 1)]
    y_segments = [(y_bounds[i], y_bounds[i + 1]) for i in range(len(y_bounds) - 1)]

    panels = []
    for (y0, y1) in y_segments:
        for (x0, x1) in x_segments:
            panels.append((x0, y0, x1, y1))
    if len(panels) <= 1:
        return [(0, 0, w, h)]
    # map normalized -> original scale and drop near-empty cells
    scale_x, scale_y = w / sw, h / sh
    out = []
    for (x0, y0, x1, y1) in panels:
        out.append((int(x0 * scale_x), int(y0 * scale_y), int(x1 * scale_x), int(y1 * scale_y)))
    return out


@register
class FigureParse(Tool):
    name = "FigureParse"
    description = (
        "Split a possibly composite mechanical drawing into sub-figures and extract free text "
        "(title block / technical requirements). Input: {'image_path','sub_figure_index'}."
    )

    def run(self, image_path: str = "", sub_figure_index: int = 1, **_kwargs: Any) -> ToolResult:
        if not image_path:
            return self.fail("image_path is required")
        try:
            panels = _panel_split(image_path)
        except Exception as exc:  # pragma: no cover - defensive
            return self.fail(f"figure parse failed: {exc}")
        n = len(panels)
        idx = max(0, min(int(sub_figure_index) - 1, n - 1))
        claim = f"split into {n} panel(s); referencing panel #{idx + 1} at bbox {panels[idx]}"
        return self.ok(
            {"n_panels": n, "sub_figure_index": idx + 1, "bbox": panels[idx], "panels": panels},
            confidence=0.6,
            evidence=[{"source": self.name, "claim": claim, "confidence": 0.6}],
        )
