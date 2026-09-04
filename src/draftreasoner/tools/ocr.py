"""Deterministic OCR-backed perception helpers (RapidOCR + OpenCV).

The vision model mis-reads tiny dimension labels; a real OCR on a tight crop of the
main drawing region is far more reliable. Used by GeometrySolve / AnnotationExtract
to feed exact dimension values into the chain solver.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

_ENGINE: Any = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINE = RapidOCR()
    return _ENGINE


def content_bbox(image_path: str, pad_ratio: float = 0.02) -> tuple[int, int, int, int] | None:
    """Find the largest interior connected component (the main view) as a crop box."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    h, w = bw.shape
    best = None
    best_area = 0
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue  # border frame / page edges
        if area > best_area:
            best_area = area
            best = (x, y, cw, ch)
    if best is None:
        return None
    x, y, cw, ch = best
    p = int(max(cw, ch) * pad_ratio)
    return (max(0, x - p), max(0, y - p), min(w, x + cw + p), min(h, y + ch + p))


def ocr(image_path: str, bbox: tuple[int, int, int, int] | None = None) -> list[tuple[str, tuple[int, int], float]]:
    path = image_path
    if bbox:
        im = Image.open(image_path).crop(tuple(int(v) for v in bbox))
        tmp = Path(tempfile.gettempdir()) / f"ocr_{Path(image_path).stem}.png"
        im.save(tmp)
        path = str(tmp)
    res, _ = _engine()(path)
    toks: list[tuple[str, tuple[int, int], float]] = []
    for box, txt, sc in res or []:
        xs = [q[0] for q in box]
        ys = [q[1] for q in box]
        toks.append((str(txt), (int(sum(xs) / 4), int(sum(ys) / 4)), float(sc)))
    return toks


def read_dimensions(
    image_path: str, bbox: tuple[int, int, int, int] | None = None
) -> list[dict[str, Any]]:
    """Return candidate diameter values (value, center, score) extracted by OCR."""
    if bbox is None:
        from PIL import Image as _Im

        w, h = _Im.open(image_path).size
        bbox = (int(0.05 * w), int(0.05 * h), int(0.95 * w), int(0.74 * h))
    box = bbox
    toks = ocr(image_path, box)
    out = []
    for txt, center, sc in toks:
        s = txt.strip()
        # diameter labels OCR as e.g. '061'/'051' (Phi -> leading 0), or explicit Phi/O
        if re.match(r"^[0OΦØDd]", s):
            digits = re.sub(r"[^0-9.]", "", s)
            if not digits:
                continue
            value = int(float(digits))
            if 1 < value < 1000:
                out.append({"value": value, "center": center, "score": sc, "raw": s})
    # keep the two largest (outer, inner) for ring parts
    out.sort(key=lambda d: -d["value"])
    return out
