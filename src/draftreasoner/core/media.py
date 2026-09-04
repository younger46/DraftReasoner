"""Shared image helpers: base64 data URI + region cropping."""
from __future__ import annotations

import base64
import mimetypes
import tempfile
from pathlib import Path

from PIL import Image


def image_data_uri(path: str) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def crop_to(path: str, bbox: tuple[int, int, int, int]) -> str:
    """Crop an image to a bbox (x0,y0,x1,y1) and return a temp file path."""
    try:
        im = Image.open(path)
        x0, y0, x1, y1 = [int(v) for v in bbox]
        im = im.crop((max(x0, 0), max(y0, 0), min(x1, im.width), min(y1, im.height)))
        tmp = Path(tempfile.gettempdir()) / f"crop_{Path(path).stem}.png"
        im.convert("RGB").save(tmp)
        return str(tmp)
    except Exception:
        return path
