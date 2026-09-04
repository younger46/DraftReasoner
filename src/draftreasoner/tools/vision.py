"""Helpers for VLM-assisted perception: crop a region, prompt the model, parse JSON."""

from __future__ import annotations

import json
import re
from typing import Any

from draftreasoner.core.media import crop_to
from draftreasoner.providers.vlm import BaseProvider, NullProvider


def has_provider(provider: Any) -> bool:
    return provider is not None and not isinstance(provider, NullProvider)


def call_json(provider: BaseProvider, prompt: str, image_path: str | None = None, bbox: Any = None) -> Any:
    """Ask a vision provider for JSON, optionally over a cropped region."""
    img_path = crop_to(image_path, bbox) if (image_path and bbox) else image_path
    raw = provider.chat([{"role": "user", "content": prompt}], image_path=img_path)
    return parse_json(raw)


def parse_json(raw: str) -> Any:
    """Tolerant JSON extraction from an LLM response."""
    if not raw:
        return {}
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        arr = re.search(r"\[.*\]", text, re.S)
        if arr:
            try:
                return json.loads(arr.group(0))
            except Exception:
                return {}
        return {}


def claims_from(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn structured annotation items into evidence claims."""
    claims = []
    for a in annotations[:16]:
        value = a.get("value") or a.get("dimension_value") or ""
        feature = a.get("feature") or a.get("location") or a.get("feature_description") or ""
        view = a.get("view") or a.get("view_name") or ""
        kind = a.get("kind") or a.get("type") or "dimension"
        parts = [f"kind={kind}", f"value={value}", f"feature={feature}", f"view={view}"]
        claims.append({"source": "AnnotationExtract", "claim": " | ".join(p for p in parts if p.split("=")[1]), "confidence": 0.8})
    return claims
