"""Configuration and path resolution.

Reads optional `.env` from the repository root so settings like the VLM backend
can be overridden without touching code. The data/ tree inside the repo root is
treated as the default source for the MechVQA benchmark.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    # <repo>/src/draftreasoner/core/config.py -> parents[3] = repo root
    return Path(__file__).resolve().parents[3]


def _load_dotenv(path: Path | None = None) -> None:
    """Tiny dotenv loader (no external dependency). Does not override real env.

    Reads `.env` if present, otherwise falls back to `.env.example`, so a filled
    example file works out of the box. Inline `#` comments are stripped.
    """
    dotenv = path or _repo_root() / ".env"
    if not dotenv.exists():
        dotenv = _repo_root() / ".env.example"
    if not dotenv.exists():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split("#", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class Settings:
    repo_root: Path = field(default_factory=_repo_root)
    data_dir: Path = field(default_factory=lambda: _repo_root() / "data")

    # provider (VLM / judge) defaults, overridable via env
    api_key: str = ""
    model: str = "gpt-4o"
    base_url: str = ""
    temperature: float = 0.0
    backend: str = "react"  # only backend now: LangGraph ReAct loop

    # agent loop tuning
    max_retries: int = 2
    retry_confidence_floor: float = 0.6
    verbose: bool = True

    @property
    def benchmark_path(self) -> Path:
        return self.data_dir / "MechVQA_test/vqa_benchmark/mechvqa_benchmark.jsonl"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "MechVQA_test/images"

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "MechVQA_train&val"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            api_key=os.getenv("DR_API_KEY", ""),
            model=os.getenv("DR_MODEL", "gpt-4o"),
            base_url=os.getenv("DR_BASE_URL", ""),
            temperature=float(os.getenv("DR_TEMPERATURE", "0.0")),
            backend=os.getenv("DR_BACKEND", "react"),
            max_retries=int(os.getenv("DR_MAX_RETRIES", "2")),
            retry_confidence_floor=float(os.getenv("DR_RETRY_CONF_FLOOR", "0.6")),
            verbose=os.getenv("DR_VERBOSE", "true").lower() == "true",
        )


def resolve_image_path(images_dir: Path, rel: str) -> Path | None:
    """Resolve a packaged relative image path (e.g. 'images/59/xxxx.jpg')."""
    candidate = (images_dir.parent / rel) if images_dir.name == "images" else (images_dir / rel)
    if candidate.exists():
        return candidate
    # fall back to a search of the hash filename
    name = Path(rel).name
    for hit in images_dir.rglob(name):
        return hit
    return None
