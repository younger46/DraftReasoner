"""Tool abstraction for MechAgent.

A tool is a small, deterministic, *verifiable* unit of work. Extending the agent
with a new capability means implementing a `Tool` subclass with a unique `name`
and registering it (see `registry.py`). Tools return structured `ToolResult`
so the agent can record evidence and reason about confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


class Tool:
    """Base class. Subclasses define `name`, `description` and `run`."""

    name: str = ""
    description: str = ""

    def run(self, **kwargs: Any) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def ok(self, data: Any, confidence: float = 1.0, evidence: list[dict] | None = None) -> ToolResult:
        return ToolResult(ok=True, data=data, confidence=confidence, evidence=evidence or [])

    def fail(self, error: str) -> ToolResult:
        return ToolResult(ok=False, error=error)
