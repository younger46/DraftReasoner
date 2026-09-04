"""Tool registry — the single extension point for new capabilities."""

from __future__ import annotations

from draftreasoner.tools.base import Tool


_REGISTRY: dict[str, type[Tool]] = {}


def register(tool_cls: type[Tool]) -> type[Tool]:
    """Class decorator that adds a Tool type to the registry by its name."""
    if not tool_cls.name:
        raise ValueError(f"{tool_cls.__name__} must define a non-empty `name`")
    _REGISTRY[tool_cls.name] = tool_cls
    return tool_cls


def get_tool(name: str) -> Tool:
    if name not in _REGISTRY:
        raise KeyError(f"unknown tool: {name!r}")
    return _REGISTRY[name]()


def has_tool(name: str) -> bool:
    return name in _REGISTRY


def build_tools() -> list[Tool]:
    """Instantiate every registered tool (ordered by registration)."""
    from draftreasoner.tools import (  # noqa: F401  (imports trigger registration)
        annotation_extract,
        figure_parse,
        geometry_solve,
        std_kb,
        view_align,
    )

    return [cls() for cls in _REGISTRY.values()]


def tool_names() -> list[str]:
    build_tools()  # ensure tool modules are imported and registered
    return list(_REGISTRY.keys())
