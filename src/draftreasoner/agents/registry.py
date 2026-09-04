"""Agent registry — register a unit agent and enumerate/instantiate them."""

from __future__ import annotations

from draftreasoner.agents.base import BaseAgent


_AGENTS: dict[str, type[BaseAgent]] = {}


def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`")
    _AGENTS[cls.name] = cls
    return cls


def get_agent(name: str) -> BaseAgent:
    if name not in _AGENTS:
        raise KeyError(f"unknown agent: {name!r}")
    return _AGENTS[name]()


def build_agents() -> dict[str, BaseAgent]:
    from draftreasoner.agents import engineer  # noqa: F401  (import triggers registration)

    return {name: cls() for name, cls in _AGENTS.items()}


def agent_names() -> list[str]:
    build_agents()  # ensure agent modules are imported and registered
    return list(_AGENTS.keys())
