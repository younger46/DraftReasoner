"""VLM provider abstraction.

All model access goes through LangChain. `create_provider(settings)` returns a
LangChain-backed provider (langchain-openai ChatOpenAI) when credentials are set,
otherwise a `NullProvider`. No direct `openai` client is used anywhere.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from draftreasoner.core.config import Settings


class BaseProvider(ABC):
    """Thin interface around a chat/vision model + an answer judge."""

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], image_path: str | None = None, **kwargs: Any) -> str: ...

    @abstractmethod
    def judge(self, question: str, reference: str, answer: str, **kwargs: Any) -> int:
        """Return 1 if `answer` is semantically correct vs `reference`, else 0."""


class NullProvider(BaseProvider):
    """Offline provider used when no credentials are configured."""

    def chat(self, messages: list[dict[str, Any]], image_path: str | None = None, **kwargs: Any) -> str:
        return "[provider not configured: set DR_API_KEY/DR_MODEL/DR_BASE_URL in .env]"

    def judge(self, question: str, reference: str, answer: str, **kwargs: Any) -> int:
        return 0


def create_provider(settings: Settings) -> BaseProvider:
    if settings.api_key:
        try:
            from draftreasoner.providers.langchain_provider import LangChainProvider

            return LangChainProvider(settings)
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("LangChain provider requires `uv sync` to install langchain-openai.") from exc
    return NullProvider()
