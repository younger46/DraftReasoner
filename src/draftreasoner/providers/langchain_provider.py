"""LangChain-backed VLM provider (uses langchain_openai.ChatOpenAI)."""

from __future__ import annotations

from typing import Any

from draftreasoner.core.config import Settings
from draftreasoner.core.media import image_data_uri
from draftreasoner.providers.vlm import BaseProvider


class LangChainProvider(BaseProvider):
    """OpenAI-compatible backend through LangChain; can attach an image to the last user turn."""

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {"model": settings.model, "api_key": settings.api_key, "temperature": settings.temperature}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self.llm = ChatOpenAI(**kwargs)

    def chat(self, messages: list[dict[str, Any]], image_path: str | None = None, **kwargs: Any) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        msgs = [
            SystemMessage(content=m["content"]) if m["role"] == "system" else HumanMessage(content=m["content"])
            for m in messages
        ]
        if image_path:
            msgs[-1] = HumanMessage(
                content=[
                    {"type": "text", "text": msgs[-1].content},
                    {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
                ]
            )
        seq = self.llm.invoke(msgs)
        return seq.content or ""

    def judge(self, question: str, reference: str, answer: str, **kwargs: Any) -> int:
        from draftreasoner.prompts.agent import JUDGE_PROMPT

        user = JUDGE_PROMPT.format(question=question, correct=reference, model=answer)
        out = self.chat([{"role": "user", "content": user}], temperature=0)
        return 1 if "1" in out else 0


def create_langchain_provider(settings: Settings) -> LangChainProvider:
    return LangChainProvider(settings)
