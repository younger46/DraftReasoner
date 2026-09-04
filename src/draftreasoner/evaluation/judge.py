"""Answer extraction + LLM-as-a-judge scoring."""

from __future__ import annotations

import re
from typing import Any

from draftreasoner.providers.vlm import BaseProvider


_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def extract_answer(text: str) -> str:
    """Pull the answer segment out of a <think>/<answer> response, with fallbacks."""
    m = _ANSWER.search(text)
    if m:
        return m.group(1).strip()
    m = _THINK.search(text)
    if m:
        return text[m.end():].strip()
    return text.strip()


def score(provider: BaseProvider, question: str, reference: str, answer: str) -> int:
    """Return 1/0 — semantic equivalence between the model answer and the reference."""
    return provider.judge(question=question, reference=reference, answer=answer)
