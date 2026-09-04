"""Load the MechVQA benchmark and measure the agent with an LLM judge."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from draftreasoner.core.agent import MechAgent
from draftreasoner.evaluation.judge import extract_answer, score
from draftreasoner.providers.vlm import BaseProvider


def load_benchmark(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _aggregate(stats: dict[str, list[int]]) -> dict[str, Any]:
    out = {}
    for key, vals in stats.items():
        get = sum(vals)
        n = len(vals)
        out[key] = {"n": n, "correct": get, "accuracy": (get / n) if n else 0.0}
    return out


def evaluate(
    agent: MechAgent,
    provider: BaseProvider,
    records: list[dict[str, Any]],
    limit: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    recs = records if limit is None else records[:limit]
    total: list[int] = []
    by_sub: dict[str, list[int]] = defaultdict(list)
    by_cap: dict[str, list[int]] = defaultdict(list)
    by_diff: dict[str, list[int]] = defaultdict(list)
    by_lang: dict[str, list[int]] = defaultdict(list)
    per_record: list[dict[str, Any]] = []

    for i, rec in enumerate(recs):
        q = rec["messages"][0]["content"]
        ref = rec["messages"][1]["content"]
        meta = rec.get("metadata") or {}
        predicted = agent.run(rec)
        ans = extract_answer(predicted)
        s = score(provider, q, ref, ans)
        total.append(s)
        by_sub[meta.get("subcategory", "?")].append(s)
        by_cap[meta.get("capability", "?")].append(s)
        by_diff[meta.get("difficulty", "?")].append(s)
        by_lang[str(meta.get("language", "?"))].append(s)
        per_record.append(
            {"index": i, "subcategory": meta.get("subcategory"), "difficulty": meta.get("difficulty"), "score": s}
        )
        if verbose:
            print(f"[{i:04d}] {meta.get('subcategory')} | {s} | {q[:60]}")

    return {
        "total": _aggregate({"total": total}),
        "by_subcategory": _aggregate(dict(by_sub)),
        "by_capability": _aggregate(dict(by_cap)),
        "by_difficulty": _aggregate(dict(by_diff)),
        "by_language": _aggregate(dict(by_lang)),
        "per_record": per_record,
    }
