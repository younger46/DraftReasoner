"""argparse-based CLI: run one record, list tools, or evaluate the benchmark."""

from __future__ import annotations

import argparse
import sys

from draftreasoner.core.agent import MechAgent
from draftreasoner.core.config import Settings
from draftreasoner.evaluation.benchmark import evaluate, load_benchmark
from draftreasoner.tools.registry import tool_names
from draftreasoner.agents.registry import agent_names


def _agent(settings: Settings) -> MechAgent:
    return MechAgent(settings)


def cmd_tools(_args: argparse.Namespace) -> None:
    print("Registered tools:")
    for n in tool_names():
        print(f"  - {n}")


def cmd_agents(_args: argparse.Namespace) -> None:
    print("Registered unit agents:")
    for n in agent_names():
        print(f"  - {n}")
    print("Default plan (single-agent): run all via [engineer]. Add agents + extend PLANS to go multi-agent.")


def cmd_run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    agent = _agent(settings)
    if args.index is not None:
        records = load_benchmark(settings.benchmark_path)
        rec = records[args.index]
        q = rec["messages"][0]["content"]
        meta = rec.get("metadata") or {}
        print(f"#{args.index} [{meta.get('subcategory')}/{meta.get('difficulty')}] {q}\n")
        print(agent.run(rec))
    else:
        q = args.question or ""
        print(agent.answer(q, image_path=args.image, metadata={"subcategory": args.subcategory}))


def cmd_eval(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    agent = _agent(settings)
    records = load_benchmark(settings.benchmark_path)
    print(f"loaded {len(records)} records")
    report = evaluate(agent, agent.provider, records, limit=args.limit, verbose=args.verbose)
    t = report["total"]["total"]
    print(f"\n== overall: n={t['n']} correct={t['correct']} accuracy={t['accuracy']:.3f}")
    for section in ("by_capability", "by_difficulty", "by_subcategory"):
        print(f"\n[{section}]")
        for k, v in sorted(report[section].items(), key=lambda kv: -kv[1]["n"]):
            print(f"  {k:<26} n={v['n']:<5} acc={v['accuracy']:.3f}")


def app() -> None:
    parser = argparse.ArgumentParser(prog="draftreasoner", description="MechAgent for mechanical drawing VQA")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tools = sub.add_parser("tools", help="list registered tools")
    p_tools.set_defaults(func=cmd_tools)

    p_agents = sub.add_parser("agents", help="list registered unit agents / current plan")
    p_agents.set_defaults(func=cmd_agents)

    p_run = sub.add_parser("run", help="answer one benchmark record (or a manual question)")
    p_run.add_argument("--index", type=int, default=None, help="benchmark record index")
    p_run.add_argument("--question", default=None, help="manual question (no index)")
    p_run.add_argument("--image", default=None, help="manual image path")
    p_run.add_argument("--subcategory", default="", help="manual subcategory")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="evaluate on the benchmark with an LLM judge")
    p_eval.add_argument("--limit", type=int, default=None, help="only first N records")
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)
