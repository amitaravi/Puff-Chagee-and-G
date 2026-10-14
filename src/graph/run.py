"""
CLI entry point.

  python -m src.graph.run --dispute DSP-0001 --config C
  python -m src.graph.run --list
  python -m src.graph.run --dispute DSP-0001 --stub     # no API key needed

Day 1 goal: `--stub` completes end to end. That proves the graph compiles and state
flows before you write a single real prompt.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..core import config
from ..tools.interface import get_tools
from .workflow import run_case


def _print(state) -> None:
    v = state.get("verdict")
    print("\n" + "=" * 68)
    print(f"CASE {state['dispute_id']}   config {state['config']}")
    print("=" * 68)

    for step in state.get("trace", []):
        stage = step["stage"]
        extra = {k: val for k, val in step.items() if k != "stage"}
        print(f"  [{stage}] {extra if extra else ''}")

    if state.get("errors"):
        print("\n  ERRORS:")
        for e in state["errors"]:
            print(f"    - {e}")

    if not v:
        print("\n  No verdict issued.")
        return

    print("\n" + "-" * 68)
    print(f"RULING:      {v.ruling}")
    print(f"AMOUNT:      ${v.amount:.2f}")
    print(f"POLICY:      {v.cited_policy_id}")
    print(f"CONFIDENCE:  {v.confidence:.2f}  {v.confidence_factors}")
    if v.provisional_ruling:
        flag = " <-- FLIPPED" if v.provisional_ruling != v.ruling else ""
        print(f"PROVISIONAL: {v.provisional_ruling} ({v.provisional_confidence}){flag}")
    print(f"ESCALATED:   {v.should_escalate}" + (f" — {v.escalation_reason}" if v.escalation_reason else ""))
    print("-" * 68)
    print("\nFINDINGS OF FACT:")
    for c in v.findings_of_fact:
        print(f"  - {c.text}  [{', '.join(c.evidence_refs)}]")
    print("\nCONCLUSIONS OF LAW:")
    for c in v.conclusions_of_law:
        print(f"  - {c}")
    print(f"\nREMEDY: {v.remedy_rationale}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Arbitra — multi-agent dispute resolution")
    ap.add_argument("--dispute", help="dispute id, e.g. DSP-0001")
    ap.add_argument("--config", default="C", choices=["A", "B", "C"], help="ablation config")
    ap.add_argument("--stub", action="store_true", help="run with canned verdicts, no API key")
    ap.add_argument("--list", action="store_true", help="list available fixtures")
    ap.add_argument("--json", action="store_true", help="dump final state as JSON")
    args = ap.parse_args(argv)

    if args.stub:
        config.set_stub(True)

    if args.list or not args.dispute:
        tools = get_tools()
        ids = tools.dispute_ids()
        print(f"Available cases ({len(ids)}, source={type(tools).__name__}):")
        for did in ids[:20]:
            print(f"  {did}")
        if len(ids) > 20:
            print(f"  ... {len(ids) - 20} more")
        if not ids:
            print("  (none — run: python scripts/generate_dataset.py)")
        return 0

    state = run_case(args.dispute, args.config)

    if args.json:
        print(json.dumps(state.get("verdict").model_dump() if state.get("verdict") else {}, indent=2))
    else:
        _print(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
