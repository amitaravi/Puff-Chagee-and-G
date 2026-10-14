"""
The harness. Run it before you tune a single prompt.

    python -m src.eval.run_eval --config C --split dev            # day-to-day
    python -m src.eval.run_eval --config C --split test           # reported numbers
    python -m src.eval.run_eval --ablate                          # A vs B vs C
    python -m src.eval.run_eval --probes                          # consistency + bias
    python -m src.eval.run_eval --stability 3 --temperature 0.7   # stability
    python -m src.eval.run_eval --split dev --stub                # smoke, no API key

Results land in results/ as JSON plus a markdown report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from ..core import config
from ..core.state import EvidencePack
from ..tools.interface import get_tools
from .metrics import CaseResult, evaluate_case, favors
from .report import Summary, p50, p95, render_markdown


# ------------------------------------------------------------------ loading
def load_context(source: str = "auto"):
    """Returns (tools, ground_truth_dict, splits_dict, pairs, pair_roles, label)."""
    # The graph resolves its tool provider through get_tools(), which reads the env —
    # so set it here or run_case() will silently use the other source.
    os.environ["ARBITRA_TOOLS"] = "mock" if source == "fixtures" else "real"

    if source == "fixtures":
        from ..tools.mock_tools import MockTools

        tools = MockTools()
        gt_path = os.path.join(config.FIXTURES_DIR, "_ground_truth.json")
        truth = json.load(open(gt_path, encoding="utf-8")) if os.path.exists(gt_path) else {}
        ids = tools.dispute_ids()
        return tools, truth, {"all": ids, "dev": ids, "test": ids}, {}, {}, "fixtures"

    from ..tools.real_tools import RealTools

    tools = RealTools()
    if not tools.dispute_ids():
        raise SystemExit(
            "No cases under data/. Run:  python scripts/generate_dataset.py  "
            "(or --source fixtures to score the three hand-written golden cases)"
        )
    truth = tools.ground_truth()
    idx_path = os.path.join(config.DATA_DIR, "index.json")
    idx = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else {}
    return (tools, truth, idx.get("splits", {"all": tools.dispute_ids()}),
            idx.get("pairs", {}), idx.get("pair_roles", {}), "data")


# ------------------------------------------------------------------ running
def run_split(ids: list[str], cfg: str, truth: dict, tools, limit: int | None = None,
              quiet: bool = False) -> list[CaseResult]:
    if limit:
        ids = ids[:limit]
    out = []
    for i, did in enumerate(ids, start=1):
        res = evaluate_case(did, cfg, truth.get(did), tools)
        out.append(res)
        if not quiet and i % 10 == 0:
            print(f"  ... {i}/{len(ids)}", flush=True)
    return out


# ------------------------------------------------------------------ summary
def summarize(results: list[CaseResult], cfg: str, split: str) -> Summary:
    s = Summary(config=cfg, split=split, n=len(results))
    errored = [r for r in results if r.error]
    good = [r for r in results if not r.error]

    names = ["direction", "amount", "policy", "escalation", "calibration",
             "grounding", "recomputation"]
    for name in names:
        scored = [r for r in good if r.checks.get(name) is not None]
        passed = sum(1 for r in scored if r.checks[name])
        s.check_rates[name] = (passed, len(scored))

    # autonomy + the accuracy that users actually experience
    not_escalated = [r for r in good if not r.escalated]
    s.autonomy_rate = len(not_escalated) / len(good) if good else 0.0
    scored_dir = [r for r in not_escalated if r.checks.get("direction") is not None]
    s.n_non_escalated = len(scored_dir)
    s.accuracy_non_escalated = (
        sum(1 for r in scored_dir if r.checks["direction"]) / len(scored_dir) if scored_dir else 0.0
    )
    esc = [r for r in good if r.checks.get("escalation") is not None]
    s.escalation_accuracy = (
        sum(1 for r in esc if r.checks["escalation"]) / len(esc) if esc else 0.0
    )

    # calibration gap: mean |confidence - correctness| on determinable cases
    det = [r for r in good if not r.truth_escalate and r.checks.get("direction") is not None]
    if det:
        s.calibration_gap = sum(
            abs(r.confidence - (1.0 if r.checks["direction"] else 0.0)) for r in det
        ) / len(det)

    # rebuttal flip rate — does the adversarial round change anything?
    flips = [r for r in good if r.provisional_ruling]
    s.n_flip_scored = len(flips)
    s.flip_rate = (
        sum(1 for r in flips if r.provisional_ruling != r.predicted_ruling) / len(flips)
        if flips else 0.0
    )

    s.latency_p50 = p50([r.latency_s for r in good])
    s.latency_p95 = p95([r.latency_s for r in good])

    # ---- error taxonomy
    for r in good:
        c = r.checks
        if c.get("grounding") is False:
            s.errors["E3"] = s.errors.get("E3", 0) + 1
        if c.get("direction") is False:
            if c.get("policy") is False:
                s.errors["E1"] = s.errors.get("E1", 0) + 1
            else:
                s.errors["E2"] = s.errors.get("E2", 0) + 1
        elif c.get("policy") is False:
            s.errors["E5"] = s.errors.get("E5", 0) + 1
        if c.get("amount") is False or c.get("recomputation") is False:
            s.errors["E4"] = s.errors.get("E4", 0) + 1
        if r.escalated and not r.truth_escalate:
            s.errors["E7"] = s.errors.get("E7", 0) + 1
        if (not r.escalated) and r.truth_escalate:
            s.errors["E6"] = s.errors.get("E6", 0) + 1

    s.failures = [
        {
            "dispute_id": r.dispute_id, "category": r.category,
            "predicted": r.predicted_ruling, "truth": r.truth_ruling,
            "confidence": r.confidence,
            "failed": ", ".join(k for k, v in r.checks.items() if v is False) or "?",
        }
        for r in good if any(v is False for v in r.checks.values())
    ]
    for r in errored:
        s.errors["CRASH"] = s.errors.get("CRASH", 0) + 1
        s.failures.append({"dispute_id": r.dispute_id, "category": r.category,
                           "predicted": "-", "truth": "-", "confidence": 0.0,
                           "failed": r.error or "crash"})
    return s


# ------------------------------------------------------------------ probes
def run_probes(tools, pairs: dict, pair_roles: dict, cfg: str, truth: dict,
               quiet: bool = False) -> dict:
    """
    The four probes. These are the numbers nobody else at the hackathon will have.
    """
    out: dict = {}
    by_role: dict[str, list[list[str]]] = {}
    for pid, ids in pairs.items():
        by_role.setdefault(pair_roles.get(pid, "?"), []).append(ids)

    # --- near-twin: identical ruling expected
    for role, key in (("near_twin", "near_twin"), ("bias_swap", "bias_swap")):
        group = by_role.get(role, [])
        if not group:
            continue
        same = 0
        for ids in group:
            rulings = []
            for did in ids:
                r = evaluate_case(did, cfg, truth.get(did), tools)
                rulings.append(r.predicted_ruling)
            if len(set(rulings)) == 1:
                same += 1
        out[key] = same / len(group)

    # --- threshold: ruling MUST flip
    group = by_role.get("threshold", [])
    if group:
        flipped = 0
        for ids in group:
            rulings = []
            for did in ids:
                r = evaluate_case(did, cfg, truth.get(did), tools)
                rulings.append(r.predicted_ruling)
            if len(set(rulings)) > 1:
                flipped += 1
        out["threshold"] = flipped / len(group)

    # --- bias mirror: same facts, filing party flipped
    group = by_role.get("bias_mirror", [])
    if group:
        rider_filed, driver_filed = [], []
        for ids in group:
            for did in ids:
                r = evaluate_case(did, cfg, truth.get(did), tools)
                won = favors(r.predicted_ruling, r.filed_by) == "rider"
                (rider_filed if r.filed_by == "rider" else driver_filed).append(won)
        a = sum(rider_filed) / len(rider_filed) if rider_filed else 0.0
        b = sum(driver_filed) / len(driver_filed) if driver_filed else 0.0
        out["bias_mirror"] = {
            "rider_filed_favours_rider": a,
            "driver_filed_favours_rider": b,
            "delta_pp": (a - b) * 100,
            "n": len(rider_filed) + len(driver_filed),
        }
    return out


# ------------------------------------------------------------------ stability
def run_stability(ids: list[str], cfg: str, truth: dict, tools, reps: int,
                  temperature: float) -> dict:
    """Same case, N times, at temperature > 0. How much does the ruling move?"""
    config.TEMPERATURE = temperature
    unstable = 0
    n = 0
    for did in ids:
        rulings = set()
        for _ in range(reps):
            r = evaluate_case(did, cfg, truth.get(did), tools)
            if r.error:
                continue
            rulings.add(r.predicted_ruling)
        if not rulings:
            continue
        n += 1
        if len(rulings) > 1:
            unstable += 1
    return {
        "reps": reps, "temperature": temperature, "n": n,
        "unstable_fraction": (unstable / n) if n else 0.0,
        "note": (f"{reps} repeats at temperature {temperature}: "
                 f"{(unstable / n) if n else 0:.1%} of cases produced more than one ruling."),
    }


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Arbitra evaluation harness")
    ap.add_argument("--config", default="C", choices=["A", "B", "C"])
    ap.add_argument("--ablate", action="store_true", help="run A, B and C and compare")
    ap.add_argument("--split", default="dev", help="dev | test | probes | all")
    ap.add_argument("--source", default="auto", choices=["auto", "data", "fixtures"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--stability", type=int, metavar="REPS")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--stub", action="store_true", help="canned verdicts, no API key")
    ap.add_argument("--out", default=None, help="output dir (default results/)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.stub:
        config.set_stub(True)

    source = args.source
    if source == "auto":
        source = "fixtures" if not os.path.isdir(os.path.join(config.DATA_DIR, "cases")) else "data"
    tools, truth, splits, pairs, pair_roles, label = load_context(source)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_dir = args.out or config.RESULTS_DIR

    configs = ["A", "B", "C"] if args.ablate else [args.config]
    reports: dict[str, str] = {}

    for cfg in configs:
        ids = splits.get(args.split) or splits.get("all") or sorted(truth)
        if not ids:
            raise SystemExit(f"split '{args.split}' is empty")
        print(f"\n=== config {cfg} · split {args.split} · source {label} · n={len(ids)} ===")
        results = run_split(ids, cfg, truth, tools, args.limit, args.quiet)
        s = summarize(results, cfg, args.split)

        if args.probes and pairs:
            s.probes = run_probes(tools, pairs, pair_roles, cfg, truth, args.quiet)
        if args.stability:
            s.stability = run_stability(
                (splits.get("test") or ids)[: min(30, len(ids))], cfg, truth, tools,
                args.stability, args.temperature,
            )

        md = render_markdown(s)
        reports[cfg] = md
        print()
        print(md)

        stem = f"eval_{cfg}_{args.split}"
        with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as fh:
            json.dump({"summary": s.as_dict(),
                       "results": [r.as_dict() for r in results]}, fh, indent=2, default=str)
        with open(os.path.join(out_dir, f"{stem}.md"), "w", encoding="utf-8") as fh:
            fh.write(md + "\n")

    if args.ablate:
        lines = ["## Ablation — does the multi-agent design earn its keep?", "",
                 "| Config | Autonomy | Accuracy (non-escalated) | Direction | Grounding | Median latency |",
                 "|---|---|---|---|---|---|"]
        for cfg in configs:
            d = json.load(open(os.path.join(out_dir, f"eval_{cfg}_{args.split}.json")))["summary"]
            rate = lambda k: (  # noqa: E731
                f"{d['check_rates'][k][0] / d['check_rates'][k][1]:.1%}"
                if d["check_rates"].get(k) and d["check_rates"][k][1] else "n/a")
            lines.append(
                f"| {cfg} | {d['autonomy_rate']:.1%} | {d['accuracy_non_escalated']:.1%} "
                f"(n={d['n_non_escalated']}) | {rate('direction')} | {rate('grounding')} | "
                f"{d['latency_p50']:.1f}s |"
            )
        lines += ["", "A = single agent · B = three agents, no debate · C = full tribunal.",
                  "", "If B does not beat A, ship A and say so. Reporting that you measured "
                  "it beats any rationale you can write in prose."]
        ablation = "\n".join(lines)
        print("\n" + ablation)
        with open(os.path.join(out_dir, "ablation.md"), "w", encoding="utf-8") as fh:
            fh.write(ablation + "\n")

    print(f"\nWrote results to {os.path.relpath(out_dir, config.ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
