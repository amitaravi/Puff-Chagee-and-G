"""Renders an eval run as the markdown you paste into the README / numbers slide."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Summary:
    config: str = "C"
    split: str = "dev"
    n: int = 0

    check_rates: dict[str, tuple[int, int]] = field(default_factory=dict)  # name -> (passed, scored)
    autonomy_rate: float = 0.0
    accuracy_non_escalated: float = 0.0
    n_non_escalated: int = 0
    escalation_accuracy: float = 0.0
    calibration_gap: float = 0.0
    flip_rate: float = 0.0
    n_flip_scored: int = 0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    errors: dict[str, int] = field(default_factory=dict)
    probes: dict[str, Any] = field(default_factory=dict)
    stability: dict[str, Any] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["check_rates"] = {k: list(v) for k, v in self.check_rates.items()}
        return d


CHECK_ORDER = [
    ("direction", "Ruling direction"),
    ("amount", "Remedy amount"),
    ("policy", "Policy cited"),
    ("escalation", "Escalation decision"),
    ("calibration", "Confidence calibration"),
    ("grounding", "Evidence grounding  (label-free)"),
    ("recomputation", "Remedy recomputation  (label-free)"),
]

ERROR_LABELS = {
    "E1": "wrong policy applied",
    "E2": "threshold / condition misapplied",
    "E3": "fabricated evidence  <- must be zero",
    "E4": "arithmetic or remedy error",
    "E5": "right outcome, wrong policy cited",
    "E6": "overconfident (should have deferred)",
    "E7": "underconfident (deferred unnecessarily)",
}


def _pct(x: float) -> str:
    return f"{x:.1%}"


def render_markdown(s: Summary, title: str = "Arbitra evaluation") -> str:
    L: list[str] = []
    L.append(f"## {title}")
    L.append("")
    L.append(f"Config **{s.config}** · split **{s.split}** · n = **{s.n}**")
    L.append("")

    # ---- the two numbers that matter, reported as a pair
    L.append("### Headline")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Autonomy rate (resolved without a human) | **{_pct(s.autonomy_rate)}** |")
    L.append(f"| Accuracy on non-escalated cases | **{_pct(s.accuracy_non_escalated)}** (n={s.n_non_escalated}) |")
    L.append(f"| Escalation decision accuracy | {_pct(s.escalation_accuracy)} |")
    L.append(f"| Median latency | {s.latency_p50:.1f}s (p95 {s.latency_p95:.1f}s) |")
    L.append("")
    L.append(
        "> Accuracy alone is a buyable metric: raise the confidence threshold and it goes "
        "up while autonomy goes down. Always report the pair. The only accuracy a user "
        "experiences is accuracy on cases that were NOT escalated."
    )
    L.append("")

    # ---- eight checks
    L.append("### Checks")
    L.append("")
    L.append("| # | Check | Pass | n | Needs a label? |")
    L.append("|---|---|---|---|---|")
    for i, (key, label) in enumerate(CHECK_ORDER, start=1):
        passed, scored = s.check_rates.get(key, (0, 0))
        labelled = "no" if "label-free" in label else "yes"
        val = f"{_pct(passed / scored)}" if scored else "n/a"
        L.append(f"| {i} | {label} | {val} | {scored} | {labelled} |")
    L.append("")

    # ---- error taxonomy
    if s.errors:
        L.append("### Error taxonomy")
        L.append("")
        L.append("| Code | Meaning | Count |")
        L.append("|---|---|---|")
        for code in sorted(s.errors):
            L.append(f"| {code} | {ERROR_LABELS.get(code, '')} | {s.errors[code]} |")
        L.append("")

    # ---- rebuttal flip rate
    if s.n_flip_scored:
        L.append("### Does the debate do anything?")
        L.append("")
        L.append(
            f"Judge emitted a provisional verdict before rebuttal and a final one after on "
            f"{s.n_flip_scored} cases. Flip rate: **{_pct(s.flip_rate)}**."
        )
        L.append("")
        L.append(
            "> 0% means the adversarial round is theatre and should be deleted. "
            "10-20% means it is doing real work — which is also the answer to "
            "*\"why three agents instead of one?\"*"
        )
        L.append("")

    # ---- probes
    if s.probes:
        L.append("### Consistency and bias probes")
        L.append("")
        L.append("| Probe | Result | Target |")
        L.append("|---|---|---|")
        p = s.probes
        if "near_twin" in p:
            L.append(f"| Near-twin consistency | **{_pct(p['near_twin'])}** | 100% (must not wobble) |")
        if "threshold" in p:
            L.append(f"| Threshold flip rate | **{_pct(p['threshold'])}** | 100% (must respect the rule) |")
        if "bias_mirror" in p:
            b = p["bias_mirror"]
            L.append(
                f"| Filing-party skew | rider-filed favours rider "
                f"**{_pct(b['rider_filed_favours_rider'])}** vs driver-filed "
                f"**{_pct(b['driver_filed_favours_rider'])}** (delta "
                f"{b['delta_pp']:+.1f}pp) | ~0pp |"
            )
        if "bias_swap" in p:
            L.append(f"| Party-attribute insensitivity | **{_pct(p['bias_swap'])}** | 100% |")
        L.append("")
        L.append(
            "> Near-twins measure whether the Judge is *random*. The bias probe measures "
            "> whether it is *prejudiced*. Neither is visible in a single-case demo."
        )
        L.append("")

    # ---- stability
    if s.stability:
        L.append("### Stability")
        L.append("")
        L.append(f"{s.stability.get('note', '')}")
        L.append("")

    # ---- failures
    if s.failures:
        L.append("### Failures")
        L.append("")
        L.append("| Case | Category | Predicted | Truth | Confidence | Failed |")
        L.append("|---|---|---|---|---|---|")
        for f in s.failures[:25]:
            L.append(
                f"| {f['dispute_id']} | {f['category']} | {f['predicted']} | {f['truth']} | "
                f"{f['confidence']:.2f} | {f['failed']} |"
            )
        L.append("")

    return "\n".join(L)


def p50(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[idx]
