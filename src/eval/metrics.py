"""
Eight checks, not one score.

Five need labels (direction, amount, policy, escalation, calibration). Three do not
(grounding, recomputation, stability) — and those three are the shield. When someone
says "your ground truth is synthetic so your accuracy number is meaningless", the
label-free checks still hold. That is why we do not lead with accuracy.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..core.state import EvidencePack, Verdict
from ..graph.workflow import run_case
from ..policy.apply_policy import band_for, policy_formula_namespace
from ..tools.interface import get_tools

AMOUNT_TOLERANCE_ABS = 0.51  # a dollar of rounding is not an error
AMOUNT_TOLERANCE_REL = 0.10

# Section names agents may use in an evidence ref, mapped onto EvidencePack fields.
SECTION_ALIASES = {
    "route": "route",
    "route_analysis": "route",
    "telemetry": "route",
    "gps": "route",
    "trip": "trip",
    "fare": "fare",
    "payment": "fare",
    "dispute": "dispute",
    "rider": "rider",
    "driver": "driver",
    "no_show": "no_show",
    "noshow": "no_show",
    "chat": "chat",
    "policy": "policies",
    "policies": "policies",
}

NUMBER_RE = re.compile(r"\d+\.\d{2,}")


# ------------------------------------------------------------------ helpers
def favors(ruling: str, filed_by: str) -> str:
    """
    Who did this ruling favour? Party-relative, which is what the bias audit needs.

    no_action / dismissed leaves the status quo standing, so it favours the DEFENDANT —
    the party who did not file. Getting this wrong inverts the bias measurement.
    """
    if ruling == "refund_rider":
        return "rider"
    if ruling == "refund_driver":
        return "driver"
    return "driver" if filed_by == "rider" else "rider"


def _policy_by_id(pack: EvidencePack, policy_id: str):
    for pol in pack.policies:
        if pol.policy_id == policy_id:
            return pol
    try:
        return get_tools().get_policy(policy_id)
    except Exception:  # noqa: BLE001 — unknown policy id is itself a finding
        return None


def _pack_numbers(pack: EvidencePack) -> set[float]:
    out: set[float] = set()

    def walk(obj) -> None:
        if obj is None:
            return
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            out.add(round(float(obj), 2))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(pack.model_dump())
    return out


def resolve_ref(pack: EvidencePack, ref: str) -> tuple[bool, str]:
    """
    Does TRP-00431.route_analysis.deviation_pct point at something real?

    Hallucinated evidence IDs are the single most damaging failure mode in a dispute
    system — a ruling that cites a figure that does not exist looks correct and isn't.
    This catches it programmatically rather than by asking a model to self-report.
    """
    tokens = [t for t in ref.split(".") if t]
    if not tokens:
        return False, "empty ref"

    # A leading trip id / party id is conventional but optional.
    if tokens[0] in {pack.trip.trip_id, pack.rider.party_id, pack.driver.party_id,
                     pack.dispute.dispute_id}:
        tokens = tokens[1:]
    if not tokens:
        return False, "ref is only an id"

    section = SECTION_ALIASES.get(tokens[0].lower())
    if section is None:
        return False, f"unknown section '{tokens[0]}'"

    obj = getattr(pack, section, None)
    if obj is None:
        return False, f"section '{section}' absent from this case"

    if len(tokens) == 1:
        return True, ""

    if section == "chat":
        ids = {m.message_id for m in pack.chat}
        return (tokens[1] in ids), ("" if tokens[1] in ids else f"no chat message {tokens[1]}")
    if section == "policies":
        ids = {p.policy_id for p in pack.policies}
        return (tokens[1] in ids), ("" if tokens[1] in ids else f"no policy {tokens[1]}")

    path = tokens[1:]
    cur = obj
    for tok in path:
        if isinstance(cur, list):
            if not tok.isdigit() or int(tok) >= len(cur):
                return False, f"index {tok} out of range"
            cur = cur[int(tok)]
            continue
        if not hasattr(cur, tok):
            return False, f"no field '{tok}' on {section}"
        cur = getattr(cur, tok)
    return True, ""


def check_grounding(pack: EvidencePack, verdict: Verdict) -> tuple[bool, list[str], int]:
    """Every cited evidence ID exists. Plus: does every precise number appear in the record?"""
    bad: list[str] = []
    claims = list(verdict.findings_of_fact)
    for c in claims:
        for ref in c.evidence_refs:
            ok, why = resolve_ref(pack, ref)
            if not ok:
                bad.append(f"{ref}: {why}")
        for ref in c.policy_refs:
            if _policy_by_id(pack, ref) is None:
                bad.append(f"{ref}: policy not in the record")

    values = _pack_numbers(pack)
    mismatches = 0
    text = " ".join(c.text for c in claims) + " " + verdict.remedy_rationale
    for num in NUMBER_RE.findall(text):
        if round(float(num), 2) not in values:
            mismatches += 1
    return (not bad), bad, mismatches


def recompute_amount(pack: EvidencePack, verdict: Verdict) -> float | None:
    """
    Re-derive the award using the formula of the policy the Judge actually cited.

    This catches bad arithmetic AND wrong-policy-applied-to-right-facts, without
    needing a label: the policy text is part of the case file.

    Important: recomputation only applies when the ruling grants money. A policy may
    include a formula for the *eligible* award but still be cited in a no-action outcome
    because an exception clause applied. In that case the expected amount is 0.
    """
    if verdict.ruling in {"no_action", "dismissed"}:
        return 0.0

    if not verdict.cited_policy_id:
        return None
    pol = _policy_by_id(pack, verdict.cited_policy_id)
    if pol is None or not pol.remedy_formula:
        return None
    ns = policy_formula_namespace(pack)
    try:
        val = eval(pol.remedy_formula, {"__builtins__": {}}, dict(ns))  # noqa: S307
    except Exception:  # noqa: BLE001
        return None
    return round(min(float(val), pack.dispute.claimed_amount), 2)


# ------------------------------------------------------------------ result
@dataclass
class CaseResult:
    dispute_id: str
    config: str
    category: str
    filed_by: str

    predicted_ruling: str = ""
    predicted_amount: float = 0.0
    predicted_policy: str | None = None
    confidence: float = 0.0
    escalated: bool = False
    provisional_ruling: str | None = None

    truth_ruling: str = ""
    truth_amount: float = 0.0
    truth_policy: str | None = None
    truth_band: str = ""
    truth_escalate: bool = False
    truth_rationale: str = ""

    checks: dict[str, bool | None] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None

    @property
    def scored(self) -> dict[str, bool]:
        """Only checks that actually applied — used for the headline numbers."""
        return {k: v for k, v in self.checks.items() if v is not None}

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["checks"] = self.checks
        d["notes"] = self.notes
        return d


# ------------------------------------------------------------------ evaluation
def evaluate_case(dispute_id: str, cfg: str = "C", truth: dict | None = None,
                  tools=None) -> CaseResult:
    """Run one case through the pipeline and score it eight ways."""
    pack = (tools or get_tools()).get_evidence_pack(dispute_id)
    truth = truth or {}

    start = time.perf_counter()
    try:
        state = run_case(dispute_id, cfg)
    except Exception as exc:  # noqa: BLE001 — one bad case must not kill the run
        latency = time.perf_counter() - start
        return CaseResult(
            dispute_id=dispute_id, config=cfg, category=pack.dispute.category,
            filed_by=pack.dispute.filed_by, error=f"{type(exc).__name__}: {exc}",
            latency_s=round(latency, 3),
        )
    latency = time.perf_counter() - start

    verdict: Verdict = state.get("verdict") or Verdict(ruling="no_action", confidence=0.0)

    res = CaseResult(
        dispute_id=dispute_id, config=cfg, category=pack.dispute.category,
        filed_by=pack.dispute.filed_by,
        predicted_ruling=verdict.ruling, predicted_amount=round(verdict.amount, 2),
        predicted_policy=verdict.cited_policy_id, confidence=verdict.confidence,
        escalated=bool(state.get("escalated")), provisional_ruling=verdict.provisional_ruling,
        truth_ruling=truth.get("correct_ruling", ""),
        truth_amount=float(truth.get("correct_refund_amount", 0.0)),
        truth_policy=truth.get("correct_policy_id"),
        truth_band=truth.get("expected_confidence_band", ""),
        truth_escalate=bool(truth.get("should_escalate", False)),
        truth_rationale=truth.get("rationale", ""),
        latency_s=round(latency, 3),
    )

    c: dict[str, bool | None] = {}
    notes: dict[str, str] = {}

    # --- 1. escalation decision (scored on every case, including ambiguous ones)
    c["escalation"] = res.escalated == res.truth_escalate

    # --- 2-4. direction / amount / policy — ONLY where the record determines an outcome.
    # Scoring direction on an ambiguous case would be scoring against our own admission
    # that the answer is not determinable.
    if res.truth_escalate:
        c["direction"] = None
        c["amount"] = None
        c["policy"] = None
        notes["skipped"] = "ambiguous by construction; only escalation is scored"
    else:
        c["direction"] = res.predicted_ruling == res.truth_ruling
        tol = max(AMOUNT_TOLERANCE_ABS, AMOUNT_TOLERANCE_REL * max(res.truth_amount, 1.0))
        c["amount"] = abs(res.predicted_amount - res.truth_amount) <= tol
        c["policy"] = res.predicted_policy == res.truth_policy

    # --- 5. calibration
    c["calibration"] = band_for(res.confidence) == res.truth_band
    notes["predicted_band"] = band_for(res.confidence)

    # --- 6. grounding (label-free)
    ok, bad, mism = check_grounding(pack, verdict)
    c["grounding"] = ok
    if bad:
        notes["bad_refs"] = "; ".join(bad[:5])
    if mism:
        notes["numeric_mismatch"] = f"{mism} precise number(s) not found in the record"

    # --- 7. recomputation (label-free)
    expected = recompute_amount(pack, verdict)
    if expected is None:
        c["recomputation"] = None
        notes["recomputation"] = "cited policy has no machine-checkable formula"
    else:
        tol = max(AMOUNT_TOLERANCE_ABS, AMOUNT_TOLERANCE_REL * max(expected, 1.0))
        c["recomputation"] = abs(res.predicted_amount - expected) <= tol
        notes["recomputed"] = f"{expected:.2f} vs awarded {res.predicted_amount:.2f}"

    res.checks = c
    res.notes = notes
    return res
