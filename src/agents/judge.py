"""The Judge. Impartial arbitrator, separate context, strongest model.

Emits a PROVISIONAL verdict before rebuttal and a FINAL one after, so the rebuttal
round can be measured rather than asserted (see Differentiation.md, Test 3).

Confidence is HYBRID: the model's self-report, adjusted by deterministic factors.
Pure self-reported confidence is unreliable and undefensible under questioning.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..core import config
from ..core.llm import get_llm, model_ready
from langchain_openai.chat_models.base import OpenAIInvalidRequestError, OpenAIRateLimitError
from ..core.state import (
    CaseBrief,
    CaseState,
    Claim,
    ClaimAssessment,
    ConfidenceFactor,
    Verdict,
    log,
)
from .prompts import (
    JUDGE_SYSTEM,
    JUDGE_USER,
    SINGLE_AGENT_SYSTEM,
    SINGLE_AGENT_USER,
    render_brief,
    render_evidence,
    render_fraud,
    render_policies,
    render_precedents,
    render_trip,
    stub_verdict,
)
from .rule_based import deterministic_verdict

# deterministic confidence adjustments — tunable, and you can show the arithmetic live
ADJUSTMENTS = {
    "missing_key_evidence": -0.15,
    "precedent_conflict": -0.10,
    "high_fraud_risk": -0.20,
    "advocates_converge": +0.05,
    "uncited_claims_struck": -0.05,
}


def _missing_key_evidence(state: CaseState) -> bool:
    pack = state["pack"]
    category = pack.dispute.category
    if category == "route_deviation" and pack.route is None:
        return True
    if category == "no_show" and pack.no_show is None:
        return True
    return False


def _precedent_conflict(pack, verdict: Verdict) -> bool:
    rulings = {p.ruling for p in pack.precedents if p.category == pack.dispute.category}
    return len(rulings) > 1


def finalize_confidence(state: CaseState, verdict: Verdict) -> Verdict:
    """Blend self-reported confidence with deterministic factors. Clamped to [0, 1]."""
    pack = state["pack"]
    base = float(verdict.confidence or 0.5)
    factors: list[ConfidenceFactor] = [ConfidenceFactor(factor="self_report", delta=round(base, 3))]
    total = base

    def apply(key: str, condition: bool) -> None:
        nonlocal total
        if condition:
            delta = ADJUSTMENTS[key]
            total += delta
            factors.append(ConfidenceFactor(factor=key, delta=delta))

    apply("missing_key_evidence", _missing_key_evidence(state))
    apply("precedent_conflict", _precedent_conflict(pack, verdict))
    apply("high_fraud_risk", bool(pack.fraud and pack.fraud.risk_level == "high"))
    apply(
        "advocates_converge",
        "rider_brief" in state
        and "driver_brief" in state
        and _same_ruling(state["rider_brief"], state["driver_brief"]),
    )
    apply("uncited_claims_struck", any("struck" in e for e in state.get("errors", [])))

    verdict.confidence = max(0.0, min(1.0, round(total, 3)))
    verdict.confidence_factors = factors
    return verdict


def _same_ruling(a: CaseBrief, b: CaseBrief) -> bool:
    """Crude convergence check: both sides effectively want the same outcome."""
    return a.requested_remedy == b.requested_remedy


def _valid_evidence_ref(pack, ref: str) -> bool:
    tokens = [t for t in ref.split(".") if t]
    if not tokens:
        return False
    if tokens[0] in {pack.trip.trip_id, pack.dispute.dispute_id, pack.rider.party_id, pack.driver.party_id}:
        tokens = tokens[1:]
    if not tokens:
        return False

    aliases = {
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
    }
    section = aliases.get(tokens[0].lower())
    if section is None:
        return False

    obj = getattr(pack, section, None)
    if obj is None:
        return False

    if len(tokens) == 1:
        return True

    if section == "chat":
        return tokens[1] in {m.message_id for m in pack.chat}

    cur = obj
    for tok in tokens[1:]:
        if isinstance(cur, list):
            if not tok.isdigit() or int(tok) >= len(cur):
                return False
            cur = cur[int(tok)]
            continue
        if not hasattr(cur, tok):
            return False
        cur = getattr(cur, tok)
    return True


def _sanitize_verdict(state: CaseState, verdict: Verdict) -> tuple[Verdict, list[str]]:
    pack = state["pack"]
    errors: list[str] = []

    valid_policy_ids = {p.policy_id for p in pack.policies}
    if verdict.cited_policy_id and verdict.cited_policy_id not in valid_policy_ids:
        errors.append(f"judge: invalid cited_policy_id '{verdict.cited_policy_id}' replaced")
        verdict.cited_policy_id = pack.policies[0].policy_id if pack.policies else None

    cleaned_findings = []
    for c in verdict.findings_of_fact:
        c.evidence_refs = [r for r in c.evidence_refs if _valid_evidence_ref(pack, r)]
        c.policy_refs = [r for r in c.policy_refs if r in valid_policy_ids]
        if c.evidence_refs or c.policy_refs:
            cleaned_findings.append(c)
    if len(cleaned_findings) != len(verdict.findings_of_fact):
        errors.append("judge: removed finding(s) with invalid or missing references")
    verdict.findings_of_fact = cleaned_findings

    if not verdict.findings_of_fact:
        fallback_ref = (
            f"{pack.trip.trip_id}.route_analysis.deviation_pct"
            if pack.route else
            f"{pack.trip.trip_id}.no_show.driver_min_distance_to_pickup_m"
            if pack.no_show else
            f"{pack.trip.trip_id}.fare.total_charged"
        )
        verdict.findings_of_fact = [Claim(text="Record-driven fallback finding inserted by guardrail.", evidence_refs=[fallback_ref])]
        errors.append("judge: inserted fallback finding to avoid uncited judgment")

    verdict.amount = max(0.0, min(round(float(verdict.amount or 0.0), 2), pack.dispute.claimed_amount))

    # --- Validate claim assessments (judge's inner monologue) ---
    cleaned_assessments = []
    for a in verdict.claim_assessments:
        # Validate evidence_checked refs — drop invalid ones but keep the assessment
        a.evidence_checked = [r for r in a.evidence_checked if _valid_evidence_ref(pack, r) or r in valid_policy_ids]
        cleaned_assessments.append(a)
    if len(cleaned_assessments) != len(verdict.claim_assessments):
        errors.append("judge: cleaned invalid evidence refs in claim assessments")
    verdict.claim_assessments = cleaned_assessments

    return verdict, errors


def _judge_user(state: CaseState, include_rebuttal: bool) -> str:
    pack = state["pack"]
    d = pack.dispute
    rider: CaseBrief | None = state.get("rider_brief")
    driver: CaseBrief | None = state.get("driver_brief")

    rebuttal_block = ""
    if include_rebuttal and rider and driver:
        parts = []
        for b in (rider, driver):
            if b.rebuttal:
                parts.append(f"{b.side.upper()} ADVOCATE'S REPLY:\n{b.rebuttal}")
        rebuttal_block = "\n\n".join(parts) + "\n\n" if parts else ""

    return JUDGE_USER.format(
        dispute_id=d.dispute_id,
        category=d.category,
        claimed_amount=d.claimed_amount,
        evidence_block=render_evidence(pack),
        policy_block=render_policies(pack),
        rider_brief=render_brief(rider) if rider else "(not filed)",
        driver_brief=render_brief(driver) if driver else "(not filed)",
        rebuttal_block=rebuttal_block,
        precedent_block=render_precedents(pack) if config.flags_for(state["config"]).use_precedent else "",
        fraud_block=render_fraud(pack) if config.flags_for(state["config"]).use_fraud else "",
    )


def _call_judge(state: CaseState, include_rebuttal: bool) -> Verdict:
    if config.is_stub():
        return stub_verdict()
    if not model_ready():
        return deterministic_verdict(state["pack"])
    llm = get_llm("judge").with_structured_output(Verdict, method="function_calling")
    try:
        return llm.invoke(
            [
                SystemMessage(content=JUDGE_SYSTEM),
                HumanMessage(content=_judge_user(state, include_rebuttal)),
            ]
        )
    except (OpenAIInvalidRequestError, OpenAIRateLimitError):
        return deterministic_verdict(state["pack"])


def provisional_verdict(state: CaseState) -> dict:
    """Before rebuttal. Logged so we can measure whether the debate changes anything."""
    verdict = _call_judge(state, include_rebuttal=False)
    return {
        "provisional": verdict,
        **log(state, "provisional_verdict", ruling=verdict.ruling, confidence=verdict.confidence),
    }


def judge(state: CaseState) -> dict:
    """After rebuttal. Final ruling."""
    verdict = _call_judge(state, include_rebuttal=config.flags_for(state["config"]).use_rebuttal)
    verdict.provisional_ruling = state.get("provisional", Verdict(ruling="no_action")).ruling
    verdict.provisional_confidence = state.get("provisional", Verdict(ruling="no_action")).confidence
    verdict, v_errors = _sanitize_verdict(state, verdict)
    verdict = finalize_confidence(state, verdict)
    return {
        "verdict": verdict,
        "errors": v_errors,
        **log(state, "judge", ruling=verdict.ruling, confidence=verdict.confidence),
    }


def single_agent(state: CaseState) -> dict:
    """Config A baseline: one model, one pass, no adversarial structure."""
    pack = state["pack"]
    d = pack.dispute
    user = SINGLE_AGENT_USER.format(
        dispute_id=d.dispute_id,
        category=d.category,
        claimed_amount=d.claimed_amount,
        evidence_block=render_evidence(pack) + "\n\n" + render_trip(pack),
        policy_block=render_policies(pack),
    )
    if config.is_stub():
        verdict = stub_verdict()
    elif not model_ready():
        verdict = deterministic_verdict(pack)
    else:
        llm = get_llm("judge").with_structured_output(Verdict, method="function_calling")
        try:
            verdict = llm.invoke([SystemMessage(content=SINGLE_AGENT_SYSTEM), HumanMessage(content=user)])
        except (OpenAIInvalidRequestError, OpenAIRateLimitError):
            verdict = deterministic_verdict(pack)
    verdict, v_errors = _sanitize_verdict(state, verdict)
    verdict = finalize_confidence(state, verdict)
    return {
        "verdict": verdict,
        "errors": v_errors,
        **log(state, "single_agent", ruling=verdict.ruling, confidence=verdict.confidence),
    }
