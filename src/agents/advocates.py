"""Rider Advocate and Driver Advocate.

They run CONCURRENTLY (thread pool inside one node) and receive the IDENTICAL redacted
evidence pack. Same data, opposed objectives, separate context windows — that separation
is the whole mechanism. See Why_Multi_Agent.md.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage

from ..core import config
from ..core.llm import get_llm, model_ready
from langchain_openai.chat_models.base import OpenAIInvalidRequestError, OpenAIRateLimitError
from ..core.state import (
    CaseBrief,
    CaseState,
    EvidencePack,
    advocate_view,
    log,
)
from .prompts import (
    ADVOCATE_SYSTEM,
    ADVOCATE_USER,
    REBUTTAL_SYSTEM,
    REBUTTAL_USER,
    render_brief,
    render_evidence,
    render_policies,
    render_trip,
    stub_brief,
)
from .rule_based import deterministic_brief, deterministic_rebuttal

ROLE = {
    "rider": "counsel for the rider (claimant)",
    "driver": "counsel for the driver (respondent)",
}
SIDES = ("rider", "driver")

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
}



def _advocate_user(pack: EvidencePack, side: str) -> str:
    d = pack.dispute
    return ADVOCATE_USER.format(
        dispute_id=d.dispute_id,
        filed_by=d.filed_by,
        category=d.category,
        claimed_amount=d.claimed_amount,
        rider_statement=d.rider_statement or "(none)",
        driver_statement=d.driver_statement or "(none)",
        trip_block=render_trip(pack),
        evidence_block=render_evidence(pack),
        policy_block=render_policies(pack),
        side=side,
    )


def _valid_evidence_ref(pack: EvidencePack, ref: str) -> bool:
    tokens = [t for t in ref.split(".") if t]
    if not tokens:
        return False

    if tokens[0] in {pack.trip.trip_id, pack.dispute.dispute_id, pack.rider.party_id, pack.driver.party_id}:
        tokens = tokens[1:]
    if not tokens:
        return False

    section = SECTION_ALIASES.get(tokens[0].lower())
    if section is None:
        return False

    obj = getattr(pack, section, None)
    if obj is None:
        return False

    if len(tokens) == 1:
        return True

    if section == "chat":
        ids = {m.message_id for m in pack.chat}
        return tokens[1] in ids

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


def _sanitize_brief(brief: CaseBrief, pack: EvidencePack, side: str) -> tuple[CaseBrief, int, int]:
    valid_policy_ids = {p.policy_id for p in pack.policies}
    dropped_uncited = 0
    dropped_badref = 0

    cleaned = []
    for claim in brief.claims:
        had_any_ref = bool(claim.evidence_refs or claim.policy_refs)
        claim.evidence_refs = [r for r in claim.evidence_refs if _valid_evidence_ref(pack, r)]
        claim.policy_refs = [r for r in claim.policy_refs if r in valid_policy_ids]

        if not claim.is_cited():
            if had_any_ref:
                dropped_badref += 1
            else:
                dropped_uncited += 1
            continue

        cleaned.append(claim)

    brief.claims = cleaned
    brief.side = side  # type: ignore[assignment]
    brief.requested_amount = max(0.0, min(float(brief.requested_amount or 0.0), pack.dispute.claimed_amount))
    return brief, dropped_uncited, dropped_badref


def run_advocate(state: CaseState, side: str) -> tuple[CaseBrief, int, int]:
    """
    One advocate. Called twice, concurrently, with identical redacted evidence.
    Returns (brief, num_uncited_claims_struck, num_badref_claims_struck).
    """
    pack = advocate_view(state["pack"])  # identical for both sides; fraud redacted
    system = ADVOCATE_SYSTEM.format(role=ROLE[side])
    user = _advocate_user(pack, side)

    if config.is_stub():
        brief = stub_brief(side)
    elif not model_ready() or config.flags_for(state["config"]).use_advocates is False:
        brief = deterministic_brief(pack, side)
    else:
        llm = get_llm("advocate").with_structured_output(CaseBrief, method="function_calling")
        try:
            brief: CaseBrief = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except (OpenAIInvalidRequestError, OpenAIRateLimitError):
            # Common provider failures: structured-output truncation and OTPM throttling.
            # Fall back to deterministic brief instead of failing the whole case.
            brief = deterministic_brief(pack, side)

    brief, uncited, badref = _sanitize_brief(brief, pack, side)
    return brief, uncited, badref


def run_rebuttal(state: CaseState, side: str) -> CaseBrief:
    """Right of reply. Powers the flip-rate test (provisional vs final verdict)."""
    yours: CaseBrief = state[f"{side}_brief"]
    other = "driver" if side == "rider" else "rider"
    theirs: CaseBrief = state[f"{other}_brief"]

    system = REBUTTAL_SYSTEM.format(role=ROLE[side])
    user = REBUTTAL_USER.format(
        your_brief=render_brief(yours),
        opponent=other,
        opponent_brief=render_brief(theirs),
    )

    if config.is_stub():
        reply = "[STUB] placeholder rebuttal."
    elif not model_ready():
        reply = deterministic_rebuttal(side, yours, theirs)
    else:
        llm = get_llm("advocate")
        try:
            reply = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content
        except OpenAIRateLimitError:
            reply = deterministic_rebuttal(side, yours, theirs)

    yours.rebuttal = str(reply)
    return yours


def _parallel(fn, state: CaseState):  # noqa: ANN001, ANN201
    """Both sides at once. Latency is one round-trip, not two."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {side: pool.submit(fn, state, side) for side in SIDES}
        return {side: fut.result() for side, fut in futures.items()}


def pleadings(state: CaseState) -> dict:
    """Both advocates file simultaneously, each in its own context window."""
    results = _parallel(run_advocate, state)

    delta: dict = {
        "rider_brief": results["rider"][0],
        "driver_brief": results["driver"][0],
        "trace": [],
        "errors": [],
    }
    for side in SIDES:
        brief, struck_uncited, struck_badref = results[side]
        delta["trace"] += log(
            state, f"{side}_advocate", theory=brief.theory, claims=len(brief.claims)
        )["trace"]
        if struck_uncited:
            delta["errors"].append(f"{side}_advocate: struck {struck_uncited} uncited claim(s)")
        if struck_badref:
            delta["errors"].append(f"{side}_advocate: struck {struck_badref} invalid-reference claim(s)")
    return delta


def rebuttal(state: CaseState) -> dict:
    """Both advocates reply simultaneously."""
    briefs = _parallel(run_rebuttal, state)
    delta: dict = {"rider_brief": briefs["rider"], "driver_brief": briefs["driver"], "trace": []}
    for side in SIDES:
        delta["trace"] += log(
            state, f"{side}_rebuttal", chars=len(briefs[side].rebuttal or "")
        )["trace"]
    return delta
