"""
The tribunal workflow.

  Filing -> Docket -> Discovery -> [Motions] -> [Settlement]
         -> Pleadings (rider + driver, parallel)
         -> [Provisional verdict] -> Rebuttal (parallel) -> Judge -> Gate -> Judgment

Config A collapses to one node. Config B drops the fast paths and the debate.
Config C is the full tribunal. Same graph, flags decide the path — so the ablation
is a CLI flag, not a rewrite.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from ..agents.advocates import pleadings, rebuttal
from ..agents.judge import judge, provisional_verdict, single_agent
from ..agents.officers import motions, settlement
from ..core import config
from ..core.state import CaseState, Verdict, log
from ..tools.interface import get_tools


# ---------------------------------------------------------------- nodes

def docket(state: CaseState) -> dict:
    dispute_id = state["dispute_id"]
    pack = get_tools().get_evidence_pack(dispute_id)
    return {"pack": pack, **log(state, "docket", dispute=dispute_id, category=pack.dispute.category)}


def discovery(state: CaseState) -> dict:
    """
    Mutual discovery. Both sides are served the identical pack — fairness is enforced
    by construction (advocate_view redacts judge-only fields), not by prompt.
    """
    pack = state["pack"]
    flags = config.flags_for(state["config"])

    if flags.use_precedent and not pack.precedents:
        pack.precedents = get_tools().search_precedents(
            {"category": pack.dispute.category}, k=3
        )
    if flags.use_fraud and pack.fraud is None:
        pack.fraud = get_tools().get_fraud_signals(state["dispute_id"])

    return {
        "pack": pack,
        **log(
            state,
            "discovery",
            served_to="both parties (identical)",
            precedents=len(pack.precedents),
            fraud=bool(pack.fraud),
        ),
    }


def gate(state: CaseState) -> dict:
    """Confidence gate. Below threshold -> appellate review by a human."""
    verdict: Verdict = state.get("verdict") or Verdict(ruling="no_action", confidence=0.0)
    escalate = verdict.confidence < config.CONFIDENCE_THRESHOLD

    if config.CRITICAL_PRIORITY_ESCALATES and state["pack"].dispute.priority == "critical":
        escalate = True
        verdict.escalation_reason = "Priority critical — human review required regardless of confidence."

    if escalate:
        verdict.should_escalate = True
        verdict.escalation_reason = verdict.escalation_reason or (
            f"Confidence {verdict.confidence:.2f} below threshold "
            f"{config.CONFIDENCE_THRESHOLD:.2f}. Referred to appellate review."
        )

    return {
        "verdict": verdict,
        "escalated": escalate,
        **log(state, "gate", escalated=escalate, confidence=verdict.confidence),
    }


def judgment(state: CaseState) -> dict:
    v = state["verdict"]
    return log(
        state,
        "judgment",
        ruling=v.ruling,
        amount=v.amount,
        confidence=v.confidence,
        escalated=state.get("escalated", False),
    )


# ---------------------------------------------------------------- routing

def after_docket(state: CaseState) -> str:
    return "single_agent" if config.flags_for(state["config"]).use_advocates is False else "discovery"


def after_discovery(state: CaseState) -> str:
    f = config.flags_for(state["config"])
    if f.use_motions:
        return "motions"
    return "settlement" if f.use_settlement else "pleadings"


def after_motions(state: CaseState) -> str:
    if state.get("motion_outcome") in {"dismissed", "summary_judgment"}:
        return "gate"
    return "settlement" if config.flags_for(state["config"]).use_settlement else "pleadings"


def after_settlement(state: CaseState) -> str:
    return "gate" if state.get("settlement_accepted") else "pleadings"


def after_pleadings(state: CaseState) -> str:
    return "provisional_verdict" if config.flags_for(state["config"]).use_rebuttal else "judge"


# ---------------------------------------------------------------- graph

def build_graph():
    g = StateGraph(CaseState)

    g.add_node("docket", docket)
    g.add_node("discovery", discovery)
    g.add_node("motions", motions)
    g.add_node("settlement", settlement)
    # Both advocates run concurrently INSIDE this node (thread pool). Modelling them as
    # two LangGraph nodes fanning into one makes each read a stale snapshot of the other's
    # brief, which is exactly the contamination we are trying to avoid.
    g.add_node("pleadings", pleadings)
    g.add_node("provisional_verdict", provisional_verdict)
    g.add_node("rebuttal", rebuttal)
    g.add_node("judge", judge)
    g.add_node("single_agent", single_agent)
    g.add_node("gate", gate)
    g.add_node("judgment", judgment)

    g.set_entry_point("docket")

    g.add_conditional_edges(
        "docket",
        after_docket,
        {"single_agent": "single_agent", "discovery": "discovery"},
    )
    g.add_conditional_edges(
        "discovery",
        after_discovery,
        {"motions": "motions", "settlement": "settlement", "pleadings": "pleadings"},
    )
    g.add_conditional_edges(
        "motions",
        after_motions,
        {"gate": "gate", "settlement": "settlement", "pleadings": "pleadings"},
    )
    g.add_conditional_edges(
        "settlement",
        after_settlement,
        {"gate": "gate", "pleadings": "pleadings"},
    )

    g.add_conditional_edges(
        "pleadings",
        after_pleadings,
        {"provisional_verdict": "provisional_verdict", "judge": "judge"},
    )

    g.add_edge("provisional_verdict", "rebuttal")
    g.add_edge("rebuttal", "judge")

    g.add_edge("single_agent", "gate")
    g.add_edge("judge", "gate")
    g.add_edge("gate", "judgment")
    g.add_edge("judgment", END)

    return g.compile()


def run_case(dispute_id: str, cfg: str = "C") -> CaseState:
    state: CaseState = {
        "dispute_id": dispute_id,
        "config": cfg.upper(),
        "trace": [],
        "errors": [],
        "settlement_accepted": False,
    }
    return build_graph().invoke(state)
