"""
state.py — THE SHARED CONTRACT.

Person B (agents) drafts. Person A (data) reviews for "can I actually generate this?"
After both sign off, this file is FROZEN. Changing it means telling the other person.

Everything else in the repo depends on this file. Read it top to bottom once.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

# ============================================================================
# 1. EVIDENCE — what Person A's tool layer produces
# ============================================================================


class Stop(BaseModel):
    lat: float
    lng: float
    duration_sec: int
    at_km: float


class RouteAnalysis(BaseModel):
    """The derived, agent-readable route summary. Agents reason over THIS, not polylines."""

    trip_id: str
    optimal_distance_km: float
    actual_distance_km: float
    deviation_km: float
    deviation_pct: float
    duration_delta_min: float
    traffic_delay_min: float
    delay_attributable_to_traffic: bool
    rider_requested_detour: bool
    unexpected_stops: list[Stop] = Field(default_factory=list)
    num_stops: int = 0
    longest_stop_sec: int = 0
    deviation_justified_by: Optional[Literal["traffic", "rider_request", "road_closure"]] = None
    route_efficiency_score: float


class NoShowFacts(BaseModel):
    driver_min_distance_to_pickup_m: float
    wait_time_min: float
    rider_gps_approaching: bool
    no_show_fee_charged: float


class ChatMessage(BaseModel):
    message_id: str
    sender: Literal["rider", "driver", "system"]
    timestamp: str
    message: str
    sentiment: float = 0.0
    contains_threat: bool = False
    contains_agreement: bool = False


class FareBreakdown(BaseModel):
    trip_id: str
    base_fare: float
    distance_fare: float
    time_fare: float
    booking_fee: float
    surge_multiplier: float
    surge_amount: float
    tolls: float = 0.0
    promo_code: Optional[str] = None
    promo_discount: float = 0.0
    total_charged: float
    estimated_fare_at_booking: float
    surge_at_booking: float
    currency: str = "USD"


class PartyProfile(BaseModel):
    party_id: str
    role: Literal["rider", "driver"]
    name: str
    account_age_days: int
    total_trips: int
    avg_rating: float
    dispute_count_90d: int
    dispute_win_rate: float
    cancellation_rate: float
    is_flagged_abuse: bool = False


class Trip(BaseModel):
    trip_id: str
    rider_id: str
    driver_id: str
    requested_at: str
    started_at: str
    ended_at: str
    pickup_address: str
    dropoff_address: Optional[str] = None  # null on no_show / cancelled trips
    status: Literal["completed", "cancelled_by_rider", "cancelled_by_driver", "no_show"]
    estimated_distance_km: float
    actual_distance_km: float
    estimated_duration_min: float
    actual_duration_min: float
    city: str = "San Francisco"
    traffic_conditions: Literal["light", "moderate", "heavy"] = "moderate"


class Dispute(BaseModel):
    dispute_id: str
    trip_id: str
    filed_by: Literal["rider", "driver"]
    filed_at: str
    category: Literal["route_deviation", "no_show", "fare_surge", "damage_cleaning"]
    claimed_amount: float
    rider_statement: Optional[str] = None
    driver_statement: Optional[str] = None
    priority: Literal["low", "medium", "high", "critical"] = "medium"


class Policy(BaseModel):
    policy_id: str
    category: str
    title: str
    rule_text: str
    conditions: dict[str, Any] = Field(default_factory=dict)
    remedy_type: Literal["partial_refund", "full_refund", "no_action", "compensate_driver"]
    remedy_formula: Optional[str] = None
    version: str = "v1"


class Precedent(BaseModel):
    precedent_id: str
    category: str
    fact_summary: str
    key_facts: dict[str, Any] = Field(default_factory=dict)
    ruling: str
    reasoning: str
    cited_policy_id: str
    confidence: float
    decided_at: str


class FraudSignals(BaseModel):
    dispute_id: str
    rider_claims_30d: int = 0
    driver_claims_30d: int = 0
    rider_repeat_claimer: bool = False
    collusion_score: float = 0.0
    device_fingerprint_shared: bool = False
    photo_authenticity_score: Optional[float] = None
    risk_level: Literal["low", "medium", "high"] = "low"


class EvidencePack(BaseModel):
    """The full case file. NOTE: contains judge-only fields — see advocate_view()."""

    dispute: Dispute
    trip: Trip
    fare: FareBreakdown
    rider: PartyProfile
    driver: PartyProfile
    route: Optional[RouteAnalysis] = None
    no_show: Optional[NoShowFacts] = None
    chat: list[ChatMessage] = Field(default_factory=list)
    policies: list[Policy] = Field(default_factory=list)

    # ---- JUDGE ONLY (never served to advocates) ----
    fraud: Optional[FraudSignals] = None
    precedents: list[Precedent] = Field(default_factory=list)


def advocate_view(pack: EvidencePack) -> EvidencePack:
    """
    Procedural fairness enforced by construction, not by prompt.

    Both advocates receive the IDENTICAL, REDACTED pack. Fraud signals are a
    presentence report, not discovery — hand them to an advocate and every fare
    dispute degenerates into a character trial.
    """
    data = pack.model_dump()
    data["fraud"] = None
    data["precedents"] = []
    return EvidencePack(**data)


# ============================================================================
# 2. ARGUMENT — what the advocates produce
# ============================================================================


class Claim(BaseModel):
    """A single assertion. evidence_refs are mandatory — uncited claims get struck."""

    text: str
    evidence_refs: list[str] = Field(default_factory=list)
    policy_refs: list[str] = Field(default_factory=list)

    def is_cited(self) -> bool:
        return bool(self.evidence_refs) or bool(self.policy_refs)


class ClaimAssessment(BaseModel):
    """The judge's reaction to a single advocate claim — the inner monologue."""

    side: Literal["rider", "driver"]
    claim_text: str
    verdict: Literal["accepted", "rejected", "questioned", "struck"]
    reasoning: str
    evidence_checked: list[str] = Field(default_factory=list)


class CaseBrief(BaseModel):
    side: Literal["rider", "driver"]
    theory: str
    claims: list[Claim] = Field(default_factory=list)
    concessions: list[Claim] = Field(default_factory=list)
    requested_remedy: str
    requested_amount: float = 0.0
    rebuttal: Optional[str] = None


# ============================================================================
# 3. JUDGMENT — what the Judge produces
# ============================================================================


class ConfidenceFactor(BaseModel):
    factor: str
    delta: float


class Verdict(BaseModel):
    ruling: Literal["refund_rider", "refund_driver", "no_action", "dismissed"]
    amount: float = 0.0
    cited_policy_id: Optional[str] = None

    findings_of_fact: list[Claim] = Field(default_factory=list)
    conclusions_of_law: list[str] = Field(default_factory=list)
    remedy_rationale: str = ""

    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)

    confidence: float = 0.0
    confidence_factors: list[ConfidenceFactor] = Field(default_factory=list)

    # provisional_* powers the rebuttal flip-rate test. Do not remove.
    provisional_ruling: Optional[str] = None
    provisional_confidence: Optional[float] = None

    should_escalate: bool = False
    escalation_reason: Optional[str] = None


# ============================================================================
# 4. GRAPH STATE — what flows through LangGraph
# ============================================================================


class CaseState(TypedDict, total=False):
    dispute_id: str
    config: str  # "A" single-agent | "B" three-agent | "C" full tribunal

    pack: EvidencePack
    rider_brief: CaseBrief
    driver_brief: CaseBrief

    motion_outcome: str  # "none" | "dismissed" | "summary_judgment"
    motion_reason: str
    settlement_offer: Optional[float]
    settlement_accepted: bool

    provisional: Verdict
    verdict: Verdict
    escalated: bool

    # Reducers, not plain lists: concurrent nodes each emit a delta and LangGraph
    # concatenates them. Without operator.add, one node's writes silently overwrite
    # the other's and entries vanish from the trace.
    trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]


def log(state: CaseState, stage: str, **payload: Any) -> dict[str, Any]:
    """
    Returns a state DELTA, it does not mutate. Merge it into your node's return:

        return {**log(state, "stage", foo=1), "my_key": value}

    Mutating state["trace"] in place does not survive across nodes.
    """
    return {"trace": [{"stage": stage, **payload}]}
