"""
apply_policy() — THE ANSWER KEY.

~200 lines of plain Python implementing the policy book. This is what "correct" means
in this repo, and it is deliberately independent of the system under test:

  - The agents never see this file.
  - No LLM is involved in producing a label.

Both properties are load-bearing. If an LLM wrote the labels, our eval would only
measure "how often does the Judge agree with GPT-4".

The generator calls this to LABEL cases, and the eval harness calls it to SCORE them.
Same function both times, so the two can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..core.state import EvidencePack

# ------------------------------------------------------------------ thresholds
# Single source of truth. These numbers appear in the policy text, the generator,
# the Motions Officer and the eval. Change them here only.
DEVIATION_FLOOR = 0.05  # below this a deviation claim is not justiciable
DEVIATION_THRESHOLD = 0.15  # POL-4.3
TRAFFIC_SHARE_FULL = 0.80  # >= this share of delay attributable to traffic => excused
TRAFFIC_SHARE_PARTIAL = 0.40  # >= this but < FULL => genuinely uncertain

NO_SHOW_ABSENT_M = 100.0  # driver further than this => fee refunded (POL-2.1a)
NO_SHOW_PRESENT_M = 50.0  # driver within this => fee stands (POL-2.1b)
NO_SHOW_WAIT_MIN = 5.0
NO_SHOW_WAIT_GREY_MIN = 3.0

CLEANING_CAP = 25.00
PHOTO_GENUINE = 0.70
PHOTO_FABRICATED = 0.40

SURGE_TOLERANCE = 0.01
FARE_ESTIMATE_TOLERANCE = 0.05  # an estimate is not a quote; 5% drift is normal

BAND_MIDPOINT = {"high": 0.92, "medium": 0.72, "low": 0.48}


@dataclass
class Truth:
    """One case's answer key."""

    ruling: str
    amount: float
    policy_id: Optional[str]
    confidence_band: str  # high | medium | low
    should_escalate: bool
    rationale: str
    confidence_target: float = 0.0  # band midpoint, for the calibration check
    pair_id: Optional[str] = None
    pair_role: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.confidence_target:
            self.confidence_target = BAND_MIDPOINT.get(self.confidence_band, 0.5)

    def as_dict(self) -> dict:
        return {
            "correct_ruling": self.ruling,
            "correct_refund_amount": round(self.amount, 2),
            "correct_policy_id": self.policy_id,
            "expected_confidence_band": self.confidence_band,
            "should_escalate": self.should_escalate,
            "rationale": self.rationale,
            "confidence_target": self.confidence_target,
            "pair_id": self.pair_id,
            "pair_role": self.pair_role,
        }


def band_for(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


def _escalate(reason: str, policy_id: str | None, likely_ruling: str, likely_amount: float) -> Truth:
    """
    The record does not determine the outcome.

    Note what we do and don't claim: should_escalate is TRUE and that is what gets
    scored. `ruling`/`amount` are recorded as the most likely human outcome purely for
    reference — the eval deliberately does NOT score direction on escalated cases,
    because "correct" is not defined for a case we just said is undeterminable.
    """
    return Truth(
        ruling=likely_ruling,
        amount=likely_amount,
        policy_id=policy_id,
        confidence_band="low",
        should_escalate=True,
        rationale=reason,
    )


# ------------------------------------------------------------------ categories


def _route_deviation(pack: EvidencePack) -> Truth:
    r = pack.route
    claimed = pack.dispute.claimed_amount
    fare = pack.fare

    if r is None:
        return _escalate(
            "Route telemetry missing; deviation cannot be computed from the record.",
            None,
            "no_action",
            0.0,
        )

    pct = r.deviation_pct
    excess_km = max(0.0, r.deviation_km)
    refund = round(fare.distance_fare * (excess_km / r.actual_distance_km), 2) if r.actual_distance_km else 0.0
    refund = min(refund, claimed)  # you never award more than was asked for

    if r.rider_requested_detour:
        return Truth(
            ruling="no_action",
            amount=0.0,
            policy_id="POL-4.5",
            confidence_band="high",
            should_escalate=False,
            rationale=(
                f"Deviation of {pct:.1%} is rider-requested. POL-4.5: a detour the rider "
                f"asked for is not an unjustified deviation, regardless of magnitude."
            ),
        )

    if pct < DEVIATION_FLOOR:
        return Truth(
            ruling="dismissed",
            amount=0.0,
            policy_id="POL-4.3",
            confidence_band="high",
            should_escalate=False,
            rationale=(
                f"Deviation of {pct:.1%} is below the {DEVIATION_FLOOR:.0%} floor at which a "
                f"deviation claim is justiciable. No policy provision is engaged."
            ),
        )

    if pct < DEVIATION_THRESHOLD:
        band = "high" if pct < 0.12 else "medium"
        return Truth(
            ruling="no_action",
            amount=0.0,
            policy_id="POL-4.3",
            confidence_band=band,
            should_escalate=False,
            rationale=(
                f"Deviation of {pct:.1%} is below the {DEVIATION_THRESHOLD:.0%} threshold in "
                f"POL-4.3. No refund."
            ),
        )

    # >= threshold from here
    share = (r.traffic_delay_min / r.duration_delta_min) if r.duration_delta_min > 0 else 0.0

    if share >= TRAFFIC_SHARE_FULL:
        return Truth(
            ruling="no_action",
            amount=0.0,
            policy_id="POL-4.3",
            confidence_band="high" if share >= 1.0 else "medium",
            should_escalate=False,
            rationale=(
                f"Deviation of {pct:.1%} exceeds the threshold, but {r.traffic_delay_min:.0f} of "
                f"{r.duration_delta_min:.0f} excess minutes ({share:.0%}) are attributable to "
                f"documented traffic. POL-4.3 exception applies."
            ),
        )

    if share >= TRAFFIC_SHARE_PARTIAL:
        return _escalate(
            f"Deviation {pct:.1%} exceeds threshold, but traffic accounts for only {share:.0%} "
            f"of the excess ({r.traffic_delay_min:.0f} of {r.duration_delta_min:.0f} min). "
            f"Partially excused under POL-4.3 — not determinable on the record.",
            "POL-4.3",
            "refund_rider",
            round(refund * (1 - share), 2),
        )

    return Truth(
        ruling="refund_rider",
        amount=refund,
        policy_id="POL-4.3",
        confidence_band="high",
        should_escalate=False,
        rationale=(
            f"Deviation {pct:.1%} exceeds the {DEVIATION_THRESHOLD:.0%} threshold. Traffic "
            f"accounts for {r.traffic_delay_min:.0f} of {r.duration_delta_min:.0f} excess minutes "
            f"and no rider detour was requested. POL-4.3: {fare.distance_fare} * "
            f"({excess_km:.2f} / {r.actual_distance_km:.2f}) = {refund:.2f}."
        ),
    )


def _no_show(pack: EvidencePack) -> Truth:
    n = pack.no_show
    if n is None:
        return _escalate("No-show facts absent from the record.", None, "no_action", 0.0)

    fee = n.no_show_fee_charged
    dist = n.driver_min_distance_to_pickup_m
    wait = n.wait_time_min

    if dist > NO_SHOW_ABSENT_M:
        return Truth(
            ruling="refund_rider",
            amount=round(fee, 2),
            policy_id="POL-2.1a",
            confidence_band="high",
            should_escalate=False,
            rationale=(
                f"Driver never came within {NO_SHOW_ABSENT_M:.0f}m of the pin (closest approach "
                f"{dist:.0f}m). POL-2.1a: no-show fee of ${fee:.2f} refunded."
            ),
        )

    if dist <= NO_SHOW_PRESENT_M and wait >= NO_SHOW_WAIT_MIN and not n.rider_gps_approaching:
        return Truth(
            ruling="no_action",
            amount=0.0,
            policy_id="POL-2.1b",
            confidence_band="high",
            should_escalate=False,
            rationale=(
                f"Driver within {dist:.0f}m, waited {wait:.0f} min, rider GPS static. All "
                f"POL-2.1b conditions met — the fee stands."
            ),
        )

    # Grey band. Genuinely undeterminable: this is the case the demo opens on.
    reasons = []
    if NO_SHOW_PRESENT_M < dist <= NO_SHOW_ABSENT_M:
        reasons.append(f"closest approach {dist:.0f}m falls in the {NO_SHOW_PRESENT_M:.0f}-{NO_SHOW_ABSENT_M:.0f}m band")
    if NO_SHOW_WAIT_GREY_MIN <= wait < NO_SHOW_WAIT_MIN:
        reasons.append(f"wait of {wait:.0f} min falls in the {NO_SHOW_WAIT_GREY_MIN:.0f}-{NO_SHOW_WAIT_MIN:.0f} min band")
    if n.rider_gps_approaching:
        reasons.append("rider GPS shows movement toward the pickup, but arrival is unconfirmed")
    if not reasons:
        reasons.append("facts do not meet any POL-2.1 condition cleanly")

    return _escalate(
        "Grey band: " + "; ".join(reasons) + ". POL-2.1b does not determine the outcome — "
        f"referred to appellate review. (Most likely human outcome: refund of ${fee:.2f}.)",
        "POL-2.1b",
        "refund_rider",
        round(fee, 2),
    )


def _fare_surge(pack: EvidencePack) -> Truth:
    f = pack.fare
    claimed = pack.dispute.claimed_amount

    if f.surge_multiplier > f.surge_at_booking + SURGE_TOLERANCE:
        owed = f.surge_amount * (1 - (f.surge_at_booking / f.surge_multiplier))
        amount = round(min(owed, claimed), 2)
        return Truth(
            ruling="refund_rider",
            amount=amount,
            policy_id="POL-3.2",
            confidence_band="medium",
            should_escalate=False,
            rationale=(
                f"Charged surge {f.surge_multiplier:.2f}x exceeds the {f.surge_at_booking:.2f}x "
                f"disclosed at booking. POL-3.2: rider owes only the disclosed surge; "
                f"excess of ${amount:.2f} refunded."
            ),
        )

    drift = (f.total_charged - f.estimated_fare_at_booking) / max(f.estimated_fare_at_booking, 1e-9)
    if drift > FARE_ESTIMATE_TOLERANCE:
        # An estimate is not a quote. Duration overrun is priced normally.
        return _escalate(
            f"Charged ${f.total_charged:.2f} exceeds the booking estimate of "
            f"${f.estimated_fare_at_booking:.2f} by {drift:.0%}, but surge was disclosed and "
            f"consistent. POL-3.1: an estimate is not a binding quote. Outcome turns on whether "
            f"the overrun is explained by trip duration — not determinable on the record.",
            "POL-3.1",
            "no_action",
            0.0,
        )

    return Truth(
        ruling="no_action",
        amount=0.0,
        policy_id="POL-3.1",
        confidence_band="high",
        should_escalate=False,
        rationale=(
            f"Surge of {f.surge_multiplier:.2f}x matches the {f.surge_at_booking:.2f}x disclosed "
            f"at booking and the total is within {FARE_ESTIMATE_TOLERANCE:.0%} of the estimate. "
            f"POL-3.1: no refund."
        ),
    )


def _damage_cleaning(pack: EvidencePack) -> Truth:
    claimed = pack.dispute.claimed_amount
    score = pack.fraud.photo_authenticity_score if pack.fraud else None

    if score is None:
        return _escalate(
            "Damage claim unsupported by a verifiable photograph. No authenticity score on the "
            "record — requires human review.",
            "POL-5.1",
            "no_action",
            0.0,
        )
    if score >= PHOTO_GENUINE:
        return Truth(
            ruling="refund_driver",
            amount=round(min(claimed, CLEANING_CAP), 2),
            policy_id="POL-5.1",
            confidence_band="medium",
            should_escalate=False,
            rationale=(
                f"Photograph verified (authenticity {score:.2f}, matches trip timestamp). "
                f"POL-5.1: cleaning fee awarded, capped at ${CLEANING_CAP:.2f}."
            ),
        )
    if score < PHOTO_FABRICATED:
        return Truth(
            ruling="dismissed",
            amount=0.0,
            policy_id="POL-5.2",
            confidence_band="high",
            should_escalate=False,
            rationale=(
                f"Photograph authenticity {score:.2f} is below the {PHOTO_FABRICATED:.2f} floor "
                f"— likely fabricated or AI-generated. POL-5.2: claim dismissed."
            ),
        )
    return _escalate(
        f"Photograph authenticity {score:.2f} is inconclusive. Not determinable on the record.",
        "POL-5.1",
        "no_action",
        0.0,
    )


_HANDLERS = {
    "route_deviation": _route_deviation,
    "no_show": _no_show,
    "fare_surge": _fare_surge,
    "damage_cleaning": _damage_cleaning,
}


def apply_policy(pack: EvidencePack, pair_id: str | None = None, pair_role: str | None = None) -> Truth:
    """The one function that defines 'correct'. Deterministic, auditable, no LLM."""
    handler = _HANDLERS.get(pack.dispute.category)
    if handler is None:
        return _escalate(f"Unknown category '{pack.dispute.category}'.", None, "no_action", 0.0)

    truth = handler(pack)

    # Safety-critical disputes always go to a human, whatever the arithmetic says.
    if pack.dispute.priority == "critical":
        truth.should_escalate = True
        truth.confidence_band = "low"
        truth.confidence_target = BAND_MIDPOINT["low"]
        truth.rationale += " Priority critical — human review required regardless."

    truth.pair_id = pair_id
    truth.pair_role = pair_role
    return truth


def truth_for_pack(pack: EvidencePack, **kw) -> dict:
    return apply_policy(pack, **kw).as_dict()


# Kept separate so the recomputation check can reuse the same field namespace.
def policy_formula_namespace(pack: EvidencePack) -> dict[str, float]:
    """Flat namespace of the numbers a remedy_formula may reference."""
    ns: dict[str, float] = {}
    for obj in (pack.fare, pack.route, pack.no_show):
        if obj is None:
            continue
        for key, val in obj.model_dump().items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                ns[key] = float(val)
    ns["claimed_amount"] = float(pack.dispute.claimed_amount)
    return ns
