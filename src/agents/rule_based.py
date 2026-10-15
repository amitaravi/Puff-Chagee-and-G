"""Deterministic fallback reasoning for environments without model keys.

This is NOT the default production path. It exists so the full pipeline can still run
end-to-end when no LLM key is available, and to keep the demo unblocked.
"""

from __future__ import annotations

from ..core.state import CaseBrief, Claim, EvidencePack, Verdict
from ..policy.apply_policy import BAND_MIDPOINT, apply_policy


def _remedy_for(side: str, ruling: str) -> str:
    if side == "rider":
        if ruling == "refund_rider":
            return "refund_rider"
        if ruling == "dismissed":
            return "dismiss_case"
        return "no_action"
    # driver side
    if ruling == "refund_driver":
        return "refund_driver"
    if ruling == "dismissed":
        return "dismiss_case"
    return "no_action"


def _route_claims(pack: EvidencePack) -> list[Claim]:
    r = pack.route
    if not r:
        return [Claim(text="Route telemetry is missing from the record.", evidence_refs=[f"{pack.trip.trip_id}.trip.status"])]
    tid = pack.trip.trip_id
    claims = [
        Claim(
            text=(
                f"Route deviation is {r.deviation_pct:.1%} ({r.deviation_km:.2f} km over the optimal route)."
            ),
            evidence_refs=[f"{tid}.route_analysis.deviation_pct", f"{tid}.route_analysis.deviation_km"],
        ),
        Claim(
            text=(
                f"Traffic explains {r.traffic_delay_min:.1f} of {r.duration_delta_min:.1f} extra minutes."
            ),
            evidence_refs=[f"{tid}.route_analysis.traffic_delay_min", f"{tid}.route_analysis.duration_delta_min"],
        ),
        Claim(
            text=f"Rider-requested detour flag is {r.rider_requested_detour}.",
            evidence_refs=[f"{tid}.route_analysis.rider_requested_detour"],
        ),
    ]
    return claims


def _no_show_claims(pack: EvidencePack) -> list[Claim]:
    n = pack.no_show
    tid = pack.trip.trip_id
    if not n:
        return [Claim(text="No-show telemetry is missing from the record.", evidence_refs=[f"{tid}.trip.status"])]
    return [
        Claim(
            text=(
                f"Closest driver distance to pickup was {n.driver_min_distance_to_pickup_m:.1f}m."
            ),
            evidence_refs=[f"{tid}.no_show.driver_min_distance_to_pickup_m"],
        ),
        Claim(
            text=f"Driver waited {n.wait_time_min:.1f} minutes.",
            evidence_refs=[f"{tid}.no_show.wait_time_min"],
        ),
        Claim(
            text=f"Rider GPS approaching flag is {n.rider_gps_approaching}.",
            evidence_refs=[f"{tid}.no_show.rider_gps_approaching"],
        ),
    ]


def _fare_claims(pack: EvidencePack) -> list[Claim]:
    f = pack.fare
    tid = pack.trip.trip_id
    return [
        Claim(
            text=(
                f"Charged total is {f.total_charged:.2f} vs booking estimate {f.estimated_fare_at_booking:.2f}."
            ),
            evidence_refs=[f"{tid}.fare.total_charged", f"{tid}.fare.estimated_fare_at_booking"],
        ),
        Claim(
            text=(
                f"Applied surge is {f.surge_multiplier:.2f}x vs booking surge {f.surge_at_booking:.2f}x."
            ),
            evidence_refs=[f"{tid}.fare.surge_multiplier", f"{tid}.fare.surge_at_booking"],
        ),
    ]


def deterministic_brief(pack: EvidencePack, side: str) -> CaseBrief:
    truth = apply_policy(pack)
    category = pack.dispute.category

    if category == "route_deviation":
        claims = _route_claims(pack)
    elif category == "no_show":
        claims = _no_show_claims(pack)
    else:
        claims = _fare_claims(pack)

    policy_refs = [truth.policy_id] if truth.policy_id else []
    for c in claims:
        c.policy_refs = [*policy_refs]

    favourable = (
        (side == "rider" and truth.ruling == "refund_rider")
        or (side == "driver" and truth.ruling == "refund_driver")
        or (truth.ruling in {"no_action", "dismissed"} and side != pack.dispute.filed_by)
    )

    if favourable:
        theory = (
            f"The documentary record supports a {truth.ruling} outcome under {truth.policy_id or 'the applicable policy'}"
        )
        requested_amount = truth.amount
        concessions: list[Claim] = []
    else:
        theory = (
            "The record contains points that favour the opponent; the requested relief should be narrowed accordingly."
        )
        requested_amount = 0.0
        concessions = [
            Claim(
                text="I concede the strongest record-backed point raised by the opposing side.",
                policy_refs=[truth.policy_id] if truth.policy_id else [],
            )
        ]

    return CaseBrief(
        side=side,  # type: ignore[arg-type]
        theory=theory,
        claims=claims,
        concessions=concessions,
        requested_remedy=_remedy_for(side, truth.ruling),
        requested_amount=round(min(requested_amount, pack.dispute.claimed_amount), 2),
    )


def deterministic_rebuttal(side: str, your_brief: CaseBrief, opp_brief: CaseBrief) -> str:
    if your_brief.requested_remedy == opp_brief.requested_remedy:
        return (
            f"{side.title()} counsel narrows the dispute: both briefs now converge on "
            f"{your_brief.requested_remedy}."
        )
    return (
        f"{side.title()} counsel contests the opponent's remedy request and maintains "
        f"{your_brief.requested_remedy} based on the cited record."
    )


def _verdict_findings(pack: EvidencePack) -> list[Claim]:
    tid = pack.trip.trip_id
    if pack.dispute.category == "route_deviation" and pack.route:
        r = pack.route
        return [
            Claim(
                text=f"Route deviation measured {r.deviation_pct:.1%} ({r.deviation_km:.2f}km).",
                evidence_refs=[f"{tid}.route_analysis.deviation_pct", f"{tid}.route_analysis.deviation_km"],
            ),
            Claim(
                text=f"Traffic accounted for {r.traffic_delay_min:.1f}/{r.duration_delta_min:.1f} extra minutes.",
                evidence_refs=[f"{tid}.route_analysis.traffic_delay_min", f"{tid}.route_analysis.duration_delta_min"],
            ),
        ]
    if pack.dispute.category == "no_show" and pack.no_show:
        n = pack.no_show
        return [
            Claim(
                text=f"Closest driver distance to pickup: {n.driver_min_distance_to_pickup_m:.1f}m.",
                evidence_refs=[f"{tid}.no_show.driver_min_distance_to_pickup_m"],
            ),
            Claim(
                text=f"Driver wait time: {n.wait_time_min:.1f} minutes.",
                evidence_refs=[f"{tid}.no_show.wait_time_min"],
            ),
        ]
    return [
        Claim(
            text=f"Total fare charged: {pack.fare.total_charged:.2f}.",
            evidence_refs=[f"{tid}.fare.total_charged"],
        )
    ]


def deterministic_verdict(pack: EvidencePack) -> Verdict:
    truth = apply_policy(pack)
    confidence = BAND_MIDPOINT.get(truth.confidence_band, 0.5)
    findings = _verdict_findings(pack)

    return Verdict(
        ruling=truth.ruling,  # type: ignore[arg-type]
        amount=round(min(truth.amount, pack.dispute.claimed_amount), 2),
        cited_policy_id=truth.policy_id,
        findings_of_fact=findings,
        conclusions_of_law=[truth.rationale],
        remedy_rationale=(
            f"Applied {truth.policy_id or 'the applicable policy'} and reached {truth.ruling} "
            f"with remedy amount {round(min(truth.amount, pack.dispute.claimed_amount), 2):.2f}."
        ),
        confidence=confidence,
        should_escalate=truth.should_escalate,
        escalation_reason=(truth.rationale if truth.should_escalate else None),
    )
