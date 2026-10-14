"""
Motions Officer and Settlement Officer — the two fast paths.

These are what turn a 5-minute median from aspiration into arithmetic: expect 40-60%
of filings to close with NO adjudication at all. They are also the cheapest phases to
build (mostly rules over the evidence pack), so do not leave them to last.

Both are rule-based in the MVP. An LLM pass is optional and probably unnecessary.
"""

from __future__ import annotations

from ..core.state import CaseState, Verdict, log
from ..policy.apply_policy import (
    DEVIATION_FLOOR,
    DEVIATION_THRESHOLD,
    NO_SHOW_ABSENT_M,
    NO_SHOW_PRESENT_M,
    NO_SHOW_WAIT_GREY_MIN,
    NO_SHOW_WAIT_MIN,
    TRAFFIC_SHARE_FULL,
    TRAFFIC_SHARE_PARTIAL,
)

# Frivolous if the claim is contradicted by a bright-line figure in the record.
FRIVOLOUS_DEVIATION_PCT = DEVIATION_FLOOR
SETTLEMENT_CAP = 8.00  # nuisance-value claims: settle rather than adjudicate
SETTLEMENT_FRACTION = 0.6

# Continuous remedies (a partial refund is meaningful) are settleable.
# Binary remedies (does the no-show fee stand, yes or no?) are not — you do not
# settle half a yes/no question.
SETTLEABLE_CATEGORIES = {"route_deviation", "fare_surge"}

# Deviation is "grey" when it sits near the threshold, or when traffic accounts for
# a substantial but INCOMPLETE share of the delay.
#
# Both bounds are imported from the policy engine rather than re-declared, because a
# disagreement between the two is exactly how settlement ends up paying money on a
# case the policy says is clear-cut. TRAFFIC_SHARE_FULL is the line above which the
# deviation is excused outright — that is a determinable outcome, not an uncertain one.
NEAR_THRESHOLD_BAND = (DEVIATION_THRESHOLD - 0.01, DEVIATION_THRESHOLD + 0.01)
AMBIGUOUS_TRAFFIC_SHARE = (TRAFFIC_SHARE_PARTIAL, TRAFFIC_SHARE_FULL)


def is_ambiguous(pack) -> bool:
    """
    Deterministic read on whether the outcome is genuinely uncertain.

    This matters more than it looks: settlement must fire ONLY on uncertain cases.
    Settle a clear-cut case and you (a) pay money you don't owe and (b) never show
    the adversarial hearing in the demo.
    """
    category = pack.dispute.category

    if category == "route_deviation" and pack.route:
        r = pack.route
        if NEAR_THRESHOLD_BAND[0] <= r.deviation_pct <= NEAR_THRESHOLD_BAND[1]:
            return True
        if r.duration_delta_min > 0:
            share = r.traffic_delay_min / r.duration_delta_min
            if AMBIGUOUS_TRAFFIC_SHARE[0] <= share < AMBIGUOUS_TRAFFIC_SHARE[1]:
                return True
        return False

    if category == "no_show" and pack.no_show:
        n = pack.no_show
        return (NO_SHOW_PRESENT_M < n.driver_min_distance_to_pickup_m <= NO_SHOW_ABSENT_M
                or NO_SHOW_WAIT_GREY_MIN <= n.wait_time_min < NO_SHOW_WAIT_MIN)

    return False


def motions(state: CaseState) -> dict:
    """Dismiss claims the record plainly refutes; summarily resolve undisputed facts."""
    pack = state["pack"]
    d = pack.dispute
    outcome = "none"
    reason = ""

    if d.category == "route_deviation" and pack.route:
        r = pack.route
        if r.deviation_pct < FRIVOLOUS_DEVIATION_PCT and not r.rider_requested_detour:
            outcome = "dismissed"
            reason = (
                f"Claim alleges route deviation; the record shows deviation of "
                f"{r.deviation_pct:.1%}, below the {FRIVOLOUS_DEVIATION_PCT:.0%} floor at which "
                f"a deviation claim is justiciable. No policy provision is engaged."
            )

    if d.category == "no_show" and pack.no_show:
        n = pack.no_show
        if n.driver_min_distance_to_pickup_m > NO_SHOW_ABSENT_M:
            outcome = "summary_judgment"
            reason = (
                f"Undisputed: driver never came within {NO_SHOW_ABSENT_M:.0f}m of the pin "
                f"({n.driver_min_distance_to_pickup_m}m). POL-2.1a requires full refund; "
                f"no material fact is in dispute."
            )
        elif (n.driver_min_distance_to_pickup_m <= NO_SHOW_PRESENT_M
              and n.wait_time_min >= NO_SHOW_WAIT_MIN and not n.rider_gps_approaching):
            outcome = "summary_judgment"
            reason = (
                f"Undisputed: driver within {n.driver_min_distance_to_pickup_m}m, waited "
                f"{n.wait_time_min} min, rider GPS static. All POL-2.1 conditions met; fee stands."
            )

    if outcome == "summary_judgment":
        # issue the ruling directly — no hearing
        absent = pack.no_show.driver_min_distance_to_pickup_m > NO_SHOW_ABSENT_M
        verdict = Verdict(
            ruling="refund_rider" if absent else "no_action",
            amount=round(pack.no_show.no_show_fee_charged if absent else 0.0, 2),
            cited_policy_id="POL-2.1a" if absent else "POL-2.1b",
            conclusions_of_law=[reason],
            remedy_rationale="Resolved on the pleadings; no material fact in dispute.",
            confidence=0.95,
        )
        return {
            "motion_outcome": outcome,
            "motion_reason": reason,
            "verdict": verdict,
            **log(state, "motions", outcome=outcome, reason=reason),
        }

    return {
        "motion_outcome": outcome,
        "motion_reason": reason,
        **log(state, "motions", outcome=outcome, reason=reason or "no threshold met; proceeding"),
    }


def settlement(state: CaseState) -> dict:
    """
    Nuisance-value settlement conference: offer a fraction of the claim now, or proceed
    to a hearing. Deterministic in the MVP — swap in a learned acceptance model later.

    The economics: referring an uncertain $5 dispute to a human costs more than $5. So
    uncertain + small settles; uncertain + large goes to a human. Certain cases go to a
    hearing, because you do not pay money you do not owe.
    """
    pack = state["pack"]
    claimed = pack.dispute.claimed_amount

    if pack.dispute.category not in SETTLEABLE_CATEGORIES:
        return {
            "settlement_offer": None,
            "settlement_accepted": False,
            **log(state, "settlement", offered=None, accepted=False, note="binary remedy, not settleable"),
        }
    if not is_ambiguous(pack):
        return {
            "settlement_offer": None,
            "settlement_accepted": False,
            **log(state, "settlement", offered=None, accepted=False, note="outcome not in doubt"),
        }
    if claimed > SETTLEMENT_CAP:
        return {
            "settlement_offer": None,
            "settlement_accepted": False,
            **log(state, "settlement", offered=None, accepted=False, note="above nuisance cap"),
        }

    offer = round(claimed * SETTLEMENT_FRACTION, 2)
    # MVP rule: claimant accepts a nuisance offer when it is the majority of a small claim.
    accepted = offer > 0
    if accepted:
        verdict = Verdict(
            ruling="refund_rider" if pack.dispute.filed_by == "rider" else "refund_driver",
            amount=offer,
            cited_policy_id=None,
            conclusions_of_law=["Resolved by settlement conference; no adjudication on the merits."],
            remedy_rationale=f"Offer of ${offer:.2f} accepted against a claim of ${claimed:.2f}.",
            confidence=1.0,
        )
        return {
            "settlement_offer": offer,
            "settlement_accepted": True,
            "verdict": verdict,
            **log(state, "settlement", offered=offer, accepted=True),
        }
    return {
        "settlement_offer": offer,
        "settlement_accepted": False,
        **log(state, "settlement", offered=offer, accepted=False),
    }
