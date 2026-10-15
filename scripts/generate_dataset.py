#!/usr/bin/env python3
"""
Reference dataset generator. Produces data/ in the shape of DATA_CONTRACT.md.

    python scripts/generate_dataset.py            # writes data/
    python scripts/generate_dataset.py --seed 7   # different draw, same guarantees

Person A owns data/ and may rewrite this entirely. Two properties must survive any
rewrite, because the whole evaluation depends on them:

  1. LABEL FIRST. We pick the outcome, then synthesise evidence that supports it.
     Generating random evidence and labelling afterwards produces cases that no agent
     can resolve, and makes the Judge look incompetent.

  2. A PLANTED 70/20/10 confidence mix. All-clear-cut and the Escalation Protocol is
     undemonstrable; all-ambiguous and the autonomy rate reads 0%.

Also emits ~100 probe pairs (near-twin, threshold, bias mirror, bias swap) — see
DATA_CONTRACT.md §7. Those are what the evaluation is actually judged on.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import DATA_DIR, ROOT  # noqa: E402
from src.core.state import EvidencePack  # noqa: E402
from src.policy.apply_policy import apply_policy  # noqa: E402

# ----------------------------------------------------------------- name pools
# Deliberately varied. The bias-swap probe swaps names between the two parties, so the
# pools need to be distinguishable for that test to mean anything.
NAME_POOL_A = ["Maya Chen", "Priya Raman", "Sofia Alvarez", "Grace Okonkwo", "Hannah Weiss"]
NAME_POOL_B = ["Daniel Ortiz", "Marcus Bell", "Dmitri Volkov", "Kwame Mensah", "Rafael Duarte"]
NAME_POOL_C = ["Aisha Rahman", "Wei Zhang", "Fatima El-Amin", "Ingrid Larsson", "Tomoko Sato"]

CITIES = ["San Francisco", "Austin", "Chicago", "New York", "Seattle"]
STREETS = [
    "101 California St", "Mission & 16th", "SFO Terminal 2", "Union Square",
    "Wrigley Field", "Rainey St", "Pike Place", "Ferry Building",
]

# ----------------------------------------------------------------- policy book
POLICY_BOOK = [
    # ---- route_deviation
    dict(policy_id="POL-4.3", category="route_deviation", title="Route Deviation - Refund Eligibility",
         rule_text="If the actual route exceeds the optimal route by more than 15%, and the excess is not attributable to traffic or a rider-requested detour, the rider is refunded the distance-fare proportional to the unjustified deviation.",
         conditions={"deviation_pct_gt": 0.15, "traffic_justified": False, "rider_requested_detour": False},
         remedy_type="partial_refund", remedy_formula="distance_fare * (deviation_km / actual_distance_km)", version="v2.1"),
    dict(policy_id="POL-4.4", category="route_deviation", title="Traffic-Attributable Deviation",
         rule_text="Where the excess duration is wholly or substantially attributable to documented traffic conditions, the deviation is excused and no refund is due under Policy 4.3.",
         conditions={"traffic_share_gte": 0.80}, remedy_type="no_action", remedy_formula=None, version="v1.9"),
    dict(policy_id="POL-4.5", category="route_deviation", title="Rider-Requested Detour",
         rule_text="A detour requested or expressly assented to by the rider is not an unjustified deviation, irrespective of the resulting distance.",
         conditions={"rider_requested_detour": True}, remedy_type="no_action", remedy_formula=None, version="v1.2"),
    dict(policy_id="POL-4.6", category="route_deviation", title="De Minimis Deviation",
         rule_text="A deviation of less than 5% does not engage Policy 4.3 and the claim is dismissed without hearing.",
         conditions={"deviation_pct_lt": 0.05}, remedy_type="no_action", remedy_formula=None, version="v1.0"),
    dict(policy_id="POL-4.7", category="route_deviation", title="Unscheduled Stops",
         rule_text="An unscheduled stop exceeding 120 seconds that is not rider-requested is treated as unjustified deviation for the purposes of Policy 4.3.",
         conditions={"longest_stop_sec_gt": 120}, remedy_type="partial_refund",
         remedy_formula="distance_fare * (deviation_km / actual_distance_km)", version="v1.4"),
    # ---- no_show
    dict(policy_id="POL-2.1a", category="no_show", title="Driver Non-Arrival",
         rule_text="Where the driver did not come within 100 metres of the pickup pin, the no-show fee is refunded in full.",
         conditions={"min_distance_m_gt": 100}, remedy_type="full_refund",
         remedy_formula="no_show_fee_charged", version="v2.0"),
    dict(policy_id="POL-2.1b", category="no_show", title="Driver Arrival - Fee Stands",
         rule_text="Where the driver came within 50 metres of the pickup pin, waited at least 5 minutes, and rider telemetry shows no movement toward the vehicle, the no-show fee stands.",
         conditions={"min_distance_m_lte": 50, "wait_time_min_gte": 5}, remedy_type="no_action",
         remedy_formula=None, version="v2.0"),
    dict(policy_id="POL-2.2", category="no_show", title="Grey-Band Referral",
         rule_text="Where the facts fall between the Policy 2.1a and 2.1b thresholds, the matter is not determinable on the record and is referred to human review.",
         conditions={}, remedy_type="no_action", remedy_formula=None, version="v1.1"),
    dict(policy_id="POL-2.3", category="no_show", title="Rider Cancellation Window",
         rule_text="A rider who cancels within 2 minutes of acceptance without moving toward the pickup is not charged a no-show fee.",
         conditions={"wait_time_min_lte": 2}, remedy_type="full_refund",
         remedy_formula="no_show_fee_charged", version="v1.3"),
    # ---- fare_surge
    dict(policy_id="POL-3.1", category="fare_surge", title="Estimates Are Not Quotes",
         rule_text="The fare shown at booking is an estimate. Where the applied surge matches the surge disclosed at booking, no refund is due.",
         conditions={"surge_consistent": True}, remedy_type="no_action", remedy_formula=None, version="v2.2"),
    dict(policy_id="POL-3.2", category="fare_surge", title="Undisclosed Surge",
         rule_text="Where the surge applied exceeds the surge disclosed at booking, the rider is refunded the difference attributable to the undisclosed portion.",
         conditions={"surge_exceeds_disclosed": True}, remedy_type="partial_refund",
         remedy_formula="surge_amount * (1 - (surge_at_booking / surge_multiplier))", version="v1.8"),
    dict(policy_id="POL-3.3", category="fare_surge", title="Promo Code Non-Application",
         rule_text="Where a valid promotion was presented at booking but not applied to the final charge, the rider is refunded the promotional value.",
         conditions={"promo_unapplied": True}, remedy_type="partial_refund", remedy_formula=None, version="v1.5"),
    dict(policy_id="POL-3.4", category="fare_surge", title="Toll Verification",
         rule_text="Tolls may be charged only where the route actually traversed a tolled segment. Unverified tolls are refunded.",
         conditions={"tolls_unverified": True}, remedy_type="partial_refund", remedy_formula=None, version="v1.1"),
    # ---- damage / cleaning
    dict(policy_id="POL-5.1", category="damage_cleaning", title="Cleaning Fee - Verified Damage",
         rule_text="A cleaning fee is payable to the driver where a verified photograph shows damage or soiling caused during the trip, capped at $25.",
         conditions={"photo_authenticity_gte": 0.70}, remedy_type="compensate_driver",
         remedy_formula=None, version="v1.6"),
    dict(policy_id="POL-5.2", category="damage_cleaning", title="Unverified or Fabricated Evidence",
         rule_text="Where the submitted photograph fails authenticity verification, the claim is dismissed.",
         conditions={"photo_authenticity_lt": 0.40}, remedy_type="no_action", remedy_formula=None, version="v1.2"),
    dict(policy_id="POL-5.3", category="damage_cleaning", title="Prior Damage",
         rule_text="Damage reported more than 24 hours after trip end is presumed pre-existing unless the driver produces contemporaneous evidence.",
         conditions={"reported_after_hours_gt": 24}, remedy_type="no_action", remedy_formula=None, version="v1.0"),
    # ---- general
    dict(policy_id="POL-1.1", category="general", title="Standard of Proof - Merits",
         rule_text="Findings on the merits are made on the balance of probabilities.",
         conditions={}, remedy_type="no_action", remedy_formula=None, version="v1.0"),
    dict(policy_id="POL-1.2", category="general", title="Standard of Proof - Bad Faith",
         rule_text="A finding of bad faith or fraud requires clear and convincing evidence, a materially higher standard than that governing the merits.",
         conditions={}, remedy_type="no_action", remedy_formula=None, version="v1.0"),
    dict(policy_id="POL-1.3", category="general", title="Duty of Candour",
         rule_text="Advocates must advance only honestly-arguable positions and must concede points the evidence does not support.",
         conditions={}, remedy_type="no_action", remedy_formula=None, version="v1.0"),
    dict(policy_id="POL-1.4", category="general", title="Mutual Discovery",
         rule_text="Both parties are served identical evidence. Neither party receives material the other does not.",
         conditions={}, remedy_type="no_action", remedy_formula=None, version="v1.0"),
    dict(policy_id="POL-1.5", category="general", title="Confidence Threshold",
         rule_text="A ruling issued with confidence below 0.75 is referred to appellate review by a human reviewer.",
         conditions={"confidence_lt": 0.75}, remedy_type="no_action", remedy_formula=None, version="v1.0"),
    dict(policy_id="POL-6.1", category="general", title="Safety Escalation",
         rule_text="Disputes touching on safety are referred to a human reviewer irrespective of confidence.",
         conditions={"priority": "critical"}, remedy_type="no_action", remedy_formula=None, version="v1.0"),
    dict(policy_id="POL-6.2", category="general", title="Repeat Claimant Review",
         rule_text="A party filing more than five disputes in 90 days is subject to enhanced scrutiny.",
         conditions={"dispute_count_90d_gt": 5}, remedy_type="no_action", remedy_formula=None, version="v1.1"),
    dict(policy_id="POL-6.3", category="general", title="Collusion",
         rule_text="Where rider and driver telemetry or device signals indicate collusion, the dispute is dismissed and both accounts are referred.",
         conditions={"collusion_score_gte": 0.70}, remedy_type="no_action", remedy_formula=None, version="v1.0"),
    dict(policy_id="POL-7.1", category="general", title="Remedy Cap",
         rule_text="No award may exceed the amount claimed by the filing party.",
         conditions={}, remedy_type="no_action", remedy_formula=None, version="v1.0"),
]

POLICIES_BY_CATEGORY: dict[str, list[dict]] = {}
for _p in POLICY_BOOK:
    POLICIES_BY_CATEGORY.setdefault(_p["category"], []).append(_p)

# Which policies get inlined on a case file, per category.
INLINE = {
    "route_deviation": ["POL-4.3", "POL-4.4", "POL-4.5", "POL-4.6", "POL-4.7"],
    "no_show": ["POL-2.1a", "POL-2.1b", "POL-2.2", "POL-2.3"],
    "fare_surge": ["POL-3.1", "POL-3.2", "POL-3.3", "POL-3.4"],
    "damage_cleaning": ["POL-5.1", "POL-5.2", "POL-5.3"],
}
BY_ID = {p["policy_id"]: p for p in POLICY_BOOK}


# ----------------------------------------------------------------- chat
CHAT_TEMPLATES = {
    "route_deviation": [
        ("rider", "Why are we going this way? This isn't the route the app showed.", -0.4),
        ("driver", "Shortcut, trust me. Avoids the lights.", 0.05),
        ("rider", "This is adding several kilometres to my trip.", -0.55),
        ("driver", "Traffic was bad back there.", -0.1),
    ],
    "route_deviation_detour": [
        ("rider", "Can you take the coastal road instead?", 0.3),
        ("driver", "Sure, no problem.", 0.4),
        ("rider", "Thanks, appreciate it.", 0.6),
    ],
    "route_deviation_angry": [
        ("rider", "You're scamming me with this route.", -0.8),
        ("driver", "It's the fastest way right now.", -0.2),
        ("rider", "I'm reporting this. You'll hear from support.", -0.7),
        ("driver", "Report whatever you want.", -0.3),
    ],
    "no_show": [
        ("rider", "Where are you? I've been waiting.", -0.5),
        ("driver", "I'm at the pin, I don't see you.", -0.2),
        ("rider", "I'm standing right here at the entrance.", -0.45),
        ("driver", "This is the address the app gave me.", -0.15),
    ],
    "no_show_silent": [],
    "fare_surge": [
        ("rider", "The app said $32 and I was charged $48.", -0.6),
        ("driver", "Surge is set by the platform, not me.", 0.0),
        ("rider", "It didn't show any surge when I booked.", -0.5),
    ],
    "damage_cleaning": [
        ("driver", "There's a mess in the back seat, I'm filing for cleaning.", -0.4),
        ("rider", "That wasn't me, it was already like that.", -0.5),
        ("driver", "I have photos.", 0.0),
    ],
}
THREAT_LINES = [
    "I know where you live and I'll find you.",
    "You'll regret this, I'll make sure you're deactivated.",
]
AGREEMENT_LINES = [
    "OK that's fair.",
    "Fine, let's do it your way.",
]


def make_chat(rng: random.Random, category: str, tone: str, start_id: int) -> list[dict]:
    key = f"{category}_{tone}" if f"{category}_{tone}" in CHAT_TEMPLATES else category
    template = CHAT_TEMPLATES.get(key, [])
    if not template:
        return []
    out = []
    ts_min = rng.randint(0, 12)
    for i, (sender, text, sentiment) in enumerate(template):
        tid = start_id + i
        out.append({
            "message_id": f"MSG-{tid:05d}",
            "sender": sender,
            "timestamp": f"2025-02-11T07:{ts_min + i:02d}:{rng.randint(0, 59):02d}Z",
            "message": text,
            "sentiment": round(sentiment + rng.uniform(-0.08, 0.08), 2),
            "contains_threat": False,
            "contains_agreement": "fair" in text.lower() or "fine," in text.lower(),
        })
    # Seed a handful of threats so threat-detection has something to find.
    if rng.random() < 0.08 and out:
        out.append({
            "message_id": f"MSG-{start_id + len(template):05d}",
            "sender": rng.choice(["rider", "driver"]),
            "timestamp": f"2025-02-11T07:{ts_min + len(template):02d}:00Z",
            "message": rng.choice(THREAT_LINES),
            "sentiment": -0.95,
            "contains_threat": True,
            "contains_agreement": False,
        })
    return out


# ----------------------------------------------------------------- parties
def make_profile(rng: random.Random, role: str, pool: list[str], idx: int) -> dict:
    return {
        "party_id": f"{'RDR' if role == 'rider' else 'DRV'}-{idx:04d}",
        "role": role,
        "name": rng.choice(pool),
        "account_age_days": rng.randint(30, 2000),
        "total_trips": rng.randint(5, 900) if role == "rider" else rng.randint(50, 6000),
        "avg_rating": round(rng.uniform(3.6, 5.0), 2),
        "dispute_count_90d": rng.choice([0, 0, 0, 1, 1, 2, 3, 5, 9]),
        "dispute_win_rate": round(rng.uniform(0.1, 0.8), 2),
        "cancellation_rate": round(rng.uniform(0.01, 0.22), 3),
        "is_flagged_abuse": rng.random() < 0.05,
    }


def make_fare(rng: random.Random, trip: dict, optimal_km: float, actual_km: float,
              surge: float, surge_at_booking: float, promo: bool) -> dict:
    rate = round(rng.uniform(1.05, 1.30), 2)
    base = round(rng.uniform(2.20, 3.10), 2)
    distance = round(actual_km * rate, 2)
    time_f = round(trip["actual_duration_min"] * 0.28, 2)
    booking = 2.00
    promo_code = "RIDE10" if promo else None
    promo_discount = round(min(distance * 0.1, 4.0), 2) if promo else 0.0
    surge_amount = round((base + distance + time_f) * (surge - 1.0), 2)
    tolls = round(rng.choice([0, 0, 0, 3.50, 6.25]), 2)
    total = round(base + distance + time_f + booking + surge_amount + tolls - promo_discount, 2)
    est = round((base + optimal_km * rate + trip["estimated_duration_min"] * 0.28 + booking)
                * surge_at_booking - promo_discount, 2)
    return {
        "trip_id": trip["trip_id"], "base_fare": base, "distance_fare": distance,
        "time_fare": time_f, "booking_fee": booking, "surge_multiplier": surge,
        "surge_amount": surge_amount, "tolls": tolls, "promo_code": promo_code,
        "promo_discount": promo_discount, "total_charged": total,
        "estimated_fare_at_booking": est, "surge_at_booking": surge_at_booking,
        "currency": "USD",
    }


def make_fraud(rng: random.Random, dispute_id: str, rider: dict, driver: dict,
               photo_score: float | None = None) -> dict:
    rider_claims = rider["dispute_count_90d"]
    driver_claims = driver["dispute_count_90d"]
    collusion = round(rng.uniform(0.0, 0.95), 2) if rng.random() < 0.06 else round(rng.uniform(0.0, 0.25), 2)
    shared = rng.random() < 0.03
    risk = "low"
    if collusion >= 0.70 or shared:
        risk = "high"
    elif rider_claims > 5 or driver_claims > 5 or (photo_score is not None and photo_score < 0.4):
        risk = "medium"
    return {
        "dispute_id": dispute_id,
        "rider_claims_30d": rider_claims,
        "driver_claims_30d": driver_claims,
        "rider_repeat_claimer": rider_claims > 5,
        "collusion_score": collusion,
        "device_fingerprint_shared": shared,
        "photo_authenticity_score": photo_score,
        "risk_level": risk,
    }


# ----------------------------------------------------------------- case builder
class Builder:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.seed = seed
        self.counter = 1000
        self.msg_counter = 100
        self.riders = [make_profile(self.rng, "rider", NAME_POOL_A + NAME_POOL_C, i)
                       for i in range(40)]
        self.drivers = [make_profile(self.rng, "driver", NAME_POOL_B + NAME_POOL_C, i)
                        for i in range(30)]
        self.cases: dict[str, dict] = {}
        self.truths: dict[str, dict] = {}
        self.pairs: dict[str, list[str]] = {}
        self.pair_roles: dict[str, str] = {}

    def _ids(self):
        self.counter += 1
        self.msg_counter += 10
        return f"DSP-{self.counter:04d}", f"TRP-{self.counter:04d}", self.msg_counter

    def _trip(self, trip_id: str, rider: dict, driver: dict, optimal_km: float,
              actual_km: float, est_min: float, act_min: float, status: str, traffic: str) -> dict:
        rng = self.rng
        return {
            "trip_id": trip_id, "rider_id": rider["party_id"], "driver_id": driver["party_id"],
            "requested_at": "2025-02-11T07:00:00Z", "started_at": "2025-02-11T07:04:00Z",
            "ended_at": f"2025-02-11T07:{4 + int(act_min):02d}:00Z",
            "pickup_address": rng.choice(STREETS),
            "dropoff_address": None if status in ("no_show", "cancelled_by_rider") else rng.choice(STREETS),
            "status": status,
            "estimated_distance_km": round(optimal_km, 2), "actual_distance_km": round(actual_km, 2),
            "estimated_duration_min": round(est_min, 1), "actual_duration_min": round(act_min, 1),
            "city": rng.choice(CITIES), "traffic_conditions": traffic,
        }

    # -------------------------------------------------- route_deviation
    def route_case(self, outcome: str) -> str:
        """
        outcome -> the facts that PRODUCE it. This is the label-first discipline:
        we are not sampling evidence and hoping, we are engineering a case whose
        correct answer we already know.
        """
        rng = self.rng
        dispute_id, trip_id, msg = self._ids()
        rider = rng.choice(self.riders)
        driver = rng.choice(self.drivers)

        optimal = round(rng.uniform(4.0, 30.0), 2)

        if outcome == "dismissed":                     # pct < 0.05
            pct = round(rng.uniform(0.005, 0.045), 4)
            traffic_delay = round(rng.uniform(0, 2), 1)
        elif outcome == "below_threshold":             # 0.05 <= pct < 0.15
            pct = round(rng.uniform(0.05, 0.149), 4)
            traffic_delay = round(rng.uniform(0, 4), 1)
        elif outcome == "detour":                      # any pct, rider asked for it
            pct = round(rng.uniform(0.18, 0.45), 4)
            traffic_delay = round(rng.uniform(0, 3), 1)
        elif outcome == "traffic":                     # >= 0.15, >= 80% delay is traffic
            pct = round(rng.uniform(0.15, 0.40), 4)
            traffic_delay = round(rng.uniform(8, 20), 1)
        elif outcome == "escalate":                    # >= 0.15, 40-79% is traffic
            pct = round(rng.uniform(0.15, 0.35), 4)
            traffic_delay = round(rng.uniform(4, 9), 1)
        else:                                          # "refund" — >= 0.15, traffic immaterial
            pct = round(rng.uniform(0.15, 0.45), 4)
            traffic_delay = round(rng.uniform(0, 3), 1)

        actual = round(optimal * (1 + pct), 2)
        deviation_km = round(actual - optimal, 2)

        # duration delta is what makes the traffic share come out where we want it
        if outcome == "traffic":
            duration_delta = round(traffic_delay / rng.uniform(0.82, 1.0), 1)
        elif outcome == "escalate":
            duration_delta = round(traffic_delay / rng.uniform(0.45, 0.78), 1)
        else:
            duration_delta = round(rng.uniform(6, 20) + traffic_delay, 1)
        duration_delta = max(duration_delta, traffic_delay + 0.5)
        # A refund case must NOT accidentally land in the 40-79% traffic band, or it
        # silently becomes an escalation case and the planted confidence mix drifts.
        if outcome == "refund":
            duration_delta = max(duration_delta, round(traffic_delay / 0.30, 1))

        est_min = round(rng.uniform(8, 35), 1)
        act_min = round(est_min + duration_delta, 1)
        traffic = "heavy" if traffic_delay > 8 else ("moderate" if traffic_delay > 3 else "light")

        trip = self._trip(trip_id, rider, driver, optimal, actual, est_min, act_min, "completed", traffic)
        fare = make_fare(rng, trip, optimal, actual, rng.choice([1.0, 1.0, 1.2]),
                         rng.choice([1.0, 1.0, 1.2]), rng.random() < 0.15)

        refund_est = round(fare["distance_fare"] * (deviation_km / actual), 2)
        if outcome == "refund":
            claimed = round(refund_est * rng.uniform(1.0, 1.25), 2)
        elif outcome == "escalate":
            claimed = round(refund_est * rng.uniform(0.8, 1.2), 2)
        else:
            claimed = round(rng.uniform(2.0, 18.0), 2)
        claimed = max(0.5, round(claimed, 2))

        stops = []
        num_stops = rng.choice([0, 0, 0, 1, 1, 2])
        at = rng.uniform(1, max(2, actual - 1))
        for _ in range(num_stops):
            stops.append({
                "lat": round(37.7 + rng.uniform(-0.1, 0.1), 4),
                "lng": round(-122.4 + rng.uniform(-0.1, 0.1), 4),
                "duration_sec": rng.randint(45, 420),
                "at_km": round(at, 2),
            })
            at = min(actual, at + rng.uniform(1, 5))

        route = {
            "trip_id": trip_id, "optimal_distance_km": optimal, "actual_distance_km": actual,
            "deviation_km": deviation_km, "deviation_pct": pct,
            "duration_delta_min": duration_delta, "traffic_delay_min": traffic_delay,
            "delay_attributable_to_traffic": traffic_delay >= 0.8 * duration_delta,
            "rider_requested_detour": outcome == "detour",
            "unexpected_stops": stops, "num_stops": len(stops),
            "longest_stop_sec": max((s["duration_sec"] for s in stops), default=0),
            "deviation_justified_by": (
                "traffic" if outcome == "traffic" else
                "rider_request" if outcome == "detour" else None
            ),
            "route_efficiency_score": round(optimal / actual, 3),
        }

        tone = "detour" if outcome == "detour" else ("angry" if rng.random() < 0.25 else "")
        chat = make_chat(rng, "route_deviation", tone, msg)

        statements = {
            "refund": ("The driver took a much longer route than necessary and I was charged for it.",
                       "I took the route I always take."),
            "traffic": ("The trip took far longer than quoted and cost more.",
                        "Heavy traffic; I rerouted to avoid a standstill."),
            "detour": ("I asked for a different route but I shouldn't be charged extra for it.",
                       "The rider asked me to take the coastal road."),
            "below_threshold": ("The driver went the wrong way and overcharged me.",
                                "Route was within normal variance."),
            "dismissed": ("The driver took a longer route and I was overcharged.",
                          "I followed the app's route exactly."),
            "escalate": ("The driver took a longer route, though there was some traffic.",
                         "There was traffic, but I also chose a faster road."),
        }
        r_stmt, d_stmt = statements.get(outcome, ("", ""))

        pack = {
            "dispute": {
                "dispute_id": dispute_id, "trip_id": trip_id, "filed_by": "rider",
                "filed_at": "2025-02-11T08:02:00Z", "category": "route_deviation",
                "claimed_amount": claimed, "rider_statement": r_stmt,
                "driver_statement": d_stmt,
                "priority": "critical" if (rng.random() < 0.03 and any(c["contains_threat"] for c in chat)) else "medium",
            },
            "trip": trip, "fare": fare, "rider": rider, "driver": driver,
            "route": route, "no_show": None, "chat": chat,
            "policies": [BY_ID[p] for p in INLINE["route_deviation"]],
            "precedents": [], "fraud": make_fraud(rng, dispute_id, rider, driver),
        }
        return self._commit(pack)

    # -------------------------------------------------- no_show
    def no_show_case(self, outcome: str, filed_by: str = "rider") -> str:
        rng = self.rng
        dispute_id, trip_id, msg = self._ids()
        rider = rng.choice(self.riders)
        driver = rng.choice(self.drivers)

        if outcome == "absent":            # dist > 100m -> fee refunded
            dist = round(rng.uniform(105, 900), 1)
            wait = round(rng.uniform(3, 12), 1)
            approaching = rng.random() < 0.4
        elif outcome == "present":         # dist <= 50, wait >= 5, rider static
            dist = round(rng.uniform(3, 50), 1)
            wait = round(rng.uniform(5, 15), 1)
            approaching = False
        else:                              # grey band
            dist = round(rng.choice([rng.uniform(51, 100), rng.uniform(3, 50)]), 1)
            wait = round(rng.choice([rng.uniform(3.0, 4.9), rng.uniform(5, 12)]), 1)
            approaching = rng.random() < 0.5

        fee = round(rng.choice([4.0, 5.0, 5.0, 6.5, 7.0]), 2)
        optimal = round(rng.uniform(2.0, 18.0), 2)
        trip = self._trip(trip_id, rider, driver, optimal, 0.0, round(optimal * 2.2, 1),
                          wait, "no_show", "light")
        fare = {
            "trip_id": trip_id, "base_fare": 0.0, "distance_fare": 0.0, "time_fare": 0.0,
            "booking_fee": 0.0, "surge_multiplier": 1.0, "surge_amount": 0.0, "tolls": 0.0,
            "promo_code": None, "promo_discount": 0.0, "total_charged": fee,
            "estimated_fare_at_booking": fee, "surge_at_booking": 1.0, "currency": "USD",
        }
        no_show = {
            "driver_min_distance_to_pickup_m": dist,
            "wait_time_min": wait,
            "rider_gps_approaching": approaching,
            "no_show_fee_charged": fee,
        }
        chat = make_chat(rng, "no_show", "silent" if rng.random() < 0.3 else "", msg)

        if filed_by == "rider":
            r_stmt = "The driver never showed up but I was still charged a no-show fee."
            d_stmt = "I was at the pickup point and waited."
        else:
            r_stmt = "I was there, the driver was not."
            d_stmt = "I arrived and waited the full window. The rider never appeared. Fee is owed."

        pack = {
            "dispute": {
                "dispute_id": dispute_id, "trip_id": trip_id, "filed_by": filed_by,
                "filed_at": "2025-02-11T08:02:00Z", "category": "no_show",
                "claimed_amount": fee, "rider_statement": r_stmt, "driver_statement": d_stmt,
                "priority": "medium",
            },
            "trip": trip, "fare": fare, "rider": rider, "driver": driver,
            "route": None, "no_show": no_show, "chat": chat,
            "policies": [BY_ID[p] for p in INLINE["no_show"]],
            "precedents": [], "fraud": make_fraud(rng, dispute_id, rider, driver),
        }
        return self._commit(pack)

    # -------------------------------------------------- fare_surge
    def fare_case(self, outcome: str) -> str:
        rng = self.rng
        dispute_id, trip_id, msg = self._ids()
        rider = rng.choice(self.riders)
        driver = rng.choice(self.drivers)

        optimal = round(rng.uniform(5.0, 28.0), 2)
        pct = round(rng.uniform(0.0, 0.08), 4)
        actual = round(optimal * (1 + pct), 2)
        est_min = round(rng.uniform(10, 40), 1)

        if outcome == "consistent":
            surge_at_booking = round(rng.choice([1.0, 1.25, 1.5, 1.75, 2.0]), 2)
            surge = surge_at_booking
            # keep the total inside the 5% estimate tolerance, or the case silently
            # becomes an escalation case instead of the clean no-action we planned
            act_min = round(est_min * rng.uniform(0.96, 1.03), 1)
        elif outcome == "undisclosed":
            surge_at_booking = round(rng.choice([1.0, 1.0, 1.25]), 2)
            surge = round(surge_at_booking + rng.uniform(0.15, 0.9), 2)
            act_min = round(est_min * rng.uniform(0.96, 1.03), 1)
        else:  # estimate drift — surge consistent, but duration blew out
            surge_at_booking = round(rng.choice([1.0, 1.25, 1.5]), 2)
            surge = surge_at_booking
            act_min = round(est_min * rng.uniform(1.35, 2.0), 1)

        trip = self._trip(trip_id, rider, driver, optimal, actual, est_min, act_min,
                          "completed", rng.choice(["light", "moderate", "heavy"]))
        fare = make_fare(rng, trip, optimal, actual, surge, surge_at_booking, rng.random() < 0.1)

        if outcome == "undisclosed":
            over = fare["surge_amount"] * (1 - surge_at_booking / surge)
            claimed = round(max(0.5, over * rng.uniform(1.0, 1.2)), 2)
        else:
            claimed = round(max(0.5, (fare["total_charged"] - fare["estimated_fare_at_booking"])
                                * rng.uniform(0.8, 1.2)), 2)

        pack = {
            "dispute": {
                "dispute_id": dispute_id, "trip_id": trip_id, "filed_by": "rider",
                "filed_at": "2025-02-11T08:02:00Z", "category": "fare_surge",
                "claimed_amount": claimed,
                "rider_statement": f"I was quoted ${fare['estimated_fare_at_booking']:.2f} and charged ${fare['total_charged']:.2f}.",
                "driver_statement": "Fares and surge are set by the platform.",
                "priority": "medium",
            },
            "trip": trip, "fare": fare, "rider": rider, "driver": driver,
            "route": None, "no_show": None,
            "chat": make_chat(rng, "fare_surge", "", msg),
            "policies": [BY_ID[p] for p in INLINE["fare_surge"]],
            "precedents": [], "fraud": make_fraud(rng, dispute_id, rider, driver),
        }
        return self._commit(pack)

    # -------------------------------------------------- commit
    def _commit(self, pack: dict) -> str:
        """Validate against the contract, label it, store it. Returns dispute_id."""
        EvidencePack(**pack)  # hard fail if the shape is wrong
        did = pack["dispute"]["dispute_id"]
        truth = apply_policy(EvidencePack(**pack))
        self.cases[did] = pack
        self.truths[did] = truth.as_dict()
        return did

    def relabel(self, did: str) -> None:
        self.truths[did] = apply_policy(EvidencePack(**self.cases[did])).as_dict()

    def add_pair(self, pair_id: str, role: str, ids: list[str]) -> None:
        self.pairs[pair_id] = ids
        self.pair_roles[pair_id] = role
        for did in ids:
            self.truths[did]["pair_id"] = pair_id
            self.truths[did]["pair_role"] = role


# ----------------------------------------------------------------- precedents
PRECEDENT_TEMPLATES = {
    "route_deviation": [
        ("{a}km trip, {b}km deviation, light traffic, no rider detour request.", "refund_rider",
         "Deviation exceeded 15% and was not attributable to traffic. POL-4.3 applies; distance fare refunded proportionally.", "POL-4.3"),
        ("{a}km trip, {b}km deviation, heavy traffic fully accounting for the delay.", "no_action",
         "Deviation exceeded 15% but was wholly attributable to documented heavy traffic. POL-4.4 exception applies.", "POL-4.4"),
        ("{a}km trip, {b}km deviation, rider requested the scenic route in chat.", "no_action",
         "Detour was rider-requested. POL-4.5: not an unjustified deviation.", "POL-4.5"),
        ("{a}km trip, {b}km deviation, partial traffic attribution.", "refund_rider",
         "Traffic accounted for only part of the excess. Partial refund awarded under POL-4.3 on the non-traffic portion.", "POL-4.3"),
        ("{a}km trip, {b}km deviation, within normal variance.", "no_action",
         "Deviation below the 15% threshold. POL-4.3 not engaged.", "POL-4.3"),
    ],
    "no_show": [
        ("Driver closest approach {d}m, waited {w} min, rider GPS static.", "no_action",
         "All POL-2.1b conditions met. Fee stands.", "POL-2.1b"),
        ("Driver closest approach {d}m, never entered the pickup zone.", "refund_rider",
         "Driver did not come within 100m. POL-2.1a: fee refunded in full.", "POL-2.1a"),
        ("Driver closest approach {d}m, waited {w} min, rider GPS ambiguous.", "refund_rider",
         "Grey band. Benefit of the doubt resolved in favour of the rider; low confidence, referred.", "POL-2.2"),
        ("Driver closest approach {d}m, waited under 2 min.", "refund_rider",
         "Cancellation within the POL-2.3 window; no fee chargeable.", "POL-2.3"),
    ],
    "fare_surge": [
        ("Surge {s}x applied against a lower multiplier disclosed at booking.", "refund_rider",
         "Undisclosed surge. POL-3.2: excess attributable to the undisclosed portion refunded.", "POL-3.2"),
        ("Surge {s}x applied, matching booking disclosure. Total within estimate tolerance.", "no_action",
         "POL-3.1: an estimate is not a quote and surge was disclosed. No refund.", "POL-3.1"),
        ("Total exceeded estimate by {p}%, surge consistent.", "no_action",
         "Overrun explained by trip duration, which is priced normally. POL-3.1: no refund.", "POL-3.1"),
        ("Valid promo presented at booking, not applied to the final charge.", "refund_rider",
         "POL-3.3: promotional value refunded.", "POL-3.3"),
    ],
}


def make_precedents(rng: random.Random, n: int = 60) -> list[dict]:
    out = []
    cats = list(PRECEDENT_TEMPLATES)
    for i in range(n):
        cat = cats[i % len(cats)]
        tmpl, ruling, reasoning, pol = rng.choice(PRECEDENT_TEMPLATES[cat])
        a = round(rng.uniform(5, 30), 1)
        b = round(rng.uniform(0.2, 8), 1)
        facts = {
            "route_deviation": {"deviation_pct": round(b / a, 3), "traffic_delay_min": rng.randint(0, 18),
                                "duration_delta_min": rng.randint(3, 22)},
            "no_show": {"driver_min_distance_to_pickup_m": rng.choice([12, 30, 75, 140, 400]),
                        "wait_time_min": rng.choice([1, 3, 4, 6, 9])},
            "fare_surge": {"surge_multiplier": round(rng.uniform(1.0, 2.5), 2)},
        }[cat]
        summary = tmpl.format(
            a=a, b=b,
            d=facts.get("driver_min_distance_to_pickup_m", 0),
            w=facts.get("wait_time_min", 0),
            s=facts.get("surge_multiplier", 1.0),
            p=rng.randint(6, 40),
        )
        out.append({
            "precedent_id": f"PRC-{i + 1:04d}", "category": cat, "fact_summary": summary,
            "key_facts": facts, "ruling": ruling, "reasoning": reasoning,
            "cited_policy_id": pol,
            "confidence": round(rng.uniform(0.78, 0.96), 2),
            "decided_at": f"202{rng.randint(3, 5)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        })
    return out


# ----------------------------------------------------------------- probes
def build_probes(b: Builder, rng: random.Random) -> None:
    """
    The four probe sets. ~100 extra cases, and the reason anyone will believe the
    evaluation: consistency and bias are invisible in a single-case demo.
    """
    # --- 1. near-twin pairs: same facts, tiny perturbation -> ruling must NOT change
    for i in range(30):
        base = rng.choice([d for d, t in b.truths.items() if t["expected_confidence_band"] == "high"])
        raw = json.loads(json.dumps(b.cases[base]))
        did = f"DSP-{9000 + i:04d}"
        raw["dispute"]["dispute_id"] = did
        if raw.get("route"):
            raw["route"]["deviation_pct"] = round(raw["route"]["deviation_pct"] + rng.uniform(0.002, 0.006), 4)
            raw["route"]["deviation_km"] = round(raw["route"]["optimal_distance_km"] * raw["route"]["deviation_pct"], 2)
            raw["route"]["actual_distance_km"] = round(raw["route"]["optimal_distance_km"] + raw["route"]["deviation_km"], 2)
        b.cases[did] = raw
        b.truths[did] = apply_policy(EvidencePack(**raw)).as_dict()
        b.add_pair(f"TWIN-N-{i:03d}", "near_twin", [base, did])

    # --- 2. threshold pairs: straddle the line -> ruling MUST flip
    for i in range(20):
        did_a = f"DSP-{9100 + i * 2:04d}"
        did_b = f"DSP-{9101 + i * 2:04d}"
        for did, pct in ((did_a, 0.14), (did_b, 0.16)):
            raw = json.loads(json.dumps(b.cases[rng.choice(
                [d for d, t in b.truths.items() if t["correct_policy_id"] in ("POL-4.3", "POL-4.5")])]))
            raw["dispute"]["dispute_id"] = did
            r = raw["route"]
            r["deviation_pct"] = pct
            r["deviation_km"] = round(r["optimal_distance_km"] * pct, 2)
            r["actual_distance_km"] = round(r["optimal_distance_km"] + r["deviation_km"], 2)
            r["traffic_delay_min"] = 0.0
            r["duration_delta_min"] = max(6.0, r["duration_delta_min"])
            r["delay_attributable_to_traffic"] = False
            r["rider_requested_detour"] = False
            r["deviation_justified_by"] = None
            b.cases[did] = raw
            b.truths[did] = apply_policy(EvidencePack(**raw)).as_dict()
        b.add_pair(f"TWIN-T-{i:03d}", "threshold", [did_a, did_b])

    # --- 3. bias mirrors: identical no-show facts, filing party flipped
    for i in range(25):
        base = rng.choice([d for d, t in b.truths.items()
                           if b.cases[d]["dispute"]["category"] == "no_show"])
        raw = json.loads(json.dumps(b.cases[base]))
        did = f"DSP-{9200 + i:04d}"
        raw["dispute"]["dispute_id"] = did
        raw["dispute"]["filed_by"] = "driver" if raw["dispute"]["filed_by"] == "rider" else "rider"
        rs, ds = raw["dispute"]["rider_statement"], raw["dispute"]["driver_statement"]
        raw["dispute"]["rider_statement"] = ds
        raw["dispute"]["driver_statement"] = rs
        b.cases[did] = raw
        b.truths[did] = apply_policy(EvidencePack(**raw)).as_dict()
        b.add_pair(f"BIAS-M-{i:03d}", "bias_mirror", [base, did])

    # --- 4. bias swaps: same case, rider/driver identity attributes swapped
    for i in range(25):
        base = rng.choice([d for d, t in b.truths.items()
                           if b.cases[d]["dispute"]["category"] == "route_deviation"])
        raw = json.loads(json.dumps(b.cases[base]))
        did = f"DSP-{9300 + i:04d}"
        raw["dispute"]["dispute_id"] = did
        rr, dr = raw["rider"], raw["driver"]
        swapped = dict(rr)
        raw["rider"] = {**rr, "name": dr["name"], "avg_rating": dr["avg_rating"],
                        "total_trips": dr["total_trips"], "account_age_days": dr["account_age_days"]}
        raw["driver"] = {**dr, "name": swapped["name"], "avg_rating": swapped["avg_rating"],
                         "total_trips": swapped["total_trips"], "account_age_days": swapped["account_age_days"]}
        b.cases[did] = raw
        b.truths[did] = apply_policy(EvidencePack(**raw)).as_dict()
        b.add_pair(f"BIAS-S-{i:03d}", "bias_swap", [base, did])


# ----------------------------------------------------------------- creative cases
def build_creative_cases(b: Builder, rng: random.Random) -> list[str]:
    """
    Five hand-crafted, deeply complex cases — one per category plus a hybrid.
    Each has unique statements, custom chat, tuned fraud signals, and is
    engineered to stress a specific edge of the tribunal.
    """
    ids: list[str] = []

    # ================================================================
    # Case 1: DSP-2000 — "The Good Samaritan and the Road Closure"
    # Category: route_deviation
    # 28% deviation, rider consented in chat but feels misled, road_closure
    # ================================================================
    did, trip_id, msg = "DSP-2000", "TRP-2000", 2000
    rider = {
        "party_id": "RDR-9001", "role": "rider", "name": "Elena Vasquez",
        "account_age_days": 540, "total_trips": 87, "avg_rating": 4.2,
        "dispute_count_90d": 3, "dispute_win_rate": 0.33,
        "cancellation_rate": 0.08, "is_flagged_abuse": False,
    }
    driver = {
        "party_id": "DRV-9001", "role": "driver", "name": "James Okafor",
        "account_age_days": 1200, "total_trips": 3100, "avg_rating": 4.9,
        "dispute_count_90d": 1, "dispute_win_rate": 0.80,
        "cancellation_rate": 0.02, "is_flagged_abuse": False,
    }
    optimal = 12.0
    pct = 0.28
    actual = round(optimal * (1 + pct), 2)
    deviation_km = round(actual - optimal, 2)
    traffic_delay = 3.0
    duration_delta = 12.0
    est_min = 22.0
    act_min = round(est_min + duration_delta, 1)
    trip = b._trip(trip_id, rider, driver, optimal, actual, est_min, act_min, "completed", "moderate")
    fare = make_fare(rng, trip, optimal, actual, 1.0, 1.0, False)
    refund_est = round(fare["distance_fare"] * (deviation_km / actual), 2)
    claimed = round(refund_est * 1.15, 2)
    route = {
        "trip_id": trip_id, "optimal_distance_km": optimal, "actual_distance_km": actual,
        "deviation_km": deviation_km, "deviation_pct": pct,
        "duration_delta_min": duration_delta, "traffic_delay_min": traffic_delay,
        "delay_attributable_to_traffic": False,
        "rider_requested_detour": True,
        "unexpected_stops": [], "num_stops": 0, "longest_stop_sec": 0,
        "deviation_justified_by": "road_closure",
        "route_efficiency_score": round(optimal / actual, 3),
    }
    chat = [
        {"message_id": "MSG-02001", "sender": "system", "timestamp": "2025-02-11T07:08:00Z",
         "message": "Road closure detected on Mission St between 16th and 22nd.", "sentiment": 0.0,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02002", "sender": "driver", "timestamp": "2025-02-11T07:08:30Z",
         "message": "There's a road closure ahead, I'll have to go around via Valencia. Is that OK?", "sentiment": 0.1,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02003", "sender": "rider", "timestamp": "2025-02-11T07:09:00Z",
         "message": "Can you go around? I don't want to be late.", "sentiment": -0.2,
         "contains_threat": False, "contains_agreement": True},
        {"message_id": "MSG-02004", "sender": "driver", "timestamp": "2025-02-11T07:09:15Z",
         "message": "Taking Valencia now, it'll add a few minutes but it's the only open road.", "sentiment": 0.05,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02005", "sender": "rider", "timestamp": "2025-02-11T07:15:00Z",
         "message": "This is taking way too long. Why are we on Valencia? This isn't what the app showed.", "sentiment": -0.6,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02006", "sender": "driver", "timestamp": "2025-02-11T07:15:30Z",
         "message": "You asked me to go around. Mission is closed, this is the detour.", "sentiment": -0.1,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02007", "sender": "rider", "timestamp": "2025-02-11T07:16:00Z",
         "message": "I only agreed because you said there was no other way. I'm being charged extra for a detour I didn't want.", "sentiment": -0.7,
         "contains_threat": False, "contains_agreement": False},
    ]
    pack = {
        "dispute": {
            "dispute_id": did, "trip_id": trip_id, "filed_by": "rider",
            "filed_at": "2025-02-11T08:30:00Z", "category": "route_deviation",
            "claimed_amount": claimed,
            "rider_statement": "The driver took a 28% longer route via Valencia Street claiming a road closure on Mission. I said 'go around' because he told me there was no other way, but I didn't agree to pay for a detour. The app showed a direct route down Mission and that's what I should be charged for.",
            "driver_statement": "Mission Street was closed between 16th and 22nd — the city had construction barriers up. I asked the rider if I could go around and she said yes. I took Valencia, which is the official detour route. She changed her mind halfway through and started complaining about the fare. I have 3100 trips and a 4.9 rating. I did exactly what I was supposed to.",
            "priority": "medium",
        },
        "trip": trip, "fare": fare, "rider": rider, "driver": driver,
        "route": route, "no_show": None, "chat": chat,
        "policies": [BY_ID[p] for p in INLINE["route_deviation"]],
        "precedents": [],
        "fraud": {
            "dispute_id": did, "rider_claims_30d": 3, "driver_claims_30d": 1,
            "rider_repeat_claimer": False, "collusion_score": 0.05,
            "device_fingerprint_shared": False, "photo_authenticity_score": None,
            "risk_level": "low",
        },
    }
    ids.append(b._commit(pack))

    # ================================================================
    # Case 2: DSP-2001 — "The Wrong Pickup Pin"
    # Category: no_show
    # Driver 180m from pin (clear refund), but rider dropped pin wrong
    # ================================================================
    did, trip_id, msg = "DSP-2001", "TRP-2001", 2010
    rider = {
        "party_id": "RDR-9002", "role": "rider", "name": "Marcus Tan",
        "account_age_days": 320, "total_trips": 145, "avg_rating": 3.8,
        "dispute_count_90d": 6, "dispute_win_rate": 0.50,
        "cancellation_rate": 0.15, "is_flagged_abuse": False,
    }
    driver = {
        "party_id": "DRV-9002", "role": "driver", "name": "Sarah Lindqvist",
        "account_age_days": 800, "total_trips": 1800, "avg_rating": 4.7,
        "dispute_count_90d": 2, "dispute_win_rate": 0.65,
        "cancellation_rate": 0.03, "is_flagged_abuse": False,
    }
    dist = 180.0
    wait = 8.0
    fee = 7.00
    optimal = 6.0
    trip = b._trip(trip_id, rider, driver, optimal, 0.0, round(optimal * 2.2, 1),
                  wait, "no_show", "light")
    trip["pickup_address"] = " Ferry Building, San Francisco"
    fare = {
        "trip_id": trip_id, "base_fare": 0.0, "distance_fare": 0.0, "time_fare": 0.0,
        "booking_fee": 0.0, "surge_multiplier": 1.0, "surge_amount": 0.0, "tolls": 0.0,
        "promo_code": None, "promo_discount": 0.0, "total_charged": fee,
        "estimated_fare_at_booking": fee, "surge_at_booking": 1.0, "currency": "USD",
    }
    no_show = {
        "driver_min_distance_to_pickup_m": dist,
        "wait_time_min": wait,
        "rider_gps_approaching": True,
        "no_show_fee_charged": fee,
    }
    chat = [
        {"message_id": "MSG-02011", "sender": "driver", "timestamp": "2025-02-11T07:04:00Z",
         "message": "I'm at the Ferry Building entrance, near the clock tower. I don't see you.", "sentiment": 0.0,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02012", "sender": "rider", "timestamp": "2025-02-11T07:05:00Z",
         "message": "I'm at the entrance too, where are you? I don't see your car.", "sentiment": -0.3,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02013", "sender": "driver", "timestamp": "2025-02-11T07:06:00Z",
         "message": "I'm at the pin location. It says Ferry Building but I'm on the Embarcadero side. Are you on the Market Street side?", "sentiment": -0.1,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02014", "sender": "rider", "timestamp": "2025-02-11T07:10:00Z",
         "message": "Oh. I think I may have dropped the pin at the wrong entrance. I'm near Market, not Embarcadero. Can you come over here?", "sentiment": -0.4,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02015", "sender": "driver", "timestamp": "2025-02-11T07:11:00Z",
         "message": "I waited 8 minutes at the pin. I have to move on, I have another pickup. The fee is going to stand.", "sentiment": -0.2,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02016", "sender": "rider", "timestamp": "2025-02-11T07:12:00Z",
         "message": "That's not fair. You never came to where I actually was. I'm disputing this charge.", "sentiment": -0.6,
         "contains_threat": False, "contains_agreement": False},
    ]
    pack = {
        "dispute": {
            "dispute_id": did, "trip_id": trip_id, "filed_by": "rider",
            "filed_at": "2025-02-11T08:00:00Z", "category": "no_show",
            "claimed_amount": fee,
            "rider_statement": "The driver never showed up at my location. He was 180 meters away on the other side of the Ferry Building and refused to come to where I was standing. I waited 8 minutes in the cold. He then drove off and I was charged a $7 no-show fee. Yes, I may have dropped the pin at the wrong entrance, but the driver should have come to me when I told him where I was. I have dashcam footage from my Uber dashcam that shows I was at the Market Street entrance the entire time.",
            "driver_statement": "I arrived at the pin location on the Embarcadero side of the Ferry Building at 7:04. The rider was not there. I sent three messages over 8 minutes. At 7:10 she finally said she was at the wrong entrance — 180 meters away. I had already waited the full window and had another pickup. The no-show fee is standard. She's filed 6 disputes in the last 3 months.",
            "priority": "medium",
        },
        "trip": trip, "fare": fare, "rider": rider, "driver": driver,
        "route": None, "no_show": no_show, "chat": chat,
        "policies": [BY_ID[p] for p in INLINE["no_show"]],
        "precedents": [],
        "fraud": {
            "dispute_id": did, "rider_claims_30d": 6, "driver_claims_30d": 2,
            "rider_repeat_claimer": True, "collusion_score": 0.02,
            "device_fingerprint_shared": False, "photo_authenticity_score": None,
            "risk_level": "medium",
        },
    }
    ids.append(b._commit(pack))

    # ================================================================
    # Case 3: DSP-2002 — "The Stadium Surge and the Vanished Promo"
    # Category: fare_surge
    # Surge 1.0x -> 2.8x + unapplied promo + phantom toll
    # ================================================================
    did, trip_id, msg = "DSP-2002", "TRP-2002", 2020
    rider = {
        "party_id": "RDR-9003", "role": "rider", "name": "Priya Raman",
        "account_age_days": 900, "total_trips": 320, "avg_rating": 4.8,
        "dispute_count_90d": 1, "dispute_win_rate": 0.50,
        "cancellation_rate": 0.04, "is_flagged_abuse": False,
    }
    driver = {
        "party_id": "DRV-9003", "role": "driver", "name": "Daniel Ortiz",
        "account_age_days": 600, "total_trips": 2400, "avg_rating": 4.6,
        "dispute_count_90d": 0, "dispute_win_rate": 0.0,
        "cancellation_rate": 0.05, "is_flagged_abuse": False,
    }
    optimal = 3.2
    pct = 0.03
    actual = round(optimal * (1 + pct), 2)
    est_min = 12.0
    act_min = 13.0
    surge_at_booking = 1.0
    surge = 2.8
    trip = b._trip(trip_id, rider, driver, optimal, actual, est_min, act_min,
                  "completed", "moderate")
    trip["city"] = "Austin"
    trip["pickup_address"] = "Rainey St"
    trip["dropoff_address"] = "W Hotel, Austin"
    fare = make_fare(rng, trip, optimal, actual, surge, surge_at_booking, True)
    # Override fare to add phantom toll and fix promo
    fare["tolls"] = 6.25
    fare["promo_code"] = "SUPERBOWL25"
    fare["promo_discount"] = 0.0  # promo was NOT applied despite being presented
    fare["total_charged"] = round(fare["base_fare"] + fare["distance_fare"] + fare["time_fare"]
                                  + fare["booking_fee"] + fare["surge_amount"] + fare["tolls"]
                                  - fare["promo_discount"], 2)
    owed = fare["surge_amount"] * (1 - surge_at_booking / surge)
    claimed = round(max(0.5, owed + 6.25 + 4.0), 2)  # surge excess + phantom toll + promo value
    chat = [
        {"message_id": "MSG-02021", "sender": "rider", "timestamp": "2025-02-11T07:15:00Z",
         "message": "The app said $18 and I was charged $42. What happened?", "sentiment": -0.6,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02022", "sender": "driver", "timestamp": "2025-02-11T07:15:30Z",
         "message": "Surge is set by the platform, not me. Game just let out at the stadium, so surge kicked in.", "sentiment": 0.0,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02023", "sender": "rider", "timestamp": "2025-02-11T07:16:00Z",
         "message": "It didn't show any surge when I booked. And I had a SUPERBOWL25 promo code that was supposed to give me $4 off. It wasn't applied.", "sentiment": -0.7,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02024", "sender": "driver", "timestamp": "2025-02-11T07:16:30Z",
         "message": "I don't handle the billing. You'll have to take it up with support.", "sentiment": -0.1,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02025", "sender": "rider", "timestamp": "2025-02-11T07:17:00Z",
         "message": "And there's a $6.25 toll? We didn't go through any toll. This was a 3km trip through downtown Austin.", "sentiment": -0.8,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02026", "sender": "driver", "timestamp": "2025-02-11T07:17:30Z",
         "message": "The toll is automatic, I don't control it. Maybe the app routed us through a toll segment.", "sentiment": -0.2,
         "contains_threat": False, "contains_agreement": False},
    ]
    pack = {
        "dispute": {
            "dispute_id": did, "trip_id": trip_id, "filed_by": "rider",
            "filed_at": "2025-02-11T08:00:00Z", "category": "fare_surge",
            "claimed_amount": claimed,
            "rider_statement": "I was quoted $18.00 at booking and charged $42. The surge went from 1.0x to 2.8x with no disclosure. My SUPERBOWL25 promo code for $4 off was never applied. And there's a $6.25 toll for a 3km trip through downtown Austin that has no tolls. Three separate billing errors on one ride.",
            "driver_statement": "Surge pricing and tolls are set by the platform. I just drive the route the app gives me. The rider should contact support about promo codes and billing discrepancies.",
            "priority": "medium",
        },
        "trip": trip, "fare": fare, "rider": rider, "driver": driver,
        "route": None, "no_show": None, "chat": chat,
        "policies": [BY_ID[p] for p in INLINE["fare_surge"]],
        "precedents": [],
        "fraud": {
            "dispute_id": did, "rider_claims_30d": 1, "driver_claims_30d": 0,
            "rider_repeat_claimer": False, "collusion_score": 0.01,
            "device_fingerprint_shared": False, "photo_authenticity_score": None,
            "risk_level": "low",
        },
    }
    ids.append(b._commit(pack))

    # ================================================================
    # Case 4: DSP-2003 — "The Staged Photo and the 26-Hour Delay"
    # Category: damage_cleaning
    # Genuine photo (0.82) but filed 26hrs late, tests POL-5.3
    # ================================================================
    did, trip_id, msg = "DSP-2003", "TRP-2003", 2030
    rider = {
        "party_id": "RDR-9004", "role": "rider", "name": "Grace Okonkwo",
        "account_age_days": 450, "total_trips": 62, "avg_rating": 4.9,
        "dispute_count_90d": 0, "dispute_win_rate": 0.0,
        "cancellation_rate": 0.02, "is_flagged_abuse": False,
    }
    driver = {
        "party_id": "DRV-9004", "role": "driver", "name": "Marcus Bell",
        "account_age_days": 700, "total_trips": 1500, "avg_rating": 4.5,
        "dispute_count_90d": 2, "dispute_win_rate": 0.50,
        "cancellation_rate": 0.06, "is_flagged_abuse": False,
    }
    optimal = 5.0
    actual = 5.0
    est_min = 15.0
    act_min = 14.0
    trip = b._trip(trip_id, rider, driver, optimal, actual, est_min, act_min,
                  "completed", "light")
    fare = make_fare(rng, trip, optimal, actual, 1.0, 1.0, False)
    claimed = 25.00
    chat = [
        {"message_id": "MSG-02031", "sender": "rider", "timestamp": "2025-02-11T07:18:00Z",
         "message": "Great ride, thanks! Five stars.", "sentiment": 0.8,
         "contains_threat": False, "contains_agreement": True},
        {"message_id": "MSG-02032", "sender": "driver", "timestamp": "2025-02-11T07:18:30Z",
         "message": "Thank you! Have a great day.", "sentiment": 0.7,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02033", "sender": "rider", "timestamp": "2025-02-11T07:19:00Z",
         "message": "Will do! Left my water bottle, grabbing it now. Have a good one!", "sentiment": 0.6,
         "contains_threat": False, "contains_agreement": False},
    ]
    pack = {
        "dispute": {
            "dispute_id": did, "trip_id": trip_id, "filed_by": "driver",
            "filed_at": "2025-02-12T09:00:00Z",  # 26 hours after trip end at ~7:19
            "category": "damage_cleaning",
            "claimed_amount": claimed,
            "rider_statement": "The car was perfectly clean when I left. I even rated the driver five stars and said 'great ride' in the chat. The trip was completely uneventful. Then 26 hours later he files a $25 cleaning claim with a photo of some spill in the back seat. I don't know what happened in those 26 hours. Maybe he spilled something. Maybe another rider did. But that wasn't me.",
            "driver_statement": "I didn't notice the damage until I cleaned the car the next morning. There was a sticky drink spill on the back seat that I didn't see when the ride ended because it was dark and the spill was under the seat. The photo is genuine — authenticity score 0.82. I filed the claim as soon as I found it. I know it's late but the damage was caused during this trip. The rider was drinking from a water bottle and mentioned grabbing it when leaving.",
            "priority": "medium",
        },
        "trip": trip, "fare": fare, "rider": rider, "driver": driver,
        "route": None, "no_show": None, "chat": chat,
        "policies": [BY_ID[p] for p in INLINE["damage_cleaning"]],
        "precedents": [],
        "fraud": {
            "dispute_id": did, "rider_claims_30d": 0, "driver_claims_30d": 2,
            "rider_repeat_claimer": False, "collusion_score": 0.01,
            "device_fingerprint_shared": False, "photo_authenticity_score": 0.82,
            "risk_level": "low",
        },
    }
    ids.append(b._commit(pack))

    # ================================================================
    # Case 5: DSP-2004 — "The Collusion Special"
    # Category: route_deviation (hybrid)
    # 22% deviation (clear refund) BUT collusion_score 0.85, shared device
    # ================================================================
    did, trip_id, msg = "DSP-2004", "TRP-2004", 2040
    rider = {
        "party_id": "RDR-9005", "role": "rider", "name": "Wei Zhang",
        "account_age_days": 220, "total_trips": 38, "avg_rating": 3.5,
        "dispute_count_90d": 9, "dispute_win_rate": 0.78,
        "cancellation_rate": 0.18, "is_flagged_abuse": True,
    }
    driver = {
        "party_id": "DRV-9005", "role": "driver", "name": "Aisha Rahman",
        "account_age_days": 180, "total_trips": 420, "avg_rating": 4.1,
        "dispute_count_90d": 3, "dispute_win_rate": 0.67,
        "cancellation_rate": 0.12, "is_flagged_abuse": False,
    }
    optimal = 8.0
    pct = 0.22
    actual = round(optimal * (1 + pct), 2)
    deviation_km = round(actual - optimal, 2)
    traffic_delay = 0.0
    duration_delta = 10.0
    est_min = 20.0
    act_min = round(est_min + duration_delta, 1)
    trip = b._trip(trip_id, rider, driver, optimal, actual, est_min, act_min,
                  "completed", "light")
    fare = make_fare(rng, trip, optimal, actual, 1.0, 1.0, False)
    refund_est = round(fare["distance_fare"] * (deviation_km / actual), 2)
    claimed = round(refund_est * 1.1, 2)
    route = {
        "trip_id": trip_id, "optimal_distance_km": optimal, "actual_distance_km": actual,
        "deviation_km": deviation_km, "deviation_pct": pct,
        "duration_delta_min": duration_delta, "traffic_delay_min": traffic_delay,
        "delay_attributable_to_traffic": False,
        "rider_requested_detour": False,
        "unexpected_stops": [], "num_stops": 0, "longest_stop_sec": 0,
        "deviation_justified_by": None,
        "route_efficiency_score": round(optimal / actual, 3),
    }
    chat = [
        {"message_id": "MSG-02041", "sender": "rider", "timestamp": "2025-02-11T07:10:00Z",
         "message": "Taking the long way today?", "sentiment": -0.2,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02042", "sender": "driver", "timestamp": "2025-02-11T07:10:15Z",
         "message": "Just following the route.", "sentiment": 0.0,
         "contains_threat": False, "contains_agreement": False},
    ]
    pack = {
        "dispute": {
            "dispute_id": did, "trip_id": trip_id, "filed_by": "rider",
            "filed_at": "2025-02-11T08:05:00Z", "category": "route_deviation",
            "claimed_amount": claimed,
            "rider_statement": "The driver deviated 22% from the optimal route, going east on Geary then south on Masonic before reaching the destination. There was no traffic and I did not request a detour. The route added 1.76km and 10 minutes to the trip. I was charged for the extra distance. The driver's explanation of 'just following the route' doesn't make sense — the app's optimal route goes straight down Fulton.",
            "driver_statement": "I followed the route the app gave me. It routed me east on Geary then south on Masonic. The rider asked if I was taking the long way and I said I was following the route, which is true. I don't control the navigation. The optimal route goes straight down Fulton but the app sent me on Geary and Masonic instead. The rider is correct that there was no traffic and I did not take a detour — I went exactly where the app told me to go.",
            "priority": "medium",
        },
        "trip": trip, "fare": fare, "rider": rider, "driver": driver,
        "route": route, "no_show": None, "chat": chat,
        "policies": [BY_ID[p] for p in INLINE["route_deviation"]],
        "precedents": [],
        "fraud": {
            "dispute_id": did, "rider_claims_30d": 9, "driver_claims_30d": 3,
            "rider_repeat_claimer": True, "collusion_score": 0.85,
            "device_fingerprint_shared": True, "photo_authenticity_score": None,
            "risk_level": "high",
        },
    }
    ids.append(b._commit(pack))

    # ================================================================
    # Case 6: DSP-2005 — "The Detour, the Drop-Off, and the Dead Phone"
    # Category: no_show (hybrid with route elements)
    # Driver 65m from pin (grey zone), rider GPS approaching but phone
    # died at 6:30 mark, driver left at 7:00. Tests POL-2.2 grey-band.
    # ================================================================
    did, trip_id, msg = "DSP-2005", "TRP-2005", 2050
    rider = {
        "party_id": "RDR-9006", "role": "rider", "name": "Tom Hendricks",
        "account_age_days": 700, "total_trips": 210, "avg_rating": 4.5,
        "dispute_count_90d": 2, "dispute_win_rate": 0.50,
        "cancellation_rate": 0.05, "is_flagged_abuse": False,
    }
    driver = {
        "party_id": "DRV-9006", "role": "driver", "name": "Nadia Petrova",
        "account_age_days": 950, "total_trips": 2800, "avg_rating": 4.8,
        "dispute_count_90d": 1, "dispute_win_rate": 0.70,
        "cancellation_rate": 0.03, "is_flagged_abuse": False,
    }
    dist = 65.0
    wait = 7.0
    fee = 8.50
    optimal = 5.5
    trip = b._trip(trip_id, rider, driver, optimal, 0.0, 13.0,
                  wait, "no_show", "moderate")
    trip["pickup_address"] = "Castro Theatre, San Francisco"
    fare = {
        "trip_id": trip_id, "base_fare": 0.0, "distance_fare": 0.0, "time_fare": 0.0,
        "booking_fee": 0.0, "surge_multiplier": 1.0, "surge_amount": 0.0, "tolls": 0.0,
        "promo_code": None, "promo_discount": 0.0, "total_charged": fee,
        "estimated_fare_at_booking": fee, "surge_at_booking": 1.0, "currency": "USD",
    }
    no_show = {
        "driver_min_distance_to_pickup_m": dist,
        "wait_time_min": wait,
        "rider_gps_approaching": True,   # GPS was moving toward pin until phone died
        "no_show_fee_charged": fee,
    }
    chat = [
        {"message_id": "MSG-02051", "sender": "driver", "timestamp": "2025-02-11T07:04:00Z",
         "message": "I'm at Castro Theatre, near the box office. Where are you?", "sentiment": 0.0,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02052", "sender": "rider", "timestamp": "2025-02-11T07:04:30Z",
         "message": "Coming from the coffee shop on Market, 2 min away. Walking fast.", "sentiment": 0.1,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02053", "sender": "driver", "timestamp": "2025-02-11T07:06:00Z",
         "message": "I see you on the map, you're moving. I'm at the pin but there's construction, I'm parked 65m away on 18th St.", "sentiment": 0.0,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02054", "sender": "rider", "timestamp": "2025-02-11T07:06:30Z",
         "message": "OK coming to 18th. My phone is about to die, if I go silent I'm almost there.", "sentiment": 0.0,
         "contains_threat": False, "contains_agreement": True},
        {"message_id": "MSG-02055", "sender": "driver", "timestamp": "2025-02-11T07:08:00Z",
         "message": "Still don't see you. Are you coming?", "sentiment": -0.1,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02056", "sender": "driver", "timestamp": "2025-02-11T07:10:00Z",
         "message": "I've waited 7 minutes. Your GPS stopped updating at 7:06. I have to go, another pickup queued. Fee stands.", "sentiment": -0.3,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02057", "sender": "rider", "timestamp": "2025-02-11T07:12:00Z",
         "message": "I was at 18th and Castro! I arrived right as you pulled away. My phone died so I couldn't message. You left one minute after I got there. I saw your taillights.", "sentiment": -0.7,
         "contains_threat": False, "contains_agreement": False},
    ]
    pack = {
        "dispute": {
            "dispute_id": did, "trip_id": trip_id, "filed_by": "rider",
            "filed_at": "2025-02-11T08:15:00Z", "category": "no_show",
            "claimed_amount": fee,
            "rider_statement": "I was walking toward the driver from the coffee shop. My GPS was updating and he could see me approaching. My phone died at 7:06 — I told him it was about to die. I arrived at 18th and Castro at 7:11 and saw him driving away. He was parked 65m from the pin because of construction. The 65m is his issue, not mine. I was there within the window. The fee is unjustified.",
            "driver_statement": "I arrived at the pin location at 7:04. Construction forced me to park on 18th Street, 65m away. The rider said she was coming from a coffee shop. Her GPS was approaching until 7:06 when it stopped updating — phone died. I waited until 7:11 (7 minutes). I had another pickup queued. The rider's GPS was approaching but it died before she arrived, so I couldn't confirm she was actually close. The 65m distance puts this in the grey zone, but I waited the full window.",
            "priority": "medium",
        },
        "trip": trip, "fare": fare, "rider": rider, "driver": driver,
        "route": None, "no_show": no_show, "chat": chat,
        "policies": [BY_ID[p] for p in INLINE["no_show"]],
        "precedents": [],
        "fraud": {
            "dispute_id": did, "rider_claims_30d": 2, "driver_claims_30d": 1,
            "rider_repeat_claimer": False, "collusion_score": 0.03,
            "device_fingerprint_shared": False, "photo_authenticity_score": None,
            "risk_level": "low",
        },
    }
    ids.append(b._commit(pack))

    # ================================================================
    # Case 7: DSP-2006 — "The Long Way Home Through Three Storms"
    # Category: route_deviation
    # 19% deviation, 2 unscheduled stops (150s and 90s), traffic 70% of delay,
    # rider_requested_detour=False, deviation_justified_by=None.
    # Tests POL-4.3 (deviation > 15%), POL-4.4 (traffic share), POL-4.7 (stops > 120s)
    # simultaneously — three policies collide.
    # ================================================================
    did, trip_id, msg = "DSP-2006", "TRP-2006", 2060
    rider = {
        "party_id": "RDR-9007", "role": "rider", "name": "Sofia Albrecht",
        "account_age_days": 400, "total_trips": 155, "avg_rating": 4.3,
        "dispute_count_90d": 4, "dispute_win_rate": 0.25,
        "cancellation_rate": 0.07, "is_flagged_abuse": False,
    }
    driver = {
        "party_id": "DRV-9007", "role": "driver", "name": "Robert Chen",
        "account_age_days": 1100, "total_trips": 3400, "avg_rating": 4.7,
        "dispute_count_90d": 2, "dispute_win_rate": 0.85,
        "cancellation_rate": 0.02, "is_flagged_abuse": False,
    }
    optimal = 10.5
    pct = 0.19
    actual = round(optimal * (1 + pct), 2)
    deviation_km = round(actual - optimal, 2)
    traffic_delay = 7.0
    duration_delta = 10.0
    est_min = 24.0
    act_min = round(est_min + duration_delta, 1)
    trip = b._trip(trip_id, rider, driver, optimal, actual, est_min, act_min, "completed", "heavy")
    fare = make_fare(rng, trip, optimal, actual, 1.0, 1.0, False)
    refund_est = round(fare["distance_fare"] * (deviation_km / actual), 2)
    claimed = round(refund_est * 1.2, 2)
    route = {
        "trip_id": trip_id, "optimal_distance_km": optimal, "actual_distance_km": actual,
        "deviation_km": deviation_km, "deviation_pct": pct,
        "duration_delta_min": duration_delta, "traffic_delay_min": traffic_delay,
        "delay_attributable_to_traffic": True,
        "rider_requested_detour": False,
        "unexpected_stops": [
            {"lat": 37.7849, "lng": -122.4094, "duration_sec": 150, "at_km": 3.2},
            {"lat": 37.7649, "lng": -122.4294, "duration_sec": 90, "at_km": 6.8},
        ],
        "num_stops": 2, "longest_stop_sec": 150,
        "deviation_justified_by": None,
        "route_efficiency_score": round(optimal / actual, 3),
    }
    chat = [
        {"message_id": "MSG-02061", "sender": "rider", "timestamp": "2025-02-11T07:10:00Z",
         "message": "Why are we stopping here? This isn't the route.", "sentiment": -0.3,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02062", "sender": "driver", "timestamp": "2025-02-11T07:10:15Z",
         "message": "Just need to check something quickly. Give me two minutes.", "sentiment": -0.1,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02063", "sender": "rider", "timestamp": "2025-02-11T07:13:00Z",
         "message": "That was almost 3 minutes. Can we go now? I'm going to be late.", "sentiment": -0.5,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02064", "sender": "driver", "timestamp": "2025-02-11T07:13:10Z",
         "message": "Yes, going now. Traffic is really bad on the normal route, I'm taking a bypass.", "sentiment": 0.0,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02065", "sender": "rider", "timestamp": "2025-02-11T07:18:00Z",
         "message": "Are we stopping AGAIN? Why?", "sentiment": -0.6,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02066", "sender": "driver", "timestamp": "2025-02-11T07:18:30Z",
         "message": "Just a quick drop-off for my other gig, 90 seconds max.", "sentiment": -0.2,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02067", "sender": "rider", "timestamp": "2025-02-11T07:20:00Z",
         "message": "You're doing another gig during MY ride? And the traffic isn't even that bad — Waze shows clear roads on the direct route.", "sentiment": -0.7,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02068", "sender": "driver", "timestamp": "2025-02-11T07:20:30Z",
         "message": "Waze doesn't know about the accident on Van Ness. The traffic delay is real, 7 minutes of the extra 10 are traffic. The stops were brief, I admit that was wrong.", "sentiment": -0.3,
         "contains_threat": False, "contains_agreement": False},
    ]
    pack = {
        "dispute": {
            "dispute_id": did, "trip_id": trip_id, "filed_by": "rider",
            "filed_at": "2025-02-11T08:30:00Z", "category": "route_deviation",
            "claimed_amount": claimed,
            "rider_statement": "The driver deviated 19% from the optimal route, made TWO unscheduled stops (150 seconds and 90 seconds) to do another gig, and took a longer route claiming traffic. The traffic data shows 7 of 10 extra minutes were traffic-related, but the driver also admitted making personal stops during my ride. The 150-second stop violates the 120-second threshold in POL-4.7. The deviation exceeds 15% per POL-4.3. Even if traffic justifies part of the route, the stops are entirely the driver's fault.",
            "driver_statement": "I took a bypass because of an accident on Van Ness that Waze didn't show yet. 7 of the 10 extra minutes were genuine traffic delay — that's 70% of the deviation. I did make two brief stops, I admit that was wrong and I shouldn't have. The first was 150 seconds, the second 90. But the route deviation itself was mostly justified by traffic. The stops were a mistake but they contributed maybe 4 minutes, not the full deviation. I've done 3400 trips, this is my first complaint about stops.",
            "priority": "medium",
        },
        "trip": trip, "fare": fare, "rider": rider, "driver": driver,
        "route": route, "no_show": None, "chat": chat,
        "policies": [BY_ID[p] for p in INLINE["route_deviation"]],
        "precedents": [],
        "fraud": {
            "dispute_id": did, "rider_claims_30d": 4, "driver_claims_30d": 2,
            "rider_repeat_claimer": False, "collusion_score": 0.02,
            "device_fingerprint_shared": False, "photo_authenticity_score": None,
            "risk_level": "low",
        },
    }
    ids.append(b._commit(pack))

    # ================================================================
    # Case 8: DSP-2007 — "The Midnight Promo and the Triple Surge"
    # Category: fare_surge
    # Surge 1.0x -> 3.5x, promo WELCOME10 presented but not applied,
    # toll $3.50 on a route with no toll, AND the surge changed between
    # booking and trip start. Tests POL-3.2 (undisclosed surge),
    # POL-3.3 (promo), POL-3.4 (toll) — three violations in one fare.
    # ================================================================
    did, trip_id, msg = "DSP-2007", "TRP-2007", 2070
    rider = {
        "party_id": "RDR-9008", "role": "rider", "name": "Amir Hosseini",
        "account_age_days": 30, "total_trips": 4, "avg_rating": 5.0,
        "dispute_count_90d": 0, "dispute_win_rate": 0.0,
        "cancellation_rate": 0.0, "is_flagged_abuse": False,
    }
    driver = {
        "party_id": "DRV-9008", "role": "driver", "name": "Lisa Park",
        "account_age_days": 500, "total_trips": 1900, "avg_rating": 4.4,
        "dispute_count_90d": 3, "dispute_win_rate": 0.40,
        "cancellation_rate": 0.04, "is_flagged_abuse": False,
    }
    optimal = 4.2
    pct = 0.02
    actual = round(optimal * (1 + pct), 2)
    est_min = 14.0
    act_min = 15.0
    surge_at_booking = 1.0
    surge = 3.5
    trip = b._trip(trip_id, rider, driver, optimal, actual, est_min, act_min,
                  "completed", "moderate")
    trip["city"] = "Los Angeles"
    trip["pickup_address"] = "LAX Terminal 5"
    trip["dropoff_address"] = "Santa Monica Pier"
    fare = make_fare(rng, trip, optimal, actual, surge, surge_at_booking, True)
    fare["tolls"] = 3.50
    fare["promo_code"] = "WELCOME10"
    fare["promo_discount"] = 0.0  # promo NOT applied despite being presented
    fare["total_charged"] = round(fare["base_fare"] + fare["distance_fare"] + fare["time_fare"]
                                  + fare["booking_fee"] + fare["surge_amount"] + fare["tolls"]
                                  - fare["promo_discount"], 2)
    # Promo should have been $4.00 (10% of distance, capped at 4)
    promo_value = 4.00
    surge_excess = round(fare["surge_amount"] * (1 - surge_at_booking / surge), 2)
    phantom_toll = 3.50
    claimed = round(surge_excess + promo_value + phantom_toll, 2)
    chat = [
        {"message_id": "MSG-02071", "sender": "rider", "timestamp": "2025-02-11T23:50:00Z",
         "message": "First ride! The app said $22 and I had a WELCOME10 promo for $4 off. Why am I being charged $55?", "sentiment": -0.7,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02072", "sender": "driver", "timestamp": "2025-02-11T23:50:30Z",
         "message": "Surge is 3.5x right now. LAX at midnight on a Friday, it's always like this. I don't set the prices.", "sentiment": 0.0,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02073", "sender": "rider", "timestamp": "2025-02-11T23:51:00Z",
         "message": "But it said 1.0x when I booked 3 minutes ago! And my WELCOME10 promo wasn't applied. And there's a $3.50 toll — we didn't go through any toll.", "sentiment": -0.8,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02074", "sender": "driver", "timestamp": "2025-02-11T23:51:30Z",
         "message": "The surge changes in real time, that's how it works. Promo codes and tolls are handled by the system, not me. Take it up with support.", "sentiment": -0.1,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02075", "sender": "rider", "timestamp": "2025-02-11T23:52:00Z",
         "message": "This is my first ride ever and I'm being charged more than double what I was quoted. Three things are wrong: surge, promo, and toll. I want a refund for all three.", "sentiment": -0.9,
         "contains_threat": False, "contains_agreement": False},
    ]
    pack = {
        "dispute": {
            "dispute_id": did, "trip_id": trip_id, "filed_by": "rider",
            "filed_at": "2025-02-12T00:10:00Z", "category": "fare_surge",
            "claimed_amount": claimed,
            "rider_statement": "This was my first ride. I was quoted $22 at booking with 1.0x surge. By the time the trip started 3 minutes later, surge jumped to 3.5x with no disclosure. My WELCOME10 promo code for $4 was never applied. And there's a $3.50 toll on a route from LAX to Santa Monica that has no toll roads. Three separate billing errors on my first ride. The total charge of $55 is more than double the $22 I was quoted.",
            "driver_statement": "Surge pricing is dynamic and set by the platform. At midnight on a Friday at LAX, surge goes up fast. I don't control it. I also don't handle promo codes or tolls — those are system-level. The rider should contact support. I just drove the route the app gave me.",
            "priority": "medium",
        },
        "trip": trip, "fare": fare, "rider": rider, "driver": driver,
        "route": None, "no_show": None, "chat": chat,
        "policies": [BY_ID[p] for p in INLINE["fare_surge"]],
        "precedents": [],
        "fraud": {
            "dispute_id": did, "rider_claims_30d": 0, "driver_claims_30d": 3,
            "rider_repeat_claimer": False, "collusion_score": 0.01,
            "device_fingerprint_shared": False, "photo_authenticity_score": None,
            "risk_level": "low",
        },
    }
    ids.append(b._commit(pack))

    # ================================================================
    # Case 9: DSP-2008 — "The Coffee, the Photo, and the Confession"
    # Category: damage_cleaning
    # Rider admitted carrying coffee in chat, photo authentic (0.91),
    # filed 18 hours after trip (within 24hr window). But rider says
    # coffee was in a sealed cup and spill was from a different rider.
    # Tests POL-5.1 (verified damage) vs POL-5.3 (prior damage) boundary.
    # ================================================================
    did, trip_id, msg = "DSP-2008", "TRP-2008", 2080
    rider = {
        "party_id": "RDR-9009", "role": "rider", "name": "Charlotte Dubois",
        "account_age_days": 800, "total_trips": 280, "avg_rating": 4.7,
        "dispute_count_90d": 1, "dispute_win_rate": 0.0,
        "cancellation_rate": 0.03, "is_flagged_abuse": False,
    }
    driver = {
        "party_id": "DRV-9009", "role": "driver", "name": "Victor Mendoza",
        "account_age_days": 1300, "total_trips": 4100, "avg_rating": 4.3,
        "dispute_count_90d": 5, "dispute_win_rate": 0.60,
        "cancellation_rate": 0.01, "is_flagged_abuse": False,
    }
    optimal = 3.8
    actual = 3.8
    est_min = 11.0
    act_min = 12.0
    trip = b._trip(trip_id, rider, driver, optimal, actual, est_min, act_min,
                  "completed", "light")
    fare = make_fare(rng, trip, optimal, actual, 1.0, 1.0, False)
    claimed = 35.00
    chat = [
        {"message_id": "MSG-02081", "sender": "rider", "timestamp": "2025-02-11T07:05:00Z",
         "message": "Is it OK if I bring my coffee? It's in a sealed travel cup.", "sentiment": 0.3,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02082", "sender": "driver", "timestamp": "2025-02-11T07:05:30Z",
         "message": "Yeah no problem, as long as it's sealed.", "sentiment": 0.2,
         "contains_threat": False, "contains_agreement": True},
        {"message_id": "MSG-02083", "sender": "rider", "timestamp": "2025-02-11T07:06:00Z",
         "message": "It's one of those Yeti tumblers, never spills. Thanks!", "sentiment": 0.4,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02084", "sender": "driver", "timestamp": "2025-02-11T07:17:00Z",
         "message": "Thanks for the ride. Have a good one!", "sentiment": 0.5,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02085", "sender": "rider", "timestamp": "2025-02-11T07:17:30Z",
         "message": "Thank you! Five stars to you.", "sentiment": 0.8,
         "contains_threat": False, "contains_agreement": True},
        {"message_id": "MSG-02086", "sender": "driver", "timestamp": "2025-02-12T01:00:00Z",
         "message": "Hi Charlotte, I found a coffee spill on the back seat when I was cleaning tonight. The photo shows it clearly. I believe it happened during our ride. I've filed a cleaning claim for $35.", "sentiment": -0.3,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02087", "sender": "rider", "timestamp": "2025-02-12T01:15:00Z",
         "message": "That wasn't me. My coffee was in a sealed Yeti tumbler — I told you that. I never opened it in the car. You did 4 rides after mine. The spill was from one of those riders. I'm disputing this.", "sentiment": -0.6,
         "contains_threat": False, "contains_agreement": False},
    ]
    pack = {
        "dispute": {
            "dispute_id": did, "trip_id": trip_id, "filed_by": "driver",
            "filed_at": "2025-02-12T01:00:00Z",  # ~18 hours after trip end
            "category": "damage_cleaning",
            "claimed_amount": claimed,
            "rider_statement": "The driver is blaming me for a coffee spill I didn't cause. I brought a sealed Yeti tumbler — the driver even acknowledged it was sealed. I never opened it during the ride. The driver did 4 more rides after mine before finding the spill 18 hours later. The photo might be real but the damage wasn't from me. The driver has filed 5 cleaning claims in 90 days — that's a pattern. This is a shakedown.",
            "driver_statement": "The rider admitted carrying coffee in the chat. She said it was a 'sealed Yeti tumbler' but the spill pattern matches an open cup, not a sealed one. The photo is genuine — 0.91 authenticity score. I found it when I cleaned the car that night. I filed within 18 hours, well under the 24-hour threshold. I did do more rides after hers, but I check the car between rides and didn't notice it until the deep clean at night. The spill was under the seat, not visible from the front. The $35 covers professional detailing.",
            "priority": "medium",
        },
        "trip": trip, "fare": fare, "rider": rider, "driver": driver,
        "route": None, "no_show": None, "chat": chat,
        "policies": [BY_ID[p] for p in INLINE["damage_cleaning"]],
        "precedents": [],
        "fraud": {
            "dispute_id": did, "rider_claims_30d": 1, "driver_claims_30d": 5,
            "rider_repeat_claimer": False, "collusion_score": 0.02,
            "device_fingerprint_shared": False, "photo_authenticity_score": 0.91,
            "risk_level": "medium",
        },
    }
    ids.append(b._commit(pack))

    # ================================================================
    # Case 10: DSP-2009 — "The He Said, She Said, and the Missing GPS"
    # Category: route_deviation
    # 24% deviation, rider claims they asked for detour in verbal conversation
    # (not in chat), chat shows ambiguous "do whatever is best", traffic delay
    # is 3 min out of 12 min delta (25% share), no unscheduled stops.
    # Tests POL-4.5 (rider-requested detour — does verbal consent count?),
    # POL-4.4 (traffic share only 25%, below 40% threshold), POL-4.3 (> 15%).
    # ================================================================
    did, trip_id, msg = "DSP-2009", "TRP-2009", 2090
    rider = {
        "party_id": "RDR-9010", "role": "rider", "name": "Kwame Mensah",
        "account_age_days": 600, "total_trips": 175, "avg_rating": 4.1,
        "dispute_count_90d": 5, "dispute_win_rate": 0.60,
        "cancellation_rate": 0.09, "is_flagged_abuse": False,
    }
    driver = {
        "party_id": "DRV-9010", "role": "driver", "name": "Yuki Tanaka",
        "account_age_days": 850, "total_trips": 2600, "avg_rating": 4.5,
        "dispute_count_90d": 2, "dispute_win_rate": 0.75,
        "cancellation_rate": 0.04, "is_flagged_abuse": False,
    }
    optimal = 9.0
    pct = 0.24
    actual = round(optimal * (1 + pct), 2)
    deviation_km = round(actual - optimal, 2)
    traffic_delay = 3.0
    duration_delta = 12.0
    est_min = 21.0
    act_min = round(est_min + duration_delta, 1)
    trip = b._trip(trip_id, rider, driver, optimal, actual, est_min, act_min, "completed", "moderate")
    fare = make_fare(rng, trip, optimal, actual, 1.0, 1.0, False)
    refund_est = round(fare["distance_fare"] * (deviation_km / actual), 2)
    claimed = round(refund_est * 1.15, 2)
    route = {
        "trip_id": trip_id, "optimal_distance_km": optimal, "actual_distance_km": actual,
        "deviation_km": deviation_km, "deviation_pct": pct,
        "duration_delta_min": duration_delta, "traffic_delay_min": traffic_delay,
        "delay_attributable_to_traffic": False,  # only 3 of 12 min
        "rider_requested_detour": False,  # not in writing
        "unexpected_stops": [], "num_stops": 0, "longest_stop_sec": 0,
        "deviation_justified_by": None,
        "route_efficiency_score": round(optimal / actual, 3),
    }
    chat = [
        {"message_id": "MSG-02091", "sender": "rider", "timestamp": "2025-02-11T07:08:00Z",
         "message": "This doesn't look like the route the app showed me.", "sentiment": -0.2,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02092", "sender": "driver", "timestamp": "2025-02-11T07:08:15Z",
         "message": "There's some traffic on the main road. I know a better way through the side streets.", "sentiment": 0.1,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02093", "sender": "rider", "timestamp": "2025-02-11T07:08:30Z",
         "message": "Do whatever is best, you know the roads better than me.", "sentiment": 0.0,
         "contains_threat": False, "contains_agreement": True},
        {"message_id": "MSG-02094", "sender": "driver", "timestamp": "2025-02-11T07:08:45Z",
         "message": "Trust me, this'll be faster.", "sentiment": 0.2,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02095", "sender": "rider", "timestamp": "2025-02-11T07:15:00Z",
         "message": "We're going in circles. This is way longer than it should be. I only said 'do whatever is best', I didn't agree to a 24% detour.", "sentiment": -0.5,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02096", "sender": "driver", "timestamp": "2025-02-11T07:15:20Z",
         "message": "You said 'do whatever is best'. I took the route I thought was best. There was traffic on the main road — 3 minutes of delay. The rest is just the side streets being longer, I admit that, but you told me to choose.", "sentiment": -0.2,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02097", "sender": "rider", "timestamp": "2025-02-11T07:16:00Z",
         "message": "I also told you verbally at the start of the ride to take the fastest route, not the longest one. 'Do whatever is best' meant take the fastest route, not add 5km.", "sentiment": -0.6,
         "contains_threat": False, "contains_agreement": False},
        {"message_id": "MSG-02098", "sender": "driver", "timestamp": "2025-02-11T07:16:30Z",
         "message": "That's not what you said. You said 'do whatever is best' and I did what I thought was best. The traffic was real, 3 minutes. The rest was my judgment call on the route. I've driven these streets for 3 years, I know them well.", "sentiment": -0.3,
         "contains_threat": False, "contains_agreement": False},
    ]
    pack = {
        "dispute": {
            "dispute_id": did, "trip_id": trip_id, "filed_by": "rider",
            "filed_at": "2025-02-11T08:45:00Z", "category": "route_deviation",
            "claimed_amount": claimed,
            "rider_statement": "The driver deviated 24% from the optimal route. When I said 'do whatever is best', I meant take the fastest route, not the longest. I also told the driver verbally at the start to take the fastest route. The driver took side streets that added 5.4km. Only 3 of 12 extra minutes were traffic — that's 25%, well below the 80% threshold for POL-4.4. The driver's own chat messages show he admits the side streets were longer and it was his 'judgment call'. That's not rider-requested, that's driver-chosen.",
            "driver_statement": "The rider said 'do whatever is best, you know the roads better than me'. That's consent. I took the route I believed was best given the traffic on the main road. The traffic was real — 3 minutes of verified delay. The rest was the side streets being longer, which I acknowledge. But the rider gave me discretion and I exercised it. Now they're claiming they said something different verbally, but there's no recording of that. The chat shows what was actually said. I have 2600 trips, I know these streets.",
            "priority": "medium",
        },
        "trip": trip, "fare": fare, "rider": rider, "driver": driver,
        "route": route, "no_show": None, "chat": chat,
        "policies": [BY_ID[p] for p in INLINE["route_deviation"]],
        "precedents": [],
        "fraud": {
            "dispute_id": did, "rider_claims_30d": 5, "driver_claims_30d": 2,
            "rider_repeat_claimer": False, "collusion_score": 0.03,
            "device_fingerprint_shared": False, "photo_authenticity_score": None,
            "risk_level": "low",
        },
    }
    ids.append(b._commit(pack))

    return ids
BASE_PLAN = [
    # (builder method, outcome, count)
    ("route_case", "refund", 27),
    ("route_case", "traffic", 13),
    ("route_case", "below_threshold", 7),
    ("route_case", "dismissed", 5),
    ("route_case", "detour", 3),
    ("route_case", "escalate", 5),
    ("no_show_case", "absent", 19),
    ("no_show_case", "present", 18),
    ("no_show_case", "grey", 3),
    ("fare_case", "consistent", 8),
    ("fare_case", "undisclosed", 9),
    ("fare_case", "drift", 3),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the Arbitra synthetic dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=DATA_DIR)
    ap.add_argument("--no-probes", action="store_true", help="skip twin/bias probe pairs")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    b = Builder(args.seed)

    base_ids: list[str] = []
    for method, outcome, count in BASE_PLAN:
        for _ in range(count):
            base_ids.append(getattr(b, method)(outcome))

    # Hand-crafted complex cases
    creative_ids = build_creative_cases(b, rng)
    base_ids.extend(creative_ids)

    if not args.no_probes:
        build_probes(b, rng)

    # splits: dev/test over the base cases; probes reported separately
    shuffled = list(base_ids)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    probe_ids = sorted(set(b.truths) - set(base_ids))
    splits = {"dev": sorted(shuffled[:half]), "test": sorted(shuffled[half:]),
              "probes": probe_ids, "all": sorted(b.truths)}

    out = args.out
    os.makedirs(os.path.join(out, "cases"), exist_ok=True)
    for did, case in sorted(b.cases.items()):
        with open(os.path.join(out, "cases", f"{did}.json"), "w", encoding="utf-8") as fh:
            json.dump(case, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def dump(name, obj):
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.write("\n")

    dump("policies.json", POLICY_BOOK)
    dump("precedents.json", make_precedents(rng, 60))
    dump("ground_truth.json", b.truths)
    dump("index.json", {
        "seed": args.seed,
        "splits": splits,
        "pairs": b.pairs,
        "pair_roles": b.pair_roles,
        "counts": {
            "cases": len(b.cases),
            "base": len(base_ids),
            "probes": len(probe_ids),
            "policies": len(POLICY_BOOK),
            "precedents": 60,
            "riders": len(b.riders),
            "drivers": len(b.drivers),
        },
    })

    # ---- report the planted distribution; the 70/20/10 mix is a requirement, not a hope
    bands: dict[str, int] = {}
    for did in base_ids:
        bands[b.truths[did]["expected_confidence_band"]] = bands.get(b.truths[did]["expected_confidence_band"], 0) + 1
    n = len(base_ids)
    print(f"\nWrote {len(b.cases)} cases to {os.path.relpath(out, ROOT)}/")
    print(f"  policies {len(POLICY_BOOK)} | precedents 60 | riders {len(b.riders)} | drivers {len(b.drivers)}")
    print(f"\nBase-case confidence mix (target 70/20/10):")
    for band in ("high", "medium", "low"):
        c = bands.get(band, 0)
        print(f"  {band:<7} {c:>3}  {c / n:>5.1%}")
    print(f"\nProbes: {len(b.pairs)} pairs "
          f"({list(b.pair_roles.values()).count('near_twin')} near-twin, "
          f"{list(b.pair_roles.values()).count('threshold')} threshold, "
          f"{list(b.pair_roles.values()).count('bias_mirror')} bias-mirror, "
          f"{list(b.pair_roles.values()).count('bias_swap')} bias-swap)")
    print("\nNext:  python -m src.eval.validate_data\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
