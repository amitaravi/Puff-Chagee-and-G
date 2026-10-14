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


# ----------------------------------------------------------------- main
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
