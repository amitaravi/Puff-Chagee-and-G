# DATA CONTRACT — read this before writing a line of generator code

**Owner of this contract:** Person B (agents). **Implementer:** Person A (data).

You already own `src/core/state.py`. That Pydantic file **is** the contract — this
document is just the on-disk encoding of it, so nobody has to reverse-engineer
Pydantic to produce JSON.

> The deal: produce files that match this. If you do, `ARBITRA_TOOLS=real` works and
> nothing in `agents/` changes. If you want to change a field, that's a contract
> change — say so, because prompts depend on field names.

---

## 1. Directory layout

```
data/
  policies.json        # list[Policy]          — the full policy book
  precedents.json      # list[Precedent]       — the RAG knowledge base
  cases/
    DSP-0001.json      # one file per dispute, EvidencePack-shaped
    DSP-0002.json
    ...
  ground_truth.json    # {dispute_id: Truth}   — NEVER loaded by src/tools/
  index.json           # splits + pair metadata (twin pairs, bias pairs)
```

That's it. Five artefacts. Everything else (raw `riders.json`, `drivers.json`,
`gps_telemetry.jsonl`, your generator internals) lives wherever you like — the
pipeline never reads it.

---

## 2. The rules that actually matter

| # | Rule | Why |
|---|---|---|
| 1 | **Every field in `state.py` must be present, spelled exactly.** Unknown keys are fine (ignored); missing keys that aren't Optional are a hard error. | Prompts render field names directly |
| 2 | **`ground_truth.json` must never be reachable from `src/tools/`.** | A leaked label makes the entire eval worthless and you'd never notice. `tests/test_leakage.py` asserts this. |
| 3 | **Every case file must load as `EvidencePack(**json)`.** | One-line validation, see §6 |
| 4 | **Generate the label FIRST, then the evidence to support it.** | Label-after-evidence produces cases no agent can resolve |
| 5 | **Plant the 70/20/10 confidence mix on purpose.** | All-clear-cut ⇒ escalation is undemonstrable. All-ambiguous ⇒ autonomy reads 0% |
| 6 | **Deterministic: fixed seed, sorted keys.** | Re-running must give byte-identical output or the eval isn't reproducible |

---

## 3. Field reference (mirrors `state.py`, grouped by who reads it)

### 3.1 `dispute` — required

```jsonc
"dispute": {
  "dispute_id": "DSP-0001",              // str, unique, filenames match
  "trip_id": "TRP-00431",
  "filed_by": "rider",                   // "rider" | "driver"
  "filed_at": "2025-02-11T07:48:00Z",    // ISO-8601
  "category": "route_deviation",         // see §4
  "claimed_amount": 3.25,                // float, USD
  "rider_statement": "...",              // optional but strongly preferred
  "driver_statement": "...",             // optional but strongly preferred
  "priority": "medium"                   // low|medium|high|critical
}
```

`priority: "critical"` forces escalation regardless of confidence. Seed a few.

### 3.2 `trip` — required

```jsonc
"trip": {
  "trip_id": "TRP-00431",
  "rider_id": "RDR-0007",
  "driver_id": "DRV-0012",
  "requested_at": "...", "started_at": "...", "ended_at": "...",
  "pickup_address": "SFO Terminal 2",
  "dropoff_address": "101 California St",   // null on no_show / cancelled
  "status": "completed",                    // completed|cancelled_by_rider|cancelled_by_driver|no_show
  "estimated_distance_km": 20.4,
  "actual_distance_km": 24.0,
  "estimated_duration_min": 28,
  "actual_duration_min": 41,
  "city": "San Francisco",
  "traffic_conditions": "light"             // light|moderate|heavy
}
```

### 3.3 `route` — **required iff `category == "route_deviation"`**

Agents reason over this derived object, **not** raw polylines.

```jsonc
"route": {
  "trip_id": "TRP-00431",
  "optimal_distance_km": 20.4,
  "actual_distance_km": 24.0,
  "deviation_km": 3.6,
  "deviation_pct": 0.176,               // FRACTION not percent. 0.176 == 17.6%
  "duration_delta_min": 13,
  "traffic_delay_min": 2,
  "delay_attributable_to_traffic": false,
  "rider_requested_detour": false,
  "unexpected_stops": [                 // [] is fine
    { "lat": 37.6213, "lng": -122.379, "duration_sec": 180, "at_km": 8.2 }
  ],
  "num_stops": 1,
  "longest_stop_sec": 180,
  "deviation_justified_by": null,       // null|"traffic"|"rider_request"|"road_closure"
  "route_efficiency_score": 0.85        // 0..1
}
```

**Gotcha that has bitten us already:** `deviation_pct` is a fraction. `0.176`, not `17.6`.
Consistency check: `deviation_km == actual_distance_km - optimal_distance_km` and
`deviation_pct ≈ deviation_km / optimal_distance_km`.

### 3.4 `no_show` — **required iff `category == "no_show"`**

```jsonc
"no_show": {
  "driver_min_distance_to_pickup_m": 80.0,
  "wait_time_min": 4.0,
  "rider_gps_approaching": false,
  "no_show_fee_charged": 5.0
}
```

### 3.5 `fare` — always required

```jsonc
"fare": {
  "trip_id": "TRP-00431",
  "base_fare": 2.55, "distance_fare": 21.67, "time_fare": 9.40,
  "booking_fee": 2.00,
  "surge_multiplier": 1.0, "surge_amount": 0.0,
  "tolls": 0.0,
  "promo_code": null, "promo_discount": 0.0,
  "total_charged": 35.62,
  "estimated_fare_at_booking": 32.00,
  "surge_at_booking": 1.0,
  "currency": "USD"
}
```

Consistency: `total_charged ≈ base + distance + time + booking + surge + tolls - promo`.

### 3.6 `rider` / `driver` — always required

```jsonc
"rider": {
  "party_id": "RDR-0007", "role": "rider", "name": "Maya Chen",
  "account_age_days": 1024, "total_trips": 312, "avg_rating": 4.9,
  "dispute_count_90d": 2, "dispute_win_rate": 0.5,
  "cancellation_rate": 0.04, "is_flagged_abuse": false
}
```

### 3.7 `chat` — always required (`[]` allowed)

```jsonc
"chat": [
  { "message_id": "MSG-00914", "sender": "rider", "timestamp": "2025-02-11T07:14:22Z",
    "message": "Why are we going this way?",
    "sentiment": -0.42, "contains_threat": false, "contains_agreement": false }
]
```

`sentiment` is −1..1. Seed at least a handful with `contains_threat: true`.

### 3.8 `policies` — always required

The **relevant** policies for this case, inlined. Not the whole book.

```jsonc
"policies": [
  { "policy_id": "POL-4.3", "category": "route_deviation",
    "title": "Route Deviation - Refund Eligibility",
    "rule_text": "If the actual route exceeds the optimal route by more than 15%, and ...",
    "conditions": { "deviation_pct_gt": 0.15, "traffic_justified": false },
    "remedy_type": "partial_refund",                       // partial_refund|full_refund|no_action|compensate_driver
    "remedy_formula": "distance_fare * (deviation_km / actual_distance_km)",
    "version": "v2.1" }
]
```

`remedy_formula` is **executed** by the eval harness's recomputation check (§5, check 7).
It must be a valid Python expression over `fare.*` and `route.*` field names — e.g.
`distance_fare`, `deviation_km`, `actual_distance_km`, `total_charged`,
`estimated_fare_at_booking`, `surge_amount`, `surge_multiplier`, `surge_at_booking`,
`no_show_fee_charged`. No function calls, no imports.

### 3.9 `precedents` — optional on the case file

Usually left `[]`; the tool layer retrieves them at runtime from `precedents.json`.
If you inline them, they're treated as already-retrieved.

```jsonc
"precedents": [
  { "precedent_id": "PRC-0088", "category": "route_deviation",
    "fact_summary": "18.1km trip, 3.2km deviation, light traffic.",
    "key_facts": { "deviation_pct": 0.177 },
    "ruling": "refund_rider", "reasoning": "...",
    "cited_policy_id": "POL-4.3", "confidence": 0.93, "decided_at": "2024-11-19" }
]
```

**Seed at least two precedents per category with the SAME key facts and OPPOSITE
rulings.** That's what makes the "precedent conflict" confidence penalty fire, and
it's the honest representation of a real precedent base.

### 3.10 `fraud` — optional, **JUDGE ONLY**

```jsonc
"fraud": {
  "dispute_id": "DSP-0001",
  "rider_claims_30d": 1, "driver_claims_30d": 0,
  "rider_repeat_claimer": false,
  "collusion_score": 0.06,
  "device_fingerprint_shared": false,
  "photo_authenticity_score": null,     // 0..1, only for damage/cleaning
  "risk_level": "low"                   // low|medium|high
}
```

`advocate_view()` strips this before either advocate sees the pack. Don't duplicate
any of it into `rider`/`driver`/`chat` — a leak there defeats the redaction.

---

## 4. Categories

| `category` | Status | Needs | Remedy shape |
|---|---|---|---|
| `route_deviation` | **MVP** | `route` | continuous (partial refund) |
| `no_show` | **MVP** | `no_show` | binary (fee stands or not) |
| `fare_surge` | 3rd (nice to have) | `fare` | continuous |
| `damage_cleaning` | stretch | `fraud.photo_authenticity_score` | continuous, capped |

MVP gate is two categories. `fare_surge` is cheap to add and gives a third;
`damage_cleaning` needs the multi-modal path, so leave it unless there's time.

---

## 5. `ground_truth.json`

```jsonc
{
  "DSP-0001": {
    "correct_ruling": "refund_rider",       // refund_rider|refund_driver|no_action|dismissed
    "correct_refund_amount": 3.25,
    "correct_policy_id": "POL-4.3",
    "expected_confidence_band": "high",     // high >=0.85 | medium 0.60-0.85 | low <0.60
    "should_escalate": false,
    "rationale": "Deviation 17.6% exceeds 15%...",
    "pair_id": null,                        // see §7
    "pair_role": null                       // "near_twin" | "threshold" | "bias_mirror" | "bias_swap"
  }
}
```

**Generated by `src/policy/apply_policy.py`, never by an LLM.** If an LLM writes your
labels, you have only measured "how often does our judge agree with GPT-4".

---

## 6. Validate before you hand it over

```bash
python -m src.eval.validate_data
```

Checks: every file parses as `EvidencePack`, `route`/`no_show` present for their
categories, fare arithmetic consistent, `deviation_pct` is a fraction, no ground-truth
keys anywhere in the case files, 70/20/10 band distribution, every `policy_id` cited in
ground truth exists. It exits non-zero on any failure — run it in CI.

---

## 7. The three extra case sets (do these — they're the differentiator)

Generated by the same `make_case()` core, with `pair_id`/`pair_role` set:

| Set | Count | Construction | What it measures |
|---|---|---|---|
| **near-twin pairs** | 30 pairs | Same facts, ±0.4pp perturbation (deviation 0.176 → 0.180). Both must get the **same** ruling. | Is the judge wobbling? |
| **threshold pairs** | 20 pairs | Straddle the policy line (0.14 vs 0.16 around the 0.15 threshold). Ruling must **flip**. | Is the judge actually applying the rule? |
| **bias mirrors** | 25 pairs | Identical no-show facts, `filed_by` flipped rider ↔ driver. | Does the judge favour a side? |
| **bias swaps** | 25 pairs | Same route case, rider/driver names + ratings swapped. | Is the judge sensitive to irrelevant party attributes? |

Each pair shares a `pair_id`. ~220 cases total.

---

## 8. Interface you implement

Four functions in `src/tools/real_tools.py`:

```python
def get_evidence_pack(self, dispute_id: str) -> EvidencePack
def get_policy(self, policy_id: str) -> Policy
def search_precedents(self, facts: dict, k: int = 3) -> list[Precedent]
def get_fraud_signals(self, dispute_id: str) -> FraudSignals
```

A reference implementation already exists — read it before writing your own, it's
~120 lines and does exactly this. Replacing it with a version that reads your own
table layout is fine; **keep the four signatures**.

---

## 9. Volumes

| Thing | Count |
|---|---|
| riders | 40 |
| drivers | 30 |
| trips | 200 |
| base disputes | 120 (60 route_deviation / 40 no_show / 20 fare_surge) |
| + pair sets | ~100 (see §7) |
| policies | 25 |
| precedents | 60 |

Target total repo size under 10 MB so it commits cleanly.
