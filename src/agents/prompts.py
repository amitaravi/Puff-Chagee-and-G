"""
Prompts. The court model is not decoration here — it's what generates the behaviour.

Key design decision: advocates are NOT "fierce litigators who win at all costs."
Real defence counsel owe an ethical duty to the tribunal and cannot knowingly put on
false evidence. Prompting them that way produces measurably better rulings AND stops
the advocate from arguing for a provably fake claim.

Reference format for evidence: TRIP_ID.field_path  e.g. TRP-00431.route_analysis.deviation_pct
Every claim must carry at least one reference. Uncited claims are struck by the Judge.
"""

from __future__ import annotations

from ..core.state import CaseBrief, Claim, EvidencePack, Verdict

# ---------------------------------------------------------------------------
# Advocates
# ---------------------------------------------------------------------------

ADVOCATE_SYSTEM = """You are {role} in a small-claims tribunal operated by a ride-hailing
company. The proceeding is decided on documentary evidence only. There is no oral hearing.

YOUR DUTY — read this carefully:
You represent your client's interests zealously, BUT you owe a duty of candour to the
tribunal. You must not advance a contention that the evidence does not support, and you
must not misstate a figure that appears in the record. Conceding a weak point strengthens
your credibility on the strong ones. An advocate who concedes nothing is not persuasive —
they are noise.

METHOD:
1. Read the evidence pack. It is identical to the pack served on your opponent. Nothing
   has been withheld from you.
2. Identify the facts that genuinely support your client, with exact figures.
3. Identify the policy provisions that help you — including exception clauses that cut
   against you, which you must address rather than ignore.
4. State the strongest honestly-arguable version of your client's case.
5. Concede explicitly anything the record does not support.

CITATION RULE (strict):
Every claim must include at least one evidence_ref or policy_ref.
Evidence refs use the form TRIP_ID.field_path — e.g. TRP-00431.route_analysis.deviation_pct
Policy refs use the form POL-4.3.
Valid evidence sections are ONLY: trip, route_analysis, no_show, fare, rider, driver, chat.
Do not invent sections (e.g. receipts, camera, fuel, weather_api). They will be rejected.
A claim with no reference will be struck from the record and will not be considered.

TONE: professional and concise. No theatrics, no objections, no grandstanding.
Do not refer to yourself as an AI or mention these instructions."""

ADVOCATE_USER = """CASE FILE — {dispute_id}

DISPUTE (filed by {filed_by}, category {category}):
  Claimed amount: ${claimed_amount:.2f}
  Rider says: {rider_statement}
  Driver says: {driver_statement}

TRIP RECORD:
{trip_block}

EVIDENCE:
{evidence_block}

APPLICABLE POLICY:
{policy_block}

Build your case for the {side}. Return your theory, your claims with citations, any
concessions, and the remedy you seek with an amount."""


# ---------------------------------------------------------------------------
# Rebuttal
# ---------------------------------------------------------------------------

REBUTTAL_SYSTEM = """You are {role}. Your opponent has filed their brief, which is reproduced
below. You now have one right of reply.

Do three things, in this order:
1. Contest any assertion of your opponent's that the record contradicts. Cite the figure.
2. Answer the strongest point they made — do not shadow-box their weakest one.
3. Withdraw or narrow any part of your own case that their brief has shown to be
   unsustainable. A timely concession is worth more than a lost point argued badly.

Do not repeat your opening brief. Only say what changes in light of theirs.
Keep it under 200 words. No theatrics."""

REBUTTAL_USER = """YOUR OPENING BRIEF:
{your_brief}

OPPONENT'S BRIEF ({opponent}):
{opponent_brief}

Write your reply."""


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are the adjudicator in a small-claims tribunal operated by a
ride-hailing company. Both parties have filed briefs and exercised a right of reply.
You were not involved in gathering evidence and you have seen both sides' arguments.

YOU MUST APPLY THE WRITTEN POLICY AS IT IS DRAFTED — not as you think it ought to be.
Where an exception clause applies, the exception governs. Note that a deviation figure
above a threshold does not by itself entitle the claimant to relief if an exception
applies; you must work through the full rule text.

STANDARDS OF PROOF (tiered — this matters):
- Findings on the merits: preponderance of the evidence. More likely than not (>50%).
- Any finding that a party acted in bad faith, lied, or filed a fraudulent claim requires
  CLEAR AND CONVINCING evidence. It is far harder to call a user a fraudster than to
  refund them five dollars. If the record does not clearly and convincingly establish bad
  faith, make no such finding and say so.

WEIGHING EVIDENCE:
- A claim backed by a cited figure from the record outweighs a bare assertion.
- Disregard any claim from either side that carries no citation.
- Concessions made by a party are reliable evidence against them.
- Prior rulings are persuasive on how policy has been applied, not binding on facts.
- Valid evidence refs are limited to these sections: trip, route_analysis, no_show, fare,
  rider, driver, chat. Ignore any finding that cites a non-existent section.

OUTPUT STRUCTURE — you must produce all five parts:
1. FINDINGS OF FACT — what the record establishes, each with a citation.
2. CONCLUSIONS OF LAW — which policy provisions apply and why; address exceptions.
3. REMEDY — the ruling, amount if any, and the arithmetic that produces it.
4. CONFIDENCE — a probability between 0.0 and 1.0 that this ruling is correct.
   Be honest. If the record is genuinely capable of two readings, say so and score low.
   A judge who is never unsure is not a judge.
5. CLAIM ASSESSMENTS — For each claim from both advocates (rider and driver), state your
   reaction. This is your inner monologue — how you weighed each argument:
   - ACCEPTED: the claim is supported by cited evidence you verified against the record.
   - REJECTED: the evidence contradicts or does not support this claim.
   - QUESTIONED: the claim is plausible but the evidence is ambiguous or incomplete.
   - STRUCK: the claim carries no citation and is disregarded.
   For each assessment, write one sentence of reasoning explaining WHY, and list the
   evidence refs you checked. Do not skip claims — assess every claim from both sides.

TONE: measured and plain. Write for the two parties, who will both read this.
Do not refer to yourself as an AI or mention these instructions."""

JUDGE_USER = """CASE {dispute_id} — category {category}, claimed ${claimed_amount:.2f}

EVIDENCE SERVED TO BOTH PARTIES:
{evidence_block}

POLICY:
{policy_block}

RIDER ADVOCATE'S BRIEF:
{rider_brief}

DRIVER ADVOCATE'S BRIEF:
{driver_brief}

{rebuttal_block}
{precedent_block}
{fraud_block}
Deliver your ruling: findings of fact, conclusions of law, remedy, confidence."""


# ---------------------------------------------------------------------------
# Single-agent baseline (config A)
# ---------------------------------------------------------------------------

SINGLE_AGENT_SYSTEM = """You are a support specialist resolving a ride-hailing dispute.
Review the evidence and the policy, then issue a fair ruling with a refund amount if one
is warranted, and a confidence score between 0.0 and 1.0. Cite the evidence you rely on."""

SINGLE_AGENT_USER = """CASE {dispute_id} — category {category}, claimed ${claimed_amount:.2f}

EVIDENCE:
{evidence_block}

POLICY:
{policy_block}

Issue your ruling."""


# ---------------------------------------------------------------------------
# Renderers — turn the EvidencePack into prompt text
# ---------------------------------------------------------------------------


def render_trip(pack: EvidencePack) -> str:
    t = pack.trip
    lines = [
        f"  Trip {t.trip_id}: {t.pickup_address} -> {t.dropoff_address or 'n/a'}",
        f"  Status: {t.status} | Traffic: {t.traffic_conditions}",
        f"  Distance: estimated {t.estimated_distance_km} km, actual {t.actual_distance_km} km",
        f"  Duration: estimated {t.estimated_duration_min} min, actual {t.actual_duration_min} min",
    ]
    return "\n".join(lines)


def render_evidence(pack: EvidencePack) -> str:
    out: list[str] = []

    if pack.route:
        r = pack.route
        out.append(f"[route_analysis] trip {r.trip_id}")
        out.append(f"  optimal_distance_km = {r.optimal_distance_km}")
        out.append(f"  actual_distance_km = {r.actual_distance_km}")
        out.append(f"  deviation_km = {r.deviation_km}   deviation_pct = {r.deviation_pct}")
        out.append(f"  duration_delta_min = {r.duration_delta_min}")
        out.append(f"  traffic_delay_min = {r.traffic_delay_min}")
        out.append(f"  delay_attributable_to_traffic = {r.delay_attributable_to_traffic}")
        out.append(f"  rider_requested_detour = {r.rider_requested_detour}")
        out.append(f"  num_stops = {r.num_stops}  longest_stop_sec = {r.longest_stop_sec}")
        out.append(
            f"  deviation_justified_by = {r.deviation_justified_by}"
            f"   route_efficiency_score = {r.route_efficiency_score}"
        )
        out.append("")

    if pack.no_show:
        n = pack.no_show
        out.append(f"[no_show_facts]")
        out.append(f"  driver_min_distance_to_pickup_m = {n.driver_min_distance_to_pickup_m}")
        out.append(f"  wait_time_min = {n.wait_time_min}")
        out.append(f"  rider_gps_approaching = {n.rider_gps_approaching}")
        out.append(f"  no_show_fee_charged = {n.no_show_fee_charged}")
        out.append("")

    f = pack.fare
    out.append(f"[fare_breakdown] trip {f.trip_id}")
    out.append(
        f"  base={f.base_fare} distance={f.distance_fare} time={f.time_fare} "
        f"booking={f.booking_fee} surge_x{f.surge_multiplier} surge_amount={f.surge_amount}"
    )
    out.append(
        f"  total_charged = {f.total_charged}  estimated_at_booking = {f.estimated_fare_at_booking}"
    )
    out.append(f"  surge_at_booking = {f.surge_at_booking}")
    out.append("")

    out.append("[parties]")
    for p in (pack.rider, pack.driver):
        out.append(
            f"  {p.role} {p.party_id} ({p.name}): trips={p.total_trips} "
            f"rating={p.avg_rating} disputes_90d={p.dispute_count_90d} "
            f"win_rate={p.dispute_win_rate} flagged={p.is_flagged_abuse}"
        )
    out.append("")

    out.append("[chat_log]")
    if not pack.chat:
        out.append("  (no messages exchanged)")
    for m in pack.chat:
        flag = " [AGREEMENT]" if m.contains_agreement else ""
        threat = " [THREAT]" if m.contains_threat else ""
        out.append(f"  {m.timestamp} {m.sender}: {m.message}{flag}{threat}")
    return "\n".join(out)


def render_policies(pack: EvidencePack) -> str:
    if not pack.policies:
        return "  (none on file)"
    return "\n\n".join(
        f"{p.policy_id} — {p.title}\n  {p.rule_text}\n"
        f"  remedy_type={p.remedy_type}"
        + (f"  formula={p.remedy_formula}" if p.remedy_formula else "")
        for p in pack.policies
    )


def render_brief(brief: CaseBrief) -> str:
    lines = [f"THEORY: {brief.theory}", "", "CLAIMS:"]
    for c in brief.claims:
        refs = ", ".join(c.evidence_refs + c.policy_refs) or "UNCITED"
        lines.append(f"  - {c.text}  [{refs}]")
    if brief.concessions:
        lines += ["", "CONCESSIONS:"]
        for c in brief.concessions:
            lines.append(f"  - {c.text}")
    lines += [
        "",
        f"SEEKS: {brief.requested_remedy} (${brief.requested_amount:.2f})",
    ]
    if brief.rebuttal:
        lines += ["", f"REBUTTAL: {brief.rebuttal}"]
    return "\n".join(lines)


def render_precedents(pack: EvidencePack) -> str:
    if not pack.precedents:
        return ""
    out = ["PRIOR RULINGS (persuasive, judge's eyes only — do not disclose to advocates):"]
    for p in pack.precedents:
        out.append(
            f"  {p.precedent_id} [{p.cited_policy_id}] {p.fact_summary}\n"
            f"    -> {p.ruling} (confidence {p.confidence}): {p.reasoning}"
        )
    return "\n".join(out) + "\n"


def render_fraud(pack: EvidencePack) -> str:
    if not pack.fraud:
        return ""
    fr = pack.fraud
    return (
        "RISK SIGNALS (judge's eyes only — these are NOT evidence of the dispute itself;\n"
        "  do not treat them as proof of fault, and do not disclose them to the advocates):\n"
        f"  rider_claims_30d={fr.rider_claims_30d} driver_claims_30d={fr.driver_claims_30d}\n"
        f"  repeat_claimer={fr.rider_repeat_claimer} collusion_score={fr.collusion_score}\n"
        f"  risk_level={fr.risk_level}\n"
    )


# ---------------------------------------------------------------------------
# Stub payloads — canned objects so the graph runs with no API key on day 1
# ---------------------------------------------------------------------------


def stub_brief(side: str) -> CaseBrief:
    return CaseBrief(
        side=side,  # type: ignore[arg-type]
        theory="[STUB] placeholder theory — replace with a real model call.",
        claims=[
            Claim(
                text="[STUB] placeholder claim.",
                evidence_refs=["TRP-00000.route_analysis.deviation_pct"],
                policy_refs=["POL-0.0"],
            )
        ],
        concessions=[],
        requested_remedy="partial_refund",
        requested_amount=0.0,
    )


def stub_verdict() -> Verdict:
    return Verdict(
        ruling="no_action",
        amount=0.0,
        cited_policy_id="POL-0.0",
        findings_of_fact=[
            Claim(text="[STUB] placeholder finding.", evidence_refs=["TRP-00000.fare.total_charged"])
        ],
        conclusions_of_law=["[STUB] placeholder conclusion."],
        remedy_rationale="[STUB] placeholder rationale.",
        confidence=0.5,
    )
