"""
Arbitra — Multi-agent autonomous dispute resolution for ride-hailing.

A Rider Advocate and Driver Advocate build opposed cases from an identical evidence
pack; a Judge applies written policy and issues a four-part opinion with a calibrated
confidence score. Modeled on a small-claims tribunal.

Tab 1 — Live Court: a music-player-style live tribunal. Hit Play, watch it unfold.
Tab 2 — Case Analysis: visual breakdown of the verdict — advocate face-off,
         claim resolution, and whether the debate moved the needle.

Run:  streamlit run demo/app.py
"""

from __future__ import annotations

import os
import sys
import time

# Streamlit runs this file directly, so the repo root isn't on sys.path by default.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from src.agents.judge import ADJUSTMENTS
from src.core.config import CONFIDENCE_THRESHOLD, flags_for
from src.core.state import CaseBrief, ClaimAssessment, EvidencePack, Verdict, Claim, CaseState
from src.graph.workflow import build_graph
from src.policy.apply_policy import band_for
from src.tools.interface import get_tools

# ============================================================ page setup
st.set_page_config(
    page_title="Arbitra",
    page_icon=":scales:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================ constants

ALL_STAGES = [
    "docket", "discovery", "motions", "settlement",
    "rider_advocate", "driver_advocate", "provisional_verdict",
    "rider_rebuttal", "driver_rebuttal", "judge", "gate", "judgment",
]

STAGE_LABELS = {
    "docket": "Docket",
    "discovery": "Discovery",
    "motions": "Motions",
    "settlement": "Settlement",
    "rider_advocate": "Rider Advocate",
    "driver_advocate": "Driver Advocate",
    "provisional_verdict": "Provisional Verdict",
    "rider_rebuttal": "Rider Rebuttal",
    "driver_rebuttal": "Driver Rebuttal",
    "judge": "Judge",
    "gate": "Gate",
    "judgment": "Judgment",
}

STAGE_DESCRIPTIONS = {
    "docket": "Case filed. Loading evidence pack and parsing dispute metadata.",
    "discovery": "Mutual discovery. Both sides served identical, redacted evidence. Precedents and fraud signals retrieved for the judge.",
    "motions": "Motions officer checks for frivolous claims (de minimis) or undisputed facts (summary judgment). Fast path #1.",
    "settlement": "Settlement conference. Nuisance-value offer for ambiguous, low-value cases. Fast path #2.",
    "rider_advocate": "Rider's counsel builds their case — theory, claims with evidence citations, concessions.",
    "driver_advocate": "Driver's counsel builds their case — opposing theory, counter-claims, concessions.",
    "provisional_verdict": "Judge issues a provisional ruling BEFORE seeing rebuttals. This is what we measure the debate against.",
    "rider_rebuttal": "Rider's advocate replies to the driver's brief. Contests, concedes, or narrows.",
    "driver_rebuttal": "Driver's advocate replies to the rider's brief. Contests, concedes, or narrows.",
    "judge": "Judge issues final ruling — findings of fact, conclusions of law, remedy, confidence.",
    "gate": "Confidence gate. Below 0.75 threshold = referred to human appellate review.",
    "judgment": "Final judgment recorded. Case closed.",
}

# Short verbs for the now-playing spinner
STAGE_ACTION = {
    "docket": "Filing case...",
    "discovery": "Serving evidence to both parties...",
    "motions": "Reviewing motions...",
    "settlement": "Conferencing settlement...",
    "rider_advocate": "Rider's counsel is building their case...",
    "driver_advocate": "Driver's counsel is building their case...",
    "provisional_verdict": "Judge is considering a preliminary ruling...",
    "rider_rebuttal": "Rider's advocate is replying...",
    "driver_rebuttal": "Driver's advocate is replying...",
    "judge": "Judge is deliberating...",
    "gate": "Checking confidence threshold...",
    "judgment": "Recording judgment...",
}

RULING_COLORS = {
    "refund_rider": ("blue", "Refund Rider"),
    "refund_driver": ("purple", "Refund Driver"),
    "no_action": ("gray", "No Action"),
    "dismissed": ("red", "Dismissed"),
}

CHECK = "\u2705"
CROSS = "\u274c"
DASH = "\u2014"

STAGE_ORDER = {s: i for i, s in enumerate(ALL_STAGES)}

# Stage accent colors for the now-playing panel
STAGE_ACCENT = {
    "docket": "#6c757d",
    "discovery": "#6c757d",
    "motions": "#fd7e14",
    "settlement": "#20c997",
    "rider_advocate": "#007bff",
    "driver_advocate": "#fd7e14",
    "provisional_verdict": "#ffc107",
    "rider_rebuttal": "#007bff",
    "driver_rebuttal": "#fd7e14",
    "judge": "#ffc107",
    "gate": "#dc3545",
    "judgment": "#28a745",
}

# ============================================================ helpers


def get_trace_map(trace: list[dict]) -> dict[str, dict]:
    return {step["stage"]: {k: v for k, v in step.items() if k != "stage"} for step in trace}


def is_fast_path(state) -> tuple[bool, str, str]:
    motion = state.get("motion_outcome", "none")
    if motion == "dismissed":
        return True, "Dismissed by Motion", state.get("motion_reason", "")
    if motion == "summary_judgment":
        return True, "Summary Judgment", state.get("motion_reason", "")
    if state.get("settlement_accepted"):
        offer = state.get("settlement_offer", 0)
        return True, "Settled", f"Settlement offer of ${offer:.2f} accepted."
    return False, "", ""


def fast_path_exit_stage(state) -> str | None:
    motion = state.get("motion_outcome", "none")
    if motion in ("dismissed", "summary_judgment"):
        return "motions"
    if state.get("settlement_accepted"):
        return "settlement"
    return None


def key_evidence_summary(pack: EvidencePack) -> str:
    """One-line summary of the core fact for the case card."""
    d = pack.dispute
    if d.category == "route_deviation" and pack.route:
        r = pack.route
        over = "above" if r.deviation_pct > 0.15 else "below"
        return f"Deviation {r.deviation_pct:.1%} ({over} 15% threshold) | Traffic share {r.traffic_delay_min / r.duration_delta_min:.0%}" if r.duration_delta_min > 0 else f"Deviation {r.deviation_pct:.1%} ({over} 15% threshold)"
    if d.category == "no_show" and pack.no_show:
        n = pack.no_show
        zone = "beyond 100m (refund)" if n.driver_min_distance_to_pickup_m > 100 else ("within 50m (fee stands)" if n.driver_min_distance_to_pickup_m <= 50 else "grey zone 50-100m")
        return f"Driver distance {n.driver_min_distance_to_pickup_m:.0f}m — {zone} | Waited {n.wait_time_min:.0f} min"
    if d.category == "fare_surge":
        f = pack.fare
        diff = "undisclosed surge" if f.surge_multiplier > f.surge_at_booking else "consistent"
        return f"Surge {f.surge_multiplier:.2f}x vs booking {f.surge_at_booking:.2f}x ({diff}) | Surge ${f.surge_amount:.2f}"
    if d.category == "damage_cleaning":
        return f"Damage/cleaning claim — ${pack.dispute.claimed_amount:.2f}"
    return f"{d.category} — ${d.claimed_amount:.2f}"


def render_progress_bar(trace_map: dict, current_stage: str | None, exit_stage: str | None) -> str:
    """Horizontal connected progress bar as HTML. Replaces pipeline pills."""
    ran_stages = set(trace_map.keys())
    n = len(ALL_STAGES)
    seg_width = 100.0 / n

    parts = ['<div style="display:flex;align-items:center;width:100%;margin:8px 0;">']
    for i, s in enumerate(ALL_STAGES):
        if s == current_stage:
            color = "#007bff"
            text_color = "white"
            extra = "box-shadow:0 0 8px rgba(0,123,255,0.6);"
        elif s in ran_stages:
            color = "#28a745"
            text_color = "white"
            extra = ""
        elif exit_stage and STAGE_ORDER[s] > STAGE_ORDER[exit_stage]:
            color = "#f0f0f0"
            text_color = "#ccc"
            extra = "text-decoration:line-through;"
        else:
            color = "#f0f0f0"
            text_color = "#bbb"
            extra = ""

        is_last = i == n - 1
        radius = "border-radius:4px 0 0 4px;" if i == 0 else ("border-radius:0 4px 4px 0;" if is_last else "border-radius:0;")
        margin = "margin-right:2px;" if not is_last else ""

        label = STAGE_LABELS.get(s, s.replace("_", " ").title())
        # Short label for the bar
        short = label.replace(" Advocate", "").replace(" Verdict", "").replace("Rebuttal", "Reb.")

        parts.append(
            f'<div style="flex:1;background:{color};color:{text_color};'
            f'padding:6px 4px;text-align:center;font-size:0.6rem;font-weight:600;'
            f'{radius}{margin}{extra}white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis;">{short}</div>'
        )
    parts.append("</div>")

    # Full labels below
    parts.append('<div style="display:flex;width:100%;margin-bottom:12px;">')
    for i, s in enumerate(ALL_STAGES):
        ran = s in ran_stages
        is_current = s == current_stage
        skipped = exit_stage and STAGE_ORDER[s] > STAGE_ORDER[exit_stage]
        if skipped:
            tc = "#ccc"
        elif is_current:
            tc = "#007bff"
        elif ran:
            tc = "#28a745"
        else:
            tc = "#bbb"
        label = STAGE_LABELS.get(s, s)
        parts.append(
            f'<div style="flex:1;text-align:center;color:{tc};font-size:0.55rem;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def render_case_card(pack: EvidencePack | None, dispute_id: str, playing: bool = False) -> None:
    """Story summary card — plain-English setup before the tribunal starts."""
    if not pack:
        with st.container(border=True):
            st.write(f"**Case {dispute_id}**")
            st.caption("Evidence pack will load when the tribunal begins.")
        return

    d = pack.dispute
    category_label = d.category.replace("_", " ").title()

    # --- Build the story narrative ---
    if d.category == "route_deviation" and pack.route:
        r = pack.route
        pct = r.deviation_pct
        over_under = "longer" if pct > 0 else "shorter"
        traffic_note = ""
        if r.traffic_delay_min > 0 and r.duration_delta_min > 0:
            share = r.traffic_delay_min / r.duration_delta_min
            traffic_note = f" Traffic accounted for {share:.0%} of the extra time."
        detour_note = ""
        if r.rider_requested_detour:
            detour_note = " The rider asked for this detour in chat."
        story = (
            f"**{pack.rider.name}** booked a ride with **{pack.driver.name}**. "
            f"The route taken was {pct:.0%} {over_under} than the optimal path "
            f"({r.optimal_distance_km:.1f}km planned, {r.actual_distance_km:.1f}km actual)."
            f"{traffic_note}{detour_note}"
        )
        question = f"Was the {pct:.0%} deviation justified, or does the rider deserve a refund?"

    elif d.category == "no_show" and pack.no_show:
        n = pack.no_show
        dist = n.driver_min_distance_to_pickup_m
        if dist > 100:
            zone = "never got close to the pickup point"
        elif dist <= 50:
            zone = "was right at the pickup point"
        else:
            zone = "was in the grey zone (50-100m from the pin)"
        story = (
            f"**{pack.rider.name}** and **{pack.driver.name}** were supposed to meet for a pickup. "
            f"The driver {zone} — closest approach was {dist:.0f}m. "
            f"The driver waited {n.wait_time_min:.0f} minutes. "
            f"A no-show fee of ${n.no_show_fee_charged:.2f} was charged."
        )
        question = f"Should the ${n.no_show_fee_charged:.2f} fee be refunded, or does it stand?"

    elif d.category == "fare_surge":
        f = pack.fare
        diff = f.total_charged - f.estimated_fare_at_booking
        surge_note = ""
        if f.surge_multiplier > f.surge_at_booking + 0.01:
            surge_note = (
                f" Surge went from {f.surge_at_booking:.1f}x at booking to "
                f"{f.surge_multiplier:.1f}x at completion — the rider says they never saw the increase."
            )
        promo_note = ""
        if f.promo_code and f.promo_discount == 0:
            promo_note = f" A promo code ({f.promo_code}) was shown at booking but never applied."
        toll_note = ""
        if f.tolls > 0:
            toll_note = f" A ${f.tolls:.2f} toll was charged."
        story = (
            f"**{pack.rider.name}** was quoted ${f.estimated_fare_at_booking:.2f} at booking "
            f"but charged ${f.total_charged:.2f} — a difference of ${diff:.2f}."
            f"{surge_note}{promo_note}{toll_note}"
        )
        question = f"Was the ${diff:.2f} price difference justified, or should the rider be refunded?"

    elif d.category == "damage_cleaning":
        story = (
            f"**{pack.driver.name}** filed a ${d.claimed_amount:.2f} cleaning/damage claim "
            f"against rider **{pack.rider.name}**, alleging damage caused during the trip."
        )
        if pack.fraud and pack.fraud.photo_authenticity_score is not None:
            score = pack.fraud.photo_authenticity_score
            if score >= 0.70:
                story += f" A photo was submitted (authenticity score {score:.2f} — verified)."
            elif score < 0.40:
                story += f" A photo was submitted (authenticity score {score:.2f} — likely fabricated)."
            else:
                story += f" A photo was submitted (authenticity score {score:.2f} — inconclusive)."
        question = f"Is the ${d.claimed_amount:.2f} damage claim valid?"

    else:
        story = f"**{pack.rider.name}** vs **{pack.driver.name}** — {category_label} dispute."
        question = "What is the correct ruling?"

    # --- Render the card ---
    with st.container(border=True):
        # Header: case ID + category tag
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
            f'<span style="font-size:1.3rem;font-weight:700;">{d.dispute_id}</span>'
            f'<span style="background:#e9ecef;padding:3px 10px;border-radius:12px;'
            f'font-size:0.75rem;font-weight:600;color:#495057;">{category_label}</span>'
            f'<span style="margin-left:auto;font-size:1.1rem;font-weight:700;color:#dc3545;">'
            f'${d.claimed_amount:.2f} at stake</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # The story
        st.markdown(story)

        # The question
        st.markdown(
            f'<div style="background:#f8f9fa;padding:10px 14px;border-radius:8px;'
            f'margin:10px 0;border-left:3px solid #007bff;">'
            f'<span style="font-size:0.85rem;color:#495057;font-weight:600;">'
            f'What the tribunal will decide:</span><br/>'
            f'<span style="font-size:0.95rem;">{question}</span></div>',
            unsafe_allow_html=True,
        )

        # Filed by + priority
        st.caption(
            f"Filed by **{d.filed_by}**  |  Priority: **{d.priority}**  |  "
            f"Rider rating: {pack.rider.avg_rating:.1f}  |  Driver rating: {pack.driver.avg_rating:.1f}"
        )


def render_now_playing(stage: str, state: CaseState, flags) -> None:
    """The main now-playing panel — rich content that swaps per stage."""
    label = STAGE_LABELS.get(stage, stage.replace("_", " ").title())
    accent = STAGE_ACCENT.get(stage, "#6c757d")
    desc = STAGE_DESCRIPTIONS.get(stage, "")

    # Header bar
    st.markdown(
        f'<div style="background:{accent};color:white;padding:10px 16px;'
        f'border-radius:8px 8px 0 0;">'
        f'<span style="font-size:1.1rem;font-weight:700;">{label}</span>'
        f'<span style="float:right;font-size:0.8rem;opacity:0.8;">{desc}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        _render_now_playing_body(stage, state, flags)


def _render_now_playing_body(stage: str, state: CaseState, flags) -> None:
    pack = state.get("pack")

    if stage == "docket":
        if pack:
            d = pack.dispute
            st.markdown(
                f"The clerk files **{d.dispute_id}** — a {d.category.replace('_', ' ')} "
                f"dispute filed by {d.filed_by}. "
                f"**{pack.rider.name}** (rider) claims **{pack.driver.name}** (driver) "
                f"owes them **${d.claimed_amount:.2f}**."
            )
        else:
            st.write("Loading evidence pack...")

    elif stage == "discovery":
        if pack:
            st.markdown(
                "Both sides are served identical evidence packs — route data, fare breakdown, "
                "chat logs, and policies. Neither side sees anything the other doesn't."
            )
            if pack.fraud:
                st.markdown(
                    f"The judge also receives **fraud signals** (risk: {pack.fraud.risk_level}, "
                    f"collusion: {pack.fraud.collusion_score:.2f}) — these are **never shown to advocates**."
                )
            if pack.precedents:
                st.caption(f"{len(pack.precedents)} precedents retrieved for the judge's reference.")

    elif stage == "motions":
        outcome = state.get("motion_outcome", "none")
        reason = state.get("motion_reason", "")
        if outcome == "none":
            st.markdown(
                "The motions officer reviews the case for **frivolous or de minimis claims** "
                "and **undisputed facts**. Neither threshold is met — the case proceeds to a full hearing."
            )
        elif outcome == "dismissed":
            st.error("Case dismissed by motion.")
            st.markdown(f"*{reason}*")
        elif outcome == "summary_judgment":
            st.warning("Summary judgment granted.")
            st.markdown(f"*{reason}*")

    elif stage == "settlement":
        offer = state.get("settlement_offer")
        accepted = state.get("settlement_accepted", False)
        if offer is None:
            st.markdown(
                "The settlement officer reviews whether this case is eligible for a quick "
                "nuisance-value settlement. It's not — the case proceeds to hearing."
            )
        elif accepted:
            st.success(f"Settled at ${offer:.2f}")
            st.markdown(
                "The claimant accepted a nuisance-value offer. No ruling on the merits — "
                "the case ends here."
            )
        else:
            st.markdown(
                f"The settlement officer offers **${offer:.2f}** to resolve the case quickly. "
                f"The claimant **rejects** it — they want their day in court."
            )

    elif stage in ("rider_advocate", "driver_advocate"):
        side = stage.split("_")[0]
        brief: CaseBrief | None = state.get(f"{side}_brief")
        if brief:
            party_name = pack.rider.name if side == "rider" else pack.driver.name
            st.markdown(f"**{side.title()}'s counsel ({party_name})** argues:")
            st.markdown(f"> {brief.theory}")
            if brief.claims:
                st.markdown("**Claims:**")
                for c in brief.claims:
                    refs = ", ".join(c.evidence_refs + c.policy_refs) or "uncited"
                    st.markdown(f"- {c.text}  *[{refs}]*")
            if brief.concessions:
                st.markdown("**Concedes:**")
                for c in brief.concessions:
                    st.markdown(f"- {c.text}")
            st.markdown(
                f"*Seeks: {brief.requested_remedy} (${brief.requested_amount:.2f})*"
            )
            st.caption("Argued concurrently from identical, redacted evidence.")
        else:
            st.write("Building case...")

    elif stage == "provisional_verdict":
        prov = state.get("provisional")
        if prov:
            _, rlabel = RULING_COLORS.get(prov.ruling, ("gray", prov.ruling))
            st.markdown(
                f"The judge issues a **provisional ruling: {rlabel}** at "
                f"{prov.confidence:.0%} confidence — **before seeing any rebuttals**. "
                f"This is the baseline the debate will be measured against."
            )
        else:
            st.write("Judge is considering...")

    elif stage in ("rider_rebuttal", "driver_rebuttal"):
        side = stage.split("_")[0]
        brief = state.get(f"{side}_brief")
        if brief and brief.rebuttal:
            st.markdown(f"**{side.title()}'s counsel replies:**")
            st.info(brief.rebuttal)
        else:
            st.write("Reply in progress...")

    elif stage == "judge":
        v = state.get("verdict")
        if v:
            _, rlabel = RULING_COLORS.get(v.ruling, ("gray", v.ruling))
            st.markdown(
                f"The judge issues the **final ruling: {rlabel}** — ${v.amount:.2f} "
                f"at {v.confidence:.0%} confidence."
            )
            if v.cited_policy_id:
                st.caption(f"Policy cited: {v.cited_policy_id}")
            if v.findings_of_fact:
                st.markdown("**Findings of Fact:**")
                for c in v.findings_of_fact:
                    refs = ", ".join(c.evidence_refs + c.policy_refs) or "no refs"
                    st.markdown(f"- {c.text}  *[{refs}]*")
            if v.conclusions_of_law:
                st.markdown("**Conclusions of Law:**")
                for i, c in enumerate(v.conclusions_of_law, 1):
                    st.markdown(f"{i}. {c}")
            if v.remedy_rationale:
                st.markdown(f"**Remedy:** {v.remedy_rationale}")
        else:
            st.write("Judge is deliberating...")

    elif stage == "gate":
        v = state.get("verdict")
        escalated = state.get("escalated", False)
        if v:
            if escalated:
                st.error(
                    f"Confidence {v.confidence:.0%} is below the {CONFIDENCE_THRESHOLD:.0%} "
                    f"threshold — **escalated to human appellate review**."
                )
                if v.escalation_reason:
                    st.caption(v.escalation_reason)
            else:
                st.success(
                    f"Confidence {v.confidence:.0%} clears the {CONFIDENCE_THRESHOLD:.0%} "
                    f"threshold — the ruling stands."
                )

    elif stage == "judgment":
        v = state.get("verdict")
        if v:
            _, rlabel = RULING_COLORS.get(v.ruling, ("gray", v.ruling))
            if v.should_escalate:
                st.error("Referred to human appellate review. Case closed.")
            else:
                st.success(f"Final judgment: {rlabel} — ${v.amount:.2f}. Case closed.")
        else:
            st.write("Recording judgment...")


def render_history_entry(step: dict) -> None:
    """Compact one-line summary for the history feed below now-playing."""
    stage = step["stage"]
    label = STAGE_LABELS.get(stage, stage.replace("_", " ").title())
    payload = {k: v for k, v in step.items() if k != "stage"}

    # Build a short summary from the payload
    summary = ""
    if stage == "docket":
        summary = f"Case {payload.get('dispute', '')} filed — {payload.get('category', '').replace('_', ' ')}"
    elif stage == "discovery":
        summary = f"Precedents: {payload.get('precedents', 0)} | Fraud: {payload.get('fraud', False)}"
    elif stage == "motions":
        outcome = payload.get("outcome", "none")
        summary = payload.get("reason", "no threshold met") if outcome != "none" else "No motion threshold met"
    elif stage == "settlement":
        if payload.get("offered") is None:
            summary = "No offer — not eligible"
        elif payload.get("accepted"):
            summary = f"Settled at ${payload.get('offered', 0):.2f}"
        else:
            summary = f"Offer ${payload.get('offered', 0):.2f} rejected"
    elif stage in ("rider_advocate", "driver_advocate"):
        summary = f"Theory: {str(payload.get('theory', ''))[:100]}"
    elif stage == "provisional_verdict":
        summary = f"Ruling: {payload.get('ruling', '?')} | Confidence: {payload.get('confidence', 0):.2f}"
    elif stage in ("rider_rebuttal", "driver_rebuttal"):
        summary = f"Reply filed ({payload.get('chars', 0)} chars)"
    elif stage == "judge":
        summary = f"Ruling: {payload.get('ruling', '?')} | Confidence: {payload.get('confidence', 0):.2f}"
    elif stage == "gate":
        summary = f"Escalated: {payload.get('escalated', False)}"
    elif stage == "judgment":
        summary = f"Ruling: {payload.get('ruling', '?')} | Amount: ${payload.get('amount', 0):.2f}"
    else:
        parts = []
        for k, v in payload.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.2f}")
            elif isinstance(v, str) and len(v) > 60:
                parts.append(f"{k}: {v[:60]}...")
            elif v is not None:
                parts.append(f"{k}: {v}")
        summary = " | ".join(parts) if parts else "completed"

    st.markdown(
        f'<div style="padding:6px 12px;margin:2px 0;color:#888;font-size:0.85rem;">'
        f'<span style="color:#28a745;font-weight:600;">{CHECK}</span> '
        f'<b style="color:#555;">{label}</b> — {summary}</div>',
        unsafe_allow_html=True,
    )


def render_verdict_banner(state: CaseState) -> None:
    """The conclusion moment after the stream ends."""
    v = state.get("verdict")
    if not v:
        st.write("Tribunal concluded.")
        return

    _, rlabel = RULING_COLORS.get(v.ruling, ("gray", v.ruling))

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;'
        f'padding:24px;border-radius:12px;text-align:center;margin:10px 0;">'
        f'<div style="font-size:0.8rem;color:#aaa;letter-spacing:2px;margin-bottom:8px;">FINAL RULING</div>'
        f'<div style="font-size:2rem;font-weight:800;margin-bottom:8px;">{rlabel}</div>'
        f'<div style="font-size:1.2rem;color:#ffc107;font-weight:600;">${v.amount:.2f}</div>'
        f'<div style="font-size:0.9rem;color:#ccc;margin-top:8px;">'
        f'Confidence: {v.confidence:.2f}'
        f'{" | Cited: " + v.cited_policy_id if v.cited_policy_id else ""}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if v.should_escalate:
        st.error("REFERRED TO HUMAN APPELLATE REVIEW")
        if v.escalation_reason:
            st.caption(v.escalation_reason)
    else:
        st.success("Ruling stands — confidence above threshold.")


def _generate_case_analysis(state: CaseState) -> str:
    """Ask the LLM to explain the tribunal's verdict in plain English."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.core.llm import get_llm

    v = state.get("verdict")
    pack = state.get("pack")
    if not v or not pack:
        return "*No verdict to analyze.*"

    d = pack.dispute
    _, rlabel = RULING_COLORS.get(v.ruling, ("gray", v.ruling))

    # Build a compact summary of the case for the LLM
    case_summary = (
        f"Case {d.dispute_id} — {d.category.replace('_', ' ')} dispute.\n"
        f"Filed by: {d.filed_by}. Claimed: ${d.claimed_amount:.2f}.\n"
        f"Rider statement: {d.rider_statement or '(none)'}\n"
        f"Driver statement: {d.driver_statement or '(none)'}\n"
    )

    if pack.route:
        r = pack.route
        case_summary += (
            f"\nRoute: optimal {r.optimal_distance_km}km, actual {r.actual_distance_km}km, "
            f"deviation {r.deviation_pct:.1%}. Traffic delay: {r.traffic_delay_min}min. "
            f"Rider detour: {r.rider_requested_detour}. Justified by: {r.deviation_justified_by}.\n"
        )
    if pack.no_show:
        n = pack.no_show
        case_summary += (
            f"\nNo-show: driver {n.driver_min_distance_to_pickup_m:.0f}m from pin, "
            f"waited {n.wait_time_min:.0f}min, fee ${n.no_show_fee_charged:.2f}.\n"
        )
    if pack.fare:
        f = pack.fare
        case_summary += (
            f"\nFare: quoted ${f.estimated_fare_at_booking:.2f}, charged ${f.total_charged:.2f}. "
            f"Surge: {f.surge_multiplier:.1f}x (booking: {f.surge_at_booking:.1f}x). "
            f"Tolls: ${f.tolls:.2f}. Promo: {f.promo_code or 'none'}.\n"
        )
    if pack.fraud:
        fr = pack.fraud
        case_summary += (
            f"\nFraud signals: risk={fr.risk_level}, collusion={fr.collusion_score:.2f}, "
            f"shared_device={fr.device_fingerprint_shared}, "
            f"photo_auth={fr.photo_authenticity_score}.\n"
        )

    # Advocate briefs
    rider_brief = state.get("rider_brief")
    driver_brief = state.get("driver_brief")
    if rider_brief:
        case_summary += f"\nRider's theory: {rider_brief.theory}\n"
        case_summary += f"Rider seeks: {rider_brief.requested_remedy} (${rider_brief.requested_amount:.2f})\n"
    if driver_brief:
        case_summary += f"\nDriver's theory: {driver_brief.theory}\n"
        case_summary += f"Driver seeks: {driver_brief.requested_remedy} (${driver_brief.requested_amount:.2f})\n"

    # Verdict details
    case_summary += (
        f"\n--- VERDICT ---\n"
        f"Ruling: {rlabel}. Amount: ${v.amount:.2f}. Confidence: {v.confidence:.2f}.\n"
        f"Cited policy: {v.cited_policy_id or 'none'}.\n"
        f"Should escalate: {v.should_escalate}.\n"
    )
    if v.findings_of_fact:
        case_summary += "Findings of fact:\n"
        for f in v.findings_of_fact:
            case_summary += f"  - {f.text}\n"
    if v.conclusions_of_law:
        case_summary += "Conclusions of law:\n"
        for c in v.conclusions_of_law:
            case_summary += f"  - {c}\n"
    if v.remedy_rationale:
        case_summary += f"Remedy rationale: {v.remedy_rationale}\n"

    system = SystemMessage(content=(
        "You are a legal analyst explaining a ride-hailing dispute tribunal's verdict "
        "in plain English. Write a clear, engaging analysis that:\n"
        "1. Restates what the dispute was about (2-3 sentences)\n"
        "2. Explains the key evidence that drove the decision\n"
        "3. Explains WHY the tribunal ruled the way it did — connect the evidence to the policy\n"
        "4. Notes any complicating factors (fraud signals, conflicting evidence, edge cases)\n"
        "5. Comments on whether the confidence level seems appropriate\n\n"
        "Write in a conversational but professional tone. Use markdown formatting. "
        "Keep it to 3-5 paragraphs. Do NOT use headers."
    ))
    human = HumanMessage(content=f"Here is the case file and verdict:\n\n{case_summary}")

    try:
        llm = get_llm("judge")
        response = llm.invoke([system, human])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"*Analysis unavailable: {e}*"


# ============================================================ verdict tab renderers


def render_evidence_tabs(pack: EvidencePack, flags) -> None:
    st.subheader("Evidence")
    tab_names = ["Key Evidence", "Fare", "Parties", "Chat", "Policies"]
    if flags.use_fraud or flags.use_precedent:
        tab_names.append("Judge-Only")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        d = pack.dispute
        if d.category == "route_deviation" and pack.route:
            r = pack.route
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Deviation %", f"{r.deviation_pct:.1%}",
                        delta="above 15% threshold" if r.deviation_pct > 0.15 else "below threshold",
                        delta_color="inverse" if r.deviation_pct > 0.15 else "normal")
            col2.metric("Deviation (km)", f"{r.deviation_km:.2f} km")
            col3.metric("Traffic Delay", f"{r.traffic_delay_min:.1f} min")
            col4.metric("Route Efficiency", f"{r.route_efficiency_score:.2f}")
            col1, col2, col3 = st.columns(3)
            traffic_share = r.traffic_delay_min / r.duration_delta_min if r.duration_delta_min > 0 else 0
            col1.metric("Traffic Share", f"{traffic_share:.0%}",
                        delta="excused" if traffic_share >= 0.80 else ("ambiguous" if traffic_share >= 0.40 else "low"))
            col2.metric("Rider Detour", "Yes" if r.rider_requested_detour else "No")
            col3.metric("Unscheduled Stops", str(r.num_stops),
                        delta=f"longest {r.longest_stop_sec}s" if r.num_stops else "")
            if r.unexpected_stops:
                st.write("**Unscheduled stops:**")
                for s in r.unexpected_stops:
                    st.write(f"  - at km {s.at_km:.1f}, {s.duration_sec}s (lat {s.lat:.4f}, lng {s.lng:.4f})")
            if r.deviation_justified_by:
                st.info(f"Deviation justified by: **{r.deviation_justified_by}**")
        elif d.category == "no_show" and pack.no_show:
            n = pack.no_show
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Driver Distance", f"{n.driver_min_distance_to_pickup_m:.0f} m",
                        delta="refund (POL-2.1a)" if n.driver_min_distance_to_pickup_m > 100 else
                              ("fee stands (POL-2.1b)" if n.driver_min_distance_to_pickup_m <= 50 else "grey zone"))
            col2.metric("Wait Time", f"{n.wait_time_min:.1f} min")
            col3.metric("Rider Approaching", "Yes" if n.rider_gps_approaching else "No")
            col4.metric("No-Show Fee", f"${n.no_show_fee_charged:.2f}")
            dist = n.driver_min_distance_to_pickup_m
            if dist <= 50:
                st.success(f"{dist:.0f}m — within 50m, fee stands (POL-2.1b)")
            elif dist > 100:
                st.error(f"{dist:.0f}m — beyond 100m, full refund (POL-2.1a)")
            else:
                st.warning(f"{dist:.0f}m — grey zone (50-100m), referred to human review")
        elif d.category == "fare_surge":
            f = pack.fare
            col1, col2, col3 = st.columns(3)
            col1.metric("Surge Multiplier", f"{f.surge_multiplier:.2f}x")
            col2.metric("Surge at Booking", f"{f.surge_at_booking:.2f}x",
                        delta="undisclosed surge" if f.surge_multiplier > f.surge_at_booking else "consistent")
            col3.metric("Surge Amount", f"${f.surge_amount:.2f}")
            if f.surge_multiplier > f.surge_at_booking:
                st.error("Surge applied exceeds surge disclosed at booking — POL-3.2 may apply")
            if f.promo_code:
                st.info(f"Promo code '{f.promo_code}' presented at booking, discount ${f.promo_discount:.2f}")
        t = pack.trip
        st.write("**Trip:**")
        st.write(f"{t.pickup_address} → {t.dropoff_address or 'n/a'} | Status: {t.status} | Traffic: {t.traffic_conditions} | City: {t.city}")
        st.write(f"Distance: est {t.estimated_distance_km} km, actual {t.actual_distance_km} km | Duration: est {t.estimated_duration_min} min, actual {t.actual_duration_min} min")

    with tabs[1]:
        f = pack.fare
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Base Fare", f"${f.base_fare:.2f}")
        col2.metric("Distance Fare", f"${f.distance_fare:.2f}")
        col3.metric("Time Fare", f"${f.time_fare:.2f}")
        col4.metric("Booking Fee", f"${f.booking_fee:.2f}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Surge Amount", f"${f.surge_amount:.2f}")
        col2.metric("Tolls", f"${f.tolls:.2f}")
        col3.metric("Promo Discount", f"${f.promo_discount:.2f}")
        col4.metric("Total Charged", f"${f.total_charged:.2f}")
        st.metric("Estimated at Booking", f"${f.estimated_fare_at_booking:.2f}",
                  delta=f"${f.total_charged - f.estimated_fare_at_booking:+.2f}")

    with tabs[2]:
        col_r, col_d = st.columns(2)
        for col, party, label in [(col_r, pack.rider, "Rider"), (col_d, pack.driver, "Driver")]:
            with col:
                st.write(f"**{label}: {party.name}** ({party.party_id})")
                st.write(f"Rating: {party.avg_rating} | Trips: {party.total_trips} | Account age: {party.account_age_days}d")
                st.write(f"Disputes 90d: {party.dispute_count_90d} | Win rate: {party.dispute_win_rate:.0%} | Cancel rate: {party.cancellation_rate:.0%}")
                if party.is_flagged_abuse:
                    st.error("Flagged for abuse")

    with tabs[3]:
        if not pack.chat:
            st.caption("No messages exchanged.")
        for msg in pack.chat:
            sender = msg.sender.title()
            color = "#e8f4ff" if msg.sender == "rider" else "#fff4e8"
            threat = " [THREAT]" if msg.contains_threat else ""
            agreement = " [AGREEMENT]" if msg.contains_agreement else ""
            sentiment_label = f"sentiment {msg.sentiment:+.2f}" if msg.sentiment else ""
            st.markdown(
                f"""<div style="background:{color};padding:8px 12px;border-radius:8px;margin:4px 0;">
                <b>{sender}</b> {msg.timestamp}<br>{msg.message}
                <br><small>{sentiment_label}{threat}{agreement}</small></div>""",
                unsafe_allow_html=True,
            )

    with tabs[4]:
        for p in pack.policies:
            with st.expander(f"{p.policy_id} — {p.title}"):
                st.write(p.rule_text)
                st.caption(f"Remedy: {p.remedy_type} | Formula: {p.remedy_formula or 'n/a'} | Version: {p.version}")
                if p.conditions:
                    st.json(p.conditions)

    if len(tabs) > 5:
        with tabs[5]:
            st.warning("JUDGE ONLY — never served to advocates. Fraud signals are a presentence report, not discovery.")
            if pack.fraud:
                fr = pack.fraud
                col1, col2, col3 = st.columns(3)
                col1.metric("Risk Level", fr.risk_level.upper())
                col2.metric("Collusion Score", f"{fr.collusion_score:.2f}")
                col3.metric("Repeat Claimer", "Yes" if fr.rider_repeat_claimer else "No")
                st.write(f"Rider claims 30d: {fr.rider_claims_30d} | Driver claims 30d: {fr.driver_claims_30d} | Device shared: {fr.device_fingerprint_shared}")
            if pack.precedents:
                st.write("**Retrieved precedents:**")
                for p in pack.precedents:
                    with st.container(border=True):
                        st.write(f"**{p.precedent_id}** [{p.cited_policy_id}] Ruling: {p.ruling} (confidence {p.confidence:.2f})")
                        st.write(f"  {p.fact_summary}")
                        st.caption(p.reasoning)
    st.divider()


def render_advocate(brief: CaseBrief, side: str) -> None:
    color = "#e8f4ff" if side == "rider" else "#fff4e8"
    st.markdown(
        f'<div style="background:{color};padding:12px;border-radius:8px;margin-bottom:10px;">'
        f'<b>{side.upper()} ADVOCATE</b></div>',
        unsafe_allow_html=True,
    )
    st.write(f"**Theory:** {brief.theory}")
    if brief.claims:
        st.write("**Claims:**")
        for c in brief.claims:
            refs = ", ".join(c.evidence_refs + c.policy_refs) or "UNCITED"
            st.write(f"- {c.text}")
            st.caption(f"  refs: {refs}")
    if brief.concessions:
        st.write("**Concessions:**")
        for c in brief.concessions:
            st.write(f"- {c.text}")
    st.metric("Seeks", brief.requested_remedy, f"${brief.requested_amount:.2f}")
    if brief.rebuttal:
        st.markdown("**Reply:**")
        st.info(brief.rebuttal)


def render_confidence_arithmetic(verdict: Verdict, threshold: float) -> None:
    st.subheader("Confidence Arithmetic")
    factors = verdict.confidence_factors or []
    fmap = {f.factor: f.delta for f in factors}
    base = fmap.get("self_report", verdict.confidence)

    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.write("**Factor**")
        st.write("Self-reported (model)")
        for key, label in [
            ("missing_key_evidence", "Missing key evidence"),
            ("precedent_conflict", "Precedent conflict"),
            ("high_fraud_risk", "High fraud risk"),
            ("advocates_converge", "Advocates converge"),
            ("uncited_claims_struck", "Uncited claims struck"),
        ]:
            st.write(label)
        st.write("**Final confidence**")
    with col2:
        st.write("**Adjustment**")
        st.write(f"{base:.2f}")
        for key, _ in [
            ("missing_key_evidence", ""),
            ("precedent_conflict", ""),
            ("high_fraud_risk", ""),
            ("advocates_converge", ""),
            ("uncited_claims_struck", ""),
        ]:
            if key in fmap:
                val = fmap[key]
                st.write(f"{val:+.2f}" if isinstance(val, (int, float)) else str(val))
            else:
                st.caption("not triggered")
        st.write(f"**{verdict.confidence:.2f}**")
    with col3:
        st.write("**Running**")
        running = base
        st.write(f"{running:.2f}")
        for key, _ in [
            ("missing_key_evidence", ""),
            ("precedent_conflict", ""),
            ("high_fraud_risk", ""),
            ("advocates_converge", ""),
            ("uncited_claims_struck", ""),
        ]:
            if key in fmap and isinstance(fmap[key], (int, float)):
                running = max(0.0, min(1.0, running + fmap[key]))
                st.write(f"{running:.2f}")
            else:
                st.write(f"{running:.2f}")
        st.write(f"**{verdict.confidence:.2f}**")

    st.write("")
    st.write(f"**Confidence:** {verdict.confidence:.2f} / Threshold: {threshold:.2f}")
    st.progress(min(verdict.confidence, 1.0))
    if verdict.confidence < threshold:
        st.error(f"BELOW THRESHOLD — {verdict.escalation_reason or 'Referred to human review.'}")
    else:
        st.success("Above threshold — ruling stands.")
    st.divider()


def render_ground_truth(verdict: Verdict, truth: dict, confidence: float) -> None:
    st.subheader("Ground Truth Comparison")
    if not truth:
        st.caption("No ground truth available for this case.")
        return
    truth_escalate = truth.get("should_escalate", False)
    if truth_escalate:
        st.info("This case is **ambiguous by construction** — direction and amount are not scored. The system is expected to escalate.")

    rows = []
    sys_ruling = verdict.ruling
    truth_ruling = truth.get("correct_ruling", DASH)
    rows.append(("Ruling", sys_ruling, truth_ruling, CHECK if sys_ruling == truth_ruling else (DASH if truth_escalate else CROSS)))
    sys_amt = f"${verdict.amount:.2f}"
    truth_amt = f"${truth.get('correct_refund_amount', 0):.2f}"
    amt_match = abs(verdict.amount - truth.get("correct_refund_amount", 0)) <= 0.51
    rows.append(("Amount", sys_amt, truth_amt, CHECK if amt_match else (DASH if truth_escalate else CROSS)))
    sys_pol = verdict.cited_policy_id or DASH
    truth_pol = truth.get("correct_policy_id", DASH)
    rows.append(("Policy", sys_pol, truth_pol, CHECK if sys_pol == truth_pol else (DASH if truth_escalate else CROSS)))
    sys_esc = str(verdict.should_escalate)
    truth_esc = str(truth_escalate)
    rows.append(("Escalation", sys_esc, truth_esc, CHECK if verdict.should_escalate == truth_escalate else CROSS))
    sys_band = band_for(confidence)
    truth_band = truth.get("expected_confidence_band", DASH)
    rows.append(("Confidence Band", sys_band, truth_band, CHECK if sys_band == truth_band else CROSS))

    col1, col2, col3, col4 = st.columns(4)
    for col, h in zip([col1, col2, col3, col4], ["Dimension", "System", "Ground Truth", "Match"]):
        col.write(f"**{h}**")
    for dim, sys_v, truth_v, match in rows:
        col1.write(dim)
        col2.write(str(sys_v))
        col3.write(str(truth_v))
        col4.write(match)

    if truth.get("rationale"):
        with st.expander("Ground truth rationale"):
            st.write(truth["rationale"])
    st.divider()


def render_verdict(verdict: Verdict, pack: EvidencePack) -> None:
    st.subheader("Judge's Opinion")
    _, label = RULING_COLORS.get(verdict.ruling, ("gray", verdict.ruling))
    col1, col2, col3 = st.columns(3)
    col1.metric("Ruling", label)
    col2.metric("Amount", f"${verdict.amount:.2f}")
    cited = verdict.cited_policy_id
    if cited:
        policy_title = next((p.title for p in pack.policies if p.policy_id == cited), cited)
        col3.metric("Cited Policy", cited)
        st.caption(f"**{policy_title}**")
    else:
        col3.metric("Cited Policy", "n/a")

    if verdict.should_escalate:
        st.error("REFERRED TO HUMAN REVIEW")
        if verdict.escalation_reason:
            st.caption(verdict.escalation_reason)
    else:
        st.success("Ruling stands — confidence above threshold.")

    if verdict.findings_of_fact:
        st.write("**Findings of Fact:**")
        for c in verdict.findings_of_fact:
            refs = ", ".join(c.evidence_refs + c.policy_refs) or "no refs"
            st.write(f"- {c.text}")
            st.caption(f"  evidence: {refs}")
    if verdict.conclusions_of_law:
        st.write("**Conclusions of Law:**")
        for i, c in enumerate(verdict.conclusions_of_law, 1):
            st.write(f"{i}. {c}")
    if verdict.remedy_rationale:
        st.info(f"**Remedy:** {verdict.remedy_rationale}")
    st.divider()


# ============================================================ sidebar
tools = get_tools()
ids = tools.dispute_ids()
if not ids:
    st.error("No cases found. Run: python scripts/generate_dataset.py")
    st.stop()

# Complex cases are the hand-crafted ones (DSP-2000 through DSP-2004)
COMPLEX_IDS = [i for i in ids if 2000 <= int(i.split("-")[1]) <= 2004]
SIMPLE_IDS = [i for i in ids if not (2000 <= int(i.split("-")[1]) <= 2004)]

# Find the case the system REFUSES to judge
refuse_case = None
gt_fn = getattr(tools, "ground_truth", None)
if callable(gt_fn):
    for did, t in sorted(gt_fn().items()):
        if t.get("should_escalate"):
            refuse_case = did
            break
refuse_case = refuse_case or ids[0]

with st.sidebar:
    st.title("Arbitra")
    st.caption("Multi-agent autonomous dispute tribunal for ride-hailing")
    st.divider()

    tier = st.radio("Case Type", ["Simple Cases", "Complex Cases"], index=1)
    tier_ids = COMPLEX_IDS if tier == "Complex Cases" else SIMPLE_IDS

    default_idx = refuse_case in tier_ids and tier_ids.index(refuse_case) or 0
    case = st.selectbox("Case", tier_ids, index=default_idx)
    cfg = st.selectbox("Config", ["C", "B", "A"], index=0,
                       help="C = Full tribunal | B = 3 agents, no debate | A = Single agent baseline")

    flags = flags_for(cfg)

    st.divider()
    st.caption(f"Opens on **{refuse_case}** — the case the system refuses to judge.")

# ============================================================ layout

tab_live, tab_analysis = st.tabs(["Live Court", "Case Analysis"])

# --------------------------------------------------------------- live court tab
with tab_live:
    # --- Session-state keys for the music-player transport ---
    SS_KEY = "court_snapshots"       # list of (stage, state_snapshot, trace_step)
    SS_IDX = "court_idx"             # current playback index
    SS_PLAYING = "court_playing"     # bool: auto-advancing
    SS_CASE = "court_case"           # which case the snapshots belong to
    SS_CFG = "court_cfg"             # which config the snapshots belong to

    # Load case card for display (before or during play)
    display_pack = None
    snapshots = st.session_state.get(SS_KEY, [])
    if snapshots and st.session_state.get(SS_CASE) == case:
        # Use the pack from the first snapshot
        display_pack = snapshots[0][1].get("pack") if snapshots else None
    else:
        # No run yet — load just for display
        try:
            display_pack = tools.get_evidence_pack(case)
        except Exception:
            pass

    render_case_card(display_pack, case)

    # --- Transport controls (music-player style) ---
    # has_run = pipeline already executed for this case+config
    # need_run = no snapshots yet (first time or case/config changed)
    has_run = bool(snapshots) and st.session_state.get(SS_CASE) == case and st.session_state.get(SS_CFG) == cfg
    need_run = not has_run

    # Playback state
    idx = st.session_state.get(SS_IDX, 0)
    is_playing = st.session_state.get(SS_PLAYING, False)
    max_idx = len(snapshots) - 1 if snapshots else 0
    at_end = idx >= max_idx if snapshots else True

    # While the pipeline is running (need_run) the Play button shows a loading
    # state and is disabled.  After the pipeline finishes it enables and becomes
    # a Play/Pause toggle.  Prev / Next / Replay are disabled until the run is
    # complete.

    t_spacer1, t_prev, t_play, t_next, t_replay, t_spacer2 = st.columns([2, 1, 2, 1, 1, 2])

    run_clicked = False
    btn_play = False
    btn_prev = False
    btn_next = False
    btn_replay = False

    with t_prev:
        btn_prev = st.button(
            "", icon=":material/skip_previous:", use_container_width=True,
            disabled=not has_run or idx == 0,
        )
    with t_play:
        if need_run:
            # First run — Play button starts the pipeline
            run_clicked = st.button(
                "", icon=":material/play_arrow:", type="primary",
                use_container_width=True,
            )
            btn_play = False
        else:
            if is_playing and not at_end:
                icon, label = ":material/pause:", "Pause"
            elif at_end:
                icon, label = ":material/replay:", "Replay"
            else:
                icon, label = ":material/play_arrow:", "Play"
            btn_play = st.button(
                label, icon=icon, type="primary", use_container_width=True,
            )
    with t_next:
        btn_next = st.button(
            "", icon=":material/skip_next:", use_container_width=True,
            disabled=not has_run or at_end,
        )
    with t_replay:
        btn_replay = st.button(
            "", icon=":material/restart_alt:", use_container_width=True,
            disabled=not has_run,
        )

    # ------------------------------------------------------------------ run
    if run_clicked and need_run:
        st.markdown(
            '<div style="text-align:center;color:#aaa;font-size:0.9rem;'
            'margin-bottom:12px;">Convening tribunal\u2026</div>',
            unsafe_allow_html=True,
        )
        progress_bar = st.progress(0.0, text="Running tribunal\u2026")
        status_text = st.empty()
        live_stages = st.empty()         # HTML stage-segment bar
        live_counter = st.empty()        # stage counter
        live_panel = st.empty()          # live now-playing content

        graph = build_graph()
        initial_state: CaseState = {
            "dispute_id": case,
            "config": cfg.upper(),
            "trace": [],
            "errors": [],
            "settlement_accepted": False,
        }
        collected: list[tuple[str, CaseState, dict]] = []
        running_state: CaseState = dict(initial_state)
        step_count = 0

        for chunk in graph.stream(initial_state, stream_mode="updates"):
            for node_name, delta in chunk.items():
                if not isinstance(delta, dict):
                    continue
                for k, v in delta.items():
                    if k == "trace" and isinstance(v, list):
                        running_state.setdefault("trace", [])
                        running_state["trace"].extend(v)
                    elif k == "errors" and isinstance(v, list):
                        running_state.setdefault("errors", [])
                        running_state["errors"].extend(v)
                    else:
                        running_state[k] = v

                new_steps = delta.get("trace", [])
                if not new_steps:
                    node_stage_map = {
                        "docket": "docket", "discovery": "discovery",
                        "motions": "motions", "settlement": "settlement",
                        "pleadings": None, "provisional_verdict": "provisional_verdict",
                        "rebuttal": None, "judge": "judge", "gate": "gate",
                        "judgment": "judgment", "single_agent": "judge",
                    }
                    mapped = node_stage_map.get(node_name)
                    if mapped:
                        new_steps = [{"stage": mapped}]

                for step in new_steps:
                    stage = step.get("stage")
                    if stage:
                        collected.append((stage, dict(running_state), step))
                        step_count += 1
                        pct = min(step_count / max(len(ALL_STAGES), 1), 1.0)
                        progress_bar.progress(
                            pct,
                            text=f"Running tribunal\u2026 [{STAGE_LABELS.get(stage, stage)}]",
                        )
                        live_counter.markdown(
                            f'<div style="text-align:center;color:#888;'
                            f'font-size:0.85rem;margin:-4px 0 8px;">'
                            f'Stage {step_count} of ~{len(ALL_STAGES)} \u2014 live</div>',
                            unsafe_allow_html=True,
                        )
                        # Render the stage-segment progress bar (green=done, blue=current)
                        trace_map = get_trace_map(running_state.get("trace", []))
                        exit_stage = fast_path_exit_stage(running_state)
                        live_stages.markdown(
                            render_progress_bar(trace_map, stage, exit_stage),
                            unsafe_allow_html=True,
                        )
                        # Render the now-playing panel live as each stage completes
                        with live_panel:
                            live_panel.empty()
                            render_now_playing(stage, dict(running_state), flags)
                            time.sleep(0.6)   # brief pause so the user can read it

        # Clear the live placeholders and show the verdict banner
        live_counter.empty()
        live_stages.empty()
        with live_panel:
            live_panel.empty()
            render_verdict_banner(running_state)
        time.sleep(1.0)

        progress_bar.progress(1.0, text="Tribunal ready.")
        progress_bar.empty()
        status_text.empty()
        live_panel.empty()

        st.session_state[SS_KEY] = collected
        st.session_state[SS_CASE] = case
        st.session_state[SS_CFG] = cfg
        st.session_state[SS_IDX] = 0
        st.session_state[SS_PLAYING] = False
        st.session_state["final_state"] = running_state
        st.session_state["run_case"] = case
        st.session_state["run_cfg"] = cfg
        st.rerun()

    # ------------------------------------------------------------------ playback buttons
    if has_run:
        if btn_prev:
            st.session_state[SS_IDX] = max(0, idx - 1)
            st.session_state[SS_PLAYING] = False
            st.rerun()
        if btn_next:
            st.session_state[SS_IDX] = min(max_idx, idx + 1)
            st.session_state[SS_PLAYING] = False
            st.rerun()
        if btn_replay:
            st.session_state[SS_IDX] = 0
            st.session_state[SS_PLAYING] = True
            st.rerun()
        # Play/Pause toggle (only meaningful when not need_run — btn_play is always defined)
        if not need_run and btn_play:
            if at_end:
                st.session_state[SS_IDX] = 0
            st.session_state[SS_PLAYING] = not is_playing if not at_end else True
            st.rerun()

    # ------------------------------------------------------------------ render
    if not has_run:
        st.markdown(
            '<div style="text-align:center;padding:40px;color:#aaa;font-size:1.1rem;">'
            'Press the <b>Play</b> button to convene the tribunal.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # re-read state (may have changed after button clicks)
    idx = st.session_state.get(SS_IDX, 0)
    is_playing = st.session_state.get(SS_PLAYING, False)
    max_idx = len(snapshots) - 1 if snapshots else 0

    if snapshots and 0 <= idx <= max_idx:
        stage, state_snap, trace_step = snapshots[idx]
    else:
        st.stop()

    # Progress bar
    trace_map = get_trace_map(state_snap.get("trace", []))
    exit_stage = fast_path_exit_stage(state_snap)
    progress_html = render_progress_bar(trace_map, stage, exit_stage)
    st.markdown(progress_html, unsafe_allow_html=True)

    # Stage counter
    status_suffix = " — playing..." if is_playing else " — paused"
    st.markdown(
        f'<div style="text-align:center;color:#888;font-size:0.85rem;margin:-4px 0 8px;">'
        f'Stage {idx + 1} of {len(snapshots)}{status_suffix}</div>',
        unsafe_allow_html=True,
    )

    # Now-playing panel
    render_now_playing(stage, state_snap, flags)

    # Verdict banner at the end
    if idx >= max_idx and not is_playing:
        st.markdown("---")
        render_verdict_banner(state_snap)

    # Auto-advance AFTER rendering so content is visible
    if is_playing and idx < max_idx:
        time.sleep(1.2)
        st.session_state[SS_IDX] = idx + 1
        st.rerun()
    elif is_playing and idx >= max_idx:
        st.session_state[SS_PLAYING] = False
        st.rerun()

# --------------------------------------------------------------- case analysis tab
with tab_analysis:
    fs = st.session_state.get("final_state")

    if not fs or st.session_state.get("run_case") != case:
        st.info("No case to analyze yet. Press **Play** in the **Live Court** tab first.")
        st.stop()

    v = fs.get("verdict")
    pack = fs.get("pack")
    rider_brief: CaseBrief | None = fs.get("rider_brief")
    driver_brief: CaseBrief | None = fs.get("driver_brief")

    if not v or not pack:
        st.error("Pipeline failed — no verdict to analyze.")
        st.stop()

    _, rlabel = RULING_COLORS.get(v.ruling, ("gray", v.ruling))
    d = pack.dispute

    # ---- Verdict summary card ----
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;'
        f'padding:20px;border-radius:12px;text-align:center;margin-bottom:20px;">'
        f'<div style="font-size:0.8rem;color:#aaa;letter-spacing:2px;margin-bottom:6px;">VERDICT</div>'
        f'<div style="font-size:1.8rem;font-weight:800;margin-bottom:6px;">{rlabel}</div>'
        f'<div style="font-size:1.1rem;color:#ffc107;font-weight:600;">${v.amount:.2f}</div>'
        f'<div style="font-size:0.85rem;color:#ccc;margin-top:6px;">'
        f'Confidence: {v.confidence:.0%}'
        f'{" | Policy: " + v.cited_policy_id if v.cited_policy_id else ""}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---- Key evidence metrics (the numbers that drove the decision) ----
    st.markdown("#### Key Evidence")
    if d.category == "route_deviation" and pack.route:
        r = pack.route
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route Deviation", f"{r.deviation_pct:.1%}",
                   delta="above threshold" if r.deviation_pct > 0.15 else "within tolerance",
                   delta_color="inverse" if r.deviation_pct > 0.15 else "normal")
        c2.metric("Extra Distance", f"{r.deviation_km:.2f} km")
        c3.metric("Traffic Delay", f"{r.traffic_delay_min:.1f} min")
        c4.metric("Rider Detour?", "Yes" if r.rider_requested_detour else "No")
    elif d.category == "no_show" and pack.no_show:
        n = pack.no_show
        c1, c2, c3 = st.columns(3)
        c1.metric("Driver Distance to Pin", f"{n.driver_min_distance_to_pickup_m:.0f} m",
                   delta="refund trigger" if n.driver_min_distance_to_pickup_m > 100 else "within range",
                   delta_color="inverse" if n.driver_min_distance_to_pickup_m > 100 else "normal")
        c2.metric("Wait Time", f"{n.wait_time_min:.0f} min")
        c3.metric("No-Show Fee", f"${n.no_show_fee_charged:.2f}")
    elif d.category == "fare_surge" and pack.fare:
        f = pack.fare
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quoted at Booking", f"${f.estimated_fare_at_booking:.2f}")
        c2.metric("Actually Charged", f"${f.total_charged:.2f}",
                   delta=f"+{(f.total_charged / f.estimated_fare_at_booking - 1):.0%}" if f.estimated_fare_at_booking > 0 else "")
        c3.metric("Surge Multiplier", f"{f.surge_multiplier:.1f}x",
                   delta="changed" if f.surge_multiplier != f.surge_at_booking else "stable")
        c4.metric("Surge at Booking", f"{f.surge_at_booking:.1f}x")
    elif d.category == "damage_cleaning":
        c1, c2 = st.columns(2)
        c1.metric("Claimed Amount", f"${d.claimed_amount:.2f}")
        if pack.fraud and pack.fraud.photo_authenticity_score is not None:
            c2.metric("Photo Authenticity", f"{pack.fraud.photo_authenticity_score:.2f}",
                       delta="flagged" if pack.fraud.photo_authenticity_score < 0.5 else "verified",
                       delta_color="inverse" if pack.fraud.photo_authenticity_score < 0.5 else "normal")
        else:
            c2.metric("Filed By", d.filed_by.title())

    # ---- Advocate face-off: two columns ----
    st.markdown("---")
    st.markdown("#### Advocate Face-Off")

    rider_name = pack.rider.name
    driver_name = pack.driver.name

    col_r, col_d = st.columns(2)

    def _render_brief_card(col, brief: CaseBrief | None, side_name: str, party_name: str, accent: str, icon: str):
        with col:
            st.markdown(
                f'<div style="background:{accent};color:white;padding:8px 14px;'
                f'border-radius:8px 8px 0 0;font-weight:700;">'
                f'{icon} {side_name}s Counsel — {party_name}</div>',
                unsafe_allow_html=True,
            )
            if not brief:
                st.markdown(
                    '<div style="border:1px solid #333;border-top:none;padding:14px;'
                    'border-radius:0 0 8px 8px;color:#888;">No brief filed.</div>',
                    unsafe_allow_html=True,
                )
                return
            # Theory
            st.markdown(
                f'<div style="border:1px solid #333;border-top:none;padding:14px;'
                f'border-radius:0 0 8px 8px;">'
                f'<div style="font-style:italic;color:#ccc;margin-bottom:10px;">'
                f'&ldquo;{brief.theory}&rdquo;</div>',
                unsafe_allow_html=True,
            )
            # Claims with evidence tags
            if brief.claims:
                st.markdown(f"**{len(brief.claims)} claim(s):**")
                for i, c in enumerate(brief.claims, 1):
                    st.markdown(f"{i}. {c.text}")
                    tags = []
                    for ref in c.evidence_refs:
                        tags.append(f"`{ref}`")
                    for ref in c.policy_refs:
                        tags.append(f"`{ref}`")
                    if tags:
                        st.markdown(f"   &nbsp; Evidence: {' '.join(tags)}")
                    else:
                        st.markdown("   &nbsp; *Uncited*")
            # Concessions
            if brief.concessions:
                st.markdown(f"**Concedes ({len(brief.concessions)}):**")
                for c in brief.concessions:
                    st.markdown(f"- ~~{c.text}~~")
            # Requested remedy
            st.markdown("")
            st.markdown(
                f'<div style="background:#1a1a2e;padding:8px 12px;border-radius:6px;'
                f'font-size:0.9rem;">'
                f'<span style="color:#aaa;">Seeks:</span> '
                f'<span style="font-weight:600;">{brief.requested_remedy}</span> '
                f'<span style="color:#ffc107;">${brief.requested_amount:.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Rebuttal
            if brief.rebuttal:
                st.markdown(f"*Rebuttal:* {brief.rebuttal}")
            st.markdown("</div>", unsafe_allow_html=True)

    _render_brief_card(col_r, rider_brief, "Rider", rider_name, "#007bff", "&#128100;")
    _render_brief_card(col_d, driver_brief, "Driver", driver_name, "#fd7e14", "&#128663;")

    # ---- Judge's inner monologue — per-claim assessments ----
    st.markdown("---")
    st.markdown("#### Judge's Inner Monologue")

    assessments = v.claim_assessments
    if not assessments:
        st.caption("No per-claim assessments recorded for this configuration.")
    else:
        rider_assessments = [a for a in assessments if a.side == "rider"]
        driver_assessments = [a for a in assessments if a.side == "driver"]

        # Badge styling
        BADGE = {
            "accepted":   ("#28a745", "ACCEPTED"),
            "rejected":   ("#dc3545", "REJECTED"),
            "questioned": ("#ffc107", "QUESTIONED"),
            "struck":     ("#6c757d", "STRUCK"),
        }

        def _render_assessment_card(a: ClaimAssessment, accent: str):
            color, label = BADGE.get(a.verdict, ("#6c757d", a.verdict.upper()))
            st.markdown(
                f'<div style="border:1px solid #333;border-left:4px solid {color};'
                f'padding:10px 14px;margin-bottom:10px;border-radius:0 6px 6px 0;">'
                f'<span style="background:{color};color:white;padding:2px 8px;'
                f'border-radius:3px;font-size:0.7rem;font-weight:700;letter-spacing:1px;">'
                f'{label}</span>'
                f'<div style="margin-top:8px;color:#ddd;font-size:0.9rem;">'
                f'&ldquo;{a.claim_text}&rdquo;</div>'
                f'<div style="margin-top:6px;color:#aaa;font-size:0.85rem;">'
                f'<em>Judge:</em> {a.reasoning}</div>',
                unsafe_allow_html=True,
            )
            if a.evidence_checked:
                refs = " ".join(f"`{r}`" for r in a.evidence_checked)
                st.markdown(f"  &nbsp; Checked: {refs}")
            st.markdown("</div>", unsafe_allow_html=True)

        col_ma_r, col_ma_d = st.columns(2)

        with col_ma_r:
            st.markdown(
                f'<div style="color:#007bff;font-weight:600;margin-bottom:8px;">'
                f'On the Rider&rsquo;s case ({len(rider_assessments)})</div>',
                unsafe_allow_html=True,
            )
            if rider_assessments:
                for a in rider_assessments:
                    _render_assessment_card(a, "#007bff")
            else:
                st.markdown("*No rider claims assessed.*")

        with col_ma_d:
            st.markdown(
                f'<div style="color:#fd7e14;font-weight:600;margin-bottom:8px;">'
                f'On the Driver&rsquo;s case ({len(driver_assessments)})</div>',
                unsafe_allow_html=True,
            )
            if driver_assessments:
                for a in driver_assessments:
                    _render_assessment_card(a, "#fd7e14")
            else:
                st.markdown("*No driver claims assessed.*")

        # Rebuttal assessment
        if v.provisional_ruling and v.provisional_confidence is not None:
            moved = v.provisional_ruling != v.ruling
            conf_delta = v.confidence - v.provisional_confidence
            st.markdown("")
            if moved:
                st.markdown(
                    f'<div style="background:#1a1a2e;color:white;padding:12px 16px;'
                    f'border-radius:8px;border-left:4px solid #ffc107;">'
                    f'<b>Rebuttal moved the court.</b> The provisional ruling '
                    f'({RULING_COLORS.get(v.provisional_ruling, ("gray", v.provisional_ruling))[1]}) '
                    f'changed to <b>{rlabel}</b>. '
                    f'Confidence shifted {conf_delta:+.1%}.'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="background:#1a1a2e;color:white;padding:12px 16px;'
                    f'border-radius:8px;border-left:4px solid #6c757d;">'
                    f'<b>Rebuttal did not move the court.</b> The ruling held at '
                    f'<b>{rlabel}</b>. Confidence shifted {conf_delta:+.1%}.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ---- Claim resolution: how the judge ruled on each side's claims ----
    st.markdown("---")
    st.markdown("#### How the Judge Ruled")

    if v.findings_of_fact:
        findings_rider = []
        findings_driver = []
        findings_neutral = []

        rider_claim_texts = {c.text for c in rider_brief.claims} if rider_brief else set()
        driver_claim_texts = {c.text for c in driver_brief.claims} if driver_brief else set()

        for f in v.findings_of_fact:
            if f.text in rider_claim_texts or any(
                kw in f.text.lower() for kw in ("rider", "passenger", "pickup")
            ):
                findings_rider.append(f)
            elif f.text in driver_claim_texts or any(
                kw in f.text.lower() for kw in ("driver", "route", "surge", "traffic")
            ):
                findings_driver.append(f)
            else:
                findings_neutral.append(f)

        col_fr, col_fd = st.columns(2)
        with col_fr:
            st.markdown(
                f'<div style="color:#007bff;font-weight:600;margin-bottom:6px;">'
                f'Rider-side findings ({len(findings_rider)})</div>',
                unsafe_allow_html=True,
            )
            for f in findings_rider:
                refs = " ".join(f"`{r}`" for r in (f.evidence_refs + f.policy_refs)) or "*uncited*"
                st.markdown(f"- {f.text}")
                st.markdown(f"  &nbsp; {refs}")
            if not findings_rider:
                st.markdown("*None*")

        with col_fd:
            st.markdown(
                f'<div style="color:#fd7e14;font-weight:600;margin-bottom:6px;">'
                f'Driver-side findings ({len(findings_driver)})</div>',
                unsafe_allow_html=True,
            )
            for f in findings_driver:
                refs = " ".join(f"`{r}`" for r in (f.evidence_refs + f.policy_refs)) or "*uncited*"
                st.markdown(f"- {f.text}")
                st.markdown(f"  &nbsp; {refs}")
            if not findings_driver:
                st.markdown("*None*")

        if findings_neutral:
            st.markdown(f"**Neutral findings ({len(findings_neutral)}):**")
            for f in findings_neutral:
                refs = " ".join(f"`{r}`" for r in (f.evidence_refs + f.policy_refs)) or "*uncited*"
                st.markdown(f"- {f.text}  {refs}")

    if v.conclusions_of_law:
        st.markdown("**Conclusions of Law:**")
        for i, c in enumerate(v.conclusions_of_law, 1):
            st.markdown(f"{i}. {c}")

    if v.remedy_rationale:
        st.info(f"**Remedy rationale:** {v.remedy_rationale}")

    # ---- Provisional vs Final (did the debate move the needle?) ----
    if v.provisional_ruling and v.provisional_confidence is not None:
        st.markdown("---")
        st.markdown("#### Did the Debate Move the Needle?")
        _, prov_label = RULING_COLORS.get(v.provisional_ruling, ("gray", v.provisional_ruling))
        moved = v.provisional_ruling != v.ruling
        conf_delta = v.confidence - v.provisional_confidence

        c1, c2, c3 = st.columns(3)
        c1.metric("Provisional Ruling", prov_label)
        c2.metric("Final Ruling", rlabel,
                   delta="changed" if moved else "unchanged",
                   delta_color="inverse" if moved else "normal")
        c3.metric("Confidence Shift", f"{conf_delta:+.1%}",
                   delta="debate helped" if conf_delta > 0 else ("debate hurt" if conf_delta < 0 else "no change"))

    # ---- Escalation status ----
    if v.should_escalate:
        st.error("**Escalated to human appellate review**")
        if v.escalation_reason:
            st.caption(v.escalation_reason)
    else:
        st.success(f"Ruling stands — confidence {v.confidence:.0%} clears the {CONFIDENCE_THRESHOLD:.0%} threshold.")

    # ---- LLM deep-dive (supplement, not the main content) ----
    st.markdown("---")
    analysis_key = f"analysis_{case}_{cfg}"
    if analysis_key in st.session_state:
        st.markdown("#### AI Analysis")
        st.markdown(st.session_state[analysis_key])
    else:
        with st.expander("Get an AI deep-dive explanation"):
            st.caption("Generates a plain-English narrative tying the evidence to the verdict.")
            if st.button("Explain the Verdict", type="primary"):
                with st.spinner("Analyzing the case..."):
                    analysis = _generate_case_analysis(fs)
                    st.session_state[analysis_key] = analysis
                st.rerun()
