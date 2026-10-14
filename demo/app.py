"""
Streamlit demo. Day 4 — build this LAST, after the agents work.

Run:  streamlit run demo/app.py

Demo script (see Differentiation.md): open on the case the system REFUSES to judge,
then close on the numbers slide. Every other team opens with a win.
"""

from __future__ import annotations

import streamlit as st

from src.core.state import CaseBrief, Verdict
from src.graph.workflow import run_case
from src.tools.interface import get_tools

st.set_page_config(page_title="Arbitra", layout="wide")
st.title("Arbitra — autonomous dispute tribunal")

# Resolves via ARBITRA_TOOLS (mock = the 3 golden fixtures, real = data/).
tools = get_tools()
ids = tools.dispute_ids()
if not ids:
    st.error("No cases found. Run: python scripts/generate_dataset.py")
    st.stop()

# Open on the case the system REFUSES to judge. Every other team opens with a win.
refuse_case = None
gt = getattr(tools, "ground_truth", None)
if callable(gt):
    for did, t in sorted(gt().items()):
        if t.get("should_escalate"):
            refuse_case = did
            break
refuse_case = refuse_case or "DSP-0003"

col1, col2 = st.columns([1, 3])
with col1:
    default_idx = ids.index(refuse_case) if refuse_case in ids else 0
    case = st.selectbox("Docket", ids, index=default_idx)
    cfg = st.selectbox("Config", ["C", "B", "A"], index=0)
    go = st.button("Convene tribunal")

if not go:
    st.info(f"Select a case and convene. Try {refuse_case} first — it is the one we refuse to judge.")
    st.stop()

state = run_case(case, cfg)
v: Verdict | None = state.get("verdict")
pack = state["pack"]

left, right = st.columns(2)

with left:
    st.subheader("Procedure")
    for step in state.get("trace", []):
        st.write(f"**{step['stage']}** — ", {k: val for k, val in step.items() if k != "stage"})

with right:
    st.subheader("Judgment")
    if not v:
        st.warning("No verdict issued.")
    else:
        if v.should_escalate:
            st.error(f"REFERRED TO HUMAN REVIEW — confidence {v.confidence:.2f}")
            if v.escalation_reason:
                st.caption(v.escalation_reason)
        else:
            st.success(f"**{v.ruling}** — ${v.amount:.2f} · confidence {v.confidence:.2f}")

        st.caption(f"Policy: {v.cited_policy_id or 'n/a'}")

        if v.provisional_ruling:
            flipped = v.provisional_ruling != v.ruling
            st.metric(
                "Provisional -> final",
                f"{v.provisional_ruling} -> {v.ruling}",
                "FLIPPED by rebuttal" if flipped else "unchanged",
            )

        if v.confidence_factors:
            st.write("**Confidence factors**")
            st.json(v.confidence_factors)

        st.write("**Findings of fact**")
        for c in v.findings_of_fact:
            st.write(f"- {c.text} `{c.evidence_refs}`")
        st.write("**Conclusions of law**")
        for c in v.conclusions_of_law:
            st.write(f"- {c}")
        st.write("**Remedy** —", v.remedy_rationale)

        truth_fn = getattr(tools, "ground_truth", None)
        if callable(truth_fn):
            t = truth_fn(case)
            if t:
                st.write("**Ground truth (for offline eval only)**")
                st.json(t)
st.divider()
st.subheader("Both sides received identical evidence")
for side in ("rider", "driver"):
    brief: CaseBrief | None = state.get(f"{side}_brief")
    if brief:
        with st.expander(f"{side.title()} advocate — {brief.theory}"):
            for c in brief.claims:
                st.write(f"- {c.text} `{c.evidence_refs + c.policy_refs}`")
            if brief.concessions:
                st.caption("Concessions")
                for c in brief.concessions:
                    st.write(f"- {c.text}")
            if brief.rebuttal:
                st.caption("Reply")
                st.write(brief.rebuttal)
