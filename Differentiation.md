# What Makes Arbitra Different

## 0. The Uncomfortable Truth

The challenge brief specifies the architecture: Rider Advocate, Driver Advocate, Judge.
Every team will build them. Every team will use LangGraph or CrewAI. Every team will
have a Streamlit demo and an architecture diagram.

**The baseline is commoditized. Architecture is not your differentiator.**

What is *not* specified anywhere in the brief:

- How you prove the system is **correct**
- How you prove it is **consistent**
- How you prove it is **fair**
- What it **costs**
- How it behaves when it **doesn't know**

Nobody is required to measure any of that. Almost nobody will.

---

## 1. The Reframe

> **Every team will show you an AI that resolves a dispute.
> You will show one that knows when it can't — and you measured how often it's right.**

Most teams demo **one** case, cherry-picked, resolving beautifully. That is a demo, not
evidence. Judges have seen a hundred of them and cannot distinguish a genuinely working
system from a lucky prompt.

Your differentiator is **measured behaviour across a corpus, including failure modes.**

---

## 2. Four Tests Nobody Else Will Run

These are cheap to build on top of infrastructure you already planned, and each one
produces a number no other team will have.

### ⭐ Test 1 — Twin-Pair Consistency

**Addresses directly:** *"Inconsistent rulings — different agents reach different
conclusions on similar cases."* It is a named pain in the brief. Nobody will measure it.

Generate **pairs** of near-identical cases:

| Pair type | Construction | Expected | Metric |
|---|---|---|---|
| **Near-twins** | Identical facts, trivial perturbation (deviation 8.0% vs 8.4%) | Same ruling | **Consistency rate** — target ~100% |
| **Threshold-straddling** | Facts either side of a policy boundary (14% vs 16%, threshold 15%) | Ruling flips | **Boundary sensitivity** — target ~100% |

**Why this is devastatingly good:** if near-twins produce different rulings, you have
*quantified the exact pain the brief complains about*. And you can then show your system
fixing it — run the test with and without the Precedent Agent and show consistency
improving. That is a clean, self-contained ablation with a direct line to a stated
customer problem.

Suggested: 30 near-twin pairs + 20 threshold-straddling pairs.

### ⭐ Test 2 — Bias Audit

Take symmetric fact patterns and **swap the parties**. Same facts, same disputed amount,
rider and driver roles exchanged. Measure rider-favourability vs driver-favourability.

- Symmetric facts should yield ~50/50 outcomes.
- Systematic skew = measurable bias.

Then report it **honestly**, whatever the number is. A team that audits its own system for
bias and publishes the result reads as engineering. A team that claims neutrality without
checking reads as marketing. If you find bias, that is a *feature* of your presentation —
you found a real problem and can name the fix.

### ⭐ Test 3 — Rebuttal Impact (proves the adversarial design earns its keep)

Instrument the Judge to emit a **provisional verdict after the pleadings**, then a
**final verdict after rebuttal**. Log both.

- **Flip rate ≈ 0%** → the rebuttal round is theatre. Report that honestly and delete it.
- **Flip rate 10–20%** → the adversarial design is doing real work, and you have the
  number to prove it.

This is the single best evidence that your architecture is justified rather than
decorative. It also directly answers *"why 3 agents?"* — see `Why_Multi_Agent.md`.

**Bonus visual for the demo:** show a case where the Rider Advocate's concession in
rebuttal visibly changed the Judge's mind. That is the debate mechanism firing in front
of the room, not described in a slide.

### Test 4 — Ablation A / B / C

Already specified in `Why_Multi_Agent.md`. Single-agent vs three-agent vs full tribunal,
measured on all 120 cases. Report accuracy, autonomy rate, latency, cost.

---

## 3. Business Metrics Nobody Shows

Technical teams forget that judges may include product and ops people.

| Metric | How | Why it lands |
|---|---|---|
| **Cost per resolved case** | tokens × model price, measured | Compare to human handling cost. Order of magnitude: cents vs dollars. |
| **Autonomy rate** | share resolved without appeal | The headline number: "X% never touch a human" |
| **Appeal precision** | of escalated cases, share that *should* have escalated | Proves escalation isn't just nervous |
| **Median time to judgment** | wall-clock p50 / p95 | vs the 24–72h status quo in the brief |

**Report actual measured numbers.** Do not invent them. Even "cost per case: $0.041,
measured over 120 cases" beats a vague "cheap."

---

## 4. Rewrite the Demo Script

**Lead with failure, not success.** This is counterintuitive and it is the most memorable
thing you can do.

### Recommended order

| # | Beat | Why this order |
|---|---|---|
| 1 | **"Here's a case we refuse to judge."** Open the ambiguous no-show (80m, 4 min). Show confidence 0.58 → auto-appeals to a human with a full record. | Immediately separates you. Every other team opens with a win. You open by proving the system knows its limits. Judges know a system that never says "I don't know" is not deployable. |
| 2 | **The docket.** 3 filings; one flagged `critical` jumps the queue. | Shows the system at volume, not one case. |
| 3 | **Discovery.** Identical evidence pack served to both sides, side by side. | Procedural fairness, visible. |
| 4 | **Motions.** A frivolous claim dismissed in 4 seconds without trial. ⚡ | Speed, and the fast-path design. |
| 5 | **The hearing.** Both advocates argue live. Rider visibly **concedes** a point. | The debate is real. |
| 6 | **The flip.** Judge's provisional verdict → rebuttal → final verdict changed. | ⭐ Proves the adversarial design does work. |
| 7 | **Judgment.** Findings → Conclusions → Remedy → Confidence 0.88. | The four-part opinion. |
| 8 | **The numbers.** 120 cases: accuracy, consistency, bias audit, cost, autonomy rate. | ⭐ Nobody else has this slide. |

Beat 8 is where you win. Everything before it is table stakes.

### Run it live

Most teams pre-record because live demos break. If yours runs live, that is a confidence
signal. Mitigate the risk: have the recording as a fallback, but go live first.

---

## 5. Secondary Differentiators

Lower priority, but they compound:

| Differentiator | Notes |
|---|---|
| **Tribunal framing, fully committed** | Docket, discovery, motions, findings/conclusions, published opinion, appeal. Other teams may gesture at "court" — you have the actual procedure with fast paths and tiered standards of proof. |
| **PRD → demo coherence** | You have a written customer journey and you demo *that* case. Unusual narrative discipline; product judges notice. |
| **Model tiering** | Cheap advocates, expensive judge. Shows cost engineering. Minor alone. |
| **Honest limitations slide** | One slide: what this does not handle. Massively increases credibility. |

---

## 6. Anti-Patterns — What Will Actively Hurt You

| ❌ Don't | Why |
|---|---|
| **Differentiate on number of stretch features** | Feature-stuffing reads as unfocused. Six half-working agents lose to three that work. |
| **Claim full autonomy** | Judges know that's naive. Owning your limits is *more* credible than denying them. |
| **Differentiate on UI polish** | Cosmetic. Judges discount it heavily against a system that can't show numbers. |
| **Cherry-pick one beautiful case** | The commodity demo. Indistinguishable from luck. |
| **Hide your failure rate** | If your accuracy is 78%, say 78% and show what you'd fix. A team that reports 78% and explains it beats a team that claims 95% with no methodology. |
| **Build stretch before MVP is green** | See `WORKPLAN.md` §7 gate. |

---

## 7. The One-Liner

> *"Every team here built three agents. We built three agents **and a way to prove
> whether they work.** We ran 120 cases: X% accuracy, Y% autonomy, 30 twin-pair
> consistency tests, a bias audit, and an ablation showing what the second agent
> actually buys you. And we'll open our demo with a case our system refuses to judge."*

---

## 8. Effort Budget

If you have limited time, this is the priority order:

| Priority | Item | Cost |
|---|---|---|
| **1** | Eval harness on all 120 cases (accuracy, autonomy, cost) | Low — Person A already owns this |
| **2** | Rebuttal flip-rate instrumentation | Low — one extra Judge output field |
| **3** | Twin-pair consistency test | Medium — needs ~50 extra generated cases |
| **4** | Bias audit | Low — party swap on existing cases |
| **5** | Ablation A/B/C | Medium — mostly prompt variants |

Items 1 and 2 are nearly free on top of what you already planned and deliver most of the
differentiation. Do those even if you skip the rest.
