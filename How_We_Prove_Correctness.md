# How We Prove the Judge Is Right

## 0. Start with the honest framing

You cannot prove a ruling is "correct" in the absolute sense. The brief itself says disputes
are *"complex, subjective, and emotionally charged."* Two fair humans given the same evidence
will sometimes disagree. If you stand on stage and claim "our AI is 94% correct," a sharp judge
will ask "correct according to whom?" and you have no answer.

So **don't claim correctness.** Claim four things, each of which is directly measurable:

| Claim | What would prove it false |
|---|---|
| **It follows the written policy** | The ruling contradicts the rule text |
| **It doesn't invent facts** | A cited fact isn't in the evidence pack |
| **It's consistent** | Near-identical cases get different rulings |
| **It knows when it's unsure** | It's confidently wrong, or it escalates cases it got right |

This is a *stronger* position than an accuracy number, because it survives hostile
questioning. "94% correct" collapses under one good question. "Zero fabricated citations
across 120 cases, verified programmatically" does not.

---

## 1. The answer key must not be an LLM

This is the structural decision everything else depends on.

Your ground truth comes from a **deterministic policy engine** — plain Python, no model:

```python
def apply_policy(ev: EvidencePack, pol: Policy) -> Verdict:
    if pol.policy_id == "POL-4.3":          # route deviation
        unjustified = (ev.route.deviation_pct > 0.15
                       and not ev.route.delay_attributable_to_traffic
                       and not ev.route.rider_requested_detour)
        if not unjustified:
            return Verdict("no_action", 0.00, "POL-4.3")
        amount = ev.fare.distance_fare * (ev.route.deviation_km
                                          / ev.route.actual_distance_km)
        return Verdict("refund_rider", round(amount, 2), "POL-4.3")
```

Three reasons this matters:

1. **Auditable.** Same input → same output, forever. A judge can read 100 lines of Python
   and decide whether *your answer key* is right. They cannot audit GPT-4's intuition.
2. **Independent of the thing under test.** If you use an LLM to generate labels, you have
   measured "how often does our judge agree with another LLM." That's circular and nearly
   meaningless.
3. **It doubles as the demo's rulebook.** You show the engine, you show the AI's ruling next
   to it. The comparison is the product.

`apply_policy()` is ~100 lines for the two MVP categories. Person A owns it. It is the single
highest-value artifact in the eval stack.

---

## 2. Eight checks, not one score

"Accuracy" is one number and it hides everything. Score each ruling on eight axes, so that
when you're wrong you know *how* you're wrong.

| # | Check | Catches | Needs labels? | Implementation |
|---|---|---|---|---|
| 1 | **Ruling direction** | Wrong side won | Yes | `ruling == correct_ruling` |
| 2 | **Remedy accuracy** | Right call, wrong amount | Yes | `abs(delta) <= max(1.00, 10% of claimed)` |
| 3 | **Policy citation** | Right outcome for the wrong reason | Yes | `cited_policy_id == correct_policy_id` |
| 4 | **Grounding / fabrication** ⭐ | Hallucinated evidence | **No** | Every `evidence_id` in Findings must exist in the pack; every numeric claim must match the source value ±ε |
| 5 | **Recomputation** ⭐ | Arithmetic errors | **No** | Re-derive the remedy from the formula of the policy it cited; compare to what it output |
| 6 | **Escalation decision** | Over- / under-escalating | Yes | `escalated == should_escalate` |
| 7 | **Calibration** | Overconfidence | Yes | Bin by confidence; accuracy in the 0.85–0.90 bin should be ≈87% |
| 8 | **Stability** ⭐ | Randomness / brittleness | **No** | Run 3× at temp 0.7 — same ruling? Plus twin-pairs and party-swap (see `Differentiation.md`) |

### ⭐ The important column is "needs labels?"

**Checks 4, 5 and 8 need no ground truth at all.** They're valid even if your labels were
wrong. That is your shield against the obvious objection — see §9.

Check 4 is the highest-leverage one. It's ~30 lines: regex the Findings of Fact for
`TRP-00431.route_analysis.deviation_pct` style references, resolve each against the actual
JSON, flag any that don't exist or whose value doesn't match. A judge who sees
*"0 fabricated citations across 120 cases, checked programmatically"* learns more about your
system than from any accuracy number.

---

## 3. The number that actually matters: accuracy **at a given autonomy rate**

Raw accuracy across 120 cases is a vanity metric, because **you can buy accuracy with the
escalation threshold.** Escalate everything → 100% accuracy on whatever you do resolve, and
0% autonomy. Any team reporting a bare accuracy number is either naive or gaming it.

Report the pair, and ideally the curve:

| Confidence threshold | Autonomy rate | Accuracy on auto-resolved | Users' experience |
|---|---|---|---|
| 0.60 | ~88% | ~89% | Fast, sloppier |
| 0.75 | ~72% | ~94% | ← recommended |
| 0.85 | ~55% | ~97% | Conservative |

Then state it as a product sentence, not a metric:

> *"At a 0.75 threshold, 72% of disputes are fully automated and those rulings are 94%
> correct. The other 28% reach a human within 90 seconds with a complete case file — versus
> 36 hours today."*

**The only accuracy a user ever experiences is accuracy on non-escalated cases.** That's
your deployment number. Lead with it.

Pair it with **appeal precision**: of the cases it escalated, what share actually needed
escalation? If that's low, you're escalating out of nervousness, not signal — and your
autonomy rate is leaving money on the table.

---

## 4. The human baseline — the move nobody else will make

Get 3 people (your teammate, a friend, anyone) to independently rule on the same **20 cases**,
blind: they see the evidence pack only — not the label, not the AI's output.

Then compute three numbers:

- **Human ↔ human agreement** — typically 70–80% on genuinely contested cases
- **AI ↔ human-majority agreement**
- **AI accuracy vs the deterministic answer key** on the same 20

This gives you the best reframe available to you:

> *"Humans agree with each other 76% of the time on these cases. Our system agrees with the
> human majority 79% of the time — and takes 90 seconds instead of 36 hours."*

That converts the conversation from "is it perfect?" (no, and nothing is) to "is it at least
as good as the status quo, ~500× faster and ~100× cheaper?" — which is the actual buying
decision a ride-hailing ops lead makes.

It also lets you say something no other team can: *"in the 5 cases where the AI disagreed
with the human majority, we read all 5. The AI was closer to policy in 3, the humans in 2."*
That is engineering rigor, and it's memorable.

---

## 5. Holdout discipline (cheap, and it's the difference between a number and a claim)

- Split **80 dev / 40 holdout**. Tune prompts on dev. Report on holdout only.
- If you end up tuning on all 120 because of time, **say so out loud.** Disclosure reads as
  integrity; hiding it doesn't. One sentence on the slide: *"prompt tuning was done on the
  dev split; holdout was untouched."*
- Fixed seed. Temperature 0 for all reported runs. Separate temp-0.7 run purely for check 8.

---

## 6. The leakage test (one pytest, and it's the first thing a sharp judge asks)

Your agents must not be able to read the answer key. Assert it in CI:

```python
def test_no_label_leakage():
    agent_readable = load_all("data/")            # everything agents can query
    forbidden = {"correct_ruling", "correct_refund_amount",
                 "should_escalate", "expected_confidence_band"}
    for f, obj in agent_readable.items():
        assert not (set(obj.keys()) & forbidden), f"leak in {f}"
```

`ground_truth_labels.json` lives in `eval/`, loaded only by the harness, never by a tool.
Ten lines. Non-negotiable — a leaked label makes your entire eval worthless and you'd never
know.

---

## 7. Failure taxonomy — read your errors, don't just count them

Every incorrect ruling gets an error code:

| Code | Meaning |
|---|---|
| E1 | Wrong policy selected |
| E2 | Right policy, threshold misapplied |
| E3 | Fabricated / mis-cited evidence |
| E4 | Remedy arithmetic wrong |
| E5 | Decisive evidence present but ignored |
| E6 | Overconfident (high confidence, wrong) |
| E7 | Underconfident (escalated, got it right) |

Then report the distribution, not just the total:

> *"12 errors on 40 holdout cases: 6× E2 (threshold misapplication on the 12–18% grey band),
> 3× E5, 2× E6, 1× E4. Zero E3 — no fabricated evidence."*

That tells the room you actually opened the failures and looked inside. It also tells *you*
exactly what to fix the next morning. E3 should be zero — that's the one error that's
unforgivable in a dispute system.

---

## 8. Bonus test: evidence ablation (no labels needed)

Take a clear-cut case, delete the **single decisive fact** from the evidence pack (e.g. the
`delay_attributable_to_traffic` field on the justified-driver case), and re-run.

Correct behaviour: **confidence drops and the case escalates.**

If the Judge still issues a confident ruling on a pack that no longer contains the fact that
decides the case, it's guessing — and you've found that out without needing a single label.
Run this on ~15 cases. Great demo beat too: *"watch what happens when we hide the fact that
decides the case."*

---

## 9. The challenge you will get, and your answer

> **"Your ground truth is synthetic, so your accuracy number is meaningless."**

This is the right objection and someone will make it. Answer:

> *"Partly agreed — which is why we don't lead with it. Three of our eight checks need no
> labels at all: fabrication, recomputation, and stability. Those hold regardless of whether
> our answer key is right. We also ran a human baseline on 20 cases. The accuracy number's
> job is to let us **compare configurations** — that's how we decide whether the second
> advocate agent is worth its cost. For a real deploy decision you need historical human
> rulings, and re-running this harness against production data is the first thing we'd do."*

That answer is better than having a great accuracy number. It shows you know the difference
between an eval and a validation.

---

## 10. The slide

One slide, four blocks. No prose.

```
┌──────────────────────────────────────────────────────────────┐
│  VERIFICATION — 40 holdout cases                             │
├───────────────────────────┬──────────────────────────────────┤
│  Fabricated citations     │  0 / 40        ← programmatic    │
│  Remedy recomputation     │  39 / 40 exact                   │
│  Twin-pair consistency    │  29 / 30       ← no labels       │
│  3× run stability         │  100%                            │
│  ─────────────────────    │                                  │
│  Accuracy @ 0.75 gate     │  94% (on 72% auto-resolved)      │
│  Autonomy rate            │  72%                             │
│  Appeal precision         │  81%                             │
│  Human↔human agreement    │  76%   ← baseline                │
│  AI↔human-majority        │  79%                             │
├───────────────────────────┴──────────────────────────────────┤
│  ERRORS: 6×E2  3×E5  2×E6  1×E4   0×E3                       │
└──────────────────────────────────────────────────────────────┘
```

Every number on that slide is measured, and every one of them is a number no other team will
have.

---

## 11. Build order (Person A)

| Phase | Item | Cost |
|---|---|---|
| 1 | `apply_policy()` deterministic engine + `ground_truth_labels.json` | Medium, do first |
| 2 | Leakage pytest | 10 min |
| 3 | Checks 1–3, 6 (label-based scoring) | Low |
| 4 | Check 4 (grounding) + 5 (recomputation) | Medium — highest value |
| 5 | Check 7 (calibration bins) + risk/coverage curve | Low |
| 6 | Check 8 (stability, twin pairs, party swap) | Low |
| 7 | Human baseline on 20 cases | 30 min of someone's time |
| 8 | Error taxonomy tagging | Low |

Phase 4 is the one to protect if time runs short. Checks 4 and 5 are what make the claim
"our judge is trustworthy" defensible instead of aspirational.
