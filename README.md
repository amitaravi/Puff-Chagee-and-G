# Arbitra

**Multi-agent autonomous dispute resolution for ride-hailing.**

Rider Advocate and Driver Advocate build opposed cases from an identical evidence pack;
a Judge applies written policy and issues a four-part opinion with a calibrated
confidence score. Modeled on a small-claims tribunal — docket, mutual discovery, motions,
settlement conference, pleadings, rebuttal, judgment, appellate review.

---

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # set OPENAI_API_KEY OR GROQ_API_KEY

# if using Groq, also set:
# OPENAI_BASE_URL=https://api.groq.com/openai/v1
# and ARBITRA_MODEL_* to Groq-supported model names

python scripts/generate_dataset.py       # 240 cases -> data/   (no API key needed)
python -m src.eval.validate_data         # contract check on data/
python -m src.graph.run --list
python -m src.graph.run --dispute DSP-1002 --config C
python -m src.graph.run --dispute DSP-1002 --stub    # no API key, canned verdicts

python -m src.eval.run_eval --config C --split dev          # day-to-day tuning
python -m src.eval.run_eval --ablate                        # A vs B vs C
python -m src.eval.run_eval --probes                        # consistency + bias
pytest                                                      # leakage + contract tests

uvicorn src.api.app:app --reload --port 8000               # API
streamlit run demo/app.py                                   # UI demo
```

`--stub` runs the whole pipeline with placeholder verdicts. **Make that work first** —
it proves the graph compiles and state flows before you write a single real prompt.

If no `OPENAI_API_KEY` is set and `--stub` is NOT enabled, Arbitra now falls back to a
deterministic rule-based reasoning path (policy-engine backed) so the full tribunal,
API and eval stack still run end-to-end.

Stub scores are meaningless by design; they are a plumbing test, not an accuracy test.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | health + tool source + stub mode |
| `GET` | `/cases` | list available case IDs |
| `POST` | `/cases/{id}/run` | run one case (`config` A/B/C, optional `stub`) |
| `GET` | `/cases/{id}/truth` | read ground truth (only when data source provides it) |
| `POST` | `/eval` | run eval on a split and return summary |

---

## Architecture

```
Filing -> Docket -> Discovery -> [Motions] -> [Settlement]
       -> Pleadings (rider + driver, CONCURRENT)
       -> [Provisional verdict] -> Rebuttal (CONCURRENT) -> Judge -> Gate -> Judgment
```

Two fast paths close cases with **no adjudication at all** — motions (frivolous claims
dismissed, undisputed facts summarily resolved) and a settlement conference. Expect 40–60%
of filings to exit before a hearing. That is what makes a 5-minute median arithmetic
rather than aspiration.

Full spec: `../System_Architecture.md` · court rationale: `../Court_Model.md`

## Six design principles

| # | Principle |
|---|---|
| P1 | **Deterministic evidence, LLM reasoning.** Retrieval is Python/JSON. Only case-building and judging are model calls. |
| P2 | **Adversarial, not parallel summaries.** Advocates see and rebut each other's briefs. |
| P3 | **Every claim cites an evidence ID.** Uncited claims are struck in code, not by prompt. |
| P4 | **Mutual discovery.** Both sides get the *identical* pack. Fairness by construction. |
| P5 | **Tiered standards of proof.** Preponderance for the merits; *clear and convincing* before any bad-faith finding. |
| P6 | **Fast paths before full trial.** |

## Ablation configs

| Config | Meaning |
|---|---|
| `A` | Single agent, one pass — the baseline we must beat |
| `B` | Three agents, no debate, no fast paths |
| `C` | Full tribunal |

`--config` is a flag, not a rewrite. Person A's eval harness runs all three.

---

## Repo layout

```
src/
  core/       state.py (FROZEN contract) · config.py · llm.py
  tools/      interface.py · mock_tools.py (fixtures) · real_tools.py (data/)
  agents/     prompts.py · advocates.py · judge.py · officers.py
  graph/      workflow.py · run.py
  api/        app.py (FastAPI endpoints)
  policy/     apply_policy.py — deterministic answer key, no LLM
  eval/       metrics.py (8 checks) · run_eval.py · report.py · validate_data.py
data/         policies.json · precedents.json · cases/*.json · ground_truth.json · index.json
scripts/      generate_dataset.py
results/      eval_<config>_<split>.{json,md} · ablation.md
demo/         Streamlit
tests/        fixtures/ (3 golden cases) · test_leakage.py
```

**Directory ownership is sacred.** Don't edit the other person's files without saying so.

---

## Person A / Person B split

| | Person A — Data | Person B — AI / agents |
|---|---|---|
| Owns | `data/` · `scripts/generate_dataset.py` | `agents/` `graph/` `api/` `demo/` `README.md` |
| Also owns | may rewrite `tools/real_tools.py` against his own tables | `src/core/state.py` (the contract) |

Person B drafted `state.py`, so **Person B defines the format and Person A conforms to
it** — not the other way round. The format is written down in `DATA_CONTRACT.md`; a
reference generator (`scripts/generate_dataset.py`) and a reference tool layer
(`src/tools/real_tools.py`) already exist, so Person A can either use them as-is or
replace them wholesale.

The only hard requirement if you replace them — keep these four signatures:

```python
get_evidence_pack(dispute_id) -> EvidencePack
get_policy(policy_id)         -> Policy
search_precedents(facts, k=3) -> list[Precedent]
get_fraud_signals(dispute_id) -> FraudSignals
```

Set `ARBITRA_TOOLS=real` to swap them in. Nothing in `agents/` changes.

### Why the answer key is Python, not an LLM

`src/policy/apply_policy.py` implements the policy book in ~200 lines of plain Python.
It **labels** the dataset and **scores** the eval — same function both times, so the two
cannot drift. No model is involved in producing a label. If an LLM wrote the labels, our
eval would only be measuring "how often does the Judge agree with GPT-4".

---

## How we prove it works

Not "our AI is 94% correct." Eight checks, three of which need no ground truth at all:

| Check | Needs labels |
|---|---|
| Ruling direction · remedy amount · policy citation · escalation · calibration | yes |
| **Grounding / fabrication** — every cited ID exists, every number matches source | **no** |
| **Recomputation** — re-derive the refund from the cited policy's formula | **no** |
| **Stability** — run 3×, twin pairs, party swap | **no** |

The label-free checks are the shield: they hold even if the answer key is wrong.

We report **accuracy at a given autonomy rate**, never raw accuracy — raise the escalation
threshold and accuracy goes up while autonomy goes down. Full framework:
`../How_We_Prove_Correctness.md`.

### The four probes (the differentiator)

Every team demos one cherry-picked case. Nobody proves the system works. These do — and
none of them is visible in a single-case demo:

| Probe | What it catches |
|---|---|
| **Near-twin consistency** | near-identical cases (deviation 17.6% vs 18.0%) must get the *same* ruling → is the Judge random? |
| **Threshold flip** | cases straddling the 15% line (14% vs 16%) must get *different* rulings → is the Judge applying the rule? |
| **Filing-party skew** | identical no-show facts, filed by rider vs driver → does the Judge favour a side? |
| **Party-attribute swap** | same case, rider/driver names and ratings swapped → is it sensitive to irrelevant attributes? |

```bash
python -m src.eval.run_eval --probes
```

Report the bias number **whatever it is**. "We found a 12-point rider skew, here's our
fix" beats "our AI is fair" every time — one is engineering, the other is marketing.

---

## Rules for the four days

1. `state.py` is frozen after day 0.
2. Integrate end of day 3. Last day is tuning, never first contact.
3. `main` is always runnable. Fixing it outranks new features.
4. Person B can run the eval from day 2 — tuning prompts without a feedback loop is guessing.
5. Build the demo UI on day 3 at the earliest. It is the biggest time sink in the project.
6. No stretch goal until the MVP checklist is green.

## MVP checklist

- [ ] Two dispute categories end to end
- [ ] Three agents, advocates concurrent, rebuttal round
- [ ] Confidence gate + appellate escalation
- [ ] Eval harness running all 120 cases
- [ ] Ablation A/B/C numbers
- [ ] Streamlit demo, opening on the case we refuse to judge
- [ ] Architecture diagram

## Not wired yet

- Token/cost accumulation (`config.Cost` is defined but not called in the agent nodes)
- Precedent retrieval is deterministic similarity on numeric key facts, not embeddings —
  deliberate for the MVP, because it makes the precedent contribution measurable
- Damage/cleaning category: `apply_policy()` supports it, the generator does not emit it
  (needs the multi-modal photo path)
- Learning feedback loop from human overrides

---

## Related docs

`PRD_Dispute_Resolution_Part1.md` · `Synthetic_Dataset_Spec.md` · `System_Architecture.md`
· `Court_Model.md` · `Why_Multi_Agent.md` · `Differentiation.md` · `How_We_Prove_Correctness.md` · `WORKPLAN.md`
