## Arbitra evaluation

Config **C** · split **dev** · n = **3**

### Headline

| Metric | Value |
|---|---|
| Autonomy rate (resolved without a human) | **100.0%** |
| Accuracy on non-escalated cases | **100.0%** (n=3) |
| Escalation decision accuracy | 100.0% |
| Median latency | 58.5s (p95 85.2s) |

> Accuracy alone is a buyable metric: raise the confidence threshold and it goes up while autonomy goes down. Always report the pair. The only accuracy a user experiences is accuracy on cases that were NOT escalated.

### Checks

| # | Check | Pass | n | Needs a label? |
|---|---|---|---|---|
| 1 | Ruling direction | 100.0% | 3 | yes |
| 2 | Remedy amount | 100.0% | 3 | yes |
| 3 | Policy cited | 66.7% | 3 | yes |
| 4 | Escalation decision | 100.0% | 3 | yes |
| 5 | Confidence calibration | 33.3% | 3 | yes |
| 6 | Evidence grounding  (label-free) | 100.0% | 3 | no |
| 7 | Remedy recomputation  (label-free) | 100.0% | 2 | no |

### Error taxonomy

| Code | Meaning | Count |
|---|---|---|
| E5 | right outcome, wrong policy cited | 1 |

### Does the debate do anything?

Judge emitted a provisional verdict before rebuttal and a final one after on 2 cases. Flip rate: **0.0%**.

> 0% means the adversarial round is theatre and should be deleted. 10-20% means it is doing real work — which is also the answer to *"why three agents instead of one?"*

### Failures

| Case | Category | Predicted | Truth | Confidence | Failed |
|---|---|---|---|---|---|
| DSP-1002 | route_deviation | refund_rider | refund_rider | 1.00 | policy |
| DSP-1004 | route_deviation | refund_rider | refund_rider | 0.82 | calibration |
| DSP-1005 | route_deviation | refund_rider | refund_rider | 0.82 | calibration |

