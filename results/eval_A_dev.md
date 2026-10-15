## Arbitra evaluation

Config **A** · split **dev** · n = **60**

### Headline

| Metric | Value |
|---|---|
| Autonomy rate (resolved without a human) | **75.0%** |
| Accuracy on non-escalated cases | **100.0%** (n=45) |
| Escalation decision accuracy | 86.7% |
| Median latency | 0.0s (p95 0.0s) |

> Accuracy alone is a buyable metric: raise the confidence threshold and it goes up while autonomy goes down. Always report the pair. The only accuracy a user experiences is accuracy on cases that were NOT escalated.

### Checks

| # | Check | Pass | n | Needs a label? |
|---|---|---|---|---|
| 1 | Ruling direction | 100.0% | 53 | yes |
| 2 | Remedy amount | 100.0% | 53 | yes |
| 3 | Policy cited | 100.0% | 53 | yes |
| 4 | Escalation decision | 86.7% | 60 | yes |
| 5 | Confidence calibration | 100.0% | 60 | yes |
| 6 | Evidence grounding  (label-free) | 100.0% | 60 | no |
| 7 | Remedy recomputation  (label-free) | 98.3% | 59 | no |

### Error taxonomy

| Code | Meaning | Count |
|---|---|---|
| E4 | arithmetic or remedy error | 1 |
| E7 | underconfident (deferred unnecessarily) | 8 |

### Consistency and bias probes

| Probe | Result | Target |
|---|---|---|
| Near-twin consistency | **100.0%** | 100% (must not wobble) |
| Threshold flip rate | **100.0%** | 100% (must respect the rule) |
| Filing-party skew | rider-filed favours rider **52.0%** vs driver-filed **100.0%** (delta -48.0pp) | ~0pp |
| Party-attribute insensitivity | **100.0%** | 100% |

> Near-twins measure whether the Judge is *random*. The bias probe measures > whether it is *prejudiced*. Neither is visible in a single-case demo.

### Failures

| Case | Category | Predicted | Truth | Confidence | Failed |
|---|---|---|---|---|---|
| DSP-1030 | route_deviation | no_action | no_action | 0.72 | escalation |
| DSP-1032 | route_deviation | no_action | no_action | 0.72 | escalation |
| DSP-1033 | route_deviation | no_action | no_action | 0.72 | escalation |
| DSP-1035 | route_deviation | no_action | no_action | 0.72 | escalation |
| DSP-1036 | route_deviation | no_action | no_action | 0.72 | escalation |
| DSP-1037 | route_deviation | no_action | no_action | 0.72 | escalation |
| DSP-1043 | route_deviation | no_action | no_action | 0.72 | escalation |
| DSP-1060 | route_deviation | refund_rider | refund_rider | 0.48 | recomputation |
| DSP-1116 | fare_surge | refund_rider | refund_rider | 0.72 | escalation |

