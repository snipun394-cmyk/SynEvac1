# Calibration Benchmark Report -- Adult_Default.walking_speed

*Generated 2026-08-01T19:09:27.979866+00:00 by Calibration Benchmark Framework V1.*

This report compares one SynEvac default parameter against one dataset-derived candidate value, across an identical, paired scenario campaign. **No SynEvac default was changed to produce this report** -- both arms were run from freshly-constructed, non-mutating objects (see `calibration_benchmark/candidates.py`).

## 1. Parameter Under Test

- **Subsystem:** Walking Model
- **Parameter:** `Adult_Default.walking_speed`
- **Calibration Mapping Report tier:** Tier 2
- **Dataset source:** Julich Pedestrian Dynamics Data Archive, stair-egress experiment 2009.04.29_Duesseldorf_Arena_Hermes (DOI 10.34735/ped.2009.5), files tu11.txt-tu14.txt -- free-flow-condition instantaneous speed (bottom quartile of concurrent occupancy per run), mean 0.649 m/s, n=1818 samples
- **Current SynEvac default:** 1.2 m/s
- **Candidate (dataset-derived) value:** 0.65 m/s
- **Rationale:** SynEvac Calibration Mapping Report, Part 3.1 (Walking Model, Tier 2): DEFAULT_PROFILE_REGISTRY's own walking_speed constants are self-disclosed as illustrative, not validated. This candidate is the free-flow speed derived directly from real, physical stair-egress trajectories -- disclosed here as a STAIR-descent value, materially slower than level-ground walking, so a large discrepancy against SynEvac's current flat (level-and-stair) constant is an expected, informative finding, not a data error.

## 2. Experiment Settings

- **Scenarios requested:** 30
- **Scenarios completed (paired, both arms):** 30
- **Design:** paired -- the same seeded scenario batch is run once under the current default and once under the candidate value; every comparison below is a paired (same-scenario) statistical test, not an independent-samples comparison.
- **Statistical test:** paired t-test (`research_framework.statistics.paired_comparison`), Cohen's d effect size, 95% confidence intervals -- all reused from this codebase's own existing statistics module, not reimplemented.

## 3. Before/After Comparison

| Metric | Baseline mean | Candidate mean | Mean difference (baseline − candidate) | Cohen's d | p-value | n |
|---|---|---|---|---|---|---|
| Evacuation Time (s) | 684.105 | 1253.859 | -569.754 | 8.297 | 0.0000 (**significant**) | 30 |
| Congestion Duration (s) | n/a | n/a | n/a | n/a | n/a | 0 |
| Queue Length (peak simultaneous occupants, worst edge) | 22.600 | 22.600 | 0.000 | 0.000 | identical values (no variance) | 30 |
| Maximum Density (peak occupancy ratio, proxy) | 1.000 | 1.000 | 0.000 | 0.000 | identical values (no variance) | 30 |
| Exit Utilization Balance (1.0 = every exit used near capacity) | 1.000 | 1.000 | 0.000 | 0.000 | identical values (no variance) | 30 |

**Additional metrics** (Prediction Accuracy / Recommendation Effectiveness -- self-contained proxies computed from this same run; see `calibration_benchmark/optional_metrics.py` for exactly what each measures and why it is scoped the way it is):

| Metric | Baseline mean | Candidate mean | Mean difference | Cohen's d | p-value | n |
|---|---|---|---|---|---|---|
| recommendation_effectiveness | 0.000 | 0.000 | 0.000 | 0.000 | identical values (no variance) | 30 |
| prediction_accuracy | 0.000 | 0.000 | 0.000 | 0.000 | identical values (no variance) | 30 |

## 4. Statistical Significance, Per Metric

- **evacuation_time** -> REJECT: p=0.0000 (n=30), Cohen's d=8.297; candidate mean (1253.859) is significantly WORSE than baseline (684.105).
- **congestion_duration** -> NOT_APPLICABLE: Fewer than 2 paired scenarios produced a value for this metric (n=0).
- **queue_length** -> INCONCLUSIVE: Baseline and candidate produced identical values in every one of the 30 paired scenarios -- no variance for a significance test to measure, and no evidence of a difference either.
- **peak_occupancy_ratio** -> INCONCLUSIVE: Baseline and candidate produced identical values in every one of the 30 paired scenarios -- no variance for a significance test to measure, and no evidence of a difference either.
- **exit_utilization_balance** -> INCONCLUSIVE: Baseline and candidate produced identical values in every one of the 30 paired scenarios -- no variance for a significance test to measure, and no evidence of a difference either.
- **recommendation_effectiveness** -> INCONCLUSIVE: Baseline and candidate produced identical values in every one of the 30 paired scenarios -- no variance for a significance test to measure, and no evidence of a difference either.
- **prediction_accuracy** -> INCONCLUSIVE: Baseline and candidate produced identical values in every one of the 30 paired scenarios -- no variance for a significance test to measure, and no evidence of a difference either.

## 5. Recommendation

**Overall verdict: REJECT**

REJECT: candidate 'Adult_Default.walking_speed' significantly worsens at least one metric. A candidate is never adopted piecemeal -- see the per-metric verdicts below for exactly which trade-off is unacceptable.

This is a recommendation only. Adopting it requires a separate, explicit, human-made change to the relevant SynEvac source file -- this framework has no code path that writes one.
