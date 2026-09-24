<!-- label_free_default_audit_v1 -->
# Label-Free Default Audit (after D1 / D2)

Read-only. No pilot, no training data, no model, no policy module implemented, no frozen specification file overwritten, no simulator / observation.py / controller / RL branch / 9,450-row dataset change (SHA-256 checked).

## Answer: **B. YES -- minimal new policy definition required**

Recommended label-free default: **SYMMETRIC_SPLIT_DEADBAND (margin 0)**.

## 1. Existing policies

| policy | hidden dependencies | label-free | verdict |
|---|---|---|---|
| capacity_aware_heuristic.CapacityAwareHeuristic / deadband_heuristic.DeadbandCapacityAwareHeuristic (GLOBAL_DEADBAND) | Q_A_at_H, Q_B_at_H (hidden target_exit labels) | False | REFERENCE ONLY (D2) |
| branch_aware_deadband_heuristic | Q_A_at_H, Q_B_at_H | False | disqualified |
| arbitration_engine / hybrid_controller / policy_selector | Q_A_at_H, Q_B_at_H | False | disqualified |
| observable_robust_differential.py (frozen controller) | Q_A_at_H, Q_B_at_H | False | disqualified (and frozen) |
| PPO checkpoints (frozen RL branch) | Q_A_at_H, Q_B_at_H | False | disqualified (and frozen) |
| NO_INTERVENTION (never relabel) | none | True | evaluated |
| CONSTANT_PARITY (baseline.baseline_action) | none | True | evaluated |
| CONSTANT_FAVOR_A / CONSTANT_FAVOR_B | none | True | evaluated for completeness |
| SYMMETRIC_SPLIT_DEADBAND (analysis-only definition, NOT implemented as a module) | none | True | evaluated (minimal new definition) |

Every existing adaptive controller routes through GLOBAL_DEADBAND and so reads Q_A/Q_B. The only existing label-free policies are constants and no-intervention.

## 2. Minimal substitute

SYMMETRIC_SPLIT_DEADBAND: the existing, unmodified `deadband_heuristic_action` applied to the observation with Q_A_at_H = Q_B_at_H = waiting_H_total / 2. Inputs: waiting_H_total and residual capacity only. The even split is a fixed prior (the parity rule's own split), not an estimate of labels or compliance. Two margins were evaluated:

- **margin 0** (= the project's original parameter-free `capacity_aware_heuristic` rule on a label-free input): PARITY unless the hazard has occurred (residual capacity 2/6) and at least one person waits at the hub, then FAVOR_B. Never FAVOR_A.
- **margin 0.75** (the dead-band used by the hidden heuristic): same, but FAVOR_B only with at least 5 people at the hub.

It is a new DEFINITION (an input adapter), not new decision logic, which is why the answer is B rather than A. It is analysis-only here and is not implemented as a module.

## 4. Hidden vs label-free (full episodes, existing configurations)

| policy | worse / tie / better vs hidden | worse vs no-intervention | mean dT100 | max dT100 | mean d hazard unmet |
|---|---|---|---|---|---|
| HIDDEN_GLOBAL_DEADBAND (reference only) | 0 / 230 / 0 | 0 | 0.0 | 0.0 | 0.0 |
| NO_INTERVENTION | 90 / 140 / 0 | 0 | 1.3696 | 4.0 | 17.37 |
| CONSTANT_PARITY (baseline.py) | 90 / 140 / 0 | 0 | 1.3696 | 4.0 | 17.37 |
| CONSTANT_FAVOR_A | 230 / 0 / 0 | 230 | 8.3 | 14.0 | 136.33 |
| CONSTANT_FAVOR_B | 216 / 0 / 14 | 140 | 1.1957 | 3.0 | -10.743 |
| SYMMETRIC_SPLIT_DEADBAND (margin 0.75) | 13 / 217 / 0 | 0 | 0.0565 | 1.0 | 0.083 |
| SYMMETRIC_SPLIT_DEADBAND (margin 0) | 0 / 230 / 0 | 0 | 0.0 | 0.0 | 0.0 |

Full episodes from each existing tick-0 configuration (S1/S2/S3 x 50 eval seeds at p=0.9; timing matrix t_h in {3,6,7,10} x 20 seeds at p=0.75), each run entirely under one policy. Outcome keys use D1 (onset-gated) hazard-edge unmet. 'worse' = lexicographically worse on (T90, T100, unmet, hazard unmet).

Selection: fixed before computing: fewest paired episodes lexicographically worse than the hidden-label heuristic (the previously accepted default); ties go to the simpler / existing policy in the order ['NO_INTERVENTION', 'CONSTANT_PARITY (baseline.py)', 'CONSTANT_FAVOR_A', 'CONSTANT_FAVOR_B', 'SYMMETRIC_SPLIT_DEADBAND (margin 0)', 'SYMMETRIC_SPLIT_DEADBAND (margin 0.75)'].

Action agreement with the hidden heuristic on the 530 existing decision states: SYMMETRIC_SPLIT_DEADBAND (margin 0): {'post_onset|agree': 113, 'post_onset|differ': 37, 'pre_onset_or_no_hazard|agree': 380}. Where the two disagree (post-onset only), the disagreement did not change any full-episode outcome on these configurations: the hidden split only matters when the hub queue's labels are uneven, which rarely happens on the default's own trajectory. It matters much more after a candidate deviation, which is where the targets change (below).

## 3/6. Target validity and impact

Protocol: state before decision -> one BASE2_298 candidate -> SYMMETRIC_SPLIT_DEADBAND (margin 0) at every later decision (identical for every candidate) -> D1 targets. The label now answers: what happens if I make this recommendation now, assuming the same observable default afterwards.

Measured on the 530 existing states (19615 unique candidate rollouts per continuation):

- identical target vector under both continuations: 64.4% of candidates
- future_T90 (label-free minus hidden): mean -0.0025, mean |d| 0.2211, range [-3.0, 2.0], unchanged 80.5%
- future_T100 (label-free minus hidden): mean -0.0162, mean |d| 0.2391, range [-3.0, 2.0], unchanged 78.6%
- future_unmet (label-free minus hidden): mean -2.2997, mean |d| 3.6044, range [-60.0, 33.0], unchanged 65.4%
- future_hazard_unmet (label-free minus hidden): mean -1.0886, mean |d| 4.0395, range [-81.0, 20.0], unchanged 77.3%
- oracle-best class identical: 62.5% of states; overlapping: 85.1%; mean within-state rank correlation 0.9081 (min 0.4)
- default action differs from the hidden heuristic's in 7.0% of states; default candidate's T100 shift {'mean': 0.0, 'max': 0.0}

Specification changes required (not applied here):

- **candidate_action_set_BASE2_298**: no change (all three BASE globals and NO_INTERVENTION remain; the default d(s)=BASE=l(s) is always in the set)
- **target_definitions**: CHANGE: continuation: GLOBAL_DEADBAND -> SYMMETRIC_SPLIT_DEADBAND (margin 0) at every later decision (identical for all candidates); future_hazard_edge_unmet_demand: onset-gated definition (D1) is final; the alternative is removed; stored_per_row: only the D1 hazard definition
- **feature_manifest_26**: no change (the default reads only admissible features; the model inputs were already label-free)
- **scenario_grid**: no change (templates, t_h and p levels do not depend on the default)
- **pre_decision_state_generation**: CHANGE: epsilon-perturbed SYMMETRIC_SPLIT_DEADBAND (margin 0) instead of the hidden heuristic, so the training state distribution matches the deployed default
- **split**: no change (split roles, seed ranges and keys do not depend on the default)
- **selector**: CHANGE: d(s) = BASE=l(s) with l = SYMMETRIC_SPLIT_DEADBAND (margin 0); the caveat about Q_A/Q_B in selector_design.json no longer applies
- **success_criteria**: CHANGE: harm is measured against the label-free default; closed loop: P2 = label-free default (primary comparator); the hidden-label heuristic becomes a reference arm P2b, reported, never a gate
- **closed_loop_evaluation**: CHANGE: add P2b (hidden heuristic, reference); P4 falls back to the label-free default
- **pilot_size**: no change (240 trajectories; revisit only if the recount's tick>=10 distinct share falls below the 25% gate)

Diversity recount with SYMMETRIC_SPLIT_DEADBAND (margin 0) as the pre-decision policy: epsilon_0.0: 67 distinct inputs, tick>=10 distinct share 3.1%; epsilon_0.3: 429 distinct inputs, tick>=10 distinct share 23.7%; epsilon_0.5: 596 distinct inputs, tick>=10 distinct share 31.1%.

## 5. Fallback validity

Selector / abstention / out-of-distribution fallback: valid. Label-free verified (invariant to any split of the hub total on every existing state: True); defined on every state: True. It is not a safety GUARANTEE. The selector's safety floor is relative to this default, so 'no harm' now means 'no worse than the label-free default'. Against the old hidden-label heuristic it was paired-worse in 0/230 existing episodes (see hidden_vs_label_free_comparison.json). No extra mechanism is added to hide that; the hidden heuristic stays in closed-loop evaluation as a reference arm so the gap stays visible.

## 7. Critical question

Can the experiment run with a genuinely label-free continuation/fallback without changing the simulator? **B. YES -- minimal new policy definition required**. No simulator, observation or controller change is needed.

## Integrity

- Protected files unchanged: **True** (64 files, including every frozen specification file and both earlier audits).
- Frozen controller MD5 match: True; RL checkpoints unchanged: True (15).
