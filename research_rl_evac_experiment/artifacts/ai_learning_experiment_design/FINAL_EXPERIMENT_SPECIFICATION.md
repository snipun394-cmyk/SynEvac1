<!-- final_experiment_specification_audit_v1 -->
# Final Experiment Specification / Data-Generation Design Audit

Read-only. No model trained, no dataset or pilot generated, no selector implemented, no simulator / observation.py / controller / frozen-RL / 9,450-row dataset file modified (SHA-256 checked; Section J).

## Status: **REQUIRES ONE MORE DESIGN DECISION**

- **D1 (target semantics)**: future_hazard_edge_unmet_demand: onset-gated (t >= t_h, 0 without hazard) or all ticks for every scenario? Recommendation: onset-gated: matches the metric's stated purpose (congestion caused by the degraded edge); pre-onset H_CE1 overflow is already counted in future_unmet_demand. Not decided silently because it changes the meaning of the 4th ranking criterion relative to every prior experiment.
- **D2 (action semantics / default policy)**: The continuation policy and the fallback d(s) are the GLOBAL_DEADBAND heuristic, which reads Q_A_at_H/Q_B_at_H (hidden labels, class D). Accept it as the simulated building's existing controller (inside label generation and as fallback), or require a label-free default first? Recommendation: accept for label generation and as fallback, disclosed: the model's own inputs stay admissible, and P2 carries the same privilege, so comparisons stay paired. A label-free default would be a new controller, which is out of scope. Not decided silently because it determines whether the AI-assisted system as evaluated is fully perception-honest.

Resolved in this audit: candidate set: absolute BASE2_298 (298 fixed slots), selected by an explicit rule; it removes the three-way semantic conflict in the existing code; continuation policy: frozen heuristic, identical across candidates; future_T90 / future_T100 / future_unmet definitions (post-decision, post-relabel measurement verified against the simulator's tick_log); features: the 26 admissible features, no trend features; pre-decision state generation: heuristic with epsilon=0.5 random global actions (diversity count); split unit and roles: trajectory; joint timing x compliance primary test; role-keyed compliance vectors; target form: conditional mean + quantiles; candidate sufficiency on existing states: BASE2_298 matches the unrestricted best in 100.0% (mean T100 regret 0.0 ticks).

## A. Experimental question

> Can a supervised model, given only the 26 admissible pre-decision features and a candidate from a fixed 298-candidate set, predict the conditional distribution of that candidate's future T90, T100, unmet demand and hazard-edge unmet demand (heuristic continuation), better than non-learning baselines on jointly held-out hazard timings and compliance levels; and does a deterministic, safety-floored selector over those predictions recommend actions whose outcomes approach the oracle-best candidate without harming relative to the heuristic default, in one-step and closed-loop evaluation?

"Learn" means, separately: **A** outcome prediction beats baselines (pinball loss, calibrated quantiles); **B** recommendations reach oracle-best outcome classes more often than the default; **C** harm rate <= 1% (upper CI); **D** A-C hold on the joint timing x compliance hold-out; **E** closed-loop episodes are no worse than the heuristic in >= 99% of paired episodes. Beating the heuristic and matching a unique oracle action are not required.

## B/D. Candidate action set (BASE2_298, 298 fixed slots)

Absolute per-zone assignment over R1, R2, C1, R3, R4, C2, H: NO_INTERVENTION, or a base global action (FAVOR_A / PARITY / FAVOR_B) on every eligible zone with up to two zones overridden to a different action (1 + 3 x (1 + 14 + 84) = 298). The heuristic's action h is always the candidate BASE=h. Duplicates are kept as fixed slots and marked by behavioral-equivalence class; on existing states a state has on average 37.01 behaviorally distinct candidates, so that is the real rollout count.

The code today has three incompatible meanings for a differential action (leave-untouched sweep; `apply_high_level_action`; env/oracle global-then-override). All are special cases of the absolute representation, which is why it is frozen here.

Selection rule: smallest fixed family with >= 99% best-match, 0 max T100 regret, and every known intervention reached (outcome at least as good). Sufficiency on existing states ({'dt450': 450, 'timing80': 80}; 19615 unique rollouts; the reference is every absolute assignment over the non-empty zones):

| set | size | best == unrestricted best | mean T100 regret | max T100 regret | distinct candidates/state | known interventions reached |
|---|---|---|---|---|---|---|
| GLOBAL4 | 4 | 69.8% | 0.2943 | 2.0 | {'min': 1, 'max': 3, 'mean': 2.65} | False |
| SIDE28 | 28 | 77.7% | 0.1472 | 2.0 | {'min': 1, 'max': 9, 'mean': 5.82} | False |
| SIDE64 | 64 | 77.7% | 0.1472 | 2.0 | {'min': 1, 'max': 9, 'mean': 5.82} | False |
| PAIRED82 | 82 | 77.5% | 0.1057 | 1.0 | {'min': 1, 'max': 9, 'mean': 5.82} | False |
| BASE1_46 | 46 | 89.2% | 0.0566 | 1.0 | {'min': 1, 'max': 27, 'mean': 13.62} | False |
| BASE2_298 | 298 | 100.0% | 0.0 | 0.0 | {'min': 1, 'max': 81, 'mean': 37.01} | True |
| E211_global_then_override | state-dependent | 90.8% | 0.0302 | 1.0 | {'min': 1, 'max': 33, 'mean': 16.22} | n/a |
| DT_leave_untouched_pairwise | state-dependent | 87.7% | 0.0604 | 1.0 | {'min': 1, 'max': 33, 'mean': 15.75} | n/a |

The wing-level SIDE28 was the first design tried; it cannot express the robust controller's S2 action (R2 and R4 -> FAVOR_B) and loses up to 2 ticks of T100. Useful splits are room-level.

Known interventions (outcome = [future T90, future T100, future unmet, future hazard unmet]) vs BASE2_298:

- robust_controller_S2_action_102 at S2|10000|tick0: [12, 13, 54, 0]; exactly expressible: True; BASE2_298 best [12, 13, 53, 4] (at least as good: True).
- robust_controller_S3_action_6 at S3|10000|tick0: [13, 14, 85, 8]; exactly expressible: True; BASE2_298 best [13, 14, 81, 16] (at least as good: True).
- dt_winner_R1=1|R2=2 at S2|10000|tick0: [12, 13, 53, 4]; exactly expressible: True; BASE2_298 best [12, 13, 53, 4] (at least as good: True).
- dt_winner_R1=2|R2=2 at S2|10001|tick0: [12, 13, 53, 1]; exactly expressible: True; BASE2_298 best [12, 12, 52, 2] (at least as good: True).

## E. Target definitions

- **future_T90**: T90_abs - t_d, where T90_abs = compute_metrics(sim.evacuated).T90 (exit tick of the ceil(0.9*40)=36th evacuee). If >= 36 occupants had exited before t_d: 0, with flag T90_reached_before_decision=True, and the criterion is constant across candidates, so the selector skips it and the loss masks it. If T90 is undefined: +inf (never observed).
- **future_T100**: T100_abs - t_d. T100_abs = last exit tick, or MAX_STEPS with T100_censored=True. A decision state requires n_active > 0, so T100 is never reached before t_d.
- **future_unmet_demand**: sum over ticks t in [t_d, end) of sum over all 10 edges of max(0, candidates_e(t) - capacity_e(t)), measured AFTER tick t's relabel (if t is a decision tick) and BEFORE its admission, i.e. on exactly the queue that admission sees. Pre-decision history is excluded.
- **future_hazard_edge_unmet_demand**: sum over t in [max(t_d, t_h), end) of max(0, H_CE1 branch-A candidates(t) - cap_H_CE1(t)), measured the same way; 0 when no hazard is scheduled or it never starts before the end. Status: REQUIRES USER DECISION (D1): both definitions remove the prior label's future-information problem; they change the meaning of the 4th ranking criterion differently.

Measurement check (post-relabel H_CE1 overflow vs simulator tick_log): {'hazard_all_ticks_matches_tick_log': 19615, 'hazard_onset_matches_tick_log': 19615, 'n': 19615}. States where T90 was already reached before the decision (existing states): 0.

## Rollout protocol

1. Compute the 26 features at the decision tick, before any relabel. 2. deepcopy the simulator (the prior audit confirmed identical starting fingerprints). 3. Apply exactly one BASE2_298 candidate via apply_differential_action. 4. Continue with the frozen GLOBAL_DEADBAND heuristic at every later decision, identical for all candidates. 5. Measure the targets above. The heuristic continuation is chosen on causal grounds: the label is then Q^heuristic(s, a), i.e. the consequence of deviating once from the default the selector falls back to; no-further-intervention would freeze a candidate's labels for the rest of the episode, which no deployed policy does, and would make the default's own label inconsistent with the default's behavior.

## C. Feature set

26 admissible features, unchanged from the pre-training audit (source inspection did not disprove it): `waiting_R1`, `waiting_R2`, `waiting_C1`, `waiting_R3`, `waiting_R4`, `waiting_C2`, `waiting_CE1`, `waiting_CE2`, `in_transit_R1_C1`, `in_transit_R2_C1`, `in_transit_C1_H_1tick`, `in_transit_C1_H_2tick`, `in_transit_R3_C2`, `in_transit_R4_C2`, `in_transit_C2_H_1tick`, `in_transit_C2_H_2tick`, `in_transit_H_CE1_1tick`, `in_transit_H_CE1_2tick`, `in_transit_H_CE2_1tick`, `in_transit_H_CE2_2tick`, `in_transit_CE1_EXITA`, `in_transit_CE2_EXITB`, `residual_cap_H_CE1_normalized`, `normalized_time`, `normalized_remaining`, `waiting_H_total`. Excluded: time_since_hazard_onset_normalized, hazard_phase (F); Q_A_at_H, Q_B_at_H (D); duplicate clocks; trend features.

## Scenario space

Varied without simulator changes: occupant distribution (the only lever that changes t=0 inputs), compliance p, hazard onset t_h, and (indirectly) decision tick and pre-decision history. Not variable: hazard location (hard-coded H_CE1), capacities, N=40, behavior beyond compliance.

Compliance vectors: nested across p for one (key, seed): True; independent of t_h: True. With one key and one seed, the p=0.5 compliant set is a subset of the p=0.9 set and the vector is identical across every t_h. Reusing a (key, seed) across a training cell and a test cell would give nearly the same hidden compliance on both sides of the split. Keys must include the split role and seed ranges must be disjoint by role.

## B. Dataset design

Full factorial over 6 train/test templates + 2 zero-shot templates x 9 onset levels x 7 compliance levels. Cells by role: {'TRAIN': 15, 'VALIDATION': 9, 'TEST_HELDOUT_COMPLIANCE': 10, 'DIAG_EXTRAPOLATION': 15, 'UNUSED_BUFFER': 4, 'TEST_HELDOUT_TIMING': 6, 'TEST_PRIMARY_JOINT': 4}. Full-phase trajectories: {'TRAIN': 1800, 'VALIDATION': 540, 'TEST_PRIMARY_JOINT': 960, 'TEST_HELDOUT_TIMING': 540, 'TEST_HELDOUT_COMPLIANCE': 900, 'DIAG_EXTRAPOLATION': 450, 'UNUSED_BUFFER': 0, 'DIAG_TEMPLATE_ZERO_SHOT': 354} (total 5544, ~5,485,011 candidate rows, ~681,209 unique rollouts, ~0.78 single-core hours). The size is driven by the primary test (safety criterion), not by training: see H.

Pre-decision states are reached by the heuristic with epsilon=0.5 random GLOBAL actions. This buys input diversity but means training states are not distributed like states under the heuristic or the AI-assisted policy; the closed-loop evaluation (G) is what measures performance on the deployed state distribution, and results are also reported on the epsilon=0 subset of test decisions (their first decision, t=0, is always on-policy).

## Diversity before simulation (design-time count, not the pilot)

Pre-decision reference trajectories only (3 seeds per non-buffer cell, all 8 templates), no candidate rollouts or targets.

| pre-decision policy | trajectories | decision states | distinct inputs | distinct share | tick>=10 distinct share | distinct by seeds/cell (1,2,3) | by tick | primary-test inputs seen in train |
|---|---|---|---|---|---|---|---|---|
| epsilon=0.0 | 1416 | 4348 | **66** | 1.5% | 3.0% | [62, 64, 66] | {'0': 8, '5': 12, '10': 42, '15': 4} | 88.2% |
| epsilon=0.3 | 1416 | 4571 | **427** | 9.3% | 23.4% | [203, 344, 427] | {'0': 8, '5': 12, '10': 356, '15': 50, '20': 1} | 52.2% |
| epsilon=0.5 | 1416 | 4699 | **604** | 12.8% | 31.3% | [263, 467, 604] | {'0': 8, '5': 12, '10': 524, '15': 59, '20': 1} | 50.0% |

Chosen: epsilon = 0.5 (smallest epsilon > 0 whose decision states at tick >= 10 are >= 25% distinct admissible inputs; else the largest counted).

Existing dataset for comparison: 450 states -> 11 distinct inputs. At t=0 (and before onset / before any hub admission) the input is identical across t_h, p and seed within a template. For those states the learnable target is the conditional outcome distribution under the TRAINING mixture of t_h and p; the joint hold-out tests robustness to a shift in that mixture, not input generalization. Every result must be reported separately for pre-onset and post-onset states.

## F. Split

Trajectory-level. Primary = t_h {4,8} x p {0.6,0.85} with unseen seeds; diagnostics: held-out seed, timing, compliance, template zero-shot, extrapolation. Keys include the split role; seed ranges are disjoint by role.

## Target identifiability

Chosen target form: **D, conditional expectation + conditional quantiles** (0.1/0.5/0.9). Point prediction alone would penalize the model for hidden compliance/timing it cannot see; pure distributions complicate the selector; a conservative bound alone discards the objective. Means drive the objective, upper quantiles drive the safety floor, and errors are reported next to the within-identical-input spread.

## Selector (design only)

Means for the tolerance-lexicographic objective (T90 -> T100 -> unmet -> hazard-edge unmet, delta 0.5 tick on T90/T100), upper quantiles for a non-regression safety floor vs d(s), abstention on out-of-support states or wide intervals, deterministic fallback to d(s). See selector_design.json.

## G. Future evaluation

Closed loop on the primary cells with common random numbers: no intervention, heuristic, frozen robust controller, AI-assisted, full-information one-step oracle. Primary endpoint: share of paired episodes where AI-assisted is strictly worse than the heuristic under the established order. See closed_loop_evaluation.json.

## H/I/J. Minimum data, pilot, go/no-go

No defensible analytic minimum exists for training size; the test size is set by the safety criterion (>= 300 independent deviating decisions -> 960 primary-test trajectories). Staged: Phase 1 pilot (240 trajectories, seeds 190000+, discarded afterwards) -> Phase 2 audit against G1-G7 in diversity_requirements.json -> Phase 3 full data only if all pass.

## Integrity

- Protected files unchanged: **True** (51 files incl. the prior audit's deliverables, script and tests; SHA-256 in dataset_design.json -> integrity).
- Frozen controller MD5 match: True; RL checkpoints unchanged: True (15).
- preregistered_success_criteria.json from the prior audit is superseded; its full content and SHA-256 are embedded under `previous_draft_superseded`.
