<!-- ai_learning_pilot_audit_v1 -->
# AI-Learning Pilot (Phase 1) Dataset Audit

Pilot only: no model trained, no selector, no closed-loop evaluation, no full dataset. Criteria are the ones pre-registered in specification v2 (committed before any pilot data existed), applied unchanged.

## Decision: **PILOT GO**

| criterion | passed | value |
|---|---|---|
| G1 | PASS | {"share": 0.5158, "min_unique_per_template": 24} |
| G2 | PASS | 0.6482 |
| G3 | PASS | {"share": 1.0, "mean_T100_regret": 0.0} |
| G4 | PASS | {"identical_rows": true, "hazard_tick_log": "19319/19319"} |
| G5 | PASS | {"states": 796, "features_identical_across_candidates": 796, "residual_ok": 796, "replay_features_match": 796, "replay_fingerprint_match": 796, "class_relabels_checked": 19319, "features_unchanged_by_relabel": 19319, "pr |
| G6 | PASS | {"share": 0.0165, "coverage": 0.7445, "evaluable": true} |
| G7 | PASS | {"2": 120} |
| X1 | PASS | {"states_checked": 556, "decomposition_exact": 556, "passed": true, "method": "replay recorded history to t_d (pre-decision unmet), roll out the default candidate with the label-free continuation; the stored pre-decision |
| X2 | PASS | "= G5" |
| X3 | PASS | "19319/19319" |
| X4 | PASS | {"n_primary_joint_cells": 4, "primary_cells": ["t_h=4|p=0.6", "t_h=4|p=0.85", "t_h=8|p=0.6", "t_h=8|p=0.85"], "test_t_h_disjoint_from_train": true, "test_p_disjoint_from_train": true, "every_template_crosses_every_level" |
| X5 | PASS | {"unique_inputs": 183, "distinct_share": 0.2299, "state_candidate_pairs": 54534} |
| X6 | PASS | "= G2" |
| X7 | PASS | {"association": {"template_vs_t_h_cramers_v": 0.0, "template_vs_p_cramers_v": 0.0, "t_h_vs_p_cramers_v": 0.0}, "perturbation_rate": 0.4982, "censored_rows": 0} |
| X8 | PASS | "= X1 decomposition" |
| X9 | PASS | {"states": 796, "default_equals_label_free_rule": 796, "default_equals_closed_form": 796, "class_reps_checked": 5216, "fast_label_map_equals_apply": 5216, "default_states_checked": 199, "default_candidate_equals_global_a |
| X10 | PASS | {"booby_trap": {"regeneration_completed_with_hidden_heuristic_disabled": true, "entry_points_disabled": ["audit_selective_guidance_control.HEURISTIC", "DeadbandCapacityAwareHeuristic.__call__/.predict", "capacity_aware_h |

## Dataset

240 trajectories, 796 decision states, 237,208 candidate rows (59 columns), 19,319 unique rollouts. File `pilot/pilot_rows.csv.gz` (SHA-256 `df6e5092af310640`).

## Diversity vs the old dataset

| measure | old dataset | pilot |
|---|---|---|
| decision states | 450 | 796 |
| unique observable inputs | 11 | 183 |
| distinct share | 2.4% | 23.0% |
| unique (input, candidate) pairs | 140 | 54,534 |
| mean feature entropy (bits) | 0.8828 | 1.1466 |

By tick: {'0': {'states': 240, 'unique': 8}, '5': {'states': 240, 'unique': 12}, '10': {'states': 240, 'unique': 129}, '15': {'states': 71, 'unique': 30}, '20': {'states': 5, 'unique': 4}}. By template: {'D0_sym': 34, 'D1_east_heavy': 31, 'D2_west_heavy': 27, 'D3_first_room_heavy': 36, 'D4_second_room_heavy': 37, 'D5_R1_concentrated': 31, 'Z1_R4_concentrated': 29, 'Z2_mixed': 36}. Ticks >= 10: {'states': 316, 'unique': 163, 'share': 0.5158, 'unique_by_template': {'D0_sym': 31, 'D1_east_heavy': 28, 'D2_west_heavy': 24, 'D3_first_room_heavy': 33, 'D4_second_room_heavy': 34, 'D5_R1_concentrated': 28, 'Z1_R4_concentrated': 26, 'Z2_mixed': 33}}. Repeated-state fraction 77.0%; repeated-trajectory fraction 38.3%.

Conditional outcome variance (irreducible share, SSW/SST over all rows; singleton groups contribute 0, so these are lower bounds):

- future_T90: given state 0.0298, given state and candidate 0.0179
- future_T100: given state 0.0384, given state and candidate 0.0247
- future_unmet: given state 0.1928, given state and candidate 0.1275
- future_hazard_unmet: given state 0.4272, given state and candidate 0.2039

G6 (pre-registered, multi-trajectory groups, post-onset): share 0.0165, coverage 74.5% of 108,472 post-onset rows; plain formula 0.0138.

## Targets

- hazard target 0 in all no-hazard rows: True; 0 whenever the episode ends before onset: True
- T100 / T90 semantics on every row: True / True
- pre-decision unmet excluded (full-episode decomposition): 556/556 states exact
- distinct target vectors per state: {'min': 1, 'median': 3.0, 'max': 61, 'states_with_at_least_3': 516, 'share_with_at_least_3': 0.6482}
- share of candidates tied with the default: {'mean': 0.4676, 'median': 0.3322}; classes per state: {'mean': 24.27}
- one continuation for all rows: True

## Joint grid

120 cells (8 templates x 5 t_h x 3 p), trajectories per cell {2: 120}; association template-t_h 0.0, template-p 0.0; seeds unique True (190000-190239). all pilot t_h and p values are TRAIN-role levels by the frozen design; the pilot is discarded after Phase 2, so it cannot leak into any full-dataset split. Primary joint test constructible at design level: True (['t_h=4|p=0.6', 't_h=4|p=0.85', 't_h=8|p=0.6', 't_h=8|p=0.85']).

## Leakage (on the rows)

passed True: {'states': 796, 'features_identical_across_candidates': 796, 'residual_ok': 796, 'replay_features_match': 796, 'replay_fingerprint_match': 796, 'class_relabels_checked': 19319, 'features_unchanged_by_relabel': 19319, 'pre_onset_states': 288, 'pre_onset_features_identical_without_hazard': 288}; feature columns exact True; forbidden feature columns []; tick-0 inputs independent of t_h and p True.

## Label-free continuation

label-free True; every row carries hash `2614f9af3d4a43e6`: True; forbidden tokens in generator source: {'run_ai_learning_pilot.py': [], 'ai_learning_label_free_default.py': []}; full regeneration with every hidden-heuristic entry point disabled completed: True and reproduced every row: True; all continuation calls had 26 features: True. Candidate semantics: {'states': 796, 'default_equals_label_free_rule': 796, 'default_equals_closed_form': 796, 'class_reps_checked': 5216, 'fast_label_map_equals_apply': 5216, 'default_states_checked': 199, 'default_candidate_equals_global_apply': 199}.

## G3 unrestricted re-check

40/40 states: BASE2_298 best == unrestricted best; mean T100 regret 0.0, max 0.0; non-empty zones {4: 40}.

## Integrity

Protected files unchanged: **True** (91 files). Controller MD5 match True; RL checkpoints unchanged True.
