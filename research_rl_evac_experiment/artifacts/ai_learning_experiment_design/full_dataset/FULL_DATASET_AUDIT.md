<!-- ai_learning_full_dataset_audit_v1 -->
# Full AI-Learning Dataset (Phase 3) Audit

No model trained, no selector, no closed-loop evaluation. G1-G7 / X1-X10 applied with the spec v2 definitions and thresholds, as declared in full_generation_record.json before generation.

## Decision: **FULL DATASET NO-GO** (failed: G1, X5, X7)

| criterion | result | value |
|---|---|---|
| G1 | FAIL | {"share": 0.1838, "min_unique_per_template": 118} |
| G2 | PASS | 0.655 |
| G3 | PASS | {"share": 1.0, "mean_T100_regret": 0.0, "nonempty": {"4": 40}} |
| G4 | PASS | {"identical_rows": true, "hazard_tick_log": "457592/457592"} |
| G5 | PASS | "see full_leakage_audit.json" |
| G6 | PASS | {"share": 0.0148, "coverage": 0.9336} |
| G7 | PASS | {"cells": 472, "trajectories": 5544} |
| X1 | PASS | {"states_checked": 12994, "exact": 12994} |
| X2 | PASS | "= G5" |
| X3 | PASS | "457592/457592" |
| X4 | PASS | {"primary_trajectories": 960, "held_out": {"primary_trajectories": 960, "primary_t_h_values": ["4", "8"], "primary_p_values": [0.6, 0.85], "primary_t_h_absent_from_train": true, "primary_p_absent_from_train": true, "zero_shot_templates_only |
| X5 | FAIL | {"unique_inputs": 1389, "distinct_share": 0.0749, "pairs": 413922} |
| X6 | PASS | "= G2" |
| X7 | FAIL | {"association_all_trajectories": {"template_vs_t_h": 0.0399, "template_vs_p": 0.0477}, "association_six_main_templates": {"template_vs_t_h": 0.0, "template_vs_p": 0.0}, "seeds_unique": true, "perturbation_rate": 0.5145, "censored_rows": 0} |
| X8 | PASS | "= X1 decomposition" |
| X9 | PASS | {"default_equals_rule": 18538, "default_equals_closed_form": 18538, "label_map_checked": 112995, "fast_label_map_equals_apply": 112995, "default_states_checked": 4635, "default_candidate_equals_global_apply": 4635} |
| X10 | PASS | {"booby_trap": true, "tokens": {"run_ai_learning_full_dataset.py": [], "run_ai_learning_pilot.py": [], "ai_learning_label_free_default.py": []}} |

Deviation from spec v2: TRAIN seeds extend from the documented 100000-100999 to 100000-101799 to give 1,800 unique TRAIN seeds; non-overlap with every other role range, the PILOT range and the Test A range 125000-125999 is preserved and verified.

## Scale

5544 trajectories, 18,538 decision states, 5,524,324 candidate rows, 457,592 unique rollouts, 1,389 unique observable inputs, 413,922 unique (input, candidate) pairs.

## Diversity

Distinct share 7.5%; tick>=10 {'states': 7450, 'unique': 1369, 'share': 0.1838, 'unique_by_template': {'D0_sym': 288, 'D1_east_heavy': 359, 'D2_west_heavy': 350, 'D3_first_room_heavy': 320, 'D4_second_room_heavy': 289, 'D5_R1_concentrated': 403, 'Z1_R4_concentrated': 118, 'Z2_mixed': 121}}; by tick {'0': {'states': 5544, 'unique': 8}, '5': {'states': 5544, 'unique': 12}, '10': {'states': 5544, 'unique': 1234}, '15': {'states': 1833, 'unique': 122}, '20': {'states': 73, 'unique': 13}}; by template {'D0_sym': 291, 'D1_east_heavy': 362, 'D2_west_heavy': 353, 'D3_first_room_heavy': 323, 'D4_second_room_heavy': 292, 'D5_R1_concentrated': 406, 'Z1_R4_concentrated': 121, 'Z2_mixed': 124}; by role {'DIAG_EXTRAPOLATION': 251, 'DIAG_TEMPLATE_ZERO_SHOT': 228, 'TEST_HELDOUT_COMPLIANCE': 455, 'TEST_HELDOUT_TIMING': 260, 'TEST_PRIMARY_JOINT': 399, 'TRAIN': 702, 'VALIDATION': 323}; repeated-state 92.5%, repeated-trajectory 68.6%; mean feature entropy 1.1319 bits (old 0.8828).

Target variation: {'states_with_ge2_outcome_vectors': 13655, 'states_with_ge3_outcome_vectors': 12143, 'share_ge2': 0.7366, 'share_ge3': 0.655, 'tied_with_default_mean': 0.4627, 'classes_per_state_mean': 24.68}. Conditional variance: {'future_T90': {'given_state': np.float64(0.0331), 'given_state_and_candidate': np.float64(0.0209)}, 'future_T100': {'given_state': np.float64(0.0399), 'given_state_and_candidate': np.float64(0.0254)}, 'future_unmet': {'given_state': np.float64(0.2071), 'given_state_and_candidate': np.float64(0.1413)}, 'future_hazard_unmet': {'given_state': np.float64(0.4828), 'given_state_and_candidate': np.float64(0.2547)}}. G6: {'post_onset_rows': 2662928, 'coverage': 0.9336, 'T100_irreducible_share': np.float64(0.0148), 'evaluable': True, 'passed': True}.

## Joint grid and seeds

Allocation {'TRAIN': 1800, 'VALIDATION': 540, 'TEST_PRIMARY_JOINT': 960, 'TEST_HELDOUT_TIMING': 540, 'TEST_HELDOUT_COMPLIANCE': 900, 'DIAG_EXTRAPOLATION': 450, 'DIAG_TEMPLATE_ZERO_SHOT': 354}; all 472 cells complete True; seeds 5544/5544 unique; effective ranges {'TRAIN': '100000-101799', 'VALIDATION': '110000-110999', 'TEST_PRIMARY_JOINT': '120000-120999', 'TEST_HELDOUT_TIMING': '130000-130999', 'TEST_HELDOUT_COMPLIANCE': '140000-140999', 'DIAG_EXTRAPOLATION': '150000-150999', 'DIAG_TEMPLATE_ZERO_SHOT': '160000-160999', 'PILOT': '190000-190999', 'TEST_A_RESERVED': '125000-125999'}; overlaps []; pilot seeds reused False; Test A range untouched True; held-out {'primary_trajectories': 960, 'primary_t_h_values': ['4', '8'], 'primary_p_values': [0.6, 0.85], 'primary_t_h_absent_from_train': True, 'primary_p_absent_from_train': True, 'zero_shot_templates_only_in_zero_shot_role': True, 'compliance_keys_role_specific': True, 'trajectory_in_exactly_one_role': True}; association {'all_trajectories': {'template_vs_t_h': 0.0399, 'template_vs_p': 0.0477}, 'six_main_templates': {'template_vs_t_h': 0.0, 'template_vs_p': 0.0}, 'every_template_has_every_t_h_and_p_level': True}.

## Leakage

passed True; {'states': 18538, 'replay_features_match': 18538, 'replay_fingerprint_match': 18538, 'pre_decision_unmet_match': 18538, 'residual_ok': 18538, 'default_equals_rule': 18538, 'default_equals_closed_form': 18538, 'class_relabels_checked': 457592, 'features_unchanged_by_relabel': 457592, 'label_map_checked': 112995, 'fast_label_map_equals_apply': 112995, 'default_states_checked': 4635, 'default_candidate_equals_global_apply': 4635, 'decomposition_states': 12994, 'decomposition_exact': 12994, 'pre_onset_states': 7584, 'pre_onset_features_identical_without_hazard': 7584, 'seconds': 99.6}; stream violations {}.

## Targets

passed True; decomposition {'states_checked': 12994, 'exact': 12994}; tick-log 457592/457592; identical starts 457592/457592; G3 40/40.

## Continuation

label-free True; semantics True; provenance [['SYMMETRIC_SPLIT_DEADBAND_margin0', '2614f9af3d4a43e61b008e863b51212acb4b50d64a859e223253758785066f85', 'v2', 'unique_per_trajectory', 5524324]]; regeneration identical True with hidden heuristic disabled.

## Integrity

103 protected files unchanged: True; controller True; checkpoints True.
