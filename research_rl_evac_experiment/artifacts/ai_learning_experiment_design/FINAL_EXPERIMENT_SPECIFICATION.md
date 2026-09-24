<!-- experiment_specification_v2_d1_d2 -->
<!-- supersedes final_experiment_specification_audit_v1 (archived: spec_versions/v1/) -->
# Final Experiment Specification, v2 (D1 and D2 applied)

v1 is archived byte-for-byte in `spec_versions/v1/`; `spec_version_record.json` lists every file's v1 and v2 SHA-256 and the fields that changed. Nothing in v2 changes the simulator, observation.py, the RL branch or the old 9,450-row dataset.

## Status: READY FOR PHASE 1 PILOT

## Decisions applied

- **D1 (approved)**: future hazard-edge unmet demand counts only ticks t >= t_h (actual onset); 0 before onset; 0 in no-hazard scenarios.
- **D2 (decided)**: continuation, selector / abstention / OOD fallback, closed-loop default and pre-decision generation all use the label-free `SYMMETRIC_SPLIT_DEADBAND_margin0` (hash `2614f9af3d4a43e6`): PARITY by default, FAVOR_B once the hazard has actually started and anyone waits at the hub. Inputs: the 26 admissible features only. The hidden-label heuristic is reference arm P2b only.

## Preserved exactly

- Candidate set BASE2_298 (`candidate_action_set.json` byte-identical to v1).
- 26-feature admissible input (`final_feature_manifest.json` byte-identical to v1).
- Scenario grid, joint timing x compliance primary test, trajectory-level split, 240-trajectory pilot.

## Evidence re-checked under the label-free continuation

BASE2_298 vs the unrestricted absolute space on the existing 530 states with the label-free continuation: 100.0% best-match, mean T100 regret 0.0, max 0.0 (v1 under the hidden heuristic: 100%, 0).

Design-time diversity with the label-free pre-decision policy: epsilon_0.0: 67 distinct inputs, tick>=10 distinct share 3.1%; epsilon_0.3: 429 distinct inputs, tick>=10 distinct share 23.7%; epsilon_0.5: 596 distinct inputs, tick>=10 distinct share 31.1%.

## Phase 1 pilot (pre-registered)

240 trajectories = 8 templates x t_h ['none', '3', '5', '7', '9'] x p [0.5, 0.7, 0.9] x 2 replicates; seed = 190000 + trajectory index (0..239): unique per trajectory, inside the PILOT range, disjoint from every full-dataset role range; compliance generate_compliance_vector('AILX_v1|PILOT|' + template, seed, p=p); pre-decision: SYMMETRIC_SPLIT_DEADBAND_margin0 with epsilon 0.5 (at each decision tick: u = rng.random(); if u < 0.5: a = int(rng.integers(0, 3)) else a = label_free_default(x); a is issued as a global action at that tick). The pilot uses only TRAIN-role t_h and p levels by design; the joint test is verified at design level (X4), and the pilot is discarded after Phase 2.

## Pilot go / no-go (fixed before any pilot data exists)

- **G1**: decision ticks >= 10: distinct admissible inputs >= 25% of those states AND >= 10 distinct per template
- **G2**: >= 50% of states have >= 3 distinct target vectors among their 298 candidates
- **G3**: BASE2_298 best == unrestricted best in >= 95% of the 40 re-check states AND mean T100 regret <= 0.1 tick
- **G4**: independent regeneration reproduces every stored row exactly AND the hazard measurement equals the simulator tick_log in 100% of rollouts
- **G5**: leakage re-audit on the rows passes: stored features are exactly the 26 admissible names; identical across a state's candidates; equal to the replayed pre-decision observation; unchanged by every candidate relabel; for every pre-onset hazard state, replaying the same history with no hazard gives identical features; residual capacity is 1 before onset and 2/6 after
- **G6**: T100 irreducible share < 0.6 on post-onset rows, computed as SSW/SST over (features, candidate_id) groups that contain rows from >= 2 distinct trajectories; if such groups cover < 10% of post-onset rows the criterion is NOT EVALUABLE and counts as not passed
- **G7**: every (template, t_h, p) pilot cell has 2 complete trajectories
- **X1**: targets valid: no-hazard rows have hazard target 0; rows whose episode ends before onset have hazard target 0; future_T100 = T100_abs - t_d and future_T90 = T90_abs - t_d (or 0 with the reached flag) on every row; for every t_d > 0 state, pre-decision unmet + future unmet of the default candidate equals the unmet of a full-episode rerun from t=0
- **X2**: zero leakage (= G5)
- **X3**: every candidate rollout starts from a state whose fingerprint equals the recorded pre-decision state (100%)
- **X4**: joint timing x compliance design valid: 4 TEST_PRIMARY_JOINT cells; test t_h and p levels disjoint from training levels; every template crosses every t_h and p; seed ranges disjoint across roles and from PILOT; compliance keys role-specific
- **X5**: diversity materially exceeds the old dataset: distinct admissible inputs >= 110 (10x old) AND distinct share >= 0.1222 (5x old) AND distinct (input, candidate) pairs >= 1400 (10x old)
- **X6**: sufficient target variation (= G2)
- **X7**: no structural confound: every template has every (t_h, p) level (association 0); seeds unique per trajectory; realized perturbation rate within 0.5 +/- 0.1; no censored T100
- **X8**: targets not dominated by unobservable history: pre-decision unmet is excluded (X1 decomposition check)
- **X9**: candidate semantics consistent: fast label maps equal apply_differential_action label maps for every class representative; the default candidate BASE=l(s) equals applying l(s) globally with actions.apply_action
- **X10**: continuation label-free: the generator and default module never reference the hidden heuristic; every continuation call receives a 26-feature vector; every row records the current continuation hash
- **rule**: GO only if G1-G7 and X1-X10 all pass. Thresholds are fixed in this v2 specification before any pilot data exists and must not be relaxed afterwards.
