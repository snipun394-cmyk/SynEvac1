# Localized Predictive Model V3 — Full Evaluation of Physically Corrected Congestion Target V2

Status: **OFFLINE RESEARCH ONLY.** Nothing in this milestone is wired into `LiveRuntime`, Recommendation, Guidance, Decision Policy, or any operator-facing workflow. `live_inference_wired = false` is written into every exported artifact explicitly.

Git commit this milestone's results were produced against: `93e3b183d97aa3aba0fafb1b3aea615654846936` (the [[predictive_congestion_target_v2_milestone]] commit — Target V2 itself is frozen and was not modified).

## 1. Purpose and scope

Answer one question rigorously: **can SynEvac reliably predict the ONSET of physically meaningful congestion (Target V2, `"v2-persistent-demand-service-imbalance"`) at Door/Exit/Stair candidates ~20 seconds before it happens?** Target V1's Door/Stair signal was already shown ([[localized_predictive_model_v2_2_milestone]]) to be a 100%-zero-duration simulator artifact; Target V2 ([[predictive_congestion_target_v2_milestone]]) replaced it with a real, physically-grounded, onset-based definition. This milestone is the first full-rigor evaluation of a model against that corrected target — the direct successor evaluation to V2.2, applied to the new target.

## 2. What this milestone reused vs. built new

Reused, byte-for-byte unchanged:
- `data/predictive_congestion_target_v2/candidate_dataset_relabeled.csv` (2,405,049 rows, 2,500 scenarios) — **no re-simulation for the primary evaluation**, confirmed sufficient during Phase 1 investigation.
- The V2.1/V2.2 12-field experimental feature schema (`predictive_model.feature_prep_v2_1.build_experimental_feature_matrix`).
- Scenario-split seed/convention (`SEED = 20260726`, 70/15/15, scenario-level only).
- `predictive_model.*` infrastructure: `scenario_split`, `imbalance`, `metrics`, `calibration`, `feature_importance`, `sanity_checks`, `operational_slices`, `topology_holdout`, `training_size_study`, `model_export`/`model_export_v2`, `tree_models`.

Built new this milestone:
- `predictive_model.baselines.DeterministicCurrentStateBaseline` — the central "does ML beat SynEvac's existing deterministic intelligence" comparison point.
- `scripts/train_localized_predictive_model_v3.py` — the main training/evaluation script (761+ lines), including lead-time bucketing, near-onset exclusion, multi-bottleneck candidate-type-combination analysis, and threshold sweeps specific to Target V2's onset semantics (meaningless for Target V1).
- `scripts/model_v3_specialization_experiment.py` — unified-vs-per-type model comparison.
- `scripts/model_v3_horizon_sensitivity.py` — reduced-scale 10s/20s/30s horizon comparison via deterministic re-simulation of a 400-scenario subset.
- Additive fields on `predictive_model.training_size_study.training_size_study` (`progress_fn` callback) and `predictive_model.model_export_v2.ExtendedModelMetadataV2` (`feature_schema_version`, `minimum_congestion_duration_seconds`, `training_scenario_count`, `calibration_status`, `metrics_by_candidate_type`, `metrics_by_topology_family`) — both backward-compatible, defaults keep every V1/V2/V2.2 call site unaffected.

## 3. Currently-congested exclusion audit

| | Count |
|---|---|
| Total candidate-time observations | 2,405,049 |
| Excluded (`currently_congested = True`, target N/A) | 674,073 |
| Eligible for prediction | 1,730,976 |
| Positives | 49,331 |
| Negatives | 1,681,645 |
| Positive rate (eligible) | 2.85% |

Trainable row counts by split: train 1,210,568 / val 257,358 / test 263,050.

## 4. Scenario-level split integrity

`split_scenarios(seed=20260726, ratios=(0.70, 0.15, 0.15))` on 2,500 scenarios → 1,750 train / 375 val / 375 test. `assert_no_scenario_overlap` mechanically verified zero shared scenario IDs across all three splits (this call raises on any overlap — the run completed, so it passed).

## 5. Class imbalance handling

`sklearn`-style `class_weight='balanced'` sample reweighting (`predictive_model.imbalance.compute_class_weight_map` / `sample_weights_from_class_weight`) for every non-trivial model — no oversampling, no SMOTE, consistent with every prior milestone.

## 6. Baselines and candidate models — full comparison

| Model | Test ROC-AUC | Test PR-AUC | Precision @ tuned threshold | Recall @ tuned threshold |
|---|---|---|---|---|
| majority_class | 0.500 | 0.0282 (= prevalence) | 0.000 | 0.000 |
| always_negative | 0.500 | 0.0282 | 0.000 | 0.000 |
| random | 0.495 | 0.0277 | 0.028 | 0.491 |
| logistic_regression | 0.933 | 0.490 | 0.472 | 0.523 |
| decision_tree (depth 6) | 0.949 | 0.441 | 0.242 | 0.761 |
| **deterministic_current_state** (new, rule-based, no learning) | 0.749 | 0.116 | 0.103 | 0.192 |
| gradient_boosting (HistGB) | 0.961 | 0.485 | 0.333 | 0.738 |
| **xgboost (WINNER)** | **0.967** | **0.615** | 0.543 | 0.573 |

RandomForest was deliberately skipped and disclosed (not silently dropped) — every prior milestone (V1/V2/V2.1/V2.2) found it a statistical near-tie with XGBoost/HistGB at 5-13x the fit cost (350-386s vs 25-30s), and this milestone's charter explicitly forbids a "giant tournament," permitting RF only "if cheap." 350s+ is not cheap on this ~7.3GB-RAM development machine, and it has never changed which model wins.

XGBoost hyperparameters: `max_depth=6, n_estimators=300, learning_rate=0.1, tree_method=hist, eval_metric=logloss`, `random_state=20260726`, class-imbalance handled via `sample_weight` (so `scale_pos_weight` forced to 1.0 to avoid double-correction). Decision threshold tuned on validation: **0.92**.

## 7. The deterministic-current-state baseline — the real bar

Per this milestone's central mandate, the question is not "does ML beat random" but **"does ML beat SynEvac's existing deterministic intelligence"** — the congestion-level classification and trend direction `crowd_intelligence` already computes and any operator can already see live. `DeterministicCurrentStateBaseline` is a fixed, unlearned rule: `score = congestion_level_rank(0-4) + 1.0·[trend=RISING] + 0.5·[trend=UNKNOWN]`, normalized to `[0,1]` by the maximum possible score (5.0). `fit()` is a genuine no-op — no parameter is ever learned from labels (verified in `tests/test_predictive_model_deterministic_baseline.py`).

**Result: PR-AUC 0.116 vs. XGBoost's 0.615 — an absolute lift of 0.499, a relative lift of 5.31x.** The deterministic baseline clears random by a wide margin (0.116 vs. 0.028 prevalence, ~4x) — confirming SynEvac's existing live signal genuinely carries real forward-looking information — but ML adds substantial, non-trivial additional value on top of it. This is the single most important comparison this milestone makes: it is not merely "the model beats chance," it is "the model beats what an operator with today's dashboard already has."

## 8. Leave-one-topology-family-out — the critical generalization gate

| Held-out family | Test ROC-AUC | Test PR-AUC | vs. in-distribution PR-AUC (0.615) |
|---|---|---|---|
| multi_exit_wide | 0.863 | 0.305 | 0.495x |
| single_exit_lowrise | 0.887 | 0.571 | 0.928x |
| twin_stair_highrise | 0.783 | 0.308 | 0.501x |
| v1_topology_fixed | 0.916 | 0.531 | 0.863x |

Two of four families (`multi_exit_wide`, `twin_stair_highrise`) show a real, substantial generalization gap — PR-AUC roughly halves when that topology family is entirely unseen during training. This is a genuine limitation, not a collapse (ROC-AUC stays high, 0.78-0.92, in every family — the model isn't randomly guessing, it's less well-calibrated to unfamiliar topology geometry). This is the single strongest piece of evidence against a D/E production-readiness verdict this milestone (see §21).

## 9. Lead-time analysis — is this useful early warning?

| Bucket | n positives | Recall | Mean predicted prob. |
|---|---|---|---|
| 0-5s before onset | 2,539 | 0.698 | 0.876 |
| 5-10s before onset | 2,028 | 0.555 | 0.835 |
| 10-15s before onset | 1,598 | 0.500 | 0.823 |
| 15-20s before onset | 1,240 | 0.440 | 0.796 |

Recall decays smoothly from 0.70 (near-immediate) to 0.44 (near the 20s horizon edge) — genuine, distributed early warning, not a model that only fires an instant before onset. Median lead time (from the exploratory model in the Target V2 milestone) was 7.6s; this full evaluation confirms that pattern holds under rigorous scenario-holdout evaluation.

## 10. Near-onset exclusion / foresight-survives audit

Re-evaluating test performance while excluding examples whose true onset is imminent (removing the "obviously about to happen" cases the model could trivially detect from current state alone):

| Min. lead time required | n | Positive rate | ROC-AUC | PR-AUC | Recall |
|---|---|---|---|---|---|
| 0.0s (full test set) | 263,050 | 2.82% | 0.967 | 0.615 | 0.573 |
| 1.0s | 262,551 | 2.63% | 0.966 | 0.590 | 0.560 |
| 2.0s | 262,037 | 2.44% | 0.966 | 0.559 | 0.545 |

PR-AUC degrades only mildly (0.615 → 0.559, a 9% relative drop) when the near-instant cases are excluded — the model is not merely a "currently-congested detector with a 1-frame lag." Genuine forward-looking skill survives this audit.

## 11. Horizon sensitivity (10s / 20s / 30s) — REDUCED SCALE

The primary dataset was relabeled at a single, fixed 20s horizon; extending to other horizons requires re-deriving labels from each scenario's raw `movement_result`, which was never persisted. Rather than re-simulate all 2,500 scenarios again (same ~5-minute cost as the original campaign, disproportionate for a secondary check), this experiment deterministically re-simulates a topology-diverse 400-scenario subset (100/family, same `master_seed=20270115`) and derives all three horizons' labels from the *same* precomputed onsets per scenario — one simulation pass, three label passes.

| Horizon | Test prevalence | Test PR-AUC | Test ROC-AUC | Relative lift (PR-AUC / prevalence) |
|---|---|---|---|---|
| 10s | 2.30% | 0.521 | 0.947 | 22.7x |
| 20s (primary) | 3.72% | 0.578 | 0.947 | 15.5x |
| 30s | 4.57% | 0.615 | 0.952 | 13.5x |

Raw PR-AUC rises with horizon, but this is partly a prevalence artifact (longer windows have structurally higher positive rates, and PR-AUC's baseline scales with prevalence). **Normalized by relative lift, prediction gets intrinsically *harder*, not easier, as the horizon lengthens** (22.7x → 15.5x → 13.5x) — exactly as expected, since more time means more opportunity for the system state to diverge unpredictably from its current trajectory. This validates 20s as a defensible middle ground: meaningfully more useful lead time than 10s, without the diminishing relative-skill returns of 30s. 20s remains the primary/production horizon; this check does not change that.

## 12. Multi-bottleneck analysis

Candidate-type combinations among simultaneously-positive (scenario, time) buckets, test split:

| Combination | Count |
|---|---|
| Door + Exit | 3,588 |
| Door + Exit + Stair | 264 |
| Door + Stair | 164 |
| Exit + Stair | 562 |

3,465 of 263,050 test rows (1.3%) belong to a multi-bottleneck bucket. Multi-bottleneck rows are dramatically harder: **false positive rate 13.9% and false negative rate 12.0%, vs. 1.19%/1.06% for single-bottleneck rows — roughly 11-12x worse**, directly consistent with the same magnitude finding from [[localized_predictive_model_v1_milestone]]'s dataset-level analysis on Target V1. This is a genuine, unresolved failure mode, not something Target V2's redesign fixed.

## 13. Occupancy-severity analysis

| Occupancy bucket | n | False positive rate | False negative rate |
|---|---|---|---|
| LOW | 87,229 | 0.19% | 0.68% |
| MEDIUM | 73,937 | 0.58% | 1.14% |
| HIGH | 101,884 | 2.92% | 1.69% |

Error rates climb with building occupancy — the model is least reliable exactly when a real evacuation would be most crowded and prediction would matter most. A secondary, related finding: **temporal phase** shows the same pattern (EARLY-phase FPR 4.80% vs. MID 0.58% vs. LATE 0.02%) — the model is noisiest before evacuation dynamics have established a clear trend, which is also structurally when HIGH-occupancy conditions are most likely to still be developing.

## 14. Single vs. multi-exit topology comparison

From the topology-holdout results (§8): `single_exit_lowrise` (held out) generalizes far better (PR-AUC 0.571, 92.8% of in-distribution) than `multi_exit_wide` (held out, PR-AUC 0.305, 49.5%). Multi-exit geometries introduce route-choice/alternative-path dynamics the model has not learned to generalize across when that exact topology family is entirely absent from training — single-exit geometry's demand/queue dynamics appear to transfer more readily to unseen scenarios of the same family shape.

## 15. Feature importance (permutation, model-agnostic)

Top features by permutation importance (baseline ROC-AUC 0.964, 20,000-row sample, 5 repeats):

| Feature | Mean importance |
|---|---|
| candidate_approaching_count | 0.1102 |
| total_active_occupant_count | 0.1052 |
| candidate_adjacent_zone_occupancy | 0.0270 |
| candidate_walking_distance | 0.0252 |
| candidate_recent_flow_rate | 0.0217 |
| candidate_congestion_trend=STABLE | 0.0138 |
| candidate_alternative_route_count | 0.0125 |
| candidate_queue_length | 0.0057 |
| candidate_type=Stair | 0.0037 |

**`candidate_queue_length` is notably NOT dominant** (rank 8, far behind `approaching_count` and `total_active_occupant_count`) — a direct, expected consequence of the currently-congested exclusion (§3): rows already showing high queue are disproportionately excluded from the eligible/predictable set, so the model is pushed toward genuine forward-looking demand signals (occupants *approaching*, not occupants *already queued*) rather than reporting already-elevated state. This matches the exploratory model's finding in [[predictive_congestion_target_v2_milestone]] exactly.

## 16. Feature-family ablation

| Family | ROC-AUC drop | PR-AUC drop |
|---|---|---|
| demand_signal (queue_length, approaching_count) | 0.0134 | 0.0665 |
| flow_and_trend (recent_flow_rate, trend one-hots) | 0.0120 | 0.0690 |
| global_and_adjacent_context | 0.0089 | 0.0214 |
| structural (capacity, distance, traversable, type) | 0.0034 | 0.0160 |
| derived_congestion_level | -0.0003 | -0.0013 |
| alternative_route_structure | 0.0000 | -0.0010 |

`demand_signal` and `flow_and_trend` are the two load-bearing families (PR-AUC drops of 0.065-0.069 each when zeroed); `derived_congestion_level` and `alternative_route_structure` contribute essentially nothing incremental once the other families are present (near-zero or slightly negative drop — noise-level, not a real negative contribution).

## 17. Current-state tautology audit

Combining §10 (near-onset exclusion: PR-AUC survives at 91% of full value even excluding <2s-lead cases) and §15 (queue_length, the most obvious "already congested" signal, ranks 8th in importance, not 1st): **the model is not merely a disguised currently-congested detector**. It relies predominantly on forward-looking demand signals (approaching occupants, building-wide occupancy) rather than already-elevated queue state.

## 18. Label-shuffle sanity test — investigated anomaly

**Result: ROC-AUC = 0.378 (not near chance; the 0.5±0.05 tolerance every prior milestone comfortably cleared — V1 0.499, V2 0.474, V2.2 0.459 — was violated).** Per this milestone's own explicit instruction ("if not near chance: investigate leakage immediately"), this was investigated in depth in isolation (not accepted or glossed over) before any production-readiness conclusion was drawn.

**Investigation findings:**
1. **Reproducible across shuffle seeds** (0.345, 0.351, 0.378, 0.392 across 4 different seeds) — a systematic, directional bias, not one seed's noise.
2. **96.5% of train rows share an exact-duplicate feature vector** with at least one other row (50,223 duplicate groups covering 1,167,598 of 1,210,568 train rows) — a structural consequence of the 27-column schema's low-cardinality one-hot/binned features.
3. **Model-dependent, sign-flipping result with the SAME data/split/seed**: XGBoost (full capacity + heavy reweighting) → 0.378 (below chance); LogisticRegression → 0.665 (above chance); XGBoost with reweighting removed → 0.462 (within the 0.05 tolerance band); XGBoost with reduced capacity (depth 3, 50 trees) → 0.567-0.578 regardless of reweighting.
4. **Mechanism isolated directly**: the extreme deviation only appears under the *combination* of severe class-imbalance reweighting (~34x, from `class_weight='balanced'` applied to a ~2.86% positive rate) and full model capacity (depth 6, 300 trees) — exactly the configuration that lets a tree ensemble overfit to noise on a feature space where the overwhelming majority of rows are duplicates.

**Conclusion**: a genuine leakage channel (row overlap, a leaked identifier) would be exploitable at *any* capacity/reweighting level and would push ROC-AUC toward 1.0 for essentially any sufficiently expressive model, not produce a model-and-hyperparameter-dependent result that flips sign. Combined with zero features flagged at the 0.9 correlation threshold (§19), proven zero scenario overlap (§4), and topology-holdout/near-onset-exclusion results showing smooth, bounded degradation rather than the near-perfect scores a real leak would preserve under those same stress tests — **this is assessed as a label-shuffle-test methodology limitation specific to the winning model's production configuration on this duplicate-heavy feature space, not evidence of data leakage.**

This does not, however, earn a clean pass. It is reported as an unresolved limitation of the current sanity-check methodology and is treated as a contributing factor (alongside §8's topology generalization gap) in the conservative production-readiness verdict (§21) — not silently waived.

## 19. Feature/target correlation audit

No feature reaches the 0.9 leakage-review threshold (`predictive_dataset.correlation.LEAKAGE_REVIEW_THRESHOLD`, reused verbatim). Highest-magnitude correlations: `candidate_congestion_trend=STABLE` (-0.433, negative — stable trend predicts *against* onset, as expected), `candidate_congestion_trend=UNKNOWN` (0.339), `candidate_recent_flow_rate` (0.278), `candidate_approaching_count` (0.266). All feature-extraction windows terminate at or before observation time `t` by construction (`predictive_dataset.simulation_extractor_v2_1`, unmodified this milestone) — no feature was found to encode future onset.

## 20. Calibration

| | Brier score | ECE |
|---|---|---|
| Before calibration | 0.0688 | 0.1105 |
| After Platt scaling | 0.0181 | 0.0099 |
| **After isotonic (recommended)** | **0.0164** | **0.0024** |

By candidate type (isotonic):

| Type | Brier (before → after) | ECE (before → after) |
|---|---|---|
| Door | 0.0454 → 0.0116 | 0.0658 → 0.0026 |
| Exit | 0.0987 → 0.0228 | 0.1639 → 0.0026 |
| Stair | 0.0177 → 0.0048 | 0.0305 → 0.0020 |

Isotonic calibration is recommended and exported (`calibrator.joblib`), fit exclusively on the validation split (never test). Raw model probabilities are poorly calibrated (ECE 0.11) and must not be used directly as risk estimates; calibrated probabilities are well-calibrated across all three candidate types (ECE ≤ 0.003 after isotonic).

## 21. Threshold analysis (no policy threshold chosen)

| Threshold | Precision | Recall | FPR | FNR | F1 |
|---|---|---|---|---|---|
| 0.1 | 0.112 | 0.988 | 0.227 | 0.012 | 0.201 |
| 0.3 | 0.152 | 0.958 | 0.155 | 0.042 | 0.262 |
| 0.5 | 0.198 | 0.903 | 0.106 | 0.097 | 0.324 |
| 0.7 | 0.308 | 0.786 | 0.051 | 0.214 | 0.442 |
| 0.9 | 0.499 | 0.617 | 0.018 | 0.383 | 0.552 |

The full sweep (0.1-0.9) is reported for future decision-makers; per the milestone charter, **no operational threshold is chosen here** — 0.92 is used only internally for this report's confusion-matrix/precision/recall breakdowns, not as a recommendation.

## 22. Live feature parity re-audit (re-verified from current code)

| Feature | Status | Basis |
|---|---|---|
| candidate_congestion_trend | EXACT | `crowd_intelligence.models.AssetApproachMetrics.trend`, shared function both sim and live |
| candidate_alternative_route_count | STRUCTURAL | Shared computation function, both sides |
| candidate_recent_flow_rate (Exit) | EXACT | `evacuation_progress.models.ExitFlow.recent_flow_per_minute` |
| candidate_recent_flow_rate (Door/Stair) | PARTIAL-UNVALIDATED | `live_occupants.history.OccupantHistory.zone_transitions` — unit-tested, never validated against a real live deployment |
| candidate_queue_length, candidate_approaching_count, total_active_occupant_count, candidate_adjacent_zone_occupancy | EXACT | Direct `BuildingState`/`crowd_intelligence` reads, unchanged since V2.1 |
| candidate_capacity, candidate_walking_distance, candidate_traversable, candidate_type | STRUCTURAL | Building topology, identical definition sim and live |

`predictive_dataset/live_extractor_v2_1.py` was re-read this milestone (not assumed from prior docs) and confirmed unchanged since [[localized_predictive_model_v2_2_milestone]] — every finding above still holds.

## 23. Model specialization experiment

One controlled experiment: unified XGBoost (§6) vs. three separate XGBoost models, each trained only on its own candidate type's train rows, evaluated against that type's test rows.

| Type | Specialized PR-AUC | Unified PR-AUC | Delta |
|---|---|---|---|
| Door | 0.557 | 0.567 | -0.0096 |
| Exit | 0.635 | 0.632 | +0.0032 |
| Stair | 0.504 | 0.538 | -0.0346 |

Mean delta: **-0.014 (specialization is a net negative)**. Stair — already the lowest-prevalence type — gets *worse* when starved of Door/Exit's larger training signal by specialization. **Verdict: prefer the unified model**, per the milestone charter's default.

## 24. Training-size study

| Fraction | Scenarios | Rows | Val PR-AUC |
|---|---|---|---|
| 10% | 175 | 130,904 | 0.559 |
| 25% | 438 | 330,583 | 0.609 |
| 50% | 875 | 616,700 | 0.628 |
| 75% | 1,312 | 903,998 | 0.636 |
| 100% | 1,750 | 1,210,568 | 0.639 |

Smooth, monotonic diminishing returns — the model benefits from more data throughout the range but the marginal gain from 75%→100% (+0.003) is much smaller than 10%→25% (+0.050). No sign of being data-starved at full scale, but no sign of having plateaued either — more scenarios would likely still help modestly.

## 25. Export and versioning

Artifacts written to `data/localized_predictive_model_v3/` (never overwriting V1/V2/V2.2 artifacts, which live in their own separate directories): `model.joblib`, `model_metadata.json` (V1-format), `model_metadata_v2.json` (extended V2 format), `calibrator.joblib`, `training_report_v3.json`. Required metadata fields, all present and verified:

- `model_version`: `"localized_predictive_model_v3"`
- `dataset_target_version`: `"v2-persistent-demand-service-imbalance"`
- `feature_schema_version`: `"2.1-experimental"`
- `dataset_campaign_version`: `"predictive_congestion_target_v2"`
- `prediction_horizon_seconds`: `20.0`
- `minimum_congestion_duration_seconds`: `3.0`
- `training_seed`: `20260726`
- `training_scenario_count`: `1750`
- `calibration_status`: `"calibrated (isotonic, fit on validation split only; ECE=0.00236, Brier=0.01639)"`
- `metrics_by_candidate_type`, `metrics_by_topology_family`: full breakdowns embedded
- `production_readiness`: `"NOT_READY"` (automated gate's own naive classification — see §26 for the human-reviewed verdict)
- **`live_inference_wired`: `false`** — explicitly patched into the written JSON, not merely absent

## 26. Production-readiness decision

The script's own automated `_assess_production_readiness()` returned **NOT_READY**, triggered specifically by the label-shuffle-test failure (§18) — its logic treats any shuffle-test failure as an automatic hard stop, regardless of explanation, since it cannot itself evaluate a qualitative investigation.

**Human-reviewed verdict: C — PROMISING BUT NEEDS MORE DATA.** This is a deliberate, reasoned override of the script's own more severe automated tag, for the same category of reason [[localized_predictive_model_v2_2_milestone]] already established as project practice (overriding an automated D to C for target-validity reasons) — the automated gate cannot evaluate a qualitative investigation, a human reviewer must.

Two independent findings support C over a NOT-READY/A verdict on one side, and rule out D/E on the other:
1. **Genuine, well-evidenced predictive skill**: 5.31x lift over the deterministic-current-state baseline (§7), PR-AUC surviving near-onset exclusion (§10), sensible feature-family ablation (§16), no leakage-correlated features (§19), calibration achievable to ECE ≤ 0.003 (§20) — this is not a degenerate or trivial model.
2. **Two real, unresolved limitations block D/E (shadow-mode readiness)**: the topology-holdout generalization gap (§8 — PR-AUC roughly halves for 2 of 4 topology families when unseen) and the label-shuffle-test anomaly (§18 — investigated and attributed to a methodology artifact, not leakage, but not a clean pass either). Per the milestone's explicit "be conservative" instruction, either finding alone would be enough to withhold D/E; both together make NOT_READY→C the only defensible verdict, not D or E.

## 27. What "shadow mode" means (not implemented this milestone)

Shadow mode would mean running the trained model's inference alongside live operation, generating predictions in real time from live `BuildingState`, **logging every prediction for later comparison against what actually happened — without ever surfacing those predictions to an operator or feeding them into any ranking/decision system.** It is the step between "offline research" and "actually influencing anything," used to validate that offline metrics (PR-AUC, calibration, lead-time distribution) hold up against real (not simulated) building telemetry before any operator-facing use is considered. This milestone's verdict (C) means shadow mode is **not yet warranted** — the topology-generalization gap and shuffle-test limitation should be resolved or better understood first. No shadow-mode code was written this milestone.

## 28. Testing

Baseline before this milestone: 4643 tests passing. Added 10 new tests (`tests/test_predictive_model_deterministic_baseline.py`: 7 tests covering `DeterministicCurrentStateBaseline`'s no-op `fit()`, monotonicity in congestion level and trend, output range/extremes, graceful handling of missing one-hot columns, and a leakage-boundary import guard; `tests/test_predictive_model_training_size_study.py`: 2 new tests for the additive `progress_fn` callback). **Full suite: 4653/4653 passing, zero regressions.**

## 29. Performance

| Measurement | Value |
|---|---|
| Single-candidate inference latency | 0.509ms (200-repeat average) |
| Batch throughput | 450,964 candidates/sec |
| Calibration overhead (isotonic, full test batch) | 10.5ms |
| Full training + evaluation wall time | 399.5s |

Inference cost is negligible relative to any plausible live-tick cadence; training/evaluation cost is measured and reported separately from inference, as required.

## 30. Known limitations (carried into exported metadata)

1. `candidate_recent_flow_rate` has a full live source only for Exit today; Door/Stair's `zone_transitions`-based mechanism is unit-tested but not yet validated against a real live deployment.
2. `multi_exit_wide` remains the hardest topology family to generalize to (§8, §14).
3. Target V2's `currently_congested` exclusion rate is high for Door (~45%), reducing usable onset-prediction examples for that type relative to its raw row count.
4. Multi-bottleneck buckets (1.3% of test rows) have 11-12x worse error rates than single-bottleneck rows (§12) — unresolved.
5. The label-shuffle-test methodology itself has a demonstrated blind spot for high-capacity, heavily-reweighted models on duplicate-heavy feature spaces (§18) — a documented limitation of the sanity check, not just of this model.

---

## Final report

### Numbered questions

1. **Does SynEvac's existing feature set + a real learned model reliably predict Target V2 congestion onset ~20s ahead?** Partially. Genuine, well-evidenced skill exists (PR-AUC 0.615 vs. 0.028 prevalence, 5.31x over the deterministic baseline) but "reliably" is too strong given the topology generalization gap (§8).
2. **Was the dataset/split/schema reused correctly?** Yes — confirmed in Phase 1 investigation before any training; no re-simulation was needed for the primary evaluation.
3. **How many rows were excluded as currently-congested, and is the exclusion rate reasonable?** 674,073 of 2,405,049 (28.0%) excluded; reasonable and expected given Target V2's onset-only semantics, though notably uneven by type (Door ~45%, see limitation #3).
4. **Is the scenario split leak-safe?** Yes, mechanically proven via `assert_no_scenario_overlap`.
5. **How was class imbalance handled?** `class_weight='balanced'` sample reweighting; no oversampling/SMOTE.
6. **What is the primary metric and value?** PR-AUC, 0.615 (test).
7. **What are the full secondary metrics?** ROC-AUC 0.967, precision 0.543, recall 0.573, F1 0.557, balanced accuracy 0.779, Brier 0.0688 (raw), ECE 0.1105 (raw) — §6, §20.
8. **How do the trivial baselines perform?** Majority-class/always-negative PR-AUC = prevalence exactly (0.0282); random PR-AUC ≈ prevalence, ROC-AUC ≈ 0.5 — all behave exactly as expected.
9. **How does Logistic Regression perform?** PR-AUC 0.490, ROC-AUC 0.933 — a strong linear baseline, beaten by XGBoost but not trivially.
10. **Which candidate model won and why was RandomForest excluded?** XGBoost; RF excluded and disclosed for cost reasons (§6), consistent with every prior milestone's finding that it's a near-tie at much higher cost.
11. **Was hyperparameter selection done without repeatedly peeking at test?** Yes — threshold tuning uses validation only (`tune_threshold` on val); test is touched only for final reporting.
12. **What are the per-candidate-type metrics?** Door PR-AUC 0.567, Exit 0.632, Stair 0.538 (§6, unified model, test split).
13. **What are the per-topology-family metrics under normal (non-holdout) evaluation?** Not separately computed outside the holdout context — topology-family breakdown is captured via the leave-one-out gate (§8), which is the more decision-relevant view.
14. **What does leave-one-topology-out show?** A real generalization gap for 2 of 4 families (§8) — the single most important finding gating the production-readiness verdict.
15. **What does lead-time-bucket analysis show?** Smooth recall decay from 0.70 (0-5s) to 0.44 (15-20s) — genuine distributed early warning (§9).
16. **Is 20s horizon still the right primary choice after the sensitivity check?** Yes — 10s is easier in raw terms but gives less warning; 30s's higher raw PR-AUC is largely a prevalence artifact, and relative lift actually favors shorter horizons (§11).
17. **How common are multi-bottleneck situations and how well does the model handle them?** 1.3% of test rows; handled far worse than single-bottleneck rows (11-12x higher error rates, §12) — a real, unresolved weakness.
18. **How does error rate vary with occupancy severity?** Rises sharply with occupancy (HIGH FPR 2.92% vs. LOW 0.19%, §13) — least reliable exactly when it matters most.
19. **How does single-exit vs. multi-exit topology compare?** Single-exit generalizes far better under holdout (92.8% vs. 49.5% of in-distribution PR-AUC, §14).
20. **What does permutation feature importance show?** `candidate_approaching_count` and `total_active_occupant_count` dominate; `candidate_queue_length` is surprisingly low (rank 8) due to the currently-congested exclusion pushing the model toward forward-looking demand signals (§15).
21. **What does feature-family ablation show?** `demand_signal` and `flow_and_trend` are load-bearing; `derived_congestion_level` and `alternative_route_structure` contribute near-nothing incrementally (§16).
22. **Is the model just detecting pre-congestion milliseconds before onset (tautology)?** No — PR-AUC survives at 91% of full value even excluding <2s-lead examples, and queue_length (the most "obviously already congested" signal) is not the dominant feature (§17).
23. **Did the label-shuffle sanity test pass?** No — ROC-AUC 0.378, outside the 0.05-tolerance chance band every prior milestone cleared. Investigated in depth (§18); attributed to a model-capacity/reweighting/feature-duplication interaction, not leakage, but treated as an unresolved limitation, not a waived failure.
24. **Does any feature correlate suspiciously with the target?** No — nothing reaches the 0.9 review threshold (§19); highest is -0.433 (STABLE trend, in the expected direction).
25. **Is the model calibrated?** Poorly before calibration (ECE 0.11); well after isotonic (ECE 0.0024 overall, ≤0.003 for every candidate type) (§20).
26. **What threshold should be used operationally?** None is recommended — a full sweep is reported (§21) for a future decision-maker, per the charter's explicit instruction not to choose a policy threshold here.
27. **Is every final feature's live-parity status re-verified from current code?** Yes (§22) — `live_extractor_v2_1.py` was re-read this milestone, not assumed from prior documentation; findings are unchanged from V2.2 but independently re-confirmed.
28. **Does ML actually beat SynEvac's existing deterministic intelligence?** Yes, decisively — 5.31x relative PR-AUC lift over `DeterministicCurrentStateBaseline` (§7), the single most important comparison in this milestone.
29. **What are the model's most important failure modes?** (1) Multi-bottleneck situations, 11-12x worse error rates; (2) high-occupancy conditions, ~15x worse FPR than low-occupancy; (3) `multi_exit_wide` and `twin_stair_highrise` topology families under holdout, ~50% PR-AUC retention; (4) early-evacuation-phase noise, FPR 4.80% vs. 0.02% late-phase (§12, §13, §8).
30. **Should the model be unified or type-specialized?** Unified — specialization is a net PR-AUC negative (mean delta -0.014, Stair actively harmed) (§23).
31. **Does the model need more training data?** Diminishing but non-zero returns at full scale (§24) — more scenarios would likely help modestly, consistent with the C verdict's "needs more data."
32. **Is this model ready for live shadow mode?** No (§26, §27) — verdict C, not D/E. Two independent, unresolved findings (topology generalization gap, shuffle-test anomaly) both need to clear before shadow mode is warranted.

### Decision questions A-L

- **A. Is the model NOT USEFUL?** No — genuine, well-evidenced skill exists (§7, §17, §19).
- **B. Is this RESEARCH ONLY with no near-term path?** No — the path forward (more scenarios, targeted topology-family augmentation, a cleaner shuffle-test methodology) is concrete and identified, not open-ended.
- **C. Is this PROMISING BUT NEEDS MORE DATA?** **Yes — this is the verdict** (§26).
- **D. Is this READY FOR LIVE SHADOW MODE?** No.
- **E. Is this READY FOR RANKING-INTEGRATION RESEARCH?** No.
- **F. Was Recommendation/Guidance/Decision Policy modified this milestone?** No, and it will not be, per the hard constraint, even though the verdict is not a hard "NOT_USEFUL."
- **G. Was RL used anywhere?** No.
- **H. Were simulator or capacity semantics changed?** No — Target V2 and the simulator were both frozen and untouched.
- **I. Were V1/V2/V2.2 artifacts overwritten?** No — V3 artifacts live in their own directory (`data/localized_predictive_model_v3/`).
- **J. Was Stair silently dropped for low prevalence?** No — Stair is fully represented in every metric, ablation, and holdout result throughout this report (§6, §8, §12, §16).
- **K. Was the model optimized solely for aggregate PR-AUC?** No — per-type, per-topology, per-occupancy, per-lead-time, and multi-bottleneck breakdowns were all required and reported (§6, §8, §9, §12, §13).
- **L. Is live inference wired anywhere?** No — `live_inference_wired = false` is explicitly present in every exported artifact (§25), and nothing in this milestone touches `LiveRuntime`, Recommendation, Guidance, or Decision Policy.
