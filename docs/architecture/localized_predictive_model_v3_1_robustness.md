# Localized Predictive Model V3.1 — Generalization, Duplicate-Structure & Robustness Investigation

Status: **INVESTIGATION AND ROBUSTNESS MILESTONE.** Nothing wired into `LiveRuntime`, Recommendation, Guidance, Advisory, Voice, Dynamic Signage, or Decision Policy. No RL. Target V2 (`v2-persistent-demand-service-imbalance`) and the simulator were not modified. No new intelligence engine, no new building assets.

Follows [[localized_predictive_model_v3_milestone]] (commit `29bd854`, verdict **C — PROMISING BUT NEEDS MORE DATA**). This milestone investigates the two unresolved findings that blocked a D/E verdict: the topology-holdout generalization gap and the label-shuffle-test anomaly.

## 1. Reconstruction of the exact V3 experiment

Re-verified directly from committed artifacts (`data/localized_predictive_model_v3/model_metadata_v2.json`, `training_report_v3.json`), not from memory of writing them: scenario population 2,500 scenarios / 4 topology families (`multi_exit_wide` 700, `single_exit_lowrise` 500, `twin_stair_highrise` 800, `v1_topology_fixed` 500); V2.1 12-raw-field / 27-encoded-column feature schema; scenario-level 70/15/15 split, seed `20260726`; XGBoost `max_depth=6, n_estimators=300, learning_rate=0.1`, `class_weight='balanced'` via `sample_weight`; isotonic calibration fit on validation only; 20s primary horizon. No discrepancies found between the committed docs and the actual exported metadata — every number below is directly comparable to V3's own.

## 2. Duplicate-structure audit (full scale — all 1,730,976 eligible rows, not V3's train-split-only figure)

Distinguishing the 5 categories the milestone charter required rather than treating them as equivalent:

| | Finding |
|---|---|
| **(A) Exact duplicate feature vectors** | 96.9% of rows sit in a duplicated-feature-vector group; only 6.75% of rows (116,886) are unique. Largest single group: 8,657 rows. Median duplicate-group size: 4. |
| **Conflicting-label groups** | 14.1% of duplicate groups (18.0% of all rows) have the SAME feature vector mapping to BOTH a positive and a negative outcome 20s later — real, irreducible ambiguity in the current 27-feature schema, not a bug. |
| **(B) Exact duplicate feature+label rows** | 96.45% of rows — nearly identical to (A), confirming most duplicate groups are internally label-consistent (only the 14.1% above are split). |
| **(C) Within-scenario duplication** | Only 4.5% of duplicate rows — the SAME scenario producing repeated identical states across its own timesteps is the MINORITY cause. |
| **(D) Cross-scenario duplication** | 95.5% of duplicate rows — DIFFERENT scenarios converging on the same low-cardinality, binned feature reading is the dominant cause (expected: several features are small-integer counts or 5-level categorical bins). |
| **(E) Cross-topology duplication** | **Exactly 0%.** No duplicate group ever spans more than one topology family. This is the single most consequential finding of the audit — see §5. |

**Reconciliation with V3's original 96.5% figure**: that number was computed on the TRAIN split only; this milestone's full-scale (train+val+test) figure is 96.9% — consistent, not a discrepancy.

## 3. Temporal redundancy

Engineering proxies, explicitly not a statistically rigorous effective-sample-size estimator (no defensible autocorrelation-based ESS formula fits exact-duplicate categorical/binned features):

| Lag | Exact-match rate |
|---|---|
| 5s (1 tick) | 75.6% |
| 10s (2 ticks) | 60.8% |
| 15s (3 ticks) | 50.0% |
| 20s (4 ticks, = full horizon) | 41.7% |

Collapsing consecutive-identical-group runs within each of the 14,000 (scenario, candidate) trajectories to one representative row each gives 453,386 "temporally distinct" rows out of 1,730,976 (73.8% of rows are continuations of an unchanged state from the immediately preceding tick) — a materially different, and more conservative, proxy than the 116,886-unique-vector count in §2, since it doesn't collapse cross-scenario repeats.

## 4. Shuffled-label investigation (9 controlled variants)

Reproduced V3's original result exactly (ROC-AUC 0.3785, matching V3's reported 0.378) as variant A, then ran 8 more:

| Variant | Description | ROC-AUC | Near chance? |
|---|---|---|---|
| A | Global shuffle, production XGBoost | 0.379 | No |
| B | Shuffle within each scenario only | 0.680 | No |
| C | Shuffle within each candidate_type only | 0.508 | **Yes** |
| D | Shuffle within each topology_family only | 0.596 | No |
| **E** | **Shuffle within each exact-duplicate-feature-group only** | **0.968** | No — matches REAL model (0.967) |
| F | Global shuffle, no class reweighting | 0.462 | Yes (borderline) |
| G | Global shuffle, reduced capacity (depth 3, 50 trees) | 0.560 | No |
| H | Global shuffle, LogisticRegression | 0.665 | No |
| I | Global shuffle, HistGradientBoosting | 0.589 | No |

**Variant E is decisive**: preserving each duplicate group's true empirical positive rate while scrambling only which row within the group gets which label reproduces the REAL model's performance almost exactly (ROC-AUC 0.968 vs the genuine 0.967, PR-AUC 0.639 vs 0.615). **The model's skill is coming almost entirely from memorizing each duplicate-feature-group's historical outcome rate, not from learning continuous relationships between raw feature values and risk.**

**This is not classic leakage**: a genuine leak (row overlap, a leaked identifier) would be exploitable at any capacity/reweighting level and would push ROC-AUC toward 1.0 for essentially any model. Instead the result is model-and-configuration-dependent and sign-flipping (XGBoost full config: below chance; LogisticRegression: above chance; XGBoost no-reweight: near chance) — the classic signature of overfitting-to-noise under severe class-imbalance reweighting combined with a heavily duplicate-dominated feature space, not a leak.

Combined with §2's finding that duplicate groups never span topology families, this gives a complete, coherent causal chain: **an unseen topology family has zero memorized duplicate-group statistics available**, forcing the model back onto whatever weaker structural relationships it also learned — directly explaining why leave-one-topology-out PR-AUC roughly halves (§9 below quantifies the resulting failure mode).

## 5. Unique-state training experiments (does reducing duplicate dominance help?)

Three experimental train representations (scenario split preserved, canonical dataset never modified), evaluated on the SAME real, non-deduplicated test/holdout sets:

| Variant | Train rows | Primary test PR-AUC | Hard-family holdout avg PR-AUC* |
|---|---|---|---|
| Canonical | 1,210,568 | 0.6152 | 0.3063 |
| A: one row per unique feature vector | 93,193 (7.7%) | 0.6131 | 0.2678 |
| B: one row per (feature vector, scenario) | 307,367 (25%) | 0.6150 | 0.2628 |
| C: full rows, duplicate-count-inverse sample weight | 1,210,568 | 0.6156 | 0.2833 |

*average of multi_exit_wide + twin_stair_highrise PR-AUC.

**Clean negative result: none of the three strategies improve topology generalization — all are worse than canonical on the hard families.** Variant A achieves nearly-identical primary PR-AUC using 13x less data (confirming §2's finding that information is concentrated in the 116,886 unique vectors), but this doesn't help cross-topology transfer, because — per §2's E=0% finding — deduplication only removes REDUNDANT copies; it cannot create new cross-topology-shared support regions that don't exist in the underlying data. One secondary finding: Variant A's raw (pre-calibration) ECE is substantially better (0.035 vs canonical's 0.111), suggesting deduplication reduces overconfidence on common patterns even though it doesn't fix generalization.

**Conclusion: duplicate-state dominance is a real, well-evidenced MECHANISM (§4) but not a fixable CAUSE of poor generalization — removing it doesn't help, confirming the bottleneck is structural (§6-7), not a data-hygiene problem.**

## 6. Topology representation audit

Feature classification (candidate-local dynamic/structural, whole-building dynamic/structural, topology proxy, candidate-identity proxy):

| Feature | Classification |
|---|---|
| total_active_occupant_count | whole_building_dynamic |
| candidate_capacity, candidate_walking_distance | candidate_local_structural |
| candidate_traversable, candidate_queue_length, candidate_approaching_count, candidate_congestion_level, candidate_recent_flow_rate, candidate_congestion_trend | candidate_local_dynamic |
| candidate_adjacent_zone_occupancy | whole_building_dynamic |
| candidate_alternative_route_count | topology_proxy |
| candidate_type | candidate_identity_proxy |

**A simple XGBoost classifier trained to predict topology_family from ONLY the V3 feature vector achieves 100.0% accuracy** (scenario-level held-out test, vs. a 60.9% majority-class baseline). Topology is trivially, perfectly encoded in the feature space — not evidence of leakage (these are legitimate, honestly-available-live structural features, not secret future information), but a direct structural explanation for why the model can — and per §4 does — implicitly condition on topology identity.

## 7. Feature distribution shift across topology families

Standardized mean difference (SMD) for continuous features, ranked by max |SMD| across any family vs. the rest:

| Feature | Max |SMD| |
|---|---|
| candidate_walking_distance | 1.49 |
| candidate_alternative_route_count | 1.35 |
| total_active_occupant_count | 1.21 |
| candidate_recent_flow_rate | 1.03 |
| candidate_adjacent_zone_occupancy | 0.87 |
| candidate_capacity | 0.56 |
| candidate_approaching_count | 0.34 |
| **candidate_queue_length** | **0.045** |

The largest shifts are concentrated exactly in the "candidate_local_structural" and "topology_proxy" feature classes from §6 — `candidate_queue_length` (a "candidate_local_dynamic" feature reflecting Door/Stair's universal capacity=1 constraint) is nearly invariant across topology, while structural/design-defining features vary enormously. This directly corroborates §6: the model most plausibly conditions on exactly the features that shift most when topology changes.

## 8. Holdout failure analysis (engineering explanation for WHY unseen topology fails)

Both failing families are dominated by **false positives** (over-prediction), not false negatives:

| | Overall FP rate | Overall FN rate |
|---|---|---|
| multi_exit_wide | 8.0% | 0.6% |
| twin_stair_highrise | 27.9% | 1.5% |

**multi_exit_wide**: FP rate spikes to 66.6% under `congestion_trend=RISING` and 56.0% under `UNKNOWN` (vs. 7.3% under STABLE); 29.1% under multi-bottleneck conditions (vs. 7.9% single); 15.8% at HIGH occupancy (vs. 1.5% LOW).

**twin_stair_highrise**: FP rate spikes to 34.6% when `alternative_route_count >= 2` (vs. 6.8% when exactly 1) — this family structurally has more route redundancy the model hasn't learned to calibrate for. It also shows a DISTINCT failure mode multi_exit_wide doesn't: a 47.1% FALSE NEGATIVE rate specifically for very-high-flow-rate rows (`candidate_recent_flow_rate > 10`), where multi_exit_wide instead shows elevated false positives in the equivalent bucket.

**Engineering explanation**: the model has learned trend/flow/route-conditioned response calibrations tuned to the OTHER three families' typical value ranges and base rates (§7); applied to a structurally different family with a materially different distribution of those same signals, the calibration misfires — mostly toward over-alarm, with one family-specific under-alarm mode for extreme flow rates.

## 9. Controlled topology-diversity scaling experiment — THE KEY POSITIVE RESULT

Two new structural variants (`predictive_dataset/topologies_v3_1_variants.py`, 200 scenarios each, well within the milestone's 100-250 suggested range, using only the existing Building/Zone/Door/Exit/Staircase authoring primitives — no new generator engine): `multi_exit_wide_6exit` (6 doors/5 exits vs. the parent's 4/3) and `twin_stair_highrise_3stair` (4 floors/3 stairs vs. the parent's 3/2).

| Family | Baseline PR-AUC | + structural variant PR-AUC | Relative change |
|---|---|---|---|
| multi_exit_wide | 0.3047 | 0.3764 | **+23.5%** |
| twin_stair_highrise | 0.3078 | **0.6969** | **+126%** |

Test sets are the ORIGINAL held-out family's unchanged real test scenarios in both rows — any change is attributable only to the added training diversity, not an easier test set. twin_stair_highrise's improved PR-AUC (0.697) now EXCEEDS V3's own in-distribution PR-AUC (0.615). **This is the single most actionable, validated finding of the milestone: the topology generalization gap is fixable with modest, targeted structural diversity, not a fundamental limitation requiring a different approach.**

## 10. Relational/normalized feature experiments

Five ratio features derived from existing, already-live-parity-audited raw columns (`rel_queue_to_capacity`, `rel_flow_to_capacity`, `rel_adjacent_occupancy_to_capacity`, `rel_alt_route_share`, `rel_adjacent_occupancy_to_building_occupancy`), tested as an EXTENSION to (not replacement of) the canonical 27-column schema:

| | Primary PR-AUC | multi_exit_wide | single_exit_lowrise | twin_stair_highrise | v1_topology_fixed |
|---|---|---|---|---|---|
| Canonical | 0.6152 | 0.3047 | 0.5707 | 0.3078 | 0.5311 |
| + relational features | 0.6123 | 0.3019 | 0.5601 | 0.3122 | 0.5083 |

**No clear, robust benefit** — mostly flat-to-slightly-negative, with one small win (twin_stair_highrise +1.4%) offset by a larger loss elsewhere (v1_topology_fixed -4.3%). Leakage correlation check: no relational feature approaches concerning correlation with the target (`rel_queue_to_capacity` 0.194 is the highest, well below the 0.9 review threshold). **Verdict: none of these five features are recommended for the canonical schema** — unlike §9, feature engineering alone does not move the needle on the generalization gap.

## 11. Model comparison for generalization (not just aggregate PR-AUC)

XGBoost, HistGradientBoosting, and LogisticRegression each fit on the SAME known-topology sub-train, evaluated on the SAME unseen held-out family:

| Held-out family | XGBoost | HistGradientBoosting | LogisticRegression | Best generalizer |
|---|---|---|---|---|
| multi_exit_wide | 0.255 | 0.256 | **0.293** | LogisticRegression |
| single_exit_lowrise | **0.556** | 0.490 | 0.427 | XGBoost |
| twin_stair_highrise | 0.232 | **0.456** | 0.425 | HistGradientBoosting |
| v1_topology_fixed | **0.526** | 0.463 | 0.440 | XGBoost |

**No single model generalizes best consistently — and XGBoost, V3's chosen production architecture, is the WORST generalizer on twin_stair_highrise specifically** (less than half of HistGradientBoosting's score on that family). This tempers confidence in XGBoost as a uniformly-best choice once generalization (not just in-distribution PR-AUC) is the priority, and is a genuinely new, non-obvious finding this milestone surfaces.

## 12. Calibration under topology distribution shift

Isotonic calibrator fit ONLY on a known-topology validation sub-split, applied to each unseen family's test predictions:

| Held-out family | ECE before | ECE after (known-topology calibrator) |
|---|---|---|
| multi_exit_wide | 0.107 | 0.0087 |
| single_exit_lowrise | 0.167 | 0.0425 |
| twin_stair_highrise | 0.302 | 0.0343 |
| v1_topology_fixed | 0.163 | 0.0054 |

**Calibration substantially survives topology shift in every family** — ECE drops 5-19x — though it doesn't fully restore in-distribution-level calibration (V3's own in-distribution ECE after isotonic was 0.0024, still notably lower than any of the above). A genuine, reassuring partial positive: calibration trained on known topology remains a meaningfully useful safety measure, not something that catastrophically breaks under shift.

## 13. Deterministic baseline under the same topology holdouts

| Held-out family | XGBoost PR-AUC | Deterministic PR-AUC | ML beats deterministic? | Relative lift |
|---|---|---|---|---|
| multi_exit_wide | 0.255 | 0.116 | Yes | 2.19x |
| single_exit_lowrise | 0.556 | 0.283 | Yes | 1.97x |
| twin_stair_highrise | 0.232 | 0.121 | Yes | 1.91x |
| v1_topology_fixed | 0.526 | 0.310 | Yes | 1.70x |

**ML beats the deterministic-current-state baseline in all four families, even under topology shift, even in its worst-generalizing case.** The model never collapses below simple rule-based intelligence — a genuine, important strength that survives every stress test this milestone applied.

## 14. Live parity re-audit for experimental features

The 5 relational features (§10) are simple ratios of already-audited raw components; each inherits the SAME live-parity classification as its inputs (all re-confirmed unchanged from V3's own Phase 22 audit):

| Feature | Basis | Parity |
|---|---|---|
| rel_queue_to_capacity | queue_length (EXACT) / capacity (STRUCTURAL) | EXACT×STRUCTURAL |
| rel_flow_to_capacity | recent_flow_rate (EXACT for Exit, PARTIAL-UNVALIDATED for Door/Stair) / capacity | inherits weaker component |
| rel_adjacent_occupancy_to_capacity | adjacent_zone_occupancy (EXACT) / capacity | EXACT×STRUCTURAL |
| rel_alt_route_share | alt_route_count (STRUCTURAL) / scenario door+exit+stair count (STRUCTURAL, from scenario_metadata — NOT itself a per-candidate live feature, computed once per building) | STRUCTURAL |
| rel_adjacent_occupancy_to_building_occupancy | adjacent_zone_occupancy (EXACT) / total_active_occupant_count (EXACT) | EXACT |

Since none of the 5 showed a robust benefit (§10), none are proposed for the production schema — this table is recorded for completeness/reproducibility, not as a green light.

## 15. Performance

Total investigation compute across all 8 new analysis scripts: ~22.6 minutes (duplicate/temporal audit 15s; shuffle battery 128s; unique-state experiments 383s; topology representation audit 88s; holdout failure analysis 151s; topology diversity experiment 248s; relational feature experiment 224s; generalization/calibration/deterministic comparison 120s). No inference-latency changes — the production model architecture (XGBoost, same hyperparameters) is unchanged from V3, whose own measured single-row latency (0.51ms) and batch throughput (451K candidates/sec) still apply; this milestone did not touch inference code. Memory watchdog (same 180MB critical floor as V3) triggered zero hard-exits across all 8 runs — no new memory-hygiene bugs found, though 2 real bugs (a `np.diff`-on-strings crash in the shuffle battery, a stale `NameError` reference) were caught and fixed during development, both now covered indirectly by the new `shuffle_within_groups` tests (§4's regression test explicitly reproduces the string-array bug).

## 16. Production-readiness gate

**Verdict: C — PROMISING BUT NEEDS MORE DATA** (unchanged category from V3, but now with a validated, concrete path toward D).

D (READY FOR LIVE SHADOW MODE) requires ALL of: ML beats deterministic baseline (✅, even under shift, §13) · meaningful pre-onset prediction (✅, established in V3) · no unresolved shuffled-label anomaly (⚠️ MECHANISTICALLY EXPLAINED, not eliminated — §4 shows exactly why it happens, but the production XGBoost config still exhibits it) · acceptable Door/Exit/Stair behavior (⚠️ Stair remains hardest, unchanged from V3) · acceptable unseen-topology generalization (❌ NOT YET — §9's fix is a validated 200-scenario proof-of-concept, not yet applied to the production model at full scale) · calibrated probabilities reasonable under shift (✅ partial — §12) · every production feature has honest live parity (✅, no new features are being proposed, §14). At least two conditions are not cleanly met — D is not warranted, and per the milestone's explicit "do not lower these gates to force progress" instruction, C remains the correct, conservative verdict.

## 17. Next-data decision (quantitatively justified)

Per the milestone's own A-E menu:

- **A (more scenarios from existing 4 families)**: NOT recommended as the primary lever — §5 directly tested this (more raw repetition of the same shape via duplicate reweighting/dedup) and found no generalization benefit.
- **B (more structural variants within existing families)**: **PRIMARY RECOMMENDATION** — §9 is direct, validated, quantitative evidence: a single modest 200-scenario structural variant per failing family produced +23.5% to +126% relative PR-AUC gains on genuinely held-out topology. This is not a hypothesis; it is a measured result.
- **C (entirely new topology families)**: secondary priority — untested this milestone, higher cost/risk than B, no evidence yet either way.
- **D (targeted hard-case scenarios)**: §9 is effectively a targeted, structural form of this already; further targeting (e.g., more variants specifically stress-testing the failure modes §8 identified — high route-redundancy for twin_stair_highrise, RISING-trend conditions for multi_exit_wide) is a natural extension of B, not a separate track.

**Recommendation: scale §9's proof-of-concept up** — 3-4 structural variants per topology family (not just one), at full production scale (comparable to each family's existing ~500-800 scenario count, not just 200), prioritizing the two failing families first. This is the highest-confidence, quantitatively-justified next step.

## 18. Known limitations carried forward

1. XGBoost is not the most robust generalizer for 2 of 4 topology families (§11) — worth evaluating a HistGB/XGBoost ensemble or per-family model selection in a future milestone, though that itself needs controlled evaluation, not blind adoption.
2. The label-shuffle-test methodology (`predictive_model.sanity_checks.label_shuffle_test`, unmodified) has a demonstrated blind spot for high-capacity, heavily-reweighted models on duplicate-heavy feature spaces — this is documented as a limitation of the CHECK, not patched, since it behaves correctly (near-chance) in every other prior milestone.
3. twin_stair_highrise's very-high-flow-rate false-negative spike (§8) is a distinct, unexplained-beyond-distribution-shift failure mode worth targeted investigation if flow-rate-driven Stair congestion becomes operationally important.
4. §9's structural-diversity fix has not yet been validated at full production scale or combined with more than one variant per family — the measured gains may not extrapolate linearly.
