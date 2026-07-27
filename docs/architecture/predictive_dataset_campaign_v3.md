# Predictive Dataset Campaign V3 — Structural Topology Diversity

Status: **data-generation and validation milestone. No Model V4 trained/exported, no scoring code changed, no live runtime touched.** Builds on the Localized Predictive Model V3.1 Robustness Investigation (commit `a625691`, 4674/4674 tests passing), which found that Dataset V2's four topology families are each a *single fixed graph* — the model's poor topology-holdout generalization (multi_exit_wide PR-AUC 0.31 held out) was driven by feature-space duplication reflecting only four recognizable templates, and a small controlled experiment (two hand-built structural variants) improved unseen-topology PR-AUC by +23.5% and +126%. This milestone generalizes that one-off experiment into a versioned, 16-template structural-diversity architecture spanning all four families, generates a full campaign against it, and runs an exploratory (not production) generalization proxy experiment to test whether it actually helps.

## 1. Topology generator audit (Phase 1)

`predictive_dataset/topologies_v2.py`'s four `TopologySpec` builders each construct exactly **one** fixed `Building` graph; only the `ScenarioDefinition` (occupancy/fire/door-state distributions) varies across the family's scenarios. `predictive_dataset/topologies_v3_1_variants.py` added two more fixed graphs (`multi_exit_wide_6exit`, `twin_stair_highrise_3stair`) as a controlled experiment, never wired into a campaign. Neither module was modified — every existing builder is imported and reused verbatim as one variant of its family.

Structural analysis modules (`topology_analysis_v2.py`, `diversity.py`) only counted `floor_count`/`exit_count`/`stair_count`/`door_count` off scenario-metadata dicts the campaign script itself populated — no `zone_count`, `candidate_count`, walking-distance aggregate, alternative-route count, or "structural signature" concept existed anywhere. The one existing route-redundancy-adjacent metric, `predictive_dataset.simulation_extractor_v2_1.build_alternative_route_counts`, is a zone-overlap count (how many other candidates share a zone with this one) — reused verbatim for Phase 3's signature rather than inventing a second definition.

## 2. Structural variant architecture (Phase 2) — `predictive_dataset/topologies_v3.py`

4 families × 4 structural variants = **16 structural templates**. Each family's `base` variant reuses its existing V2/V3.1 builder unmodified; the remaining variants are new, genuinely different graphs (never renamed ids or shifted coordinates):

| Family | Variant | Floors | Zones | Doors | Exits | Stairs | Candidates | What's different |
|---|---|---|---|---|---|---|---|---|
| single_exit_lowrise | `single_exit_lowrise` (base) | 1 | 2 | 1 | 1 | 0 | 2 | — |
| | `single_exit_deep_corridor` | 1 | 4 | 3 | 1 | 0 | 4 | Long serial corridor, 3 doors in series |
| | `single_exit_branching_deadends` | 1 | 4 | 3 | 1 | 0 | 4 | Branching (not linear) connectivity |
| | `single_exit_vertical` | 2 | 3 | 1 | 1 | 1 | 3 | Adds a floor/stair the base never has |
| twin_stair_highrise | `twin_stair_highrise` (base) | 3 | 6 | 3 | 2 | 2 | 7 | — |
| | `twin_stair_highrise_3stair` | 4 | 8 | 4 | 2 | 3 | 9 | 4th floor, 3rd independent stair |
| | `twin_stair_low` | 2 | 4 | 2 | 2 | 1 | 5 | Smaller/shorter version of the pattern |
| | `twin_stair_chained_core` | 3 | 6 | 3 | 2 | 2 | 7 | SERIAL stairs (floor3→floor2→ground), not parallel |
| multi_exit_wide | `multi_exit_wide` (base) | 1 | 5 | 4 | 3 | 0 | 7 | — |
| | `multi_exit_wide_6exit` | 1 | 7 | 6 | 5 | 0 | 11 | 2 more spokes, each with door+exit |
| | `multi_exit_linear_chain` | 1 | 4 | 3 | 2 | 0 | 5 | LINEAR (not hub/star) connectivity |
| | `multi_exit_reduced_redundancy` | 1 | 5 | 4 | 2 | 0 | 6 | Same shape, 2 of 4 spokes have no exit |
| v1_topology_fixed | `v1_topology_fixed` (base) | 2 | 4 | 2 | 2 | 1 | 5 | — |
| | `v1_fixed_dual_stair` | 2 | 4 | 2 | 2 | 2 | 6 | 2 stairs converge on same upper zone |
| | `v1_fixed_three_floor` | 3 | 4 | 1 | 1 | 2 | 4 | Chained vertical (ground→mid→upper) |
| | `v1_fixed_long_corridor` | 2 | 6 | 4 | 2 | 1 | 7 | 5-zone/4-door serial ground floor |

Every stair sets **both** `from_floor_id` and `to_floor_id` (the V1 bug's fix, inherited discipline) — mechanically confirmed for all 16 variants, including the new chained/multi-stair connectivity patterns (`tests/test_predictive_dataset_topologies_v3.py`). `with_scenario_count()` (a `dataclasses.replace` helper) lets the pilot/full-scale runners independently control campaign size without baking a fixed count into each builder.

## 3. Structural signature (Phase 3) — `predictive_dataset/topology_signature.py`

`StructuralTopologySignature`: `floor_count`, `zone_count`, `door_count`, `exit_count`, `stair_count`, `candidate_count`, `graph_node_count`, `graph_edge_count`, `mean/max_candidate_walking_distance`, `mean/max_alternative_route_count`. Pure dataset metadata — **not** added to `predictive_dataset.schema.CANDIDATE_FEATURE_SCHEMA`. `structural_key()` rounds the graph-shape fields (excluding family/variant_id identity) for duplicate detection.

## 4. Diversity verification (Phase 4) — `predictive_dataset/topology_diversity_v3.py`

`structural_diversity_report()` over all 16 variants: **16 requested, 16 distinct structural signatures, 0 duplicate groups, `all_genuinely_distinct=True`.** Family distribution 4/4/4/4. Floor-count distribution `{1: 7, 2: 5, 3: 3, 4: 1}`; exit-count `{1: 5, 2: 9, 3: 1, 5: 1}`; stair-count `{0: 7, 1: 4, 2: 4, 3: 1}`; candidate-count spans 2–11. Route redundancy (mean alternative-route count) spans 1.0–3.64 across variants. No generator needed fixing — verified distinct on the first pass. A dedicated test (`test_predictive_dataset_topology_diversity_v3.py::test_detects_a_deliberately_injected_duplicate`) proves the detector actually catches a renamed-clone failure mode, not just that the real 16 happen to pass.

## 5-7. Target V2 preserved, feature schema frozen, live parity unchanged (Phases 5-7)

`predictive_dataset/target_generator_v2.py` (`v2-persistent-demand-service-imbalance`, ≥3.0s persistence) is **untouched** by this milestone. `tests/test_predictive_dataset_target_v2_architecture_guards.py`'s 9 leakage-boundary tests pass unmodified against the new topologies, mechanically reconfirming: `target_generator_v2` is never imported by `simulation_extractor`/`live_extractor` (either version); `target_semantics_analysis` is never imported by any extractor; Target V1 remains byte-for-byte reproducible. `predictive_dataset/campaign_runner_v3.py` (the new campaign loop) imports `target_generator_v2` directly — exactly the same "designated caller" role `scripts/run_predictive_congestion_target_v2_relabel.py` already had, not a new leakage surface.

Feature schema: **unchanged** — the same V2.2/V3 experimental 12-field schema (9 base `CANDIDATE_FEATURE_SCHEMA` fields + 3 V2.1 experimental fields: `candidate_recent_flow_rate`, `candidate_congestion_trend`, `candidate_alternative_route_count`). None of V3.1's unsuccessful relational features were added back. `predictive_dataset/schema.py` and `predictive_dataset/simulation_extractor_v2_1.py` were not modified, so every field's documented live-parity source (`AIFeatureField.source`) is unchanged and still accurate — no new parity gap introduced.

## 8. Pilot campaign (Phase 8)

`scripts/run_predictive_dataset_campaign_v3_pilot.py`: 25 scenarios × 16 variants = 400 scenarios, `master_seed=20260727`.

**Result: 400/400 accepted, 0 failed, 443,622 rows, 46.5s (9,544 rows/sec).** Every variant contributed rows. Zero zero-walking-distance candidates. Target V2 positive rate 2.08% overall, present and non-trivial for every candidate type (Door 1.5%, Exit 2.8%, Stair 1.1%) — consistent with Target V2's established real-congestion range. `topology_variant_identifiability_report.json` (Phase 10, run on pilot): family-classifier accuracy 98.9% (macro-F1 0.972) vs 62.3% majority baseline; 16-way variant-classifier accuracy 78.0% (macro-F1 0.752) vs 18.6% majority baseline. Pilot quality gate: **PASS** — proceeded to full scale.

## 9. Hard-case coverage targets (Phase 9) — `predictive_dataset/campaign_config_v3.py`

Ten measurable targets defined **before** the full campaign ran (never tuned after seeing results): the six V2-style targets scaled to V3's per-variant counts, plus three new structural-specific targets V2 never needed (`multi_stair_scenarios`, `chained_stair_connectivity_scenarios`, `reduced_redundancy_exit_scenarios`). See §13 for the full-campaign pass/fail table.

## 10. Identifiability, full scale (Phase 10)

`scripts/model_v3_topology_variant_identifiability_audit.py` on the full 2,662,830-row dataset (scenario-level shuffled 80/20 split, XGBoost, same pattern as `model_v3_1_topology_representation_audit.py`'s Phase 7):

| Classifier | Classes | Accuracy | Macro-F1 | Majority baseline |
|---|---|---|---|---|
| Topology family | 4 | 99.10% | 0.976 | 61.05% |
| Structural variant | 16 | **82.74%** | 0.796 | 24.89% |

Family identity remains near-perfectly recoverable from features (expected — floor/door/exit/stair-count-correlated feature ranges are a real structural signal, not noise). The finer-grained 16-way structural-variant classification is meaningfully harder (82.7% vs. family's 99.1%) while still well above chance — quantifying, not eliminating, identifiability, exactly as Phase 10 requires.

## 11-12. Full-scale decision and memory-safe pipeline (Phases 11-12)

Pilot throughput (9,544 rows/sec, 46.5s for 400 scenarios) extrapolated to **150 scenarios/variant × 16 variants = 2,400 scenarios**, chosen for balanced structural coverage against every `COVERAGE_TARGETS_V3` minimum (not row-count maximization) — deliberately smaller than Dataset V2's 2,500 scenarios despite spanning 4× the structural templates.

`predictive_dataset/campaign_runner_v3.py` streams rows directly to CSV inside the tick/candidate loop (never accumulates rows in memory) — the same discipline `run_predictive_dataset_campaign_v2_2_fullscale.py` and the V2-relabel script established, reused rather than reinvented. A `psutil`-based `_check_memory()` guard aborts loudly below 300MB available, checked every 50-100 scenarios. Post-campaign analysis (`predictive_dataset_v3_post_campaign_analysis.py`) is chunked throughout (`pandas.read_csv(..., chunksize=250_000)`), using running sum/sumsq accumulators for distribution stats and a running hash→count dict for duplication analysis — never a single full-dataset DataFrame or list-of-dicts.

## 13. Full campaign results (Phase 13)

`scripts/run_predictive_dataset_campaign_v3_fullscale.py`, `master_seed=20260727`.

**Scenario campaign: 2,400 requested, 2,400 accepted, 0 failed.** Wall time 278.8s (9,551 rows/sec). Memory: 84.3% used / available dropped to 1,235MB at the tightest point — never crossed the 300MB abort floor.

**Row counts**: 2,662,830 total candidate-time rows. Door 1,461,802; Exit 962,539; Stair 238,489.

**Coverage verification — all 10 targets pass:**

| Target | Minimum | Actual | Passed |
|---|---|---|---|
| Every structural variant represented | 16 | 16 | ✅ |
| Single-exit family scenarios | 60 | 600 | ✅ |
| Multi-floor scenarios | 300 | 1,350 | ✅ |
| Stair rows with real demand | 1,000 | 80,635 | ✅ |
| High-occupancy scenarios (≥30) | 200 | 1,295 | ✅ |
| Multi-bottleneck rows | 2,000 | 14,179 | ✅ |
| Total-lockout scenarios with rows | 3 | 128 | ✅ |
| Multi-stair scenarios | 300 | 750 | ✅ |
| Chained-stair connectivity scenarios | 100 | 300 | ✅ |
| Reduced-redundancy exit scenarios | 100 | 300 | ✅ |

(A bug in the first coverage-report draft under-counted `total_lockout_scenarios_with_rows` by requiring a *positive-labeled* row instead of the documented "contributed any row" definition — fixed before this table; see `scripts/predictive_dataset_v3_coverage_report.py` git history.)

## 14. Structural diversity vs. Dataset V2 (Phase 14)

| Field | V2 baseline (4 graphs) | V3 (16 variants) |
|---|---|---|
| Distinct structural signatures | 4 | **16** |
| floor_count | 1–3 (mean 1.75, 3 distinct) | 1–4 (mean 1.88, **4 distinct**) |
| zone_count | 2–6 (4 distinct) | 2–8 (**7 distinct**) |
| door_count | 1–4 (4 distinct) | 1–6 (**5 distinct**) |
| exit_count | 1–3 (3 distinct) | 1–5 (**4 distinct**) |
| stair_count | 0–2 (3 distinct) | 0–3 (**4 distinct**) |
| candidate_count | 2–7 (3 distinct) | 2–11 (**8 distinct**) |
| mean_candidate_walking_distance | 17.1–49.8 (mean 29.6) | 12.3–55.8 (mean 33.5) |
| mean_alternative_route_count | 1.0–2.57 (mean 1.96) | 1.0–3.64 (mean 2.11) |

Every dimension is materially broader in V3, never narrower — the Phase 14 success gate.

## 15. Feature-distribution overlap (Phase 15)

Per-family range overlap fraction for the three strongest topology-shift features V3.1 identified: `candidate_walking_distance` **1.0** (6/6 family pairs overlap), `candidate_alternative_route_count` **1.0** (6/6), `total_active_occupant_count` **1.0** (6/6). In Dataset V2, each of these features effectively formed four isolated, family-specific clusters (a topology-family classifier could nearly read family off them directly); in Dataset V3, every family's range now overlaps every other family's range on every one of these three features. This is a direct, measured answer to Phase 15's question — the ranges no longer partition cleanly by family.

## 16. Duplication and temporal redundancy (Phase 16)

| Metric | Value |
|---|---|
| Total rows | 2,662,830 |
| Distinct feature-vector count | 386,390 |
| Duplicate feature-vector row fraction | **92.68%** (V3.1 measured 96.9% on Dataset V2) |
| Cross-variant duplicate hashes | 23,223 |
| Cross-family duplicate hashes | 1,399 |
| Within-scenario-only hashes | 303,318 |
| Cross-scenario duplicate hashes | 83,072 |
| Distinct vectors shared across ≥2 variants | 6.0% |

Duplication remains high (expected — the feature space is still low-cardinality: small integer counts and few categorical states), but is measurably lower than V2's 96.9%, and a real fraction of distinct feature vectors (23,223 of 386,390) are now shared *across different structural variants* rather than being family-siloed. Per V3.1's own finding, reducing duplication was never adopted as a success criterion — this section is composition-understanding only.

## 17. Target V2 quality (Phase 17)

Overall positive rate 2.19%; Door 1.54%, Exit 2.99%, Stair 1.32% — all within Target V2's established real-congestion range (0.9–3.0%), no candidate type collapsed to near-zero or runaway-high. Median lead time to onset: **7.56s** (mean 8.36s) across 42,965 positive-labeled rows, well inside the 20s prediction horizon. Multi-bottleneck rows: 14,179 (2+ distinct positive candidates in the same scenario/tick). Positive rate varies meaningfully by variant (0.78%–12.8%) — `single_exit_lowrise` base is highest (its stochastic sole-exit closure genuinely stresses the one path), `multi_exit_wide_6exit` is lowest (most redundant topology tested). No Door/Stair zero-duration artifact reappeared (the V1 bug this campaign's stair-authoring discipline guards against).

## 18-19. Exploratory generalization proxy experiment (NOT Model V4)

`scripts/predictive_dataset_v3_generalization_experiment.py` — same XGBoost algorithm/config Model V3 selected (`predictive_model.tree_models.build_tree_models`, no hyperparameter search), same class-weighting (`compute_class_weight_map`/`sample_weights_from_class_weight`), fit fresh at every holdout. Total wall time 649.5s for 20 fits (16 variant + 4 family holdouts).

**Phase 18A — leave-one-structural-variant-out (all 16, new `predictive_model/structural_variant_holdout.py`)**: train on every OTHER variant (including sibling variants of the same family), test on the held-out variant.

**Phase 18B — leave-one-topology-family-out (all 4, reuses `predictive_model/topology_holdout.py` unmodified)**: train on every other family entirely, test on a family the model has never seen in any form.

**Phase 18C/19 — comparison against Model V3.1's own family-holdout numbers (trained on Dataset V2's 4 fixed graphs) and against the deterministic-current-state baseline:**

| Family (held out) | V3.1 PR-AUC (Dataset V2) | V3 PR-AUC (Dataset V3, family holdout) | Δ relative | V3 PR-AUC (Dataset V3, **variant** holdout, family otherwise seen) | ML vs. deterministic (variant holdout) |
|---|---|---|---|---|---|
| multi_exit_wide | 0.305 | 0.254 | −16.6% | 0.393 | 3.25× |
| single_exit_lowrise | 0.571 | 0.513 | −10.2% | 0.559 | 1.97× |
| twin_stair_highrise | 0.308 | 0.286 | −7.1% | 0.729 | 5.82× |
| v1_topology_fixed | 0.531 | 0.497 | −6.4% | 0.589 | 1.89× |

Two distinct, honestly-reported findings, not one:

1. **Family-level generalization (an entirely unfamiliar building type) did not improve** — PR-AUC fell 6–17% relative to V3.1's Dataset-V2 numbers on the SAME four held-out families. This is a real, disclosed result, not spun away. Part of it is a harder-test-set effect (the V3 test set for a held-out family is now the union of that family's 4 structural variants, not one fixed graph — a broader, more realistic, and objectively harder test than V3.1 faced), but the headline finding stands: more structural diversity, by itself, did not make the hardest generalization case (zero prior exposure to the family) easier.
2. **Structural-VARIANT-level generalization (a novel shape of a family the model HAS otherwise seen) is dramatically stronger**, and directly validates V3.1's own controlled-experiment mechanism at full scale across all 16 variants, not just the original 2: e.g. `twin_stair_highrise` held out as a family scores PR-AUC 0.286, but the SAME graph held out only as a *variant* (with `twin_stair_chained_core`/`twin_stair_highrise_3stair`/`twin_stair_low` scenarios in training) scores **0.729** — a 2.5× improvement purely from having seen sibling structural variants of the same family. Every one of the 16 variant-holdout PR-AUC values (0.301–0.805) is at or above its family-holdout counterpart, most by a wide margin.

**ML vs. deterministic baseline (Phase 19)**: XGBoost beats the deterministic current-state baseline in **every one of the 20 holdouts**, family and variant alike — relative lift ranges 1.60×–3.56× at the family grain and 1.60×–7.15× at the variant grain. ML's advantage over deterministic congestion intelligence is not fragile to either kind of topology holdout.

## 20. Dataset readiness decision

**B — STRUCTURAL DIVERSITY IMPROVED BUT DATASET STILL INSUFFICIENT.**

Not A: structural diversity increase is real and large by every Phase 14-16 measure (16 vs. 4 distinct signatures, broadened/overlapping feature ranges, more structurally-plausible variety, lower duplication).

Not C: Phase 20's gate requires "exploratory unseen-structure generalization improves materially." That is true at the structural-variant grain (validated dramatically, generalizing V3.1's own +23.5%/+126% finding across all 16 templates) but **not** true at the family grain — the harder, arguably more important case of a building type the model has never seen in any form regressed 6-17% relative to V3.1's own numbers. A dataset is not ready to justify Model V4 production training while its headline generalization claim only holds for one of the two grains this milestone explicitly set out to test.

## 21. Versioning

| Field | Value |
|---|---|
| `campaign_version` | `predictive_dataset_campaign_v3` |
| `structural_variant_version` | `v3-structural-diversity-16-variants` |
| `schema_version` | `1.0` (unchanged) |
| `target_version` | `v2-persistent-demand-service-imbalance` (unchanged, frozen) |
| `master_seed` | `20260727` |
| Pilot scenarios/variant | 25 (400 total) |
| Full-scale scenarios/variant | 150 (2,400 total) |
| Generated | 2026-07-28 |
| Source commit | see `git rev-parse HEAD` at commit time below |

`data/predictive_dataset_campaign_v3_pilot/`, `data/predictive_dataset_campaign_v3/`, and `data/predictive_dataset_campaign_v3_generalization/` are gitignored (per this project's established per-milestone convention — every generated-artifact directory gets its own explicit `.gitignore` line, never a blanket rule). Dataset V1 and V2 artifacts are untouched.

## 22. Remaining limitations

- Family-level (entirely-unseen-building-type) generalization remains the open problem this milestone did not solve — see §20.
- The V3-vs-V3.1 family-holdout comparison (§18-19 table) is not perfectly apples-to-apples: the V3 test set for a held-out family is broader (4 variants, not 1 fixed graph) than V3.1's, which likely explains part (not necessarily all) of the observed PR-AUC decline.
- Structural-variant classification (82.7% accuracy) shows the finer-grained template is still substantially identifiable from features — diversity reduced but did not eliminate memorization risk.
- Duplicate feature-vector rate remains high (92.7%) — expected given the still-low-cardinality feature schema (frozen by design this milestone), not a regression to fix here.

## 23. Testing

New tests (41), all passing: `tests/test_predictive_dataset_topologies_v3.py` (variant registry, stair-bug regression guard across all 16 variants, structural-property spot checks, `with_scenario_count`), `tests/test_predictive_dataset_topology_signature_v3.py` (signature correctness, `structural_key` identity-exclusion, degenerate-building safety), `tests/test_predictive_dataset_topology_diversity_v3.py` (all-16-distinct gate, deliberately-injected-duplicate detection, V2-vs-V3 comparison), `tests/test_predictive_dataset_campaign_config_v3.py` (config/coverage-target construction), `tests/test_predictive_model_structural_variant_holdout.py` (partition correctness, mirroring `test_predictive_model_topology_holdout.py`'s own conventions). **Full suite: 4,715/4,715 passing (4,674 baseline + 41 new), 385.8s.**

---

## Final Report

1. **Topology families**: 4 (`single_exit_lowrise`, `twin_stair_highrise`, `multi_exit_wide`, `v1_topology_fixed`) — unchanged from V2.
2. **Structural variants**: 16 (4 per family).
3. **Genuinely distinct structural signatures**: 16 of 16 (mechanically verified, 0 duplicates).
4. **Structural dimensions that vary**: floor count, zone count, door count, exit count, stair count, candidate count, connectivity pattern (linear/branching/hub-spoke/chained-serial-stairs), exit symmetry/redundancy, walking distance, alternative-route count.
5. **Pilot**: 400 scenarios (25/variant), 400/400 accepted, 0 failed, 443,622 rows, 46.5s, every variant contributed rows, no zero-walking-distance bug, meaningful Target V2 positives in every candidate type — PASS.
6. **Full campaign**: 2,400 scenarios (150/variant), 2,400/2,400 accepted, 0 failed.
7. **Total candidate-time rows**: 2,662,830.
8. **Rows per family/variant**: see §13/§17; per-variant scenario counts uniform (150 each) by design, row counts vary with candidate density (2-11 candidates/variant).
9. **Door/Exit/Stair balance**: Door 1,461,802 (54.9%), Exit 962,539 (36.2%), Stair 238,489 (9.0%).
10. **Target V2 positive rates**: overall 2.19%; Door 1.54%, Exit 2.99%, Stair 1.32%.
11. **Hard-case coverage**: all 10 predefined targets pass (multi-bottleneck 14,179 rows, high-occupancy 1,295 scenarios, multi-stair 750, chained-stair 300, reduced-redundancy 300, total-lockout-with-rows 128).
12. **Structural diversity vs. V2**: 16 vs. 4 distinct signatures; every structural field's range materially broader (see §14 table).
13. **Candidate walking-distance distributions**: broadened AND now fully overlapping across families (overlap fraction 1.0, vs. V2's effectively-isolated per-family clusters).
14. **Alternative-route distributions**: same — broadened and fully overlapping (overlap fraction 1.0).
15. **Topology family still inferable**: yes, 99.1% accuracy (vs. V3.1's 100% on Dataset V2 — marginally reduced, still near-total).
16. **Structural variant inferable**: yes but meaningfully harder — 82.7% accuracy (16-way) vs. 24.9% majority baseline.
17. **Duplicate-feature rate**: 92.68% of rows belong to a duplicated feature vector (vs. V3.1's 96.9% on Dataset V2) — lower, not eliminated, never a success criterion.
18. **Temporal/cross redundancy**: 303,318 within-scenario-only hashes, 83,072 cross-scenario, 23,223 cross-variant, 1,399 cross-family.
19. **Target V2 regressions/artifacts**: none found — no zero-duration Door/Stair artifact reappeared, positive rates stayed within the established real-congestion range.
20. **Exploratory unseen-VARIANT generalization**: improved dramatically — every one of 16 variant holdouts (PR-AUC 0.301-0.805) scores at or above its family-holdout counterpart, up to 2.5× higher (twin_stair_highrise: 0.286 family-held-out vs. 0.729 variant-held-out).
21. **Exploratory unseen-FAMILY generalization**: did NOT improve — PR-AUC fell 6.4%-16.6% relative to Model V3.1's own Dataset-V2 family-holdout numbers on the same four families.
22. **ML vs. deterministic baseline**: ML wins in all 20 holdouts, relative lift 1.60×-7.15×.
23. **Peak memory / runtime**: full campaign 278.8s / 1,235MB available at tightest point (never breached the 300MB abort floor); generalization experiment 649.5s for 20 fits.
24. **Dataset version**: `campaign_version=predictive_dataset_campaign_v3`, `structural_variant_version=v3-structural-diversity-16-variants`, `target_version=v2-persistent-demand-service-imbalance`, `master_seed=20260727`.
25. **Readiness verdict**: **B — structural diversity improved but dataset still insufficient** (see §20).
26. **Full-suite result**: 4,715/4,715 passing (385.8s).
27. **Commit hash**: see `git log` — this milestone's commit immediately follows `a625691` (V3.1 Robustness Investigation) on `main`.

**A. Did V3 genuinely increase structural diversity?** Yes — mechanically verified (16 vs. 4 distinct signatures, broader ranges on every dimension, no accidental duplicates).

**B. Did we solve V3.1's problem, or merely create more rows?** Partially. The dataset is not merely bigger (2.4M vs. 2.5M scenarios, comparable scale) — it is structurally broader, and that breadth demonstrably helps the model transfer to unseen *variants* of a known family (validating V3.1's core hypothesis at scale). It does not yet solve transfer to an entirely unseen family — the harder half of the problem remains open.

**C. Is Target V2 still scientifically valid?** Yes — all leakage-boundary tests pass unmodified, positive rates stayed in the established real-congestion range, no zero-duration artifact reappeared.

**D. Did Door/Exit/Stair all remain usable?** Yes — all three have substantial row counts and non-degenerate positive rates.

**E. Did unseen-structure generalization improve?** Mixed, disclosed precisely: yes at the structural-variant grain, no at the topology-family grain.

**F. Does ML still provide predictive value beyond deterministic intelligence?** Yes, unambiguously — wins every one of 20 holdouts tested.

**G. Is more dataset work required before Model V4?** Yes — specifically, work targeting family-level (not just variant-level) generalization, since that is where this milestone's evidence shows the gap remains.

**H. Is Predictive Dataset V3 ready for Model V4 training?** No — verdict B, not C. Model V4 training should not proceed until family-level generalization is specifically investigated and improved.
