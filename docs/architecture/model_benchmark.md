# Predictive Model Development & Benchmark Campaign

Status: complete. This milestone develops, trains, and benchmarks multiple predictive models against
SynEvac1's existing dataset generation (`ai_registry`, Pipeline B) and evaluation
([[prediction_evaluation_milestone]]) infrastructure, selects a production model for each of the two existing
Shadow-Mode targets, and registers the winners via the existing `ai_registry.registry.ModelRegistry`.
Simulation, feature extraction, dataset generation, Recommendation, and Guidance are all unmodified.

---

## Dataset

**Pipeline**: `ai_registry` (Pipeline B, the production training-campaign pipeline — see
[[dataset_pipeline_audit_milestone]] for the full architectural map distinguishing it from Pipeline A's
research-scale `predictive_dataset` campaigns). One fixed 2-floor `Building`
(`ai_registry.training_scenario.make_training_building()`), scenario-to-scenario diversity from
`make_training_definition()`'s distributions (occupancy, fire origin/profile, door/exit/stair states,
firefighter deployment).

**Scale**: 12,000 scenarios requested, **11,997 accepted** (3 rejected by the generator's own validation
— not investigated further, a normal, expected rate), generated via `ai_registry.training.
generate_training_campaign()` in 257.7s (46.6 scenarios/sec — Pipeline B, running against one small fixed
building, is roughly 6-7x faster per scenario than Pipeline A's 24-structural-variant campaigns; see
[[dataset_pipeline_audit_milestone]] for that comparison). Master seed `20260730`.

**Splits** (documented, reproducible): `ai_training.split.make_split()`, `GroupShuffleSplit` grouped by
`scenario_id` (no scenario's row can straddle train/val/test — trivially true here since Pipeline B is
one row per scenario, but the same grouped-split code Pipeline B's own production training already uses),
`random_state=20260730`, `test_size=0.15`, `val_size=0.15`:

| Target | Total rows | Train | Validation | Test |
|---|---|---|---|---|
| `bottleneck_occurrence` | 11,997 | 8,397 | 1,800 | 1,800 |
| `evacuation_time` | 11,997 | 8,397 | 1,800 | 1,800 |

Both targets use the identical split (same `random_state`, same row count — one row per scenario for
both) — a scenario in the classification test set is also in the regression test set, deliberately, so
robustness/condition breakdowns are directly comparable across targets.

Datasets are never committed to git (established convention — see [[dataset_pipeline_audit_milestone]]):
regenerable from `scripts/run_model_benchmark_campaign.py` with the documented master seed.

---

## Model candidates (Phase 2)

Both targets already have an existing algorithm factory (`ai_training.models.base.build_classifier`/
`build_regressor`) that every `ai_training/models/*.py` model selects its estimator from by name. This
milestone **extended that same factory** (never introduced a second, parallel model-construction
mechanism) with `decision_tree`, `logistic_regression`/`linear_regression`, `mlp`, and `dummy` —
`random_forest`, `gradient_boosting`, `xgboost` already existed.

| Requested (Phase 2) | Included | Why |
|---|---|---|
| Random Forest | ✓ | pre-existing |
| Gradient Boosting | ✓ | pre-existing |
| XGBoost | ✓ | pre-existing (already an installed dependency) |
| LightGBM | ✗ | **not an installed dependency** in this environment (`import lightgbm` → `ModuleNotFoundError`); milestone brief explicitly forbids adding new external dependencies solely for this benchmark |
| MLP Neural Network | ✓ (new) | `sklearn.neural_network.MLPClassifier`/`MLPRegressor` — no new dependency (sklearn already core) |
| Logistic Regression (baseline) | ✓ (new) | classification only |
| Decision Tree (baseline) | ✓ (new) | both targets |
| Dummy classifier/regressor (control) | ✓ (new) | `sklearn.dummy` |

7 algorithms per target (14 models total), trained for **both** existing Shadow-Mode targets:
`bottleneck_occurrence` (classification, `ai_training.models.bottleneck_model.BottleneckModel`) and
`evacuation_time` (regression, `ai_training.models.evacuation_time_model.EvacuationTimeModel`) — both
classes reused completely unmodified; this milestone only supplies them a different `algorithm` in
`config`, exactly the extension point their own `default_estimator()` already existed for.

---

## Hyperparameter search (Phase 3)

Deterministic grid search (`model_benchmark/search.py`), small fixed grids (2-4 combinations per
algorithm — see `model_benchmark/algorithms.py` for the exact grids), 3-fold `GroupKFold` cross-validation
on the TRAIN split only (grouped by `scenario_id`, exactly like the outer split — `GroupKFold` has no
`random_state` at all, fully deterministic by construction). Every algorithm's every trial (params, per-
fold scores, mean, std) is recorded in `classification_results.json`/`regression_results.json`'s own
`search_trials` field. Scoring: F1 for classification CV, negative MAE for regression CV (the search
itself, at the CV-fold level, predates and is independent of this milestone's later discovery that F1 is
a poor RANKING metric for the final model-selection decision — see below; the grid search's own job is
just picking reasonable hyperparameters per algorithm, which F1 remains adequate for). `dummy` and
`linear_regression` have empty grids (no meaningful hyperparameters) and are trained directly.

Best hyperparameters found:

| Algorithm | Classification | Regression |
|---|---|---|
| Random Forest | `n_estimators=100, max_depth=None` | `n_estimators=100, max_depth=10` |
| Gradient Boosting | `learning_rate=0.05, n_estimators=100` | `learning_rate=0.05, n_estimators=100` |
| XGBoost | `max_depth=3, n_estimators=100` | `max_depth=3, n_estimators=100` |
| Decision Tree | `max_depth=5` | `max_depth=5` |
| Logistic/Linear Regression | `C=1.0` | (no hyperparameters) |
| MLP | `hidden_layer_sizes=(64, 32)` | `hidden_layer_sizes=(64, 32)` |
| Dummy | (no hyperparameters) | (no hyperparameters) |

---

## Evaluation methodology (Phase 4)

Every model's TEST-split predictions are scored via `prediction_evaluation`'s own existing metric
functions — reused directly, never reimplemented: `prediction_evaluation.classification_metrics.
compute_classification_metrics()` (precision, recall, F1, confusion matrix, ROC-AUC, calibration error,
confidence bias) and `prediction_evaluation.regression_metrics.compute_regression_metrics()` (MAE, RMSE,
MAPE, bias, mean/median/std error, 95% CI, worst/best-case error). Training time and inference latency
(both a batched-throughput figure and a single-row-call figure, matching Shadow-Mode's own per-cycle
one-`BuildingState`-at-a-time call pattern) are measured directly with `time.perf_counter()`; training
memory delta via `psutil.Process().memory_info().rss` before/after `fit()` (the same convention
[[dataset_pipeline_audit_milestone]] and [[predictive_dataset_campaign_v4_milestone]] already use).

### Classification leaderboard (`bottleneck_occurrence`, test split, n=1,800)

| Algorithm | ROC-AUC | F1 | Precision | Recall | Calib. error | Train (s) | Batch latency (ms/row) | Single-row latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.8367** | 0.9761 | 0.953 | 1.000 | **0.0739** | 0.42 | 0.008 | 2.27 |
| Logistic Regression | 0.8367 | 0.9757 | 0.954 | 0.998 | 0.1669 | 0.18 | 0.007 | 2.25 |
| XGBoost | 0.8317 | 0.9763 | 0.954 | 1.000 | 0.1812 | 0.24 | 0.013 | 3.27 |
| Random Forest | 0.8300 | 0.9760 | 0.955 | 0.998 | 0.0997 | 0.42 | 0.016 | 8.37 |
| Decision Tree | 0.8282 | 0.9760 | 0.955 | 0.998 | 0.0959 | 0.16 | 0.007 | 2.28 |
| Dummy (control) | 0.4832 | 0.9408 | 0.949 | 0.932 | 0.5127 | 0.17 | 0.009 | 2.38 |
| MLP | **0.4773** | 0.9749 | 0.951 | 1.000 | n/a (1 populated bin) | 0.44 | 0.009 | 2.25 |

### Regression leaderboard (`evacuation_time`, test split, n=1,800)

| Algorithm | MAE (s) | RMSE (s) | Bias (s) | Train (s) | Batch latency (ms/row) | Single-row latency (ms) |
|---|---|---|---|---|---|---|
| **Linear Regression** | **118.07** | 178.22 | -4.92 | 0.28 | 0.012 | 3.14 |
| Gradient Boosting | 118.19 | 178.35 | -4.78 | 0.39 | 0.008 | 2.14 |
| Decision Tree | 118.32 | 178.44 | -4.85 | 0.27 | 0.012 | 3.10 |
| XGBoost | 118.65 | 178.91 | -5.04 | 0.29 | 0.014 | 3.15 |
| MLP | 118.52 | **178.11** | **-2.98** | 1.92 | 0.015 | 3.61 |
| Random Forest | 118.85 | 179.24 | -5.14 | 0.39 | 0.015 | 8.42 |
| Dummy (control) | 128.19 | 189.72 | -1.99 | 0.30 | 0.012 | 3.70 |

---

## A critical, disclosed finding: severe class imbalance makes F1 misleading for `bottleneck_occurrence`

The classification test split is **95.1% positive** (`doors_that_became_bottlenecks` is true for the
large majority of scenarios in this small fixed building — even modest occupancy plus any partially-
blocked exit tends to produce SOME congestion somewhere during a run; consistent with the pre-existing,
already-disclosed finding in [[prediction_evaluation_milestone]] that this label is a broad, easily-
triggered whole-scenario condition). Under this imbalance, **F1 is a misleading ranking metric**: MLP
achieved F1=0.975 — near the top of the leaderboard — while its confusion matrix is `tn=0, fp=88, fn=0,
tp=1712`: it learned to predict "positive" for literally every single test row. Its ROC-AUC, 0.4773, is
*below* the Dummy control's own 0.4832 — i.e., MLP's predicted probabilities carry **less** genuine
discriminative signal than a control that doesn't even look at the features. F1 alone would never have
revealed this; only a threshold-independent metric (ROC-AUC) or explicitly inspecting the confusion
matrix does.

**Consequence for model selection**: this milestone ranks classification candidates by ROC-AUC, not F1
(see Phase 8 below) — disclosed here explicitly since it is a genuine mid-benchmark methodological
correction, not the original plan. The five genuinely-discriminating candidates (Gradient Boosting,
Logistic Regression, XGBoost, Random Forest, Decision Tree) all score ROC-AUC ≈ 0.83, meaningfully above
both Dummy (0.483) and MLP (0.477).

---

## Robustness analysis (Phase 5)

### Building conditions

Slicing is derived from the campaign's own LEGACY per-scenario fields (`ignition_zone`, `ignition_floor`,
`total_occupants`, `Exit_1_State`/`Exit_2_State`, ground-truth `exits_exceeding_capacity`/
`stairs_exceeding_capacity`) — **never** from the canonical live-feature schema a model actually trains on
(`model_benchmark/robustness.py`), so slice boundaries never leak into a model's own inputs.

**Gradient Boosting** (`bottleneck_occurrence` winner) by condition:

| Axis | Value | n | F1 | ROC-AUC |
|---|---|---|---|---|
| occupancy_tier | low (≤10) | 107 | 0.887 | 0.688 |
| occupancy_tier | medium (11-20) | 1,218 | 0.974 | 0.741 |
| occupancy_tier | high (>20) | 475 | 1.000 | n/a — zero negative-class scenarios at high occupancy in this dataset |
| floor_of_ignition | ground floor | 1,405 | 0.975 | 0.831 |
| floor_of_ignition | upper floor | 395 | 0.981 | 0.863 |
| exit_block_tier | all exits open | 1,539 | 0.972 | 0.835 |
| exit_block_tier | an exit blocked | 261 | 0.998 | 0.904 |
| fire_origin_zone | zone-office-a | 124 | 0.916 | **0.767 (weakest zone)** |
| fire_origin_zone | zone-lobby / office-b / upper | 667 / 614 / 395 | 0.976-0.984 | 0.819-0.863 |

**Linear Regression** (`evacuation_time` winner) by condition (MAE, seconds):

| Axis | Value | n | MAE |
|---|---|---|---|
| occupancy_tier | low | 107 | **74.2** (best) |
| occupancy_tier | medium | 1,218 | 110.5 |
| occupancy_tier | high | 475 | 147.4 |
| floor_of_ignition | ground floor | 1,405 | 111.3 |
| floor_of_ignition | upper floor | 395 | 142.3 |
| exit_block_tier | all open | 1,539 | 91.0 |
| exit_block_tier | blocked | 261 | **277.7 (worst — blocked-exit scenarios have far more variable evacuation dynamics)** |

Every finding above is directionally sensible (more occupants → harder to predict precisely; a blocked
exit → both easier to detect a resulting bottleneck AND harder to predict the exact evacuation time; a
fire on the upper floor, requiring stair descent, is more variable than a ground-floor fire) — none is a
surprising anomaly, but each is a genuine, quantified weak spot worth knowing before any deployment
decision.

**"Different building layouts" is explicitly NOT a testable robustness axis here** — Pipeline B trains
against exactly one fixed `Building` (`make_training_building()`). This is a structural property of
Pipeline B, not something this benchmark chose to skip; a future milestone wanting genuine cross-layout
robustness would need Pipeline A's 24-structural-variant campaign machinery feeding a compatible target
definition, which does not currently exist (see [[dataset_pipeline_audit_milestone]]'s own Pipeline A/B
independence finding).

### Prediction horizons (5s / 10s / 20s / 30s / 60s)

Exercised via the ACTUAL `prediction_evaluation` framework (`PredictionRegistry`, `GroundTruthTimeline`,
`evaluator.evaluate()`, `horizon_analysis`) — reused, not reimplemented (`scripts/
run_model_benchmark_horizon_analysis.py`). **Disclosed methodological limitation, stated plainly**:
`bottleneck_occurrence` is a whole-scenario prediction computed once from pre-simulation features
(`ai_features.CANONICAL_LIVE_SCHEMA`) — it has no genuine "predict N seconds from now" behavior the way
Pipeline A's own per-candidate-per-tick Target V2 does (see [[predictive_dataset_foundation_milestone]] and
[[prediction_evaluation_milestone]]). Tagging the SAME per-scenario prediction with each of the 5 nominal
horizons and evaluating produced **identical metrics at every horizon bucket for every model** — this is
the EXPECTED, CORRECT result given the model's inputs never vary with the horizon tag, not evidence of
genuine horizon-robustness the way a truly time-varying model could claim. Reported honestly as "not a
horizon-dependent model class," not fabricated as "robust across all horizons."

---

## Feature importance (Phase 6)

**Most important feature, both targets, every algorithm that supports native importance**:
`total_occupant_count` — dominant by a wide margin (97.0% of Gradient Boosting's native
`feature_importances_`; the single largest-magnitude coefficient, 1.43, in Logistic Regression; an 18.5-
second MAE increase when permuted out of Linear Regression, by far the largest of any feature).

**Secondary contributors** (both targets): `smoke_detector_alarm_count`, `facp_active_alarm_source_count`,
`heat_detector_alarm_count` — each single-digit-percent native importance for Gradient Boosting, small but
nonzero coefficients for Logistic/Linear Regression. Permutation importance on the REGRESSION side shows
these three fields' contribution is actually slightly **negative** for Linear Regression (permuting them
*decreased* MAE) — a legitimate "least useful, arguably noisy for this specific model" finding, not an
error.

**Least useful / effectively unused features (both targets, essentially every algorithm)**:
`occupancy_observed`, `mean_occupant_track_confidence`, all `camera_*`/`sensor_*` coverage/active/offline
counts, all `heat_detector_coverage_count`/`smoke_detector_coverage_count`/`*_fault_count`, `building_
alarm_status`, `facp_available`/`facp_panel_state`/`facp_acknowledged`/`facp_silenced`, and both
`control_*` fields — native importance ≈ 0.0 and permutation importance ≈ 0.0 across the board. **Caveat,
disclosed rather than overstated**: "unused" here means "did not show predictive signal within THIS
11,997-scenario campaign against ONE fixed small building," not "can never matter" — several of these
fields (camera/sensor health, FACP state) are near-constant in this training campaign's own scenario
distribution (cameras/sensors are simulated as reliably online in most scenarios), so there is
insufficient VARIANCE for any model to learn a relationship even if one exists in principle. A campaign
deliberately varying sensor/camera failure rates more aggressively could reveal a different picture.

**Note on classification permutation importance's own limitation**: permutation importance for
`bottleneck_occurrence` is scored by F1-drop, and — consistent with this document's own "severe class
imbalance" finding above — is heavily compressed by the same 95.1%-positive imbalance (even shuffling
`total_occupant_count`, the single most important feature by every other measure, only drops F1 by
~0.0018). **Native importance is the more reliable signal for this specific target** and is what this
section's "most important feature" claim above is primarily based on, cross-checked against Logistic
Regression's coefficients and the regression side's (unaffected by this issue) permutation importance —
all three independently agree on `total_occupant_count`'s dominance.

**Highly correlated features** (Pearson |r| ≥ 0.8, 16 numeric canonical columns considered): exactly 2
pairs — `facp_active_alarm_source_count` ↔ `smoke_detector_alarm_count` (r=0.885, sensible: this small
building's FACP alarm sourcing is substantially driven by smoke detection) and `heat_detector_alarm_count`
↔ `smoke_detector_alarm_count` (r=-0.878, sensible: a scenario's fire profile — "Flaming" vs
"Smoldering" — tends to trip one detector type preferentially over the other). No other pair of the 16
numeric canonical features exceeds the threshold — the canonical schema is, empirically, mostly
non-redundant on this campaign's data.

---

## Model comparison leaderboard (Phase 7)

Combining every axis measured above (full numbers in the two leaderboard tables further up):

| Criterion | Classification leader | Regression leader |
|---|---|---|
| Prediction accuracy (ROC-AUC / MAE) | Gradient Boosting (tied with Logistic Regression) | Linear Regression (tied with Gradient Boosting/Decision Tree) |
| Robustness (most consistent across conditions) | Gradient Boosting (no condition drops ROC-AUC below 0.69) | Linear Regression (no worse than the tree ensembles in any slice) |
| Training cost | Decision Tree (0.16s) | XGBoost/Decision Tree (~0.27-0.29s) |
| Inference cost (single-row) | Decision Tree/Logistic Regression (~2.25ms) | Gradient Boosting (2.14ms) |
| Memory footprint | Broadly flat across all non-MLP models (a few MB, often within measurement noise — see Limitations) | Same |
| Calibration quality | **Gradient Boosting (0.074, best of any real model)** | n/a (regression has no calibration-curve concept) |
| Overall suitability for live deployment | **Gradient Boosting** | **Linear Regression** |

---

## Model selection (Phase 8)

### `bottleneck_occurrence`: Gradient Boosting

Gradient Boosting and Logistic Regression are **statistically tied on ROC-AUC** (0.83669 vs 0.83668 — a
difference of ~0.00001, far inside any reasonable noise band for n=1,800). This benchmark's registration
script (`scripts/run_model_benchmark_registration.py`) implements this explicitly: it clusters every
candidate within 0.01 ROC-AUC of the best score, then breaks the tie by **calibration error, ascending**
— because Shadow-Mode's own `ai_registry.inference_service.LiveAIInferenceService` uses a fixed
probability threshold (`bottleneck_threshold=0.5`) to convert a predicted probability into a boolean
occurrence flag, meaning the model's probabilities need to actually **mean what they claim**, not merely
rank-order correctly. Gradient Boosting's calibration error (0.074) is more than double as good as
Logistic Regression's (0.167) and dramatically better than XGBoost's (0.181) — a real, quantitative,
deployment-relevant difference where ROC-AUC alone is silent.

**Why the others were not selected**: XGBoost/Random Forest/Decision Tree all score ROC-AUC ≈ 0.828-0.832
— within the same tied cluster, but each with worse calibration than Gradient Boosting (0.096-0.181).
Logistic Regression is a very close second (near-identical ROC-AUC, but meaningfully worse calibration).
Dummy and MLP are excluded entirely — both fail to beat Dummy's own eligibility bar (MLP does not even
beat Dummy itself; see the imbalance finding above).

### `evacuation_time`: Linear Regression

Linear Regression has the single best MAE (118.07s) among all seven candidates, narrowly ahead of
Gradient Boosting (118.19s) and Decision Tree (118.32s) — a margin of well under 1%, likely inside this
test set's own noise (95% CI half-widths on signed error span roughly ±8s for every model, several times
larger than the gap between candidates). Given the top several models are **effectively tied** on the
primary metric, Linear Regression is additionally attractive on Occam's-razor grounds: it is the
simplest, fastest-to-train (0.28s), most interpretable model in the field (a single coefficient per
feature — see Phase 6), with RMSE and single-row latency both competitive with the tree ensembles. This is
a genuinely meaningful finding in its own right: for `evacuation_time` on this dataset, a purely linear
model captures essentially everything the more complex tree ensembles and a neural network do — the
underlying relationship (dominated by `total_occupant_count`, per Phase 6) does not require nonlinear
modeling capacity to approximate well.

**Why the others were not selected**: Gradient Boosting/Decision Tree/XGBoost are statistically tied but
add training/inference cost and reduced interpretability for no measurable accuracy gain. MLP has the
single best bias (-2.98s) and RMSE (178.11s) — genuinely worth noting — but at 6-7x the training cost
(1.92s vs Linear Regression's 0.28s) for a MAE that is still slightly worse; not a compelling trade on
this dataset's scale. Random Forest is the single worst real-model MAE (118.85s) and by far the slowest
single-row inference (8.42ms) of any non-Dummy candidate — dominated on every axis.

---

## Model registration (Phase 9)

Both selected models were registered via the existing `ai_registry.registry.ModelRegistry` (`scripts/
run_model_benchmark_registration.py`): `register_model()`, persisted to disk via the existing `ai_registry.
metadata.save_live_model()`, and proven servable end-to-end through the existing, **completely untouched**
Shadow-Mode gateway (`live_system.live_ai_gateway.RegistryLiveAIInferenceGateway` wrapping
`ai_registry.inference_service.LiveAIInferenceService`) — a real `.predict(state, timestamp)` call against
a real `BuildingState` (built via the existing `ai_features.build_building_state_at_alarm_activation()`)
returned a valid `LiveAIPredictionSnapshot` with both a `bottleneck` and an `evacuation_time_experimental`
prediction, zero errors, zero warnings. `ai_registry.registry.ModelRegistry.get_latest_compatible_model()`
correctly returns the newly-registered models (proven in `tests/test_model_benchmark_registration.py`,
alongside a negative-control test proving a model with a corrupted `ordered_feature_names` is correctly
**rejected** by the same registry's existing compatibility check — this milestone's registration is
genuinely schema-checked, not merely "didn't crash").

**No runtime startup/wiring path was modified.** `live_runtime/factory.py`, `live_system/main.py`, and
every other Shadow-Mode wiring file from [[shadow_mode_prediction_milestone]] are untouched — this
milestone proves that IF that existing wiring's `ModelRegistry` were populated with these two newly-
trained-and-selected models, Shadow-Mode would serve them correctly; it does not itself change which
models are actually loaded at any real startup today.

---

## Known limitations

- **`bottleneck_occurrence`'s 95.1% class imbalance** (disclosed at length above) means even a well-
  calibrated model's absolute performance ceiling on this specific target/building is modest — ROC-AUC
  ≈0.84 is a real, useful signal, not a strong classifier in isolation. A future milestone with a richer
  multi-building campaign (Pipeline A-style diversity, if a compatible target were built for it) would be
  needed to know whether this ceiling is a property of the TARGET or of this ONE fixed building's limited
  scenario diversity.
- **Horizon robustness is not a meaningful axis for either currently-registrable model type** (disclosed
  above) — both are whole-scenario, pre-simulation-feature predictions, never a function of elapsed time.
- **"Different building layouts" is not a testable robustness axis** with Pipeline B (one fixed building).
- **Memory-delta measurements are noisy for small/fast-fitting models** (several entries in the leaderboard
  are negative, reflecting ordinary Python GC timing rather than a real memory decrease) — meaningful only
  in aggregate/relative terms (e.g., MLP and Random Forest visibly costing more than Decision Tree), not
  as a precise per-model figure.
- **Classification permutation importance is compressed by the same class imbalance** that motivated
  ranking by ROC-AUC instead of F1 (disclosed in Phase 6) — native importance and Logistic Regression's
  own coefficients are the more reliable signal for that target, cross-validated against the (unaffected)
  regression-side permutation importance.
- Every number in this document reflects ONE deterministic run (`master_seed=20260730`,
  `split_seed=20260730`) — reproducible by re-running `scripts/run_model_benchmark_campaign.py`, but not
  independently repeated across multiple seeds to characterize run-to-run variance.
