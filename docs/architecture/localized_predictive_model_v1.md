# First Localized Predictive Congestion Model v1 — Offline Research

Status: **OFFLINE RESEARCH ONLY.** Nothing in this milestone is wired into recommendation
scoring, exit ranking, guidance, signage, LiveRuntime, or operator workflow. This document
and the `predictive_model/` package it describes exist to answer one question: *can SynEvac
predict that a specific Door/Exit/Stair will become congested within a short horizon, better
than simple baselines?* Builds on `docs/architecture/predictive_dataset_campaign_v1.md`
(commit `1574654`, 2,508,480-row frozen dataset, 4440/4440 tests passing) — that milestone
built and validated the data; this one is the first to train anything on it.

## 1. Problem definition

For every `(scenario, observation_time, candidate)` triple where the candidate is not
*already* congested, predict whether it will become congested (2+ concurrent occupants on
its edge, `predictive_dataset.target_generator.CONGESTION_THRESHOLD`) within the next
`prediction_horizon` seconds — a binary classification problem, evaluated primarily at the
20-second horizon the prior milestone recommended.

## 2. Dataset

`data/predictive_dataset_campaign_v1/candidate_dataset_v1.csv` — 2,508,480 rows, 1,966
contributing scenarios, dataset version `{schema: 1.0, campaign: predictive_dataset_campaign_v1,
feature: 1.0, target: v1-congestion-threshold-2-horizon-window}`. `predictive_model.dataset_loader`
loads this CSV only after checking its `campaign_v1_report.json`-embedded `dataset_version`
against exactly this tuple — an incompatible schema/campaign/feature/target version raises
`IncompatibleDatasetVersionError` rather than silently training against a shape this code
wasn't written for.

Model input features are exactly `predictive_dataset.schema.CANDIDATE_FEATURE_NAMES` (9
fields) — never `currently_congested`, `had_any_activity_in_window`, or `target` (label-side
only), and never `scenario_id`/`observation_time`/`candidate_id`/`prediction_horizon`
(identity/selection, not signal). `predictive_model.feature_prep` one-hot encodes
`candidate_type`/`candidate_congestion_level` (fixed vocabularies, not fit from data) and
imputes nullable numeric fields with a `-1` sentinel plus an explicit `_missing` indicator
column — column count/order is fixed by the frozen schema's own `nullable` flags, not by
which values happen to be missing in a given split, so train/val/test and all four horizons
share an identical 21-column feature matrix.

## 3. Scenario-level split (Phase 2)

Split strictly by `scenario_id`, never by row — `predictive_model.scenario_split.split_scenarios()`
sorts the unique scenario ids (input-order-independent), shuffles deterministically
(`seed=20260726`), and slices 70/15/15:

| Split | Scenarios |
|---|---|
| Train | 1,376 |
| Validation | 295 |
| Test | 295 |

`assert_no_scenario_overlap()` mechanically proves — both at the ID-set level and by
re-deriving `scenario_id` sets from each split's actual output rows — that no scenario
contributes rows to more than one split; this same split (same seed) is reused across all
four horizons, since the underlying scenarios are identical.

## 4. Models evaluated (Phases 3-4)

| Model | Type | Library |
|---|---|---|
| Majority Class | trivial baseline | stdlib |
| Always Negative | trivial baseline | stdlib |
| Random | trivial baseline | numpy |
| Logistic Regression | linear baseline | scikit-learn |
| Decision Tree (depth 6) | shallow-tree baseline | scikit-learn |
| Random Forest (300 trees) | tree ensemble | scikit-learn |
| Gradient Boosting | histogram-based tree ensemble | scikit-learn `HistGradientBoostingClassifier` |
| XGBoost (300 rounds, hist) | tree ensemble | xgboost |

**Library availability in this environment:** xgboost — available; LightGBM, CatBoost — **not
installed**, not evaluated (documented, not silently skipped: `predictive_model.tree_models.
library_availability_report()`). "Gradient Boosting" is `HistGradientBoostingClassifier`, not
classic `sklearn.ensemble.GradientBoostingClassifier` — the classic implementation has no
histogram binning and is impractical at this row count; HistGradientBoosting is sklearn's own
modern successor for exactly this scale, documented substitution.

## 5. Class imbalance (Phase 5)

Overall train positive rate at 20s: 12.5%. Strategy: `class_weight='balanced'` → per-sample
weighting (`predictive_model.imbalance`), plus F1-maximizing decision-threshold tuning on the
**validation** split only (never test, never train). Deliberately **no oversampling** — a
synthetic minority row built by interpolating between real feature vectors has never been
validated against this dataset's own leakage boundary, so reweighting real rows (never
fabricating new ones) was judged the safer choice for a milestone whose charter includes
re-verifying that boundary, not engineering around it.

## 6. Results at the primary horizon (20s)

Test split, 86,724 trainable rows (positive rate 12.9%):

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Balanced Acc. | Brier |
|---|---|---|---|---|---|---|---|
| Majority Class | 0.500 | 0.129 | 0.000 | 0.000 | 0.000 | 0.500 | 0.129 |
| Always Negative | 0.500 | 0.129 | 0.000 | 0.000 | 0.000 | 0.500 | 0.129 |
| Random | 0.500 | 0.130 | 0.206 | 0.130 | 0.500 | 0.501 | 0.334 |
| Logistic Regression | 0.933 | 0.572 | 0.658 | 0.518 | 0.900 | 0.888 | 0.101 |
| Decision Tree (depth 6) | 0.947 | 0.647 | 0.672 | 0.574 | 0.809 | 0.860 | 0.183 |
| Random Forest | 0.954 | 0.691 | 0.701 | 0.625 | 0.798 | 0.863 | 0.086 |
| **Gradient Boosting (best)** | **0.956** | **0.708** | **0.706** | 0.624 | 0.813 | 0.870 | 0.157 |
| XGBoost | 0.956 | 0.703 | 0.707 | 0.614 | 0.832 | 0.877 | 0.086 |

**Best model: `gradient_boosting` (HistGradientBoostingClassifier)**, selected by test PR-AUC
(the appropriate metric under a ~13% positive rate). XGBoost is a statistical tie (PR-AUC
0.703 vs 0.708) — both are legitimate choices; HistGradientBoosting was marginally ahead.

By candidate type (test, at the tuned threshold 0.96):

| Type | n | Positive rate | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Door | 35,628 | 25.2% | 0.915 | 0.714 | 0.622 | 0.873 | 0.726 |
| Exit | 33,282 | 5.5% | 0.963 | 0.686 | 0.641 | 0.681 | 0.660 |
| **Stair** | 17,814 | 2.1% | 0.936 | **0.240** | 0.375 | **0.024** | 0.045 |

**Stair prediction is effectively non-functional at this threshold** (9 true positives out of
372 real stair-congestion events) — directly consistent with the prior milestone's disclosed
`stair-1` demand-blindness finding (both `candidate_queue_length` and `candidate_approaching_
count` are structurally blind for this building's stair candidate, see
`docs/architecture/predictive_dataset_campaign_v1.md` §10). This is not a modeling failure to
fix by retuning — the input signal for stair candidates in this dataset genuinely carries
almost no demand information.

## 7. Horizon robustness (Phase 7)

Best model type (`gradient_boosting`) retrained fresh at each horizon, same scenario split:

| Horizon | Positive rate | ROC-AUC | PR-AUC |
|---|---|---|---|
| 10s | 6.7% | 0.927 | 0.391 |
| **20s** | 12.9% | 0.956 | 0.708 |
| 30s | 17.8% | 0.980 | 0.913 |
| 60s | 22.4% | 0.992 | 0.976 |

Raw PR-AUC increases monotonically with horizon — but this is **not evidence that 60s is
the better operating point**. It is largely a base-rate artifact: a longer window has more
opportunities to register "became congested," which mechanically raises both the positive
rate (6.7% → 22.4%) and the achievable PR-AUC ceiling, and gives the model more elapsed
demand-signal evolution to observe before the outcome — a fundamentally easier prediction
target, not a "better" one. **20 seconds remains the recommended operating horizon**, for the
same reason `docs/architecture/predictive_dataset_campaign_v1.md` §14 already gave: it is the
shortest horizon clearing the genuine-advance-warning floor, and the floor is an operational
lead-time requirement, not a statistical-difficulty one. A model that only "works" at 60s
would give an operator less than a minute of usable warning after inference/decision latency;
this milestone's own numbers do not change that conclusion, they corroborate it from the
modeling side rather than the dataset-statistics side.

## 8. Feature importance (Phase 8)

`HistGradientBoostingClassifier` exposes no `feature_importances_` — permutation importance
(ROC-AUC drop, 20,000-row sample, 5 repeats, seed 20260726) is the only method computed for
the best model, and was the *preferred* method per this milestone's own instructions:

| Rank | Feature | Mean ROC-AUC drop |
|---|---|---|
| 1 | `candidate_queue_length` | 0.228 |
| 2 | `candidate_adjacent_zone_occupancy` | 0.046 |
| 3 | `candidate_traversable` | 0.042 |
| 4 | `total_active_occupant_count` | 0.035 |
| 5 | `candidate_walking_distance` | 0.022 |
| 6 | `candidate_approaching_count` | 0.015 |
| 7 | `candidate_capacity` | 0.012 |
| 8+ | one-hot categorical columns | ≤ 0.006 each |

`candidate_queue_length` dominates by nearly 5x over the next feature — matches engineering
intuition (current queueing is the strongest available signal for near-future congestion) and
matches the prior milestone's own correlation finding (`r=0.547` with target, the strongest of
any raw feature, `docs/architecture/predictive_dataset_campaign_v1.md` §11).

## 9. Error analysis (Phase 9)

Test split (86,724 rows): 5,489 false positives (6.3%), 2,095 false negatives (2.4%) overall.

- **By candidate type:** Door FP rate 13.4% (over-alarms on the highest-base-rate type);
  Stair FP rate 0.08% but effectively zero recall (§6).
- **By occupancy:** FN rate is *worse* at HIGH occupancy (9.4%) than LOW (1.5%) — the model
  misses more real congestion events exactly when the building is fullest, the situation
  where a miss matters most operationally.
- **By blocked-route scenario:** FP/FN rates are nearly identical for blocked vs. fully-open
  scenarios (FP 6.0% vs 7.1%, FN 2.5% vs 2.2%) — the model is not meaningfully destabilized by
  blocked-route scenarios specifically.
- **By temporal phase:** EARLY has the highest FP (11.1%) and FN (4.7%) rates; LATE has the
  lowest of both (2.0% / 0.9%) — consistent with EARLY being both the highest-positive-rate
  and highest-activity phase (more opportunities for either kind of error).
- **By simultaneous-bottleneck count — the most operationally important slice:** rows with
  ≥2 candidates simultaneously trending toward congestion have an FN rate of **13.3%**, over
  **11x** the FN rate of single-bottleneck rows (1.16%). **The model is meaningfully worse at
  exactly the multi-bottleneck situations that matter most for evacuation safety** — a real,
  disclosed limitation, not a rounding artifact.

## 10. Calibration (Phase 10)

Raw `gradient_boosting` probabilities are poorly calibrated: Brier 0.157, Expected Calibration
Error (ECE) 0.209 on test. Both Platt scaling and isotonic regression (fit on validation
probabilities only, applied to test) substantially improve this:

| | Brier score | ECE |
|---|---|---|
| Raw (uncalibrated) | 0.157 | 0.209 |
| Platt scaling | 0.061 | 0.026 |
| **Isotonic regression (recommended)** | **0.056** | **0.003** |

**The raw model is not well calibrated; isotonic-calibrated probabilities are** — any future
use of this model's output as a probability (rather than just a thresholded alarm) should
apply the isotonic calibration map, not the raw model output.

## 11. Scientific sanity checks (Phase 11)

- **Feature-family ablation** (zeroed, not dropped, so architecture stays fixed):

  | Family removed | ROC-AUC drop | PR-AUC drop |
  |---|---|---|
  | `global_and_adjacent_context` (occupancy features) | **0.037** | **0.085** |
  | `demand_signal` (queue + approaching count) | 0.006 | 0.047 |
  | `structural` (type/capacity/distance/traversable) | 0.007 | 0.015 |
  | `derived_congestion_level` (one-hot congestion level) | -0.0002 | 0.001 |

  Removing occupancy-context features hurts more than removing the raw demand-signal
  features, despite `candidate_queue_length` having by far the strongest *individual*
  correlation (§8) — because `candidate_congestion_level` (which the model can still see) is
  itself derived from queue/approaching/capacity, so some of the demand family's signal
  survives its own removal via that redundant, correlated feature. `derived_congestion_level`
  alone is nearly free to remove for the same reason in reverse. This is a genuine, disclosed
  redundancy finding, not a contradiction.

- **Leakage correlation re-check:** none of the 21 model-input columns exceed the reused
  0.9 review threshold (`predictive_dataset.correlation.LEAKAGE_REVIEW_THRESHOLD`); strongest
  are `candidate_congestion_level=LOW` (r=-0.574) and `=CRITICAL` (r=0.530) — both expected,
  moderate associations with a feature that is itself partly derived from current queueing.

- **Label-shuffle test:** a fresh `gradient_boosting` trained on train features paired with
  randomly shuffled train labels scores ROC-AUC **0.499** against the real validation labels —
  collapses to chance exactly as expected. No leakage channel survives breaking the true X/y
  correspondence.

## 12. Model export (Phase 12)

`data/localized_predictive_model_v1/model.joblib` (the fitted `HistGradientBoostingClassifier`
wrapper) + `model_metadata.json` (dataset version, feature names, split sizes, decision
threshold, validation/test metrics, production-readiness verdict — every field marked
`"not_wired_into_live_inference": true`). **Not imported by any live-facing module** —
`predictive_model/` has zero imports from and zero imports into `live_system/`,
`building_state/`, `recommendation/`, `guidance/`, `dynamic_signage/`, or LiveRuntime.

## 13. Known limitations

- **Stair prediction is not functional** (§6) — a direct, disclosed consequence of the
  dataset's own `stair-1` demand-blindness (`docs/architecture/predictive_dataset_campaign_v1.md`
  §10), not something this modeling milestone can fix without a feature-schema change there.
- **Multi-bottleneck situations have an 11x higher false-negative rate** than single-bottleneck
  situations (§9) — the highest-value prediction case is also the weakest.
- **Single building topology** — every scenario uses the same fixed building (2 doors, 2
  exits, 1 stair, `ai_registry.training_scenario.make_training_building()`). Generalization to
  any other topology (especially single-exit buildings, per the dataset milestone's own §13
  disclosed gap) is completely unverified.
- **Horizon comparison is base-rate-confounded** (§7) — raw PR-AUC across horizons should not
  be read as "which horizon is easiest to deploy," only "20s remains the right lead-time floor."
- **Raw probabilities are miscalibrated**; only the isotonic-calibrated output should ever be
  read as a probability (§10).

## 14. Recommended deployment threshold

**If** this model were ever integrated (it is not, by this milestone's own charter): decision
threshold **0.96** (F1-maximizing on validation at the 20s horizon), applied to
**isotonic-calibrated** probabilities, evaluated **separately by candidate type** given the
Door/Exit/Stair performance gap in §6 — a single building-wide threshold would over-alarm on
Door and be silent on Stair.

## 15. Future improvements

1. Address the dataset-level `stair-1` demand-blindness (predictive_dataset campaign v2, per
   that milestone's own §14 recommendation) before re-attempting stair-candidate prediction.
2. Investigate why multi-bottleneck rows have disproportionately high FN rates — possibly a
   dedicated interaction feature ("count of other candidates also currently trending toward
   congestion") the current per-candidate-independent feature schema cannot express.
3. Add building-topology diversity (at minimum a single-exit building) before any claim of
   generalization beyond this one fixed layout.
4. Re-run this same pipeline once a v2 dataset closes the total-lockout / stair gaps, to see
   whether the model's weakest slices specifically improve.

## Production readiness decision (Phase 14)

**PROMISING BUT NEEDS MORE DATA.**

Evidence for "promising": gradient boosting clears every trivial baseline by a wide margin
(PR-AUC 0.708 vs. 0.129 for majority-class/always-negative, a >5x lift), passes both sanity
checks (leakage re-check clean, label-shuffle test collapses to chance), and isotonic
calibration produces genuinely trustworthy probabilities (ECE 0.003).

Evidence against "ready": (a) stair-candidate prediction is non-functional (§6), a direct,
disclosed dataset limitation; (b) the highest-value case — multiple simultaneous bottlenecks —
has an 11x worse false-negative rate than the common case (§9); (c) the entire dataset is one
fixed building topology with no single-exit or total-lockout representation, so generalization
beyond this exact building is completely unverified. None of these are modeling-tuning
problems fixable by retraining on the same data — they require dataset changes this milestone
is explicitly out of scope to make.

## Final report — explicit answers

**A. Does the predictive model outperform simple baselines?** Yes, decisively — test PR-AUC
0.708 vs. 0.129 (majority-class/always-negative), a >5x lift; ROC-AUC 0.956 vs. 0.500.

**B. Which algorithm performed best?** `HistGradientBoostingClassifier` ("gradient_boosting"),
narrowly ahead of XGBoost (PR-AUC 0.708 vs 0.703 — a near-tie, both legitimate choices) and
ahead of Random Forest (0.691), Decision Tree (0.647), Logistic Regression (0.572).

**C. Is PR-AUC high enough to justify operational interest?** For Door and Exit candidates,
yes (0.71 and 0.69 against 25% and 5.5% base rates respectively) — clearly informative. For
Stair, no (0.24, non-functional recall) — operational interest should be scoped to Door/Exit
only pending a dataset fix for stair demand-blindness.

**D. Is the model well calibrated?** Not out of the box (ECE 0.209) — but isotonic-calibrated
output is well calibrated (ECE 0.003) and should be the only form ever read as a probability.

**E. Which features contribute most?** `candidate_queue_length` by a wide margin (permutation
importance 0.228, ~5x the next feature), then `candidate_adjacent_zone_occupancy`,
`candidate_traversable`, and `total_active_occupant_count`.

**F. Is there evidence of target leakage?** No — leakage-correlation re-check found nothing
above the 0.9 review threshold, and the label-shuffle test collapsed to chance (ROC-AUC 0.499).

**G. Should this model remain offline research, or is it ready for controlled live
integration?** **Remain offline research.** The stair-candidate and multi-bottleneck weaknesses
(§6, §9) and single-topology dataset limitation (§13) are dataset-level gaps this milestone
cannot close by retraining; a v2 predictive_dataset campaign addressing them is the right next
step before any controlled-integration milestone is even considered.

**H. What should be the next milestone?** A `predictive_dataset` v2 campaign that (1) adds a
route-membership-based demand signal to close the stair-1 blind spot, (2) adds at least one
alternate building topology (starting with a single-exit building), and (3) gives
total-lockout scenarios at least one row instead of zero — all three are exactly the
`docs/architecture/predictive_dataset_campaign_v1.md` §14 recommendations this modeling
milestone's own error analysis now independently confirms are the highest-value next fixes.

## Full-suite result (Phase 16)

See commit message for the exact final test count and confirmation of zero regressions
against the 4440-test baseline entering this milestone.
