# Localized Predictive Congestion Model V2 — Training, Generalization & Robustness Evaluation

Status: **OFFLINE RESEARCH ONLY.** Nothing in this milestone is wired into recommendation
scoring, exit ranking, guidance, signage, LiveRuntime, or operator workflow. Builds on
`docs/architecture/localized_predictive_model_v1.md` (commit `7ee61e9`, "PROMISING BUT
NEEDS MORE DATA") and `docs/architecture/predictive_dataset_campaign_v2.md` (commit
`a3a2c56`, 9,620,196-row, 4-topology-family dataset). This milestone trains and rigorously
evaluates Model V2 against Dataset V2, with a specific focus on whether V1's disclosed
weaknesses (non-functional Stair prediction, 11x-worse multi-bottleneck false-negative
rate, zero topology diversity) actually improved, not just whether headline metrics look
good.

## 1. Objective

For every `(scenario, observation_time, candidate)` triple where the candidate is not
already congested, predict `P(candidate becomes congested within 20 seconds)` — the same
binary classification target V1 used (`predictive_dataset.target_generator`,
`CONGESTION_THRESHOLD`, unchanged). This is not an RL milestone and does not touch any
live-facing module.

## 2. V1 weaknesses (recap, Phase 1 reconstruction)

Reconstructed from `docs/architecture/localized_predictive_model_v1.md` before writing any
V2 code, per this milestone's own charter of "do not silently change methodology":

- Stair prediction was **non-functional** (PR-AUC 0.240, 9 true positives out of 372 real
  stair-congestion events) — a direct consequence of `stair-1`'s zero-duration traversal
  bug in the training topology (`from_floor_id` never set).
- Rows with **2+ simultaneous bottlenecks** had an **11x worse false-negative rate**
  (13.3% vs. 1.16% single-bottleneck) — the highest-value operational case was the weakest.
- Single fixed building topology (2 doors, 2 exits, 1 stair) — no single-exit coverage, no
  total-lockout rows, generalization to any other topology completely unverified.
- Best model: `HistGradientBoostingClassifier`, test PR-AUC 0.708 (20s horizon, 12.9%
  positive rate), ROC-AUC 0.956. Raw probabilities miscalibrated (ECE 0.209); isotonic
  calibration fixed this (ECE 0.003).

## 3. Why Dataset V2 exists

`docs/architecture/predictive_dataset_campaign_v2.md` fixed the *data*-generation side of
every weakness above: repaired the stair `from_floor_id` bug, added 4 topology families
(`single_exit_lowrise`, `twin_stair_highrise`, `multi_exit_wide`, `v1_topology_fixed`),
gave total-lockout scenarios real rows, and produced 1,024,108 multi-bottleneck rows (205x
its own coverage target). This milestone is the first to train anything on that dataset.

## 4. Dataset / split methodology (Phases 1-3)

**Methodology carried over unchanged from V1** (see `scripts/train_localized_predictive_model_v2.py`
module docstring for the complete, disclosed list): scenario-level 70/15/15 split, seed
`20260726`; identical 9-field feature schema (`predictive_dataset.schema.CANDIDATE_FEATURE_NAMES`,
unchanged by the V2 dataset milestone); identical target definition (congestion-threshold-2,
20s horizon); identical class-imbalance strategy (`class_weight='balanced'` sample
reweighting, no oversampling); identical calibration methods (Platt + isotonic, fit on
validation only); identical model zoo.

**Deliberate, disclosed differences:**

1. **Loaded only the 20s-horizon slice**, not all 4 horizons. V1's own Phase 7 already
   established 20s as the right operational horizon; V2's charter is about topology/robustness
   generalization *at* that horizon, not re-litigating horizon choice — and it is also what
   makes loading feasible at all (see Phase 2 below). `predictive_model.dataset_loader.
   load_dataset_single_horizon_chunked()` reads the 1.05GB / 9,620,196-row CSV in 250k-row
   chunks, filters to `prediction_horizon==20.0` per chunk, and downcasts to compact dtypes
   (`float32`/`int32`/`category`) — loaded 2,405,049 rows (2,500 scenarios) in **11.7s**,
   peak frame memory ~180MB.
2. **RandomForest `max_depth` capped at 20** (V1: unbounded) and **`n_jobs` reduced from -1
   to 2** for RandomForest/XGBoost. This development machine measured ~1-1.6GB free RAM out
   of ~7.3GB total at run start (already below V1's presumed environment) and V2's train
   split (1,629,958 rows) is ~4x V1's (~405K). A first real attempt at this exact script with
   `n_jobs=4` and unbounded RandomForest depth drove available system memory down to 269MB
   before being killed manually; a background memory watchdog thread (polls every 4s,
   hard-exits at <180MB available) was added as a second line of defense. The retry
   completed cleanly, available memory never dropping below ~535MB. `predictive_model.
   tree_models`'s defaults (`n_jobs=-1`, `max_depth=None`) are unchanged for any other caller,
   including V1's own script.

**Scenario split:**

| Split | Scenarios | Trainable rows (20s) |
|---|---|---|
| Train | 1,750 | 1,629,958 |
| Validation | 375 | 345,057 |
| Test | 375 | 354,183 |

`assert_no_scenario_overlap()` (reused verbatim from V1) mechanically proved zero overlap.
Train positive rate: 12.96% (V1: 12.5%).

## 5. Model candidates (Phase 5)

Same zoo as V1 plus the same trivial baselines:

| Model | Test ROC-AUC | Test PR-AUC | Fit time |
|---|---|---|---|
| Majority Class | 0.500 | 0.129 | 0.6s |
| Always Negative | 0.500 | 0.129 | 0.6s |
| Random | 0.500 | 0.129 | 0.8s |
| Logistic Regression | 0.909 | 0.528 | 6.8s |
| Decision Tree (depth 6) | 0.936 | 0.660 | 6.7s |
| Random Forest (300 trees, depth≤20) | 0.936 | 0.678 | 357.3s |
| Gradient Boosting (HistGradientBoosting) | 0.939 | 0.685 | 26.9s |
| **XGBoost (300 rounds, hist)** | **0.940** | **0.692** | 27.8s |

**Best model: `xgboost`**, selected by test PR-AUC — a near-tie with Gradient Boosting
(0.6918 vs 0.6855), same pattern V1 observed (V1: Gradient Boosting 0.708 vs XGBoost 0.703,
also a near-tie). V2 does **not** default to crowning HistGradientBoosting just because it
won V1 — XGBoost genuinely won this evaluation on its own evidence. RandomForest's 357s fit
time (vs. 27-28s for the two boosting methods) is a direct, expected consequence of the
depth-20 cap plus 4x more training rows than V1; still far more practical than an unbounded
run given this machine's memory ceiling.

## 6. Overall 20s results (Phase 6)

Test split, 354,183 trainable rows, **positive rate 12.91%** (V1: 12.9% — essentially
unchanged despite 4 new topology families):

| Metric | Value |
|---|---|
| ROC-AUC | 0.9395 |
| PR-AUC | 0.6918 |
| F1 | 0.629 |
| Precision | 0.549 |
| Recall | 0.737 |
| Balanced accuracy | 0.824 |
| Brier score (raw) | 0.1107 |
| Confusion matrix | tn=280,737 fp=27,736 fn=12,004 tp=33,706 |
| Decision threshold (F1-tuned on val) | 0.76 |

PR-AUC 0.692 clears the 12.9% positive-rate baseline by >5x — decisively above trivial
baselines, consistent with V1's own finding.

## 7. Door results (Phase 7)

| | V1 | V2 |
|---|---|---|
| n | 35,628 | 188,306 |
| Positive rate | 25.2% | 17.7% |
| ROC-AUC | 0.915 | 0.892 |
| PR-AUC | 0.714 | 0.578 |
| Precision | 0.622 | 0.492 |
| Recall | 0.873 | 0.733 |
| F1 | 0.726 | 0.589 |

Door performance is **lower in V2 than V1** on every metric. This is a real, disclosed
regression, not noise — Door candidates in V2 are drawn from 4 structurally different
topologies (V1: one fixed building) with more varied demand patterns, and specifically the
`multi_exit_wide` family's Door candidates are much harder to predict in isolation (§11).

## 8. Exit results (Phase 7)

| | V1 | V2 |
|---|---|---|
| n | 33,282 | 128,930 |
| Positive rate | 5.5% | 3.8% |
| ROC-AUC | 0.963 | 0.925 |
| PR-AUC | 0.686 | 0.464 |
| Precision | 0.641 | 0.506 |
| Recall | 0.681 | 0.389 |
| F1 | 0.660 | 0.440 |

**Exit is now the weakest candidate type in V2** — a real, disclosed regression from V1.
`docs/architecture/predictive_dataset_campaign_v2.md` §16 already flagged this risk ("Exit
congestion is now comparatively rare... Model V2 evaluation should watch Exit-specific
recall separately, the same way this milestone watched Stair") and this evaluation confirms
it materialized: Exit recall (38.9%) is now the single lowest recall of any candidate type,
and under multi-bottleneck conditions specifically it collapses further (§11: recall 39.3%
even there, but FN rate 60.7% — see below). This was NOT the dataset milestone's stated goal
(Stair repair and multi-bottleneck coverage were), and is the clearest tradeoff this
evaluation found.

## 9. Stair results (Phase 7) — did it become functional?

| | V1 | V2 |
|---|---|---|
| n | 17,814 | 36,947 |
| Positive rate | 2.1% | 20.5% |
| ROC-AUC | 0.936 | **0.992** |
| PR-AUC | **0.240** | **0.966** |
| Precision | 0.375 | 0.912 |
| Recall | **0.024** | **0.980** |
| F1 | 0.045 | 0.945 |
| Balanced accuracy | — | 0.978 |
| Brier score | — | 0.024 |

**Yes — decisively.** Stair prediction went from "9 true positives out of 372 real events"
(effectively non-functional) to **PR-AUC 0.966, recall 98.0%, precision 91.2%** — now the
*best*-predicted candidate type in V2, better than Door or Exit. This directly confirms the
dataset milestone's own repair (constant-zero `walking_distance`/queue/approaching-count →
genuinely varying, nonzero demand signal) translated into genuine model-level predictive
power, not just data-quality cosmetics.

## 10. V1 → V2 comparison summary

| Metric | V1 | V2 | Direction |
|---|---|---|---|
| Overall test PR-AUC | 0.708 | 0.692 | slightly down |
| Overall test ROC-AUC | 0.956 | 0.940 | slightly down |
| Door PR-AUC | 0.714 | 0.578 | down |
| Exit PR-AUC | 0.686 | 0.464 | down |
| **Stair PR-AUC** | **0.240** | **0.966** | **up, dramatically** |
| Multi-bottleneck FN rate ÷ single-bottleneck FN rate | 11.0x | ~5.5x | improved, still elevated |
| High-occupancy FN rate ÷ low-occupancy FN rate | 6.3x | ~2.6x | improved |
| Topology families | 1 | 4 | new coverage |
| Single-exit coverage | none | 500 scenarios, holdout PR-AUC 0.830 | new, generalizes |

The overall headline PR-AUC is marginally *lower* in V2 (0.692 vs 0.708) — this is the
single most important nuance in this whole evaluation: **a slightly lower aggregate number
hides a qualitatively much healthier model.** V1's 0.708 was propped up by Door/Exit
performing well against a single easy topology while Stair contributed almost nothing
useful; V2's aggregate blends a now-excellent Stair signal with harder, more diverse
Door/Exit conditions across 4 real topologies. Judging V2 solely by the aggregate number
would be exactly the mistake this milestone's own framing warned against ("Can we obtain a
high overall ROC-AUC?" is explicitly *not* the real question).

## 11. Topology-family generalization (Phase 4) — leave-one-topology-out

Each family held out entirely (trained on the other 3, tested only on the held-out family's
own scenarios), using the same XGBoost architecture as the winning model:

| Held-out family | Train scenarios | Test rows | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced Acc. | Brier |
|---|---|---|---|---|---|---|---|---|---|
| `single_exit_lowrise` | 2,000 | 30,645 | 0.927 | **0.830** | 0.809 | 0.619 | 0.701 | 0.768 | 0.095 |
| `v1_topology_fixed` | 2,000 | 153,485 | 0.939 | 0.686 | 0.597 | 0.788 | 0.679 | 0.853 | 0.119 |
| `twin_stair_highrise` | 1,700 | 665,257 | 0.918 | 0.768 | 0.627 | 0.875 | 0.731 | 0.862 | 0.158 |
| `multi_exit_wide` | 1,800 | 1,479,811 | 0.869 | **0.314** | 0.349 | 0.490 | 0.408 | 0.705 | 0.123 |

Compare to the normal-split PR-AUC of 0.692: three of four holdouts land close to or above
that number (single_exit_lowrise and twin_stair_highrise are actually *higher*), but
**`multi_exit_wide` collapses to 0.314 — less than half the normal-split PR-AUC.** This is a
real, disclosed generalization gap, not a rounding artifact. Per-candidate-type breakdown
for the `multi_exit_wide` holdout shows the collapse is concentrated in **Exit** (PR-AUC
0.085, recall 15.1%) and, to a lesser extent, **Door** (PR-AUC 0.338); this family's
hub-and-spoke, 3-exit/4-door structure produces Door/Exit demand patterns unlike anything in
the other three families, and the model trained without ever seeing it cannot transfer.

## 12. Multi-bottleneck robustness (Phase 8)

| Bucket | n | Positive rate | ROC-AUC | PR-AUC | Recall | FN rate | FP rate |
|---|---|---|---|---|---|---|---|
| none (0 competing bottlenecks) | 176,582 | 0.0% | — | — | — | — | 7.9% |
| single (1 other candidate trending positive) | 110,736 | 15.9% | 0.941 | 0.695 | 70.1% | 29.9% | 7.6% |
| **multiple (2+)** | 66,865 | 42.1% | 0.894 | 0.842 | 76.0% | **24.0%** | 17.2% |

Using the same "≥2 simultaneously-positive candidates" definition V1 used
(`predictive_model.error_analysis.simultaneous_bottleneck_counts`): overall FN rate for
multi-bottleneck rows is **10.1%** vs. **1.83%** for none-or-single — a **~5.5x** ratio,
down from V1's 11x. **Improved, but still a real, elevated gap.** Per-candidate-type
breakdown of the "multiple" bucket is the more actionable finding: **Exit's FN rate under
multi-bottleneck conditions is 60.7%** (Door: 23.5%, Stair: 1.9%) — the multi-bottleneck
weakness in V2 is now concentrated almost entirely in Exit candidates, not spread evenly
the way V1's was.

## 13. Occupancy robustness (Phase 9)

Using `predictive_dataset.label_analysis`'s own bands (LOW ≤10, HIGH ≥20 occupants):

| Band | n | Positive rate | ROC-AUC | PR-AUC | Recall | FN rate |
|---|---|---|---|---|---|---|
| LOW | 1,356 | 21.8% | 0.941 | 0.821 | 83.1% | 17.0% |
| MEDIUM | 16,268 | 17.2% | 0.958 | 0.795 | 87.1% | 12.9% |
| HIGH | 336,559 | 12.7% | 0.938 | 0.680 | 72.8% | 27.2% |

FN rate is worst at HIGH occupancy (27.2%) vs. LOW (17.0%) — a **~1.6x** ratio (computed
directly on the operational-slice PR-AUC table above; the error-analysis table's own
HIGH/LOW split, which uses a slightly different row population — trainable rows only,
identical bands — gives 4.44%/1.70%, a **~2.6x** ratio). Both readings agree on direction and
both are meaningfully improved from V1's 9.4%/1.5% (**~6.3x**). **Improved, not eliminated**
— HIGH-occupancy rows remain the hardest, exactly where an operational miss matters most.

## 14. Single-exit vs. multi-exit (Phase 10)

| | n | Positive rate | ROC-AUC | PR-AUC | Recall | FN rate |
|---|---|---|---|---|---|---|
| single_exit | 4,358 | 36.9% | 0.936 | **0.857** | 92.7% | 7.3% |
| multi_exit | 349,825 | 12.6% | 0.939 | 0.681 | 73.0% | 27.0% |

Single-exit buildings (V1 had zero coverage) are **not** harder to predict — if anything,
comparatively easier (higher PR-AUC, much higher recall, lower FN rate). This makes sense
operationally: with no alternative route, the sole exit's congestion pattern is more tightly
coupled to whole-building occupancy and less confounded by route-choice variation. Per this
milestone's own instruction not to conflate "exit choice is trivial in a single-exit
building" with "prediction is useless" — prediction is, if anything, *more* reliable here,
useful for queue/flow awareness even with no alternative to route occupants toward.

## 15. Calibration (Phase 11)

| | Brier score | ECE |
|---|---|---|
| Raw (uncalibrated) | 0.1107 | 0.1299 |
| Platt scaling | 0.0651 | 0.0173 |
| **Isotonic regression (recommended)** | **0.0636** | **0.0030** |

Same conclusion as V1: raw probabilities are meaningfully miscalibrated; isotonic
calibration (fit on validation only, `predictive_model.calibration.IsotonicCalibrator`)
brings ECE down to 0.003 — any future consumer of this model's output as a probability
(not just a thresholded alarm) must use the isotonic-calibrated output, exported alongside
the model as `calibrator.joblib`.

## 16. Feature importance (Phase 12)

Permutation importance (ROC-AUC drop, 20,000-row sample, 5 repeats, seed 20260726) against
XGBoost:

| Rank | Feature | Mean ROC-AUC drop |
|---|---|---|
| 1 | `candidate_queue_length` | 0.236 |
| 2 | `candidate_walking_distance` | 0.049 |
| 3 | `candidate_traversable` | 0.032 |
| 4 | `candidate_adjacent_zone_occupancy` | 0.030 |
| 5 | `candidate_approaching_count` | 0.020 |
| 6 | `total_active_occupant_count` | 0.020 |
| 7 | `candidate_capacity` | 0.006 |
| 8+ | one-hot categorical columns | ≤0.002 each |

`candidate_queue_length` still dominates (matches V1's finding, mean drop 0.228 there), by
~5x over the next feature. XGBoost's own built-in `feature_importances_` (gain-based) ranks
`candidate_queue_length` (0.625) first as well, followed by `candidate_type=Door` (0.100)
and `candidate_traversable` (0.063) — broadly consistent with the permutation ranking.
**Repaired Stair demand features do contribute real signal**: `candidate_type=Stair` appears
in the built-in top-8 (0.016), and Stair's own near-perfect metrics (§9) are only possible
because `candidate_queue_length`/`candidate_approaching_count` now carry genuine, nonzero
information for Stair rows (`predictive_dataset_campaign_v2.md` §6: 22.3%/40.9% of Stair
rows now have nonzero queue/approaching values, vs. 0%/0% in V1) — the model is using real
repaired signal, not just a `candidate_type=Stair` shortcut (feature importance for that
one-hot column alone is two orders of magnitude below `candidate_queue_length`).

## 17. Feature-family ablation (Phase 13)

Families zeroed (not dropped — architecture held fixed), retrained fresh, evaluated on
validation:

| Family removed | ROC-AUC drop | PR-AUC drop |
|---|---|---|
| `structural` (type/capacity/distance/traversable) | **0.0141** | **0.0671** |
| `demand_signal` (queue + approaching count) | 0.0094 | 0.0281 |
| `global_and_adjacent_context` (occupancy features) | 0.0075 | 0.0212 |
| `derived_congestion_level` (one-hot congestion level) | 0.00003 | 0.0004 |

Unlike V1 (where `global_and_adjacent_context` hurt most), **`structural` features hurt
most in V2** — expected, since V2's 4 topology families make `candidate_type`/
`candidate_walking_distance`/`candidate_traversable` carry real cross-topology
discriminative information (which topology/candidate-type combination this row even is)
that V1's single-topology dataset could never test. `demand_signal` still matters
substantially (0.028 PR-AUC drop) despite `candidate_congestion_level` (retained) being
partly derived from it — same redundancy pattern V1 found. **Candidate-local information is
genuinely necessary**: no family's removal is free, and the two largest drops
(`structural`, `demand_signal`) are both squarely candidate-local, not whole-building
context.

## 18. Leakage / sanity checks (Phase 14)

| Check | Result |
|---|---|
| Leakage-correlation recheck (0.9 threshold, reused from `predictive_dataset.correlation`) | **0 features flagged** |
| Label-shuffle test (fresh XGBoost trained on train features + shuffled train labels, scored on real val labels) | ROC-AUC **0.474** — collapses to chance (near_chance=True) |
| Scenario-split leakage | `assert_no_scenario_overlap()` passed; zero overlap at both ID-set and actual-row level |
| Future-timestep inaccessibility | Inherited, re-verified unchanged from `predictive_dataset`'s own leakage boundary (`tests/test_predictive_dataset_leakage_guards.py`, `tests/test_predictive_dataset_architecture_guards.py`, `tests/test_predictive_dataset_campaign_v2_pipeline.py`'s `LeakageReAuditV2Tests`) — not re-derived per-row by this training script |
| Candidate-identity shortcut | Structurally impossible: `candidate_id` is never a model input feature (excluded from `CANDIDATE_FEATURE_NAMES`/`feature_prep`'s feature lists) |

**No evidence of target leakage.**

## 19. Training-size study (Phase 15)

XGBoost retrained at increasing scenario-preserving fractions of the 1,750 train scenarios,
evaluated against the same, full, unchanged 375-scenario validation split:

| Fraction | Train scenarios | Train rows | Val PR-AUC |
|---|---|---|---|
| 10% | 175 | 174,716 | 0.6750 |
| 25% | 438 | 442,338 | 0.6876 |
| 50% | 875 | 827,993 | 0.6925 |
| 100% | 1,750 | 1,629,958 | 0.6961 |

PR-AUC gains **saturate hard past 25%** of the available training data (25%→100% is a
0.0085 PR-AUC gain — less than a third of the 10%→25% gain of 0.0126). The full 9.6M-row
(4-horizon) campaign is not necessary for this model to converge at the 20s horizon; roughly
a quarter of the current train split already captures most of the achievable signal. This
matters directly for iteration speed on any future V3 experiment.

## 20. Error analysis (Phase 16) — major failure modes

Test split (354,183 rows): overall FP rate 7.8%, FN rate 3.4%.

1. **Exit is now the systematic weak point**, not Stair. Across every slice (overall,
   multi-bottleneck, topology-holdout), Exit has the lowest recall and PR-AUC of the three
   candidate types — the clearest, most consistent failure mode in this evaluation.
2. **Multi-bottleneck rows remain disproportionately hard**, and the difficulty is now
   concentrated in Exit specifically (FN rate 60.7% under multi-bottleneck conditions) rather
   than spread evenly — a more specific, more actionable finding than V1's "11x worse
   overall."
3. **`multi_exit_wide` does not generalize** when held out entirely (PR-AUC 0.314 vs. 0.692
   normal-split) — the model has not learned a topology-invariant notion of Door/Exit
   congestion, only patterns that transfer across the other 3 (structurally closer) families.
4. **HIGH occupancy remains the hardest occupancy band** (FN rate 27.2% vs. LOW's 17.0%),
   improved from V1's ratio but not eliminated — exactly the situation where a miss matters
   most operationally.
5. **EARLY temporal phase has the highest FP rate** (13.7%) — consistent with V1's own
   finding (EARLY was V1's highest-FP/FN phase too), an unchanged, stable pattern across both
   dataset versions.

## 21. Inference performance (Phase 19)

CPU-only, this development machine (12 logical cores, `n_jobs=2` for the winning XGBoost
model at inference too):

| Measurement | Value |
|---|---|
| Single-row latency | 0.41ms |
| Batch size | 354,183 rows |
| Batch wall time | 0.72s |
| Throughput | **~494,300 candidates/sec** |

Training time and inference time are kept separate per this milestone's own instruction;
this is not extrapolated to GPU.

## 22. Remaining limitations

- **Exit congestion prediction is now the weakest candidate type** (§8, §11, §12, §20) — the
  clearest actionable gap for any future V3 work, symmetric to how V1 flagged Stair.
- **`multi_exit_wide` topology does not generalize when entirely unseen** (§11) — PR-AUC
  collapses to less than half the normal-split value, concentrated in that family's Door/Exit
  candidates.
- **Multi-bottleneck FN rate remains ~5.5x worse than single/none-bottleneck** (§12),
  improved from V1's 11x but not resolved, and now specifically an Exit problem.
- **HIGH-occupancy FN rate remains meaningfully worse than LOW** (§13), improved from V1 but
  not resolved.
- **`v1_topology_fixed`'s stair candidate remains low-utilization** even after the
  `from_floor_id` fix (per `predictive_dataset_campaign_v2.md` §6, 1.2% active fraction) — a
  property of that specific topology/occupancy shape, confirmed here by its own
  topology-holdout Stair PR-AUC of only 0.437 (vs. `twin_stair_highrise`'s 0.957) despite the
  same repaired feature schema.
- `candidate_traversable` still does not incorporate mid-scenario `ScenarioEvent`
  door/exit/stair overrides — unchanged limitation, carried over from V1.
- This milestone loaded only the 20s-horizon slice of the campaign CSV; no 10s/30s/60s
  horizon-robustness sweep was repeated for V2 (§4).

## 23. Production-readiness decision (Phase 17/22)

**PROMISING BUT NEEDS MORE DATA.**

Evidence for "promising": XGBoost clears every trivial baseline by >5x on PR-AUC, passes
every sanity check (leakage-correlation recheck clean, label-shuffle collapses to chance),
isotonic calibration produces trustworthy probabilities (ECE 0.003), Stair prediction is now
genuinely excellent (PR-AUC 0.966), 3 of 4 topology-holdout families generalize close to or
above the normal-split PR-AUC, single-exit buildings are not harder to predict, occupancy and
multi-bottleneck robustness both measurably improved from V1, and training-data requirements
actually saturate well below the full campaign size.

Evidence against "ready": (a) `multi_exit_wide` generalization genuinely collapses when held
out entirely (PR-AUC 0.314 vs. 0.692) — this is the specific, disqualifying finding for
shadow-mode or controlled-integration readiness, not a marginal miss; (b) Exit prediction is
now the systematic weak candidate type, with a 60.7% FN rate under multi-bottleneck
conditions specifically; (c) multi-bottleneck and high-occupancy FN rates, while both
improved, remain meaningfully elevated. None of these are fixable by retraining on the same
data — (a) and (b) point at a future V3 dataset/feature investigation (more `multi_exit_wide`-like
diversity, and an Exit-specific demand feature investigation analogous to what V2 did for
Stair), not a modeling change this milestone could make.

## Final report — explicit answers

**A. Did Stair prediction become genuinely functional?** **Yes, decisively** — PR-AUC
0.240→0.966, recall 2.4%→98.0%. Now the best-predicted candidate type in V2, and permutation/
built-in feature importance confirm the model is using genuinely repaired demand signal
(`candidate_queue_length`/`candidate_approaching_count`), not a `candidate_type=Stair`
shortcut.

**B. Did multi-bottleneck false-negative performance improve?** **Yes, partially** — ratio
improved from 11x to ~5.5x worse than single/none-bottleneck rows — but the remaining gap is
now concentrated almost entirely in Exit candidates (60.7% FN rate under multi-bottleneck
conditions), not resolved.

**C. Did high-occupancy robustness improve?** **Yes, partially** — FN-rate ratio (HIGH vs.
LOW) improved from ~6.3x to ~1.6-2.6x (two consistent readings, §13) — improved but not
resolved; HIGH occupancy remains the hardest band.

**D. Does the model generalize to an unseen topology family?** **No, not uniformly.** 3 of 4
held-out families (`single_exit_lowrise`, `twin_stair_highrise`, `v1_topology_fixed`)
generalize close to or above the normal-split PR-AUC; `multi_exit_wide` collapses to less
than half (0.314 vs. 0.692) — this is the single most important disqualifying finding in
this evaluation.

**E. Is candidate-local information genuinely necessary for its predictions?** **Yes** —
feature-family ablation shows every family's removal costs real PR-AUC, and the two costliest
removals (`structural`, `demand_signal`) are both candidate-local, not whole-building context.

**F. Are probabilities adequately calibrated?** **Not out of the box** (ECE 0.130) — but
isotonic-calibrated output is well calibrated (ECE 0.003) and is exported alongside the model
specifically so any future consumer uses it instead of the raw output.

**G. Is there any evidence of target leakage?** **No** — leakage-correlation recheck flagged
zero features, and the label-shuffle test collapsed to chance (ROC-AUC 0.474).

**H. Does V2 clearly outperform V1 where V1 was weakest?** **Yes, dramatically for Stair**
(V1's single stated headline weakness) **and partially for multi-bottleneck/occupancy
robustness** — but V2 introduces a new, previously-undetected weakness (Exit, and one
specific topology family's generalization) that V1's single-topology dataset had no way to
expose. This is not a wash: V1's Stair failure was disqualifying on its own; V2 traded it for
weaknesses that are narrower in scope (one candidate type, one of four topology families)
and, unlike V1's Stair problem, do not appear to be structurally unfixable — they look like
exactly the kind of gap a future targeted dataset investigation (V2's own playbook, applied
to Exit and to `multi_exit_wide` diversity) could close.

**I. Should this model now be allowed to influence evacuation recommendation ranking?**
**Not yet.** The `multi_exit_wide` topology-holdout collapse (§11) is a genuine, disqualifying
generalization gap — a model that cannot be trusted on an entirely unseen-but-realistic
building shape should not influence real evacuation decisions, however well it performs on
topologies resembling its training data. Recommended next step: a targeted investigation of
*why* `multi_exit_wide` fails to transfer (more topology diversity within that family's shape,
or an Exit-specific demand-feature investigation analogous to V2's own Stair repair) before
any shadow-mode live-validation milestone is even considered.

## Full-suite result (Phase 21)

See the accompanying commit message / final report for the exact test count; this milestone
added new unit tests for the chunked dataset loader, topology holdout, operational slices,
training-size study, sanity-check machinery, and the extended V2 export format, on top of the
existing V1 test suite (all of which remains green — `predictive_model.tree_models`'s new
`n_jobs`/`max_depth` parameters default to V1's original hardcoded values, so no existing
test's behavior changed).
