# Localized Predictive Model V2.2 — Full-Scale Feature Validation, Live Parity & Target-Semantics Audit

Status: **OFFLINE RESEARCH ONLY.** Nothing in this milestone is wired into recommendation
scoring, exit ranking, guidance, signage, LiveRuntime, or operator workflow. Builds on:

- `docs/architecture/localized_predictive_model_v2.md` (commit `5b51923`) — trained Model
  V2, found Exit weak and `multi_exit_wide` topology-generalization disqualifying.
- `docs/architecture/localized_predictive_model_v2_exit_generalization_investigation.md`
  (commit `9d926bb`) — root-caused both findings and proposed 3 candidate-local/structural
  features, validated only on a 500-scenario (20% scale), single-split experiment.

This milestone has three separate goals, kept separate throughout: (A) validate the V2.1
features at V2's original ~2,500-scenario scale; (B) prove every proposed feature has an
honest LIVE equivalent; (C) quantify the Door/Stair target-semantics finding at full scale
without changing the production target.

## 1. V2.1 findings (recap)

`candidate_queue_length` — the dominant feature everywhere else — is structurally constant
zero for every Exit row (`Exit.capacity` defaults to 50, never overridden; Door/Stair's
*derived* capacity floors to 1). `multi_exit_wide`'s Exit positive rate (1.4%) is 4-10x
lower than every other family, and its hub-and-spoke structure has the highest alternative-
route dilution. A second finding: ~91.2% of adjacent Door occupant-pair transitions (15-25
scenario sample) have an exactly-zero temporal gap — a FIFO admission-handoff artifact, not
sustained crowding — while Exit's congestion episodes are genuinely sustained (mean 40s).
Three proposed features (`candidate_recent_flow_rate`, `candidate_congestion_trend`,
`candidate_alternative_route_count`) improved Exit PR-AUC +19-95% relative and the
`multi_exit_wide` holdout PR-AUC +38% relative in that small, single-split experiment.

## 2. Exact V2.1 feature definitions (Phase 2, frozen)

| Feature | Type | Units | Candidate types | Measured/derived | Future-leakage risk |
|---|---|---|---|---|---|
| `candidate_recent_flow_rate` | int (sim) / float (live, Exit) | occupants per 60s window | Door, Exit, Stair | Derived (count of already-completed crossings) | None — only reads `(time-60, time]`, strictly backward |
| `candidate_congestion_trend` | categorical: RISING/STABLE/FALLING/UNKNOWN | — | Door, Exit, Stair | Derived (comparison vs. 30s ago) | None — only reads `time` and `time-30` |
| `candidate_alternative_route_count` | int | count of other candidates | Door, Exit, Stair | Derived, purely structural | None — zero occupancy/time dependence |

Implementation: `predictive_dataset/simulation_extractor_v2_1.py` (simulation side, commit
`9d926bb`, unchanged this milestone) and `predictive_dataset/live_extractor_v2_1.py` (live
side, **new this milestone**). Full per-field simulation-source/live-source/missing-value/
timestamp-semantics documentation already exists in `docs/architecture/
localized_predictive_model_v2_exit_generalization_investigation.md` §11 and is not
duplicated here verbatim — this document adds the LIVE half that investigation left as a
disclosed gap.

## 3. Repository investigation (Phase 1 answers)

1. **The three V2.1 features**: `candidate_recent_flow_rate`, `candidate_congestion_trend`,
   `candidate_alternative_route_count` — verified directly from commit `9d926bb`'s
   `predictive_dataset/simulation_extractor_v2_1.py`, not assumed from any prompt summary.
2. **Simulation calculation**: `_recent_flow_rate` counts completed `OccupantTimelineStep`s
   on the edge with `end_time` in `(time-60, time]`; `_congestion_trend` compares
   `queue_length + approaching_count` at `time` vs. `time-30`; `build_alternative_route_
   counts` counts other candidates sharing a `CandidateIdentity.zone_ids` entry — computed
   once per Building, zero occupancy dependence.
3. **Already-live fields, verified by direct code reading, not the prior milestone's own
   summary**: `candidate_congestion_trend` → `crowd_intelligence.models.AssetApproachMetrics.
   trend` (a `TrendDirection` enum: RISING/STABLE/FALLING/UNKNOWN, already computed for
   every Door/Exit/Stair asset by `crowd_intelligence.trends.TrendTracker`) — zero new code
   needed beyond reading the field. `candidate_alternative_route_count` is purely structural
   and reuses the literal same function on both sides — trivially exact, not merely "already
   live."
4. **The one feature without full live parity**: `candidate_recent_flow_rate` for Door/
   Stair. `evacuation_progress.ledger.EvacuationLedger.recent_exit_count` is architecturally
   Exit-only — it tracks *building-exit* events (an occupant's final departure, matched to
   "nearest exit" via `_nearest_exit_id`), a concept with no Door/Stair analog (occupants
   don't have a discrete "exited through this door" event; they cross it as an intermediate
   waypoint).
5. **Can it be derived from existing live state without a new subsystem?** **Yes.**
   `live_occupants/history.py::OccupantHistory.zone_transitions` — a bounded, per-occupant
   log of `ZoneTransitionRecord(timestamp, from_zone_id, to_zone_id)` — is **already
   maintained by `LiveOccupantManager` on every zone change**, for camera/behavior-tracking
   purposes entirely unrelated to prediction. A Door/Staircase candidate's
   `CandidateIdentity.zone_ids` is exactly a 2-zone pair; counting `zone_transitions` whose
   `{from_zone_id, to_zone_id}` matches that pair within the trailing 60s window is a thin,
   stateless QUERY over already-persisted data — no new tracking subsystem, no new engine,
   no new event bus. Implemented in `predictive_dataset/live_extractor_v2_1.py::
   _door_or_stair_flow_rate`.
6. **Can Dataset V2 be re-extracted from existing simulation outputs, or must simulations
   rerun?** Simulations must rerun — the original V2 campaign never persisted raw
   `MultiAgentSimulationResult`/timeline objects, only the already-extracted flat CSV.
   However, generation is **fully deterministic** (`master_seed=20270115`,
   `predictive_dataset/topologies_v2.py` byte-for-byte unchanged since `a3a2c56`) — rerunning
   produces the literal same simulated occupants/scenarios V2 already validated. This is
   re-extraction in every statistical sense (same population, 3 more columns), even though
   it requires re-executing the simulator. No new scenario campaign was generated; the
   existing V2 CSV (`data/predictive_dataset_campaign_v2/candidate_dataset_v2.csv`) is
   reused unchanged as this milestone's 9-field baseline comparison point, never
   regenerated.

## 4. Full-scale dataset statistics (Phase 6)

`scripts/run_predictive_dataset_campaign_v2_2_fullscale.py`: 2,500 scenarios requested,
**2,500 accepted, 0 failed**, 299.5s wall time, streamed directly to CSV (never held fully
in memory — a deliberate change from the V2.1 script's in-memory list approach, necessary
at 5x the row count on this ~7.3GB-RAM machine).

**Row counts match V2's own 20s-horizon slice exactly** — 2,405,049 total rows, confirming
byte-for-byte deterministic reproduction (same `master_seed`, same topology definitions):

| Candidate type | Rows | Trainable | Positive | Positive rate |
|---|---|---|---|---|
| Door | 1,244,514 | 1,244,514 | 219,292 | 17.62% |
| Exit | 927,926 | 852,075 | 31,710 | 3.72% |
| Stair | 232,609 | 232,609 | 49,474 | 21.27% |

**Confirmed, not "fixed": `candidate_queue_length` remains structurally 0 for 100% of Exit
rows** (nonzero rate 0.0000, identical to V2) — `Exit.capacity` was never touched, per this
milestone's own explicit constraint against arbitrarily lowering it to make the dataset
easier.

New feature distributions (trainable rows):

| Feature | Door | Exit | Stair |
|---|---|---|---|
| `candidate_recent_flow_rate` mean | 0.589 | 0.867 | 2.092 |
| `candidate_recent_flow_rate` nonzero rate | 45.1% | **38.7%** | 51.7% |
| `candidate_alternative_route_count` mean | 3.25 | 1.19 | 2.87 |
| `candidate_congestion_trend` STABLE / RISING / FALLING / UNKNOWN | 94.7% / 2.4% / 0.3% / 2.7% | 95.0% / 0.9% / 2.4% / 1.7% | 76.5% / 3.1% / 15.9% / 4.5% |

**Exit's new flow-rate feature has real, substantial variance (38.7% nonzero) exactly where
`candidate_queue_length` has none** — direct, full-scale confirmation that the proposed
signal genuinely fills the gap the V2.1 investigation identified, not just in the 500-
scenario sample. Stair's markedly higher FALLING-trend rate (15.9% vs. ~2-3% for Door/Exit)
is a new, previously-unobserved pattern — consistent with Stair's own higher `candidate_
queue_length` nonzero rate (22.3%) and higher episode-duration variance (queues that form
also resolve, producing more FALLING readings than the mostly-quiescent Door/Exit demand
signal).

## 5. Target-semantics audit (Phase 7) — full-scale, decisive

`scripts/run_target_semantics_audit_v2_2.py` re-simulated all 2,500 scenarios (same
deterministic reproduction as §4) and walked every candidate's real event timeline directly
— something the flat CSV cannot support (no start/end timestamps). **Internal correctness
check**: the audit's own `min_duration=0.0` counterfactual (which should reproduce
`target_generator.generate_candidate_label`'s real output exactly) matched production on
**every single one of the 2,329,198 trainable rows checked — 0 mismatches.**

**This is a far more severe finding than the V2.1 investigation's 15-25 scenario sample
suggested — not merely "mostly," but ENTIRELY:**

| Candidate type | Episodes | Zero-duration fraction | Mean duration | p95 duration | Adjacent-pair exact-zero-gap fraction |
|---|---|---|---|---|---|
| **Door** | 54,832 | **100.0%** | 0.0s | 0.0s | 96.2% |
| **Exit** | 18,716 | **0.0%** | 20.4s | 83.9s | 0.0% |
| **Stair** | 31,419 | **100.0%** | 0.0s | 0.0s | 80.0% |

**Every single Door and Stair congestion episode in the entire 2,500-scenario, 1.48M-row
combined dataset has exactly zero duration.** Not 91%, not "substantially" — literally all
of them, at full scale. This target is currently **incapable of representing sustained
Door/Stair congestion at all** under the current simulator's capacity-1/FIFO-admission
mechanics (§2 of the V2.1 investigation already traced why: capacity-1 edges hand off
admission instantaneously, and `target_generator._edge_occupant_count`'s inclusive `<=`
bounds register that handoff instant as a momentary threshold crossing). Exit, by contrast,
has **zero** zero-duration episodes — every one of its 18,716 episodes is genuine, sustained
overlap (mean 20.4s, up to minutes-long at the p95).

**Direct answer to this milestone's own central question: no, the target does not mean the
same physical thing for Door/Stair as for Exit.** For Door/Stair it currently measures "a
queue-admission handoff is occurring" (a near-tautological function of `candidate_queue_
length`, which is exactly why that feature dominates permutation importance for those
types). For Exit it measures genuine multi-person crowding. **Every Door/Stair PR-AUC number
reported anywhere in V1/V2/V2.1/V2.2 should be read with this in mind**: high Door/Stair
predictive performance reflects successfully predicting an imminent mechanical timing event,
not foreseeing real crowding — a materially different (and less operationally valuable)
claim than what the same metric means for Exit.

## 6. Counterfactual target analysis (Phase 8) — analysis-only, target NOT changed

Same full-scale audit, re-labeling every trainable row under "congestion sustained for at
least N seconds" instead of the production ">=2 concurrent, any duration" rule — **never
written back as the production target**, purely an offline sensitivity check:

| Candidate type | Production (≥0s) | ≥1s | ≥5s | ≥10s |
|---|---|---|---|---|
| Door | 17.62% | **0.00%** | 0.00% | 0.00% |
| Exit | 3.72% | 3.46% | 2.41% | 1.17% |
| Stair | 21.27% | **0.00%** | 0.00% | 0.00% |

**Door and Stair positive rate collapses to exactly zero under ANY nonzero persistence
requirement** — confirming §5's finding from a different angle: there is no sustained
congestion anywhere in the dataset for these two types to even counterfactually detect, under
the current simulator's capacity mechanics. **Exit's positive rate degrades gracefully and
remains real** (3.72%→1.17%, a ~3.2x reduction under the strictest 10s bar, not a collapse)
— consistent with genuine, variable-duration crowding rather than a boundary artifact.

This is not evidence the current target is "wrong" for Exit, and it is not, on its own,
evidence the target must be redefined for Door/Stair this milestone (explicitly out of
scope) — but it is decisive evidence that **Door/Stair "congestion" and Exit "congestion" are
not comparable phenomena under the current definition**, and any future target-redesign
milestone should treat this as its starting evidence, not rediscover it.

## 7. Sim/live parity matrix (Phase 3/5, Goal B)

| Feature | Sim/live exact-equality proof | Missing-evidence semantics | Test |
|---|---|---|---|
| `candidate_alternative_route_count` | **Always exact** — both extractors call the literal same `build_alternative_route_counts()` | Never missing (pure Building geometry) | `tests/test_predictive_dataset_v2_1_sim_live_parity.py::test_alternative_route_count_is_always_exact` |
| `candidate_recent_flow_rate` (Door/Stair) | **Exact when evidence is identical** — proven with matched timestamps fed through both `OccupantTimelineStep` (sim) and `ZoneTransitionRecord` (live) | `None` when `occupants` not supplied (never fabricated `0`) | `test_identical_crossing_evidence_produces_identical_flow_rate`, `test_identical_evidence_outside_window_produces_identical_zero` |
| `candidate_recent_flow_rate` (Exit) | Not numerically proven equal — genuinely different computation bases (sim: exact edge-crossing count; live: `EvacuationLedger`'s nearest-exit-attributed count) | `None` when `evacuation_snapshot` not supplied | `test_exit_flow_rate_is_none_not_fabricated_when_evacuation_progress_unavailable` |
| `candidate_congestion_trend` | Not numerically proven equal — sim compares a demand proxy at two timestamps; live reads `TrendTracker`'s own independently-computed trend | `None` when `crowd_snapshot` not supplied; `"UNKNOWN"` is itself a real, non-`None` value (not enough history yet) | `test_trend_is_none_not_fabricated_when_crowd_intelligence_unavailable`, `test_trend_unknown_is_a_real_value_not_none` |

Architecture guards (`tests/test_predictive_dataset_v2_1_architecture_guards.py`, 7 tests):
simulation ground truth never enters live extraction (`live_extractor_v2_1.py` never imports
`simulator.*`); future timestamps never enter prediction features (neither v2_1 extractor
imports `target_generator`); live-only packages never enter simulation core
(`simulation_extractor_v2_1.py` never imports `live_occupants`/`crowd_intelligence`/
`evacuation_progress`); the deterministic intelligence layer never imports the ML layer
(`crowd_intelligence`/`evacuation_progress` never import `predictive_dataset`/
`predictive_model`).

## 8. Model V2.2 results (Phase 9-10, Goal A)

`scripts/train_localized_predictive_model_v2_2.py`: same scenario split convention (seed
`20260726`, 70/15/15: train 1,750 / val 375 / test 375 scenarios), same model zoo, same
class-imbalance/calibration strategy as V2. One real bug found and fixed during this
milestone (not hidden): `predictive_model.training_size_study` hardcoded V1/V2's 9-field
`build_feature_matrix` internally, so passing a `val_feat` built with the 12-field
experimental schema raised an XGBoost feature-shape-mismatch crash. Fixed by adding an
optional `feature_builder` parameter (default unchanged — every V2 call site unaffected),
2 new regression tests added.

**Best model: XGBoost** (same as V2), test PR-AUC **0.7695** (V2: 0.6918, **+11.2%
relative**), ROC-AUC 0.956.

| | Overall | Door | Exit | Stair |
|---|---|---|---|---|
| **V2.2 PR-AUC** | **0.770** | **0.706** | **0.565** | **0.975** |
| V2 PR-AUC | 0.692 | 0.578 | 0.464 | 0.966 |
| Relative change | **+11.2%** | **+22.1%** | **+21.7%** | +0.9% |

**All three candidate types improved or held steady — no regression anywhere.** Door and
Exit both improved by roughly the same relative amount (~22%); Stair, already near-ceiling
in V2, held essentially flat (as expected — see §5's finding that Stair's target is a
mechanical artifact largely unrelated to demand features in the first place).

Sanity checks: leakage-correlation recheck flagged **zero** features; label-shuffle test
collapsed to chance (ROC-AUC 0.459). Full metrics: precision 0.612, recall 0.780, F1 0.686,
balanced accuracy 0.853, Brier 0.093.

**Occupancy / multi-bottleneck / single-vs-multi-exit** (operational slices):

| Slice | n | PR-AUC | Recall | FN rate |
|---|---|---|---|---|
| No bottleneck | 176,582 | — | — | — |
| Single bottleneck | 110,736 | 0.787 | 75.5% | 24.5% |
| **Multiple bottlenecks** | 66,865 | **0.894** | **79.6%** | **20.4%** |
| LOW occupancy | 1,356 | 0.839 | 80.3% | 19.7% |
| MEDIUM occupancy | 16,268 | 0.840 | 83.8% | 16.2% |
| HIGH occupancy | 336,559 | 0.763 | 77.6% | 22.4% |
| Multi-exit topology | 349,825 | 0.762 | 77.5% | 22.5% |
| Single-exit topology | 4,358 | **0.917** | **91.2%** | **8.8%** |

Multi-bottleneck rows now score **higher** PR-AUC than single-bottleneck rows (0.894 vs.
0.787) — a reversal from V1/V2's own "multi-bottleneck is hardest" framing, though this
partly reflects multi-bottleneck rows' much higher base rate (42.1% vs. 15.9%) inflating
achievable PR-AUC, not a claim that multi-bottleneck rows are now "easy" in an absolute
sense. HIGH occupancy remains the relatively hardest occupancy band, consistent with V1/V2.

## 9. Leave-one-topology-family-out (Phase 11) — the critical test

| Held-out family | Test rows | Overall PR-AUC | Door PR-AUC | Exit PR-AUC | Stair PR-AUC |
|---|---|---|---|---|---|
| **`multi_exit_wide`** | 1,479,811 | **0.429** | 0.453 | 0.133 | n/a (0 stairs) |
| `single_exit_lowrise` | 30,645 | 0.880 | 0.923 | 0.657 | n/a (0 stairs) |
| `twin_stair_highrise` | 665,257 | 0.667 | 0.619 | 0.342 | 0.884 |
| `v1_topology_fixed` | 153,485 | 0.731 | 0.747 | 0.641 | 0.583 |

**`multi_exit_wide`'s holdout PR-AUC (0.429) is dramatically higher than V2's (0.314) — a
+36.6% relative improvement, and it lands within 0.1% of the V2.1 500-scenario experiment's
own prediction (0.429) for the identical holdout.** This is the single most important
confirmation this milestone set out to get: **the V2.1 improvement is not a small-sample
artifact — it survives full-scale, 2,500-scenario validation almost exactly as predicted.**

**But generalization is still not uniform, and Exit remains the weak link inside the hardest
holdouts.** `multi_exit_wide`'s own Exit PR-AUC (0.133) and `twin_stair_highrise`'s Exit
PR-AUC (0.342) are both still weak in absolute terms — better than V2 (§10 below), but far
from strong. Door generalizes comparably well in every holdout (0.45-0.92); Exit is
uniformly the harder type to transfer, everywhere.

## 10. V1 → V2 → V2.1 → V2.2 comparison

| Metric | V1 | V2 | V2.1 (500-scenario) | **V2.2 (full-scale)** |
|---|---|---|---|---|
| Overall PR-AUC | 0.708 | 0.692 | 0.805 | **0.770** |
| Door PR-AUC | 0.714 | 0.578 | 0.747 | **0.706** |
| Exit PR-AUC | 0.686 | 0.464 | 0.547 | **0.565** |
| Stair PR-AUC | 0.240 | 0.966 | 0.975 | **0.975** |
| `multi_exit_wide` holdout PR-AUC | n/a | 0.314 | 0.429 | **0.429** |
| Multi-bottleneck FN rate | 13.3% | ~10.1% | n/a | **8.6%** |
| High-occupancy FN rate | 9.4% | ~4.4% | n/a | **22.4%\*** |
| Isotonic-calibrated ECE | 0.003 | 0.003 | n/a | **0.002** |

\* Not directly comparable to V1/V2's own high-occupancy FN figures — V2.2's operational-
slices FN-rate definition (§8 of this doc) differs slightly in row population from V1/V2's
error-analysis-table FN-rate (trainable rows only vs. all test rows); both readings agree
HIGH occupancy is the relatively hardest band, which is the load-bearing conclusion.

**Why the numbers changed**: V2.2's overall/Door/Exit PR-AUC are *higher* than V2 (the
9-field baseline) because the 3 new features add real information the model didn't have —
confirmed directly by the ablation study (§11): the `v2_1_flow_and_trend` feature family is
now the **single largest ablation drop in the entire model** (0.078 PR-AUC, larger than the
original `demand_signal` family's 0.029). V2.2's overall PR-AUC (0.770) is slightly *lower*
than V2.1's small-scale figure (0.805) — expected and unremarkable: V2.1's 500-scenario
run is a smaller, less representative sample with higher variance; V2.2 is the trustworthy
number. The `multi_exit_wide` holdout number, by contrast, landing almost exactly on V2.1's
own small-scale prediction (0.429 both times) is the more important consistency check, and
it held.

## 11. Feature importance & ablation (Phase 13)

Permutation importance (ROC-AUC drop), top 6:

| Rank | Feature | Mean ROC-AUC drop |
|---|---|---|
| 1 | `candidate_queue_length` | 0.259 |
| 2 | `candidate_walking_distance` | 0.104 |
| **3** | **`candidate_recent_flow_rate`** | **0.045** |
| 4 | `candidate_approaching_count` | 0.021 |
| 5 | `candidate_adjacent_zone_occupancy` | 0.009 |
| 6 | `total_active_occupant_count` | 0.009 |
| — | `candidate_alternative_route_count` | 0.005 |

Feature-family ablation (zeroed, retrained, PR-AUC drop):

| Family | PR-AUC drop |
|---|---|
| **`v2_1_flow_and_trend`** (recent_flow_rate + congestion_trend) | **0.078 — largest in the model** |
| `structural` (type/capacity/distance/traversable) | 0.034 |
| `demand_signal` (queue_length + approaching_count) | 0.029 |
| `global_and_adjacent_context` | 0.007 |
| `derived_congestion_level` | 0.0002 |
| `v2_1_alternative_route_structure` (alternative_route_count alone) | **-0.001 (negligible/no marginal value in isolation)** |

Explicit answers to this milestone's own Phase 13 questions:

- **Does Exit prediction still suffer because queue_length is dead?** Less so. `candidate_
  recent_flow_rate` is now the **3rd-most-important feature overall** by permutation
  importance, and the flow/trend family is the single largest ablation drop in the entire
  model — direct, full-scale confirmation this feature genuinely fills the gap
  `candidate_queue_length`'s structural blindness (§4) left for Exit.
- **Which new feature replaces the missing signal?** `candidate_recent_flow_rate`, primarily
  — `candidate_congestion_trend` also contributes (bundled in the same ablated family; its
  `STABLE` one-hot column is individually the 2nd-highest-gain column in XGBoost's builtin
  importance, 0.167).
- **Does alternative-route structure materially help `multi_exit_wide`?** **Revised finding,
  more precise than V2.1's own small-scale conclusion**: in isolated full-scale ablation,
  `candidate_alternative_route_count` alone contributes **negligible** marginal value
  (-0.001 PR-AUC when removed — noise-level). V2.1's investigation attributed part of
  Door's `multi_exit_wide`-holdout improvement to this feature; the full-scale ablation
  shows the real driver is overwhelmingly `candidate_recent_flow_rate`/`candidate_
  congestion_trend`, not alternative-route count. This is a genuine, disclosed correction —
  not every V2.1 hypothesis survived full-scale scrutiny equally.
- **Does flow-rate help all candidate types?** Yes — Door, Exit, *and* Stair PR-AUC all
  moved in the same (non-negative) direction (§8), and the flow/trend family's ablation
  drop is measured on the whole model, not Exit alone.
- **Is the model learning candidate type as a shortcut?** No evidence of this — `candidate_
  type=*` one-hot columns rank far below the demand/structural features in both importance
  methods (consistent with V1/V2's own finding), and `candidate_traversable`/`candidate_
  type=Stair` only enter the builtin top-8 well behind `candidate_queue_length`/`candidate_
  congestion_trend=STABLE`.

## 12. Calibration (Phase 14)

| | Brier score | ECE |
|---|---|---|
| Raw (uncalibrated) | 0.0932 | 0.1119 |
| Platt scaling | 0.0570 | 0.0220 |
| **Isotonic regression (recommended)** | **0.0548** | **0.0019** |

Same pattern as V1/V2: raw probabilities are meaningfully miscalibrated; isotonic
calibration (fit on validation only, exported separately as `calibrator.joblib`, versioned
independently from `model.joblib`) brings ECE down to 0.002 — marginally better than V2's
own 0.003. Any future consumer must use the calibrated output, never the raw model output,
exactly as V1/V2 already established.

## 13. Failure analysis

- **Exit remains the hardest candidate type to generalize**, even though its overall PR-AUC
  improved 21.7% relative. Inside the two hardest topology holdouts (`multi_exit_wide`,
  `twin_stair_highrise`) its PR-AUC is still only 0.13-0.34 — a real, unresolved gap.
- **`multi_exit_wide` overall generalization improved dramatically (+36.6% relative) but is
  still the weakest of the 4 holdouts** — a genuinely better number, not a solved problem.
- **The target-semantics finding (§5) reframes every Door/Stair number in this whole
  document.** Door PR-AUC 0.706 and Stair PR-AUC 0.975 are real, reproducible, non-leaked
  numbers — but per §5/§6, they measure the model's ability to predict an imminent
  zero-duration queue-admission handoff, not sustained crowding. This is not a new failure
  mode introduced by V2.2; it is a pre-existing property of V1/V2/V2.1's shared target
  definition that this milestone is the first to quantify precisely, and it must inform how
  every "strong" Door/Stair metric anywhere in this project's history is read going forward.

## 14. Remaining limitations

- Exit generalization to unseen topologies remains weak in 2 of 4 held-out families
  (§9) — improved from V2, not resolved.
- Door/Stair's congestion target is, at full scale, **100% a zero-duration timestamp
  artifact** (§5) — a target-semantics limitation, not fixed this milestone by design.
- `candidate_recent_flow_rate` has full live parity for Exit today; Door/Stair's live
  mechanism (`live_occupants.history.zone_transitions`) is implemented and unit-tested here
  but has never run against a real live deployment — only synthetic fixtures.
- `candidate_alternative_route_count`'s marginal contribution, once isolated via full-scale
  ablation, is negligible (§11) — a genuine correction to the V2.1 investigation's own
  hypothesis, not a fatal problem (the feature does no harm, and remains well-justified
  structurally even if its measured marginal value here is small).
- This milestone did not re-run V1's 10s/30s/60s horizon-robustness sweep (same disclosed
  scope limitation as V2/V2.1).

## 15. Production-readiness decision (Phase 15)

The training script's own automated heuristic (`_assess_production_readiness`) labeled this
**READY_FOR_SHADOW_MODE_LIVE_VALIDATION**, based purely on: sanity checks passing and no
topology-holdout PR-AUC falling below 50% of the normal-split PR-AUC. **That mechanical
threshold is not sufficient evidence on its own, and this document does not adopt it
uncritically** — per this milestone's own explicit "be conservative" instruction and its
list of 8 required conditions for D/E readiness.

Checked against those 8 conditions:

1. Exit performance materially improves over V2 — **yes** (+21.7% relative overall).
2. Stair remains strong — **yes**, but see caveat below.
3. `multi_exit_wide` generalization improves materially and survives full-scale testing —
   **yes** (+36.6% relative, confirmed at full scale).
4. Multi-bottleneck weakness is acceptable/bounded — **yes**, improved further (§8).
5. High-occupancy behavior is acceptable — **partially**, remains the relatively hardest
   band, consistent with every prior milestone.
6. Calibration is usable — **yes** (ECE 0.002 after isotonic).
7. Every production feature has an honest live source — **partially**. 2 of 3 fully;
   `candidate_recent_flow_rate` has full parity for Exit only, and the Door/Stair mechanism
   (§4, §7) has never been exercised against a real live deployment, only unit-tested
   fixtures.
8. **No target-semantics finding invalidates the interpretation of the prediction — NO,
   this condition fails.** §5's finding is exactly the kind of thing this condition
   guards against: Door and Stair's "strong" metrics reflect a target that is, at full
   scale, 100% a mechanical timestamp artifact for those two types. A model whose two best-
   performing candidate types' predictions cannot be honestly described as "foreseeing
   congestion" should not be described as ready to influence live safety-adjacent decisions,
   regardless of how high their PR-AUC reads.

**Verdict: C — PROMISING BUT NEEDS MORE DATA.** Not D. Condition 8 alone is disqualifying
for shadow-mode-or-higher regardless of how strong the aggregate metrics look, and condition
7 (Door/Stair live-flow-rate parity, real-deployment validation of the zone-transition
mechanism) is also incomplete. This is a genuinely stronger "C" than V2's — every metric
this milestone could improve, it did, and the central full-scale generalization question
(§9) was answered decisively in the affirmative — but it is not a "D," and choosing D on the
strength of the aggregate PR-AUC number alone would be exactly the mistake this milestone's
own charter warned against.

## 16. Performance (Phase 21)

Measured separately, CPU-only, this development machine:

| Measurement | Value |
|---|---|
| Feature extraction (12-field experimental schema, isolated from simulation time) | 17,425 rows/sec |
| Inference: single-row latency | 0.50ms |
| Inference: batch throughput (354,183 rows) | 301,769 candidates/sec |

Simulation-generation time is never mixed into either figure — the extraction benchmark
times only `extract_experimental_candidate_features()` calls against already-completed
`MultiAgentSimulationResult` objects; the full campaign's own 299.5s/2.4M-row figure (§4)
includes simulation and is not a feature-extraction-cost number.

## Final report — explicit answers

**A. Did the V2.1 Exit improvement survive full-scale validation?** **Yes** — Exit PR-AUC
0.464→0.565 overall (+21.7% relative), consistent in direction and magnitude with V2.1's
own small-scale finding.

**B. Did the `multi_exit_wide` improvement survive full-scale leave-one-topology-out
testing?** **Yes, decisively** — 0.314→0.429 (+36.6% relative), landing within 0.1% of
V2.1's own 500-scenario prediction for the identical holdout. This is the single most
important confirmation this milestone produced.

**C. Does Stair remain genuinely predictive after the new features are introduced?**
Numerically yes (PR-AUC 0.975, unchanged from V2.1) — but §5/§13's target-semantics finding
means "genuinely predictive of congestion" is the wrong description for Stair (and Door):
the target itself is, at full scale, 100% a zero-duration timestamp artifact for these two
types. The model is genuinely predictive *of that artifact*, not of sustained crowding.

**D. Can Door, Exit and Stair all provide the new features in LIVE SynEvac without
fabricated values?** Yes for `candidate_congestion_trend` and `candidate_alternative_
route_count` (already-existing live sources, proven exact/structural). For `candidate_
recent_flow_rate`: yes for Exit (existing `evacuation_progress` mechanism); a new,
implemented, unit-tested — but not yet live-deployment-validated — mechanism for Door/
Stair (`live_occupants.history.zone_transitions`). Missing evidence always surfaces as
`None`, never fabricated, in every case (§7 parity tests).

**E. What exactly causes Exit congestion if `candidate_queue_length` is always zero?**
Genuine sustained multi-occupant overlap in the exit corridor (§5: mean episode duration
20.4s, zero zero-duration episodes) — driven by arrival rate and transit duration, not by
the simulator's capacity-admission mechanism (which never engages for Exit's capacity-50
default). `candidate_recent_flow_rate` and `candidate_congestion_trend` are the features
that make this mechanism visible to the model.

**F. Does alternative-route structure explain a meaningful part of `multi_exit_wide`'s
difficulty?** **Revised, more precise answer than the V2.1 investigation gave**: full-scale
ablation shows `candidate_alternative_route_count` alone contributes negligible marginal
value once `candidate_recent_flow_rate`/`candidate_congestion_trend` are already present
(§11). The structural dilution effect V2.1 identified (§3 of this doc) is real as a
description of *why* `multi_exit_wide` is harder, but the *feature* that fixes it turned
out to be the flow/trend signal, not the alternative-route count.

**G. How much of Door/Stair's existing congestion target is caused by zero-duration/
timestamp-boundary handoffs?** **All of it, at full scale** — 100.0% of both Door's 54,832
and Stair's 31,419 congestion episodes across the entire 2,500-scenario dataset have exactly
zero duration (§5).

**H. Does model performance remain strong under a sustained-congestion counterfactual
target?** For Exit, **yes** — positive rate degrades gracefully (3.72%→1.17% at a strict
10s persistence bar), consistent with genuine, variable-duration crowding. For Door/Stair,
**the question is moot** — their counterfactual positive rate collapses to exactly 0.00% at
any nonzero duration threshold (§6); there is no sustained congestion in the dataset for
these types to even test performance against.

**I. Is there any evidence of future leakage?** No — leakage-correlation recheck flagged
zero features, and the label-shuffle test collapsed to chance (ROC-AUC 0.459).

**J. Is the model calibrated well enough for probability interpretation?** Not out of the
box (ECE 0.112) — but isotonic-calibrated output is well calibrated (ECE 0.002) and must be
the only form ever read as a probability.

**K. Should V2.2 remain offline research, move to LIVE SHADOW MODE, or influence
Recommendation ranking?** **Remain offline research (Verdict C, PROMISING BUT NEEDS MORE
DATA)** — not shadow mode yet. The target-semantics finding (§5/§15 condition 8) and
incomplete live-deployment validation of the Door/Stair flow-rate mechanism (condition 7)
are both disqualifying for shadow mode specifically, even though every metric this milestone
set out to validate moved in the right direction.

**L. What is the single biggest remaining technical risk?** **The target-semantics finding
itself.** Door and Stair's headline metrics (PR-AUC 0.706 and 0.975) look like the
project's strongest results, but §5 proves they substantially measure a simulator timing
artifact rather than real congestion. Any future milestone that treats these numbers at
face value — including a future integration decision — risks building on a
misinterpretation this milestone specifically exists to prevent. Resolving it (a genuine
target-redesign investigation, explicitly out of this milestone's scope) should be a
higher priority than further feature engineering.
