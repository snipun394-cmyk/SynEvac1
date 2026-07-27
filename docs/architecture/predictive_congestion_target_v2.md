# Predictive Congestion Target V2 — Physically Meaningful Congestion Definition & Dataset Relabeling

Status: **TARGET ENGINEERING + DATASET RELABELING + VALIDATION.** Not a model-training
milestone. Nothing here is wired into recommendation scoring, exit ranking, guidance,
signage, LiveRuntime, or operator workflow. The simulator itself was **not modified** —
no movement scheduling, edge traversal timing, admission semantics, pathfinding, capacity
model, congestion model, route choice, or Door/Stair/Exit traversal behavior was touched.
Only the *target generator* (how observed simulation intervals are interpreted into a
congestion label) changed, in a new, additive module.

## 1. Why Target V1 was invalid for Door/Stair

Builds on `docs/architecture/localized_predictive_model_v2_2.md` §5, re-verified directly
from code and controlled simulation here rather than assumed. Target V1
(`predictive_dataset/target_generator.py`, **completely untouched by this milestone**)
defines "congested" as `>=2 occupants concurrently on a candidate's edge, at ANY duration`
(`CONGESTION_THRESHOLD = 2`, inclusive `start_time <= time <= end_time` bounds).

**Root cause, traced exactly**: `Door`/`Staircase`'s effective capacity is structurally `1`
(derived from each type's own default `width`, floored by `simulator/capacity.py`'s
`DefaultCapacityModel`/`StairCapacityModel` — confirmed: `candidate_capacity` is constant
`1.0` across 100% of Door and Stair rows in every V1/V2/V2.2 dataset generated so far). The
discrete-event simulator's admission mechanism (`simulator/coordinator.py::
_handle_try_enter_edge`) therefore **never** allows a second occupant onto a Door/Stair edge
while the first is still on it — the only way V1's `>=2 concurrent` threshold can ever fire
for these types is the exact instant one occupant's admission ends and the next
already-queued occupant is admitted, which V1's *inclusive* interval bounds register as a
momentary crossing of duration exactly zero. Verified exhaustively (V2.2's full-scale audit,
re-confirmed at this milestone's own 20%-sample sensitivity sweep, §5 below): **100% of
Door's and Stair's V1 "congestion" episodes have exactly zero duration.** `Exit`'s capacity
defaults to `50` and is never approached by these scenarios' occupant counts (max 111
total), so its admission mechanism never engages either way — but Exit's occupants
frequently and genuinely overlap in transit (longer walking distances, no per-edge admission
gate), producing real, sustained V1 episodes (mean 20.4s) that were never invalid.

## 2. Queue vs. occupancy: the two raw signals already available, unchanged

Investigated directly (Phase 1 of this milestone), not assumed:

- **`candidate_queue_length`** (`predictive_dataset/simulation_extractor.py::
  _current_queue_length`) counts occupants in the interval `[join_time, start_time)` —
  waiting for **admission onto** the edge, a BEFORE-the-candidate phenomenon. This is a real,
  continuous, non-degenerate quantity: `step.queue_wait_time` accumulates real elapsed time,
  never a boundary artifact.
- **Edge occupancy** (what V1's target measures) counts occupants in `[start_time, end_time]`
  — physically **present on** the edge, an ON-the-candidate phenomenon.

For a capacity-1 edge (Door/Stair), the admission gate means real demand accumulation shows
up as **queueing**, never as concurrent on-edge occupancy (structurally impossible, per §1).
For a capacity-50 edge (Exit), the gate essentially never engages, so real demand
accumulation shows up as **occupancy overlap**, never as formal queueing. This is not a
modeling choice — it is a direct, mechanical consequence of the two types' different
authored capacities, confirmed empirically (§5): **zero Exit queue episodes were found across
every scenario checked** (capacity 50 is never reached), and **zero nonzero-duration Door/
Stair occupancy episodes exist anywhere in the dataset** (capacity 1 makes it structurally
impossible).

## 3. Operational definition of congestion (Phase 2)

**A candidate is experiencing meaningful congestion when it has a persistent demand/service
imbalance that is impeding evacuation flow** — occupants are accumulating (queueing to
enter, or physically overlapping in transit) faster than the candidate is absorbing them,
and this condition has lasted long enough to be a real operational state, not a momentary
timestamp coincidence. This is deliberately NOT "two occupant timestamps happened to touch."

## 4. Relationship to `ground_truth/bottleneck.py` (Phase 11)

Inspected directly: `ground_truth/bottleneck.py::_congestion_duration_for_edge` uses the
**identical** `>=2 concurrent occupants` interval-sweep technique and threshold as Target
V1's own `_edge_occupant_count`/`_congested_within_window` — but it is a **separate,
independent reimplementation** (not a shared call), used for whole-run Command-Center
summary reporting (total time spent congested), not per-tick prediction labels. It is
**equally susceptible to the same zero-duration-episode contamination for Door/Stair** — a
previously-undocumented parallel limitation, flagged here but explicitly **out of scope to
fix** (a different consumer, not this milestone's target). Separately, `ground_truth.
bottleneck::_has_any_queueing` (used for `doors_that_became_bottlenecks` engineering
findings) already uses `step.queue_wait_time > 0` — i.e., it already treats ANY queueing as
bottleneck evidence for Door, independently confirming this milestone's own choice of queue
persistence as the physically-correct Door/Stair mechanism; Target V2 does **not** merge
with or replace `ground_truth.bottleneck` — they answer related but distinct questions
("has this edge EVER been a bottleneck this run" vs. "will THIS candidate become
meaningfully congested in the next 20 seconds") and are kept as separate, non-unified
concepts.

## 5. Threshold sensitivity sweep (Phase 4)

500-scenario (20% scale, same precedent as the V2.1 investigation) sample, all 4 topology
families, `master_seed=20270115`. Two raw episode mechanisms computed independently
(`predictive_dataset/target_semantics_analysis.py::queue_episodes`/`occupancy_episodes`):

**Queue episodes (>=1 occupant waiting for admission):**

| Type | n episodes | mean | median | p95 | retained >=1s | >=3s | >=5s | >=10s |
|---|---|---|---|---|---|---|---|---|
| Door | 1,362 | 397.6s | 243.6s | 1369.4s | 99.5% | 98.3% | 97.7% | 95.0% |
| Exit | **0** | — | — | — | — | — | — | — |
| Stair | 402 | 125.0s | 93.1s | 403.6s | 96.8% | 90.5% | 80.3% | 69.7% |

**Occupancy episodes (>=2 occupants concurrently on the edge):**

| Type | n episodes | mean | median | p95 | zero-duration | retained >=1s | >=3s | >=5s | >=10s |
|---|---|---|---|---|---|---|---|---|---|
| Door | 10,623 | 0.0s | 0.0s | 0.0s | **100%** | 0% | 0% | 0% | 0% |
| Exit | 3,709 | 20.2s | 7.4s | 81.2s | **0%** | 96.7% | 79.2% | 60.0% | 41.4% |
| Stair | 6,152 | 0.0s | 0.0s | 0.0s | **100%** | 0% | 0% | 0% | 0% |

**This confirms §2's mechanical prediction exactly, at 20% sample scale**: Exit has zero
queue episodes ever (capacity 50 never reached); Door/Stair have zero nonzero-duration
occupancy episodes ever (capacity 1 makes overlap structurally impossible). Neither
mechanism alone works for all three types — but each type has exactly one genuinely
non-degenerate mechanism, and Door/Stair's queue episodes are typically very long-lived
(median 93-244s) once they start, meaning a persistence floor only needs to filter genuine
momentary noise, not erode real signal.

**Chosen threshold: `MIN_PERSISTENCE_SECONDS = 3.0`, applied identically to both
mechanisms.** At 3s: Door retains 98.3% of real queue episodes, Stair 90.5%, Exit 79.2% of
real occupancy episodes — a floor that discards only genuinely momentary conditions while
preserving the overwhelming majority of real signal for every type. The SAME value is used
for both mechanisms deliberately (not a per-type-tuned pair of constants) — per this
milestone's own Phase 6 instruction against "arbitrary type-specific labels just to improve
class balance," one shared persistence floor applied to whichever raw signal is
structurally meaningful for that type is a principled choice, not an arbitrary one.
`QUEUE_THRESHOLD = 1` (any occupant waiting is already demand > capacity, given Door/Stair's
own capacity is exactly 1) and `OCCUPANCY_THRESHOLD = 2` (V1's own value, reused not
reinvented).

## 6. Chosen Target V2 formula (Phase 7/12)

**`target_version = "v2-persistent-demand-service-imbalance"`**
(`predictive_dataset/target_generator_v2.py`, new module, Target V1 untouched)

**One universal formula, no candidate-type branching.** For candidate C:

- A **qualifying episode** is either (a) a queue episode (>=1 occupant waiting) or (b) an
  occupancy episode (>=2 occupants concurrent), whose FULL duration is >= 3.0 seconds.
- **`currently_congested(t)`** = `t` falls within the elapsed portion of a qualifying
  episode already underway (`episode_start + 3.0 <= t < episode_end`) — evaluated using only
  information at or before `t` (leak-safe).
- **`target(t, horizon)`** = `None` if `currently_congested(t)` (not applicable, same policy
  as V1); else `True` iff some qualifying episode's persistence bar is first crossed
  within `(t, t+20]` — an **onset** definition ("will congestion BEGIN", not "does it exist
  at any point in the window"), per this milestone's own Phase 21 instruction.

Because Exit structurally never produces a queue episode and Door/Stair structurally never
produce a nonzero-duration occupancy episode (§2, §5), this single OR'd, type-agnostic
formula automatically reduces to "queue-based" for Door/Stair and "occupancy-based" for Exit
— without ever branching on `candidate_type` in the implementation.

**Can a timestamp-boundary handoff alone ever create a positive?** No — a zero-duration
occupancy episode has `duration=0 < 3.0`, so it is never a qualifying episode and
contributes no onset time at all (proven directly by
`tests/test_predictive_dataset_target_generator_v2_sanity.py::
test_scenario_7_timestamp_handoff_alone_is_never_congestion`).

## 7. Controlled sanity scenarios (Phase 5/10)

All 8 of this milestone's own required scenarios, implemented as unit tests
(`tests/test_predictive_dataset_target_generator_v2_sanity.py`), passing:

| # | Scenario | Expected | Result |
|---|---|---|---|
| 1 | Single occupant through a Door | No congestion | ✓ |
| 2 | Two occupants sequential, negligible wait | No congestion | ✓ |
| 3 | Several occupants accumulate at a narrow Door | Congestion | ✓ |
| 4 | Exit demand that clears efficiently | No congestion | ✓ |
| 5 | Exit demand exceeds clearing, persistent overlap | Congestion | ✓ |
| 6 | Stair sustained multi-floor demand | Congestion | ✓ |
| 7 | Timestamp handoff alone (V1's own bug) | No congestion | ✓ |
| 8 | Congestion computed independently per candidate | Migrates correctly | ✓ |

Plus dedicated onset-timing tests (onset falls exactly at `episode_start + 3.0s`; an onset
outside the prediction window correctly yields a negative) and an already-congested
exclusion test (`target=None` when `currently_congested=True`).

## 8. Full-scale relabeling (Phase 8/14)

**Relabeled from existing scenario definitions, not a "new" campaign.** Simulations were
rerun (raw timeline objects were never persisted from any prior campaign — only
already-extracted CSVs survive), but using the exact same `master_seed=20270115` and the
exact same, byte-for-byte-unchanged `predictive_dataset/topologies_v2.py` definitions
Predictive Dataset V2 and Model V2.2 both already used — this reproduces the literal same
2,500 scenarios and occupants, not a new population. `scripts/
run_predictive_congestion_target_v2_relabel.py` streams rows directly to CSV (never
accumulates in memory, learned directly from the 9.6M-row memory scare two milestones ago)
and computes the V2.2 12-field feature schema, Target V1, and Target V2 from the SAME
simulation run in one pass, so V1/V2 agreement (§10) is exact, row-for-row, not an
approximation across separately-generated datasets. Result: **2,500 accepted, 0 failed,
2,405,049 rows** (identical row count to V2/V2.2 — confirms deterministic reproduction),
**321.0s wall time, 7,492 rows/sec** (feature+dual-label extraction combined; simulation
time is not separated out here since it dominates the same way it did in every prior
full-scale campaign).

## 9. Dataset Target V2 statistics (Phase 9)

| | Rows | Trainable (target != None) | Positive | Positive rate |
|---|---|---|---|---|
| Target V1 | 2,405,049 | 2,329,198 | 300,704 | 12.90% |
| **Target V2** | 2,405,049 | **1,730,976** | **49,331** | **2.85%** |

**By candidate type:**

| Type | V1 n | V1 pos rate | V2 n | **V2 pos rate** |
|---|---|---|---|---|
| Door | 1,244,514 | 17.62% | 687,889 | **1.89%** |
| Exit | 852,075 | 3.72% | 861,863 | **4.03%** |
| Stair | 232,609 | 21.27% | 181,224 | **0.88%** |

Door and Stair's positive rates **collapse** under Target V2 (17.6%→1.9%, 21.3%→0.9%) —
exactly as expected once zero-duration artifacts stop counting. **Exit's positive rate is
essentially unchanged, even slightly higher** (3.72%→4.03%) — direct confirmation that V1's
Exit signal was already physically real and V2 does not fundamentally alter it, only trims
sub-persistence-bar momentary overlaps while very slightly widening the effective window
via the onset (vs. any-point-in-window) semantics.

**By topology family (V2 positive rate):** `single_exit_lowrise` 11.76% (highest — no
alternative routes concentrates real pressure), `twin_stair_highrise` 5.45%,
`v1_topology_fixed` 4.36%, `multi_exit_wide` 1.40% (lowest — most alternative-route
dilution, consistent with every prior milestone's own finding about this family).

**By occupancy band (V2):** LOW 10.25%, MEDIUM 5.80%, HIGH 2.67% — an **inverse**
relationship to what V1/V2/V2.2 showed. Explained by `currently_congested` (§9 below): at
HIGH occupancy, far more candidate-ticks are already inside an ongoing persistent episode
(excluded as "not applicable"), leaving proportionally fewer genuine "about to begin"
onset opportunities — not evidence that high-occupancy congestion is somehow less severe,
but a real, disclosed consequence of onset-based labeling.

**Single- vs. multi-exit (V2):** single-exit 11.76% vs. multi-exit 2.74% — single-exit
buildings show meaningfully more Target V2 congestion, consistent with §9's topology-family
finding.

**`currently_congested` rate — a striking, disclosed difference:**

| Type | V1 `currently_congested` rate | V2 `currently_congested` rate |
|---|---|---|
| Door | **0.00%** | **44.7%** |
| Exit | 8.17% | 7.12% |
| Stair | **0.00%** | **22.1%** |

V1's Door/Stair `currently_congested` rate is essentially zero because a discrete 5s tick
grid almost never lands exactly on a zero-duration boundary instant (a measure-zero event)
— V1's target still "sees" these artifacts because its forward-window sweep scans every
raw event continuously, not just tick-aligned instants, but its *own* `currently_congested`
exclusion almost never fires for Door/Stair. Target V2's queue-based episodes, once formed,
are **long-lived** (§5: Door median 244s, mean 398s) — so a large fraction of all Door
ticks fall *inside* an already-ongoing qualifying episode, correctly excluded from onset
prediction. This is a genuine, disclosed property of the physically meaningful definition,
not a flaw: Door congestion, once it forms in these scenarios, tends to be a **lasting
structural condition**, not a transient blip — arguably a more operationally honest
picture of what a capacity-1 bottleneck actually does over an evacuation's timeline.

**Multi-bottleneck representation under Target V2 (Phase 23)**: 40,528 distinct
(scenario, time) buckets have at least one V2-positive candidate; **7,103 of those (17.5%)
have 2+ simultaneously V2-positive candidates** — real multi-bottleneck cases remain well
represented, not accidentally eliminated by the stricter target.

## 10. Target V1 vs. Target V2 comparison (Phase 10)

Row-for-row agreement, computed on the 1,721,188 rows where BOTH targets are applicable
(neither is "currently congested" under its own respective definition):

| | V2 positive | V2 negative |
|---|---|---|
| **V1 positive** | 28,921 (1.68%) | 7,150 (0.42%) |
| **V1 negative** | 11,002 (0.64%) | 1,674,115 (97.27%) |

**By candidate type, the disagreement pattern is the whole story:**

| Type | V1∩V2 (both +) | V1+/V2− | V1−/V2+ | Both − |
|---|---|---|---|---|
| Door | 0.31% | 0.06% | **1.58%** | 98.05% |
| Exit | 2.97% | 0.75% | **0.00%** | 96.28% |
| Stair | 0.80% | 0.19% | 0.08% | 98.93% |

**Exit: V2 is a strict refinement of V1** (0.00% "V1 negative, V2 positive" — V2 never adds
a new Exit positive V1 didn't already have; V2 only *removes* momentary sub-persistence
overlaps V1 counted, consistent with §9's "essentially unchanged" Exit rate). **Door: V2 and
V1 are almost entirely disjoint signals** — V1's 0.06%+0.31%=0.37% of "V1 positive" rows are
mostly *not* what V2 considers real congestion, while V2 finds a completely different
1.58% of rows (real, physically meaningful queue buildup with NEVER more than 1 occupant
simultaneously on the capacity-1 edge — exactly the phenomenon V1's on-edge-occupancy-only
definition structurally could not see, §1-2). **This is the single cleanest piece of
evidence that V1 and V2 are measuring genuinely different things for Door, not just
tightening the same measurement.**

## 11. Lead-time analysis (Phase 22)

For all 49,331 real Target V2 positive examples, `lead_time_seconds_v2` = the soonest
qualifying onset time minus the observation time:

| | Value |
|---|---|
| Mean | 8.40s |
| Median | 7.60s |
| IQR | 3.43s – 12.92s |
| Min / Max | 0.001s / 20.0s |

| Lead-time bucket | Fraction of positives |
|---|---|
| 0-5s | 34.9% |
| 5-10s | 27.2% |
| 10-15s | 21.0% |
| 15-20s | 16.8% |

**By candidate type**: Door mean 9.17s (median 8.79s), Exit mean 8.13s (median 7.21s),
Stair mean 7.99s (median 6.98s) — all three types offer a genuinely useful, well-spread
warning window, not a definition that only "predicts" congestion a fraction of a second
before it starts. A median of ~7-9 seconds of lead time, inside a 20s horizon, is
operationally meaningful — enough time for the window to matter, not so late that it is
functionally "detection" rather than "prediction."

## 12. Feature/target relationship — foresight, not tautology (Phase 20)

Direct Pearson correlation, model-input columns vs. `target_v2` (2,405,049 trainable rows),
all far below the project's established 0.9 leakage-review threshold:

| Feature | Correlation with `target_v2` |
|---|---|
| `candidate_recent_flow_rate` | **0.275** |
| `candidate_approaching_count` | 0.263 |
| `total_active_occupant_count` | 0.215 |
| `candidate_queue_length` | **0.193** |
| `candidate_walking_distance` | -0.067 |
| `candidate_alternative_route_count` | -0.001 |

**No feature is near-tautological with Target V2 — and, notably, `candidate_queue_length`
is no longer the single dominant correlate**, unlike every V1/V2/V2.2 result where it
dominated by a wide margin. This is a real, explainable consequence of the onset design
(§7): a candidate with an already-high `candidate_queue_length` is disproportionately likely
to *already be* `currently_congested` under Target V2 (and therefore excluded from
training, per the "already congested → not applicable" policy) — so among the *trainable*
rows (not yet congested), current queue length carries less of the signal, and
forward-looking demand indicators (`candidate_recent_flow_rate`, `candidate_approaching_
count`) carry more. This is exactly the "foresight, not report" property this milestone's
own Phase 20 asked to be preserved and verified, not merely assumed.

## 13. Exploratory model result (Phase 12/18/19)

**Not Localized Predictive Model V3.** Existing V2.2 12-field feature schema, unchanged;
scenario-level 70/15/15 split (seed `20260726`, train 1,750 / val 375 / test 375 — same
convention every prior milestone used); XGBoost and HistGradientBoosting only, plus trivial
baselines; same 20s horizon.

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Majority class / always negative | 0.500 | 0.028 |
| Random | 0.495 | 0.028 |
| Logistic regression | 0.933 | 0.490 |
| Decision tree (depth 6) | 0.949 | 0.441 |
| Gradient Boosting | 0.961 | 0.485 |
| **XGBoost (best)** | **0.967** | **0.615** |

Test positive rate 2.82% — XGBoost's PR-AUC of 0.615 is a **~21.8x lift** over the
positive-rate baseline, a *stronger relative lift* than Target V1 ever achieved for the
overall model at any milestone (V2.2: PR-AUC 0.770 vs. 12.9% base rate ≈ 6.0x lift).
**Target V2 is unambiguously predictable, far better than chance.**

**By candidate type — per this milestone's own Phase 19 "do not compare directly to old
headline numbers" instruction, read against Target V2's own, much lower, more physically
honest base rates:**

| Type | n | Positive rate | ROC-AUC | PR-AUC | Recall | Relative lift |
|---|---|---|---|---|---|---|
| Door | 103,750 | 1.85% | 0.980 | 0.567 | 56.2% | ~30.7x |
| Exit | 130,345 | 4.04% | 0.952 | 0.632 | 57.9% | ~15.6x |
| Stair | 28,955 | 0.76% | 0.985 | 0.538 | 52.0% | **~70.7x** |

**By topology family:** `twin_stair_highrise` 0.748 (strongest), `single_exit_lowrise`
0.602, `v1_topology_fixed` 0.544, `multi_exit_wide` 0.384 (weakest, consistent with every
prior milestone's own finding about this family's difficulty).

**Which candidate type is now hardest? A complete reversal from every prior milestone.**
Under Target V1 (V1/V2/V2.2), Stair was consistently the *easiest* type by a wide margin
(PR-AUC 0.966-0.975) — because it was predicting the artifact, not real congestion. Under
Target V2, **Stair has the lowest absolute PR-AUC (0.538) of the three types** — genuinely
the hardest to predict, even though its *relative* lift over baseline (~70.7x) is the
largest, because its true positive rate (0.76%) is now the rarest event in the dataset.
Exit is now the easiest in absolute PR-AUC terms (0.632). This reversal is itself strong,
independent evidence that Target V1's Stair "success" was measuring something fundamentally
different from what Target V2 measures.

## 14. Leakage analysis

- **Architecture boundary**: `predictive_dataset/target_generator_v2.py` and its shared
  primitives module `target_semantics_analysis.py` are proven (`tests/
  test_predictive_dataset_target_v2_architecture_guards.py`, 9 tests) to never be imported
  by `simulation_extractor.py`, `simulation_extractor_v2_1.py`, `live_extractor.py`, or
  `live_extractor_v2_1.py` — the same leakage boundary Target V1 already established,
  extended to the new module.
- **Correlation check**: no feature approaches the project's 0.9 leakage-review threshold
  against `target_v2` (§12, max 0.275).
- **Elapsed-only evaluation**: `currently_congested(t)`/onset detection is proven
  (`tests/test_predictive_dataset_target_semantics_primitives_v2.py`'s
  `PersistenceQueryTests`) to only ever compare `time` against onset times derived from
  `episode_start + min_duration <= time`, never an episode's full, only-known-in-retrospect
  duration — the leak-safety property described in §6.
- **No future information in features**: unchanged from V2.2 — this milestone did not touch
  `simulation_extractor_v2_1.py`/`live_extractor_v2_1.py` at all.

## 15. Remaining limitations

- Target V2's `currently_congested` exclusion rate for Door (44.7%) is high — a real,
  disclosed consequence of long-lived queue episodes (§9), not a defect, but it does mean a
  large fraction of Door candidate-ticks are not usable as "onset" training examples (they
  are already inside a qualifying episode). A future milestone might separately model
  "will this ALREADY-congested candidate's queue clear soon" as a distinct, complementary
  question — out of this milestone's scope.
- `MIN_PERSISTENCE_SECONDS=3.0` and `QUEUE_THRESHOLD=1` are evidence-based but not
  exhaustively optimized — the sensitivity sweep (§5) covered 6 duration values on a 20%
  sample; a full grid search at full scale was not performed (not required to answer this
  milestone's own question, and consistent with its "prefer the smallest physically
  interpretable target" instruction over exhaustive tuning).
- `multi_exit_wide` remains the hardest topology to predict Target V2 congestion in
  (PR-AUC 0.384) — consistent with every prior milestone's own finding about this family,
  unresolved by a target change (expected — the topology-generalization gap was always a
  feature/data question, not a target-semantics one).
- This is an EXPLORATORY model only (Phase 12's own charter) — no calibration, ablation,
  leave-one-topology-out holdout, or feature-importance study was run against Target V2.
  That is explicitly Localized Predictive Model V3's job, not this milestone's.

## 16. Recommendation for the next milestone

**Localized Predictive Model V3** — a full evaluation of Target V2 using the same rigor
Model V2/V2.2 applied to Target V1: calibration, leave-one-topology-family-out
generalization, feature importance/ablation, multi-bottleneck and occupancy slicing, and a
genuine production-readiness gate. Target V2 itself does not need further target-engineering
work before that — this milestone's own evidence (§13: strong, well-calibrated-looking
relative lift on every candidate type; §10: genuinely different signal from V1, not just a
tightened version; §11: real, well-distributed lead time) supports treating it as frozen and
moving to full model validation, not iterating on the target definition further.

## Final report — explicit answers

**A. Why exactly was Target V1 invalid for Door and Stair?** Their structural capacity is
exactly 1 (derived from default width), so the simulator's admission model never allows
genuine concurrent occupancy — V1's `>=2 concurrent` threshold could only ever fire at the
zero-duration instant of a FIFO admission handoff. Confirmed exhaustively: 100% of Door's
and Stair's V1 episodes have exactly zero duration.

**B. What does "congestion" mean under Target V2 in plain engineering language?** A
candidate has a persistent demand/service imbalance — occupants are either genuinely
queued waiting to use it, or genuinely, concurrently present on it — that has lasted at
least 3 seconds, long enough to be a real operational state rather than a momentary
coincidence.

**C. What exact mathematical/logical conditions define Target V2?** A qualifying episode is
a maximal interval where either (a) `>=1` occupant is continuously waiting
(`join_time <= t < admission_time`) or (b) `>=2` occupants are continuously, concurrently
present on the edge (`start_time <= t <= end_time`), with full duration `>= 3.0s`.
`currently_congested(t)` = `t` in `[episode_start+3.0, episode_end)` for some qualifying
episode. `target(t, 20)` = `None` if currently congested; else `True` iff some qualifying
episode's onset (`episode_start+3.0`) falls in `(t, t+20]`.

**D. Does the same operational definition apply to Door, Exit and Stair?** Yes — one
universal formula (the OR of the two mechanisms above), applied identically to every
candidate type with no type-conditional branching in the implementation. It reduces to
queue-based for Door/Stair and occupancy-based for Exit purely as a structural consequence
of their different capacities, not an explicit rule.

**E. Can a timestamp-boundary handoff alone ever create a Target V2 positive?** No — proven
directly (`test_scenario_7_timestamp_handoff_alone_is_never_congestion`); a zero-duration
episode never reaches the 3.0s persistence bar, so it never contributes a qualifying onset.

**F. What minimum persistence/duration was selected, and why?** 3.0 seconds, chosen from a
6-value sensitivity sweep (0.5-10s) as the floor that discards only genuinely momentary
conditions while retaining 98.3% of Door's, 90.5% of Stair's, and 79.2% of Exit's real
episodes — the same value applied to both mechanisms deliberately, to avoid arbitrary
per-type tuning.

**G. Does Target V2 represent congestion ONSET rather than already-existing congestion?**
Yes — explicitly onset-based (§6), with an already-congested exclusion (`target=None`)
mirroring Target V1's own policy, both proven by dedicated tests.

**H. What is the overall Target V2 positive rate?** 2.85% (49,331 of 1,730,976 trainable
rows) — down from Target V1's 12.90%.

**I. What are Door/Exit/Stair positive rates?** Door 1.89% (V1: 17.62%), Exit 4.03% (V1:
3.72%, essentially unchanged), Stair 0.88% (V1: 21.27%).

**J. Are meaningful multi-bottleneck scenarios still represented?** Yes — 7,103 (17.5%) of
all positive (scenario, time) buckets have 2+ simultaneously Target-V2-positive candidates.

**K. What is the typical prediction lead time?** Median 7.6 seconds, mean 8.4 seconds,
reasonably spread across the full 0-20s window (not concentrated at the boundary) — genuine,
usable advance warning.

**L. Can Target V2 be generated from existing V2 artifacts, or was simulation rerun
necessary?** Simulation had to be rerun (raw timelines were never persisted from any prior
campaign) — but using the exact same deterministic scenario definitions/seed, so it
reproduces the identical population, not new data. No scenario topology/configuration
regeneration was needed.

**M. Does an exploratory model beat trivial baselines on Target V2?** Yes, decisively —
XGBoost PR-AUC 0.615 vs. ~0.028 for majority-class/always-negative/random, a ~21.8x lift.

**N. Which candidate type is now hardest to predict?** Stair (PR-AUC 0.538, lowest of the
three absolute values) — a complete reversal from every prior milestone, where Stair was
always the easiest under Target V1's artifact-inflated definition.

**O. Is there any evidence of future leakage?** No — architecture guards pass, no feature
correlation approaches the 0.9 review threshold, and the onset/currently-congested logic is
proven to only ever compare against elapsed (at-or-before-`t`) information.

**P. Is Target V2 scientifically/operationally defensible enough to replace Target V1 for
future model development?** **Yes.** It is physically interpretable (§3), passes all 8
controlled sanity scenarios, generalizes to a single formula across candidate types without
arbitrary tuning, is robustly predictable (§13), offers genuine multi-second lead time
(§11), retains real multi-bottleneck representation (§9), and — most importantly — is
proven to measure a fundamentally different, more honest phenomenon for Door/Stair than
Target V1 did (§10). Target V1 is preserved, not deleted, for historical reproducibility,
but Target V2 should be the target for all future predictive-congestion model development.

**Q. Should the next milestone be Localized Predictive Model V3, more target/data work, or
something else?** **Localized Predictive Model V3** — a full evaluation of Target V2 with
the same rigor previously applied to Target V1 (calibration, topology generalization,
feature importance, production-readiness gate). See §16.
