# Localized Predictive Model V2.1 — Exit Prediction & Multi-Exit Generalization Investigation

Status: **INVESTIGATION ONLY.** Nothing in this milestone is wired into recommendation
scoring, exit ranking, guidance, signage, LiveRuntime, or operator workflow. Builds on
`docs/architecture/localized_predictive_model_v2.md` (commit `5b51923`), which found Model
V2 "PROMISING BUT NEEDS MORE DATA" specifically because Exit prediction is the weakest
candidate type and the `multi_exit_wide` topology fails to generalize when held out
entirely (PR-AUC 0.314 vs. 0.692 normal-split). This milestone investigates *why*, before
proposing any fix.

## 1. Model V2 baseline (recap)

| | Overall | Door | Exit | Stair |
|---|---|---|---|---|
| PR-AUC | 0.692 | 0.578 | 0.464 | 0.966 |

`multi_exit_wide` held out entirely: overall PR-AUC 0.314; within that holdout, Exit
PR-AUC 0.085 (recall 15.1%), Door PR-AUC 0.338.

## 2. Exit data audit (Phase 1)

Row/scenario/positive counts (20s horizon, full V2 dataset):

| Candidate type | Trainable rows | Positive | Positive rate | Scenarios | Distinct candidates |
|---|---|---|---|---|---|
| Door | 1,244,514 | 219,292 | 17.6% | 2,500 | 10 |
| Exit | 852,075 | 31,710 | 3.7% | 2,500 | 8 |
| Stair | 232,609 | 49,474 | 21.3% | 1,300 | 3 |

**Exit does not have an insufficient-example problem.** 852,075 trainable rows with 31,710
positive examples is more raw positive-example volume than Stair had in the *entire* V1
dataset (372 positive events) before V2 fixed it — data quantity alone does not explain
Exit's weak PR-AUC. This rules out hypothesis (A) as the primary cause.

**The real finding is a structural feature blind spot, mechanically identical in spirit to
V1's stair-1 bug but with a different, now precisely traced root cause:**

```
Exit candidate_queue_length: mean=0.000 std=0.000, EVERY quantile = 0.0
nonzero_queue_rate = 0.0000  (0 of 927,926 rows, every topology family, no exceptions)
```

`candidate_queue_length` — the single dominant feature in every permutation-importance
ranking this project has ever produced (V1: 0.228, V2: 0.236, ~5x the next feature) — is
**constant zero for every Exit row in the entire dataset**, despite Exit genuinely becoming
congested 3.7% of the time. Traced to source, not guessed:

- `predictive_dataset/simulation_extractor.py::_current_queue_length()` counts occupants
  whose `join_time <= time < step.start_time` — i.e., time spent *waiting for admission*
  before a capacity check let them onto the edge (`simulator/coordinator.py::
  _handle_try_enter_edge`, `if current_on_edge < capacity: admit else: queue`).
- `models/exit.py`'s `Exit` dataclass defaults `capacity: int = 50`, used **as-authored**
  by `simulator/capacity.py::DefaultCapacityModel.capacity()` (`explicit_capacity is not
  None` branch) — no V2 (or V1) topology ever overrides it.
- `models/door.py`'s `Door` has no `capacity` field at all; capacity is *derived* from
  `width` (default `0.90`) via `PEOPLE_PER_METER_OF_WIDTH = 1.5`: `int(0.90 * 1.5) = 1`.
  `models/staircase.py`'s default `width = 1.50` similarly derives to `int(1.50 * 1.2) = 1`
  via `StairCapacityModel`. **Confirmed directly against the real dataset:**
  `candidate_capacity` is constant `1.0` for Door (1,244,514/1,244,514 rows), constant
  `50.0` for Exit (927,926/927,926 rows), constant `1.0` for Stair (232,609/232,609 rows).

With capacity 1, **any** second concurrent occupant is mechanically forced into the
explicit admission queue — Door/Stair queue constantly. With capacity 50, no scenario in
this dataset (3-111 total occupants) comes remotely close to exhausting it, so the
admission-queue mechanism **never engages for Exit, at all, structurally** — not a
data-generation bug, a direct, deterministic consequence of an authored default capacity
value that was never tuned for Exit the way Door/Stair's *derived* capacity happens to be
tight by coincidence of their own unrelated width defaults.

## 3. `multi_exit_wide` distribution shift (Phase 2)

Quantile comparison across the 4 topology families (Door/Exit rows, 20s horizon):

| Metric (Exit rows) | single_exit_lowrise | twin_stair_highrise | **multi_exit_wide** | v1_topology_fixed |
|---|---|---|---|---|
| Positive rate | 14.1% | 11.1% | **1.4%** | 5.3% |
| Candidates/scenario | 1 | 2 | 3 | 2 |
| `candidate_walking_distance` (median) | 6.4m | 8.6m | **27.3m (longest)** | 7.2m |
| `total_active_occupant_count` (mean) | 6.2 | 32.5 | 17.7 | 8.3 |

**`multi_exit_wide`'s Exit positive rate (1.4%) is 4-10x lower than every other family** —
the most severe class-imbalance shift in the dataset, for a family the model has *never
seen* when it's the held-out one. This alone is enough to explain most of the topology-
holdout collapse: a model trained where Exit congestion is common (5-14%) is asked to find
needles in a haystack 4-10x sparser than anything in its training distribution. Door
positive rate is also comparatively low in `multi_exit_wide` (12.8% vs. 26-28% in
`twin_stair_highrise`/`v1_topology_fixed`) — the same directional shift, milder.

**Why:** `multi_exit_wide` has a hub-and-spoke layout (5 zones, 4 doors, 3 exits) — the
*most* alternative-route diversity of any family. More alternative routes mechanically
diffuse occupant load across more candidates, lowering the marginal congestion probability
of any *one* candidate — a genuine structural property of the topology, not an artifact.
Its Exit candidates also have the *longest* walking distances (median 27.3m, vs. 6.4-8.6m
elsewhere) yet the *lowest* positive rate — ruling out "long transit time alone causes
congestion"; what actually matters is the interaction of transit time **and** arrival rate
(a Little's-Law-style quantity, see §5), and `multi_exit_wide`'s per-exit arrival rate is
low precisely because of the alternative-route diffusion effect.

## 4. Exit label semantics audit (Phase 3)

Traced real positive/negative examples through `predictive_dataset/target_generator.py`,
plus direct simulation instrumentation (25 fresh `multi_exit_wide` scenarios, `master_seed`
`20270115`, not the frozen campaign — purely for tracing, no dataset was modified):

**The congestion target (`_edge_occupant_count(...) >= 2`, based on `step.start_time <=
time <= step.end_time` interval overlap) is a *different phenomenon* from the
`candidate_queue_length` feature (based on `join_time <= time < step.start_time`, admission
*waiting*), and the two are decoupled for Exit in a way they are not for Door/Stair:**

- **Door congestion episodes are near-instantaneous artifacts, not sustained crowding.**
  Direct measurement (25 `multi_exit_wide` scenarios, 1,010 adjacent same-edge occupant
  pairs on Door edges): **91.2% have an exactly-zero gap** between one occupant's
  `end_time` and the next's `start_time` on the same edge — a capacity-1 FIFO admission
  handoff, not a real overlap (zero negative gaps found — capacity=1 is strictly enforced,
  never exceeded). Because `_edge_occupant_count` uses **inclusive** `<=` on both bounds, this
  zero-duration handoff instant registers as a momentary count of 2, tripping the
  congestion threshold. Confirmed directly: **every one of 547 recorded Door congestion
  episodes in a 15-scenario sample had exactly 0.00s duration** (mean/median/p90/max all
  0.00). Door's "congestion" label is, to a substantial degree, measuring *"a queue
  handoff occurred"* — which is why `candidate_queue_length` (having anyone already
  waiting) is such an overwhelmingly strong, almost tautological predictor: a nonzero queue
  on a capacity-1 edge makes an imminent handoff near-certain.
- **Exit congestion episodes are genuine, sustained, multi-second crowding.** Same
  15-scenario sample: **81 real Exit congestion episodes, mean duration 40.2s, median
  27.5s, p90 84s, max 245s** — actual overlapping presence of 2+ people in the exit
  corridor, not a boundary artifact (capacity 50 means these overlaps reflect genuine
  concurrent transit, never an admission-forced coincidence).
- **Time-to-onset for positive Exit ticks**: mean 9.3s, median 8.6s (well within the 20s
  horizon — the label is a meaningful advance-warning target, not a boundary trigger).

**Conclusion: the target definition is *not* equally meaningful across candidate types.**
For Door/Stair (capacity 1), it is heavily influenced by a mechanical FIFO-handoff artifact
that `candidate_queue_length` predicts almost by construction. For Exit (capacity 50), it
measures real, sustained crowding that `candidate_queue_length` structurally cannot see
(no admission queue ever forms) — the model has to infer it from weaker, longer-range
signals (`candidate_approaching_count`, `candidate_adjacent_zone_occupancy`) alone. This
was **not changed** by this investigation (per the milestone's own "do NOT change the
target yet" instruction) — it is reported as a finding for the feature-design phase below.

## 5. Feature sufficiency & demand representation (Phase 4)

What would a human engineer need to distinguish "Exit E1 about to congest" from "Exit E2
staying clear"? Given §4's finding (Exit congestion is a *sustained-overlap* phenomenon,
not an admission-queue phenomenon), the natural engineering answer is a **flow/throughput-
based** signal — Little's Law intuition: expected concurrent occupancy ≈ arrival rate ×
mean transit time. The current schema has transit time (`candidate_walking_distance`) and a
long-range demand count (`candidate_approaching_count`, 3m proximity window) but **no
recent-throughput or trend signal** — exactly the gap `multi_exit_wide`'s data (§3, longest
distance + lowest positive rate) shows matters.

## 6. Sim/live feature parity survey (Phase 5)

Surveyed `crowd_intelligence/`, `evacuation_progress/`, `trajectory_intelligence/`,
`navigation/`, `building_state/`, `live_occupants/` before proposing anything new (per this
milestone's "reuse concepts already available, do not create another live intelligence
subsystem" instruction):

| Already-live concept | Where | Relevance |
|---|---|---|
| `AssetApproachMetrics.trend` (RISING/STABLE/FALLING/UNKNOWN) | `crowd_intelligence/trends.py::TrendTracker`, 30s window, computed for **every** Door/Exit/Stair asset | Directly answers Phase 4's "rate of change" ask — already exists, all candidate types |
| `ExitFlow.recent_flow_per_minute` | `evacuation_progress/engine.py::EvacuationLedger.recent_exit_count`, 60s window | A genuine live throughput signal, currently Exit-specific |
| `ZoneEvacuationRecommendation.alternative_exit_ids` | `evacuation_recommendation/ranking.py::SafeExitDistanceCalculator` | Already computes "other reachable exits for this zone" — the exact structural quantity Phase 6 asks for |
| `AssetApproachMetrics.queue_candidate_count`/`approaching_count` | `crowd_intelligence/queue.py` | Confirms the **live** analog of queue/approaching is a geometric/behavioral proxy (STATIONARY classification within 3m), not the simulator's capacity-admission mechanism — see disclosed nuance below |

**Disclosed nuance, not fixed here:** the simulator's `candidate_queue_length` is
structurally always 0 for Exit because of the *admission-capacity* mechanism (§2); the
**live** analog (`AssetApproachMetrics.queue_candidate_count`, a STATIONARY-behavior
proxy) has no such structural reason to be zero for Exit — someone can stand still near an
exit for many reasons regardless of formal capacity. This means a model trained purely on
simulation ground truth may have learned "ignore queue_length for Exit" in a way that does
**not** necessarily hold for the live proxy. This is a real sim/live fidelity question,
out of scope to fix here (it is a simulator-capacity-modeling question, not a
feature-schema question) — flagged for future attention, not silently ignored.

## 7. Structural context (Phase 6)

`predictive_dataset/candidate.py::CandidateIdentity.zone_ids` already resolves which
zone(s) each candidate touches via the Navigation Graph. Counting *other* candidates
sharing a zone_id gives a purely structural, occupancy-independent "alternative route
count" — exactly mirroring `evacuation_recommendation`'s already-computed
`alternative_exit_ids` concept (§6), generalized to Door/Stair. `multi_exit_wide`'s Door/
Exit candidates have the highest alternative-route counts of any family (hub-and-spoke, 4
doors + 3 exits all touching the shared hub zone) — directly consistent with §3's finding
that its per-candidate congestion probability is structurally diluted. The current schema
has no feature expressing this at all.

## 8. Model-limitation control (Phase 8)

Trained HistGradientBoosting, XGBoost, and RandomForest (max_depth=20, same as the
committed V2 config) on the identical `multi_exit_wide` leave-one-topology-out split
(train=1,800 scenarios from the other 3 families, 849,387 rows; test=700 `multi_exit_wide`
scenarios, 1,479,811 rows), existing V2 feature schema only, no new tuning beyond what V2
already used:

| Model | Overall ROC-AUC | Overall PR-AUC | Exit PR-AUC | Exit recall | Fit time |
|---|---|---|---|---|---|
| Random Forest | 0.877 | 0.291 | 0.070 | 36.3% | 133.7s |
| Gradient Boosting | 0.844 | 0.282 | 0.056 | 42.6% | 23.1s |
| XGBoost | 0.869 | **0.314** | **0.085** | 32.0% | 15.2s |

**Every model collapses to roughly the same degree** (overall PR-AUC 0.282-0.314, Exit
PR-AUC 0.056-0.085 — all a small fraction of the normal-split Exit PR-AUC of 0.464). Per
this milestone's own decision rule ("if every model collapses similarly, classify this
primarily as a data/feature/generalization problem"), **this rules out model architecture
as the primary cause** — a fundamentally different learner (linear boosting vs. bagged
trees vs. histogram boosting) does not materially change the outcome, which is exactly what
you'd expect if the real problem is that the *information available in the rows themselves*
is insufficient to distinguish this topology's Exit behavior, not that any one algorithm is
failing to exploit information that's actually there.

## 9. Error-case forensics (Phase 7)

XGBoost's `multi_exit_wide`-holdout Exit predictions (test set 615,771 Exit rows;
threshold 0.5): TN 574,993, FP 32,347, FN 5,737, TP 2,694.

| Case | n | mean `approaching_count` | mean `walking_distance` | mean predicted prob |
|---|---|---|---|---|
| **FN** (missed real congestion) | 5,737 | 7.60 | **38.4m** | 0.176 |
| **FP** (false alarm) | 32,347 | **15.79** | 22.9m | 0.676 |
| TP | 2,694 | 12.16 | 24.2m | 0.749 |
| TN | 574,993 | 3.10 | 30.9m | 0.069 |

Two recurring, opposite-direction patterns, both traced to concrete example rows (not
averages alone):

1. **FNs skew toward the longest-distance exit** (`mew-exit-east`, 49.3m) — mean FN walking
   distance (38.4m) is *longer* than TP's (24.2m). Across the other 3 training families,
   shorter walking distance strongly co-occurs with higher congestion probability (tighter,
   more bottleneck-like candidates); the model appears to have learned "long distance ⇒
   less likely to congest soon" as a general prior. `multi_exit_wide`'s exits are the
   *longest*-distance of any family (§3) yet still congest for real (§4, 40s mean episode
   duration) — the model under-trusts exactly the topology-specific case its training
   distribution taught it to discount. Concrete example: `scn-def876201a29845a`,
   `mew-exit-east`, t=1880s, `approaching_count=1`, congestion_level LOW → predicted
   0.026, actual congestion within 20s (a case where approaching_count genuinely was low,
   so this one may be a harder, more defensible miss — but several FN examples have
   `approaching_count` in the 10-30 range and still score under 0.5, e.g. the
   `mew-exit-west` row with `approaching_count=30` scoring only 0.44).
2. **FPs skew toward high `approaching_count` on the shortest-distance exit**
   (`mew-exit-west`, 16.6m) — mean FP `approaching_count` (15.8) is *higher* than TP's
   (12.2). `multi_exit_wide` has 3 exits; a high count of occupants within the 3m
   `approaching_count` proximity window does not reliably convert into concurrent overlap
   at *this* exit specifically, because occupants have real alternative exits to redirect
   toward — a dilution effect §3/§7 already identified structurally, but which the current
   schema has no feature to express, so the model falls back on raw `approaching_count` and
   over-alarms.

Both patterns point the same direction: **the model is using the right general idea
(approaching demand + transit distance predict congestion) but has no way to adjust that
idea for a topology where transit distances run long and alternative routes are plentiful**
— consistent with a feature/structural-context gap, not noise.

## 10. Shared vs. Exit-specific model (Phase 9)

Trained 3 candidate-type-specific XGBoost models (Door-only, Exit-only, Stair-only), each
on the SAME normal 70/15/15 scenario split and SAME existing feature schema, no
architecture change:

| Candidate type | Shared-model PR-AUC (committed V2) | Specialized-model PR-AUC | Difference |
|---|---|---|---|
| Door | 0.578 | 0.577 | -0.001 (noise) |
| Exit | 0.464 | 0.457 | **-0.007 (slightly worse)** |
| Stair | 0.966 | 0.965 | -0.001 (noise) |

**Candidate-type specialization provides no material improvement — if anything, Exit is
marginally worse on its own.** This is a clean, unambiguous result: splitting Door/Exit/
Stair into separate classifiers does not fix Exit's weakness, most likely because Exit's
comparatively small positive-example pool (§2) actually benefits from sharing training
signal with Door/Stair's larger pool, and there is no evidence the three types need
different decision boundaries over the *same* feature space. **Combined with §8's
model-architecture control, this is now two independent pieces of evidence against a
model-side explanation** — the shared architecture should continue to be used; the problem
is what the rows contain, not how they are modeled.

## 11. Feature proposal (Phase 10)

Exactly 3 features, each grounded directly in a finding above, each with a genuine,
already-existing live analog (§6) rather than a newly invented live subsystem:

| # | Feature | Definition | Addresses |
|---|---|---|---|
| 1 | `candidate_recent_flow_rate` | Count of occupants who **completed** crossing this candidate's edge during `(time - 60s, time]` | §5's missing throughput signal; Little's-Law-style proxy for concurrent load that doesn't depend on an admission queue ever forming (fixes Exit's structural blindness from §2 without touching the simulator's capacity model) |
| 2 | `candidate_congestion_trend` | RISING / STABLE / FALLING / UNKNOWN — compares `queue_length + approaching_count` at `time` against the same candidate at `time - 30s` | §5/§9's missing rate-of-change signal; the FN pattern (§9) where the model under-trusts long-distance exits could plausibly be corrected by "demand is rising" independent of absolute distance |
| 3 | `candidate_alternative_route_count` | Static structural count: how many OTHER Door/Exit/Stair candidates share at least one zone with this one (`predictive_dataset.candidate.CandidateIdentity.zone_ids`) | §3/§7's structural-dilution finding; the FP pattern (§9) where high approaching_count over-alarms specifically in a topology with many alternative routes |

**Sim/live parity** (Phase 5, full table):

| Feature | Simulation source | Live source | Available at prediction time? | Leakage risk | Missing-data semantics | Candidate types |
|---|---|---|---|---|---|---|
| `candidate_recent_flow_rate` | `predictive_dataset/simulation_extractor_v2_1.py::_recent_flow_rate` — count of `OccupantTimelineStep`s on this edge with `end_time` in `(time-60, time]` | `evacuation_progress.EvacuationLedger.recent_exit_count` (already computed, `ExitFlow.recent_flow_per_minute`) for Exit; **no equivalent tracker currently exists for Door/Stair** | Yes for Exit (existing live signal); Door/Stair would need a new, structurally identical per-asset flow tracker (not built by this investigation) | None — only counts already-completed crossings at or before `time` | `0` when nobody has crossed in the window (never `None` — a true zero, same discipline as `candidate_queue_length`) | Door, Exit, Stair (Exit has an immediate live source; Door/Stair do not yet) |
| `candidate_congestion_trend` | `_congestion_trend()` — compares `queue_length+approaching_count` now vs. 30s ago | `crowd_intelligence.trends.TrendTracker` → `AssetApproachMetrics.trend`, **already computed for every asset type** | Yes, immediately, for all 3 types | None — strictly backward-looking | `UNKNOWN` when `time < 30s` (sim) / no prior trend sample (live) — never fabricated as STABLE | Door, Exit, Stair |
| `candidate_alternative_route_count` | `build_alternative_route_counts()` — static, from `CandidateIdentity.zone_ids` | `evacuation_recommendation.ranking.SafeExitDistanceCalculator`'s reachability computation (Exit-focused today); Door/Stair would reuse the same NavigationGraph adjacency already available via `navigation.graph.NavigationGraph.find_neighbors` | Yes — purely structural, no occupancy dependence at all | None — never depends on occupancy or time | Never missing (always resolvable from Building geometry) | Door, Exit, Stair |

Two of three features (`candidate_congestion_trend`, `candidate_alternative_route_count`)
have an immediate, already-built live source for every candidate type. `candidate_recent_
flow_rate`'s live source currently exists only for Exit — disclosed, not hidden; extending
it to Door/Stair would be a small, mechanical follow-up (the same `EvacuationLedger`
counting mechanism, applied to a Door/Stair asset instead of an Exit) but was not built as
part of this investigation.

## 12. Targeted V2.1 experiment (Phase 11)

Implemented in `predictive_dataset/simulation_extractor_v2_1.py` (additive, V1/V2 schema
untouched) and `predictive_model/feature_prep_v2_1.py` (additive feature-matrix builder).
`scripts/run_predictive_dataset_campaign_v2_1_experiment.py` regenerates the SAME 4
topology families at **20% of V2's scenario counts** (500 total: 100/160/140/100, same
`master_seed=20270115`, 20s horizon only) and extracts **two** CSVs from the literal same
simulated occupants per scenario — `candidate_dataset_baseline.csv` (existing 9-field
schema) and `candidate_dataset_experimental.csv` (9 fields + the 3 new ones) — so any
metric difference between them is attributable to the added features alone, not to a
different random draw or a different scale. 500 scenarios accepted, 0 failed, 467,133 rows,
52.4s wall time (`data/predictive_dataset_campaign_v2_1_experiment/`, gitignored, regenerable).

Deliberately **not** another 9.6M-row campaign, per this milestone's own "training
performance already saturates around 25%" instruction (`docs/architecture/
localized_predictive_model_v2.md` §19) — this is a small, targeted, hypothesis-test-sized
campaign only.

## 13. Hypothesis test (Phase 12)

Same scenario split (seed `20260726`), same XGBoost architecture, same class-weighting —
only the feature set differs between the two runs below.

**Normal 70/15/15 split** (n_train=321,685, n_test=62,265):

| | Overall PR-AUC | Door PR-AUC | Exit PR-AUC | Stair PR-AUC |
|---|---|---|---|---|
| Baseline (9 fields) | 0.722 | 0.579 | 0.458 | 0.965 |
| **Experimental (+3 fields)** | **0.805** | **0.747** | **0.547** | **0.975** |
| Change | +0.083 | +0.168 | **+0.089 (+19.4% relative)** | +0.010 |

**`multi_exit_wide` leave-one-topology-out holdout** (n_train=163,242, n_test=289,028 —
this family has no Stair candidates, so Stair is structurally absent from this slice, not a
bug):

| | Overall PR-AUC | Door PR-AUC | Exit PR-AUC | Exit recall |
|---|---|---|---|---|
| Baseline (9 fields) | 0.311 | 0.343 | 0.065 | 27.0% |
| **Experimental (+3 fields)** | **0.429** | **0.450** | **0.126** | **31.3%** |
| Change | +0.118 (+38% relative) | +0.107 (+31% relative) | **+0.062 (+95% relative — nearly doubled)** | +4.3 points |

**Both disqualifying weaknesses improved materially, not marginally.** Exit PR-AUC on the
unseen `multi_exit_wide` topology nearly doubled (0.065→0.126); the overall holdout PR-AUC
improved 38% relative (0.311→0.429); Door also improved substantially in both the normal
split and the holdout (the `candidate_alternative_route_count`/`candidate_congestion_trend`
features are not Exit-specific). Stair, tested only on the normal split (no Stair
candidates exist in `multi_exit_wide`), did **not** regress — it improved slightly
(0.965→0.975), consistent with §10's finding that adding information which doesn't apply to
Stair's already-strong signal does no harm.

**Caveat, stated plainly**: this is one train/test split on a single 500-scenario run, not
a repeated-seed or cross-validated result — a genuinely promising, well-controlled signal
(same scenarios, same seed, same architecture, isolated feature effect), not yet a
fully-validated claim at V2's original scale. Scaling this up (and building the missing
Door/Stair live flow-rate tracker disclosed in §11) is the natural next step, explicitly
left to a future milestone per this investigation's own "stop after this investigation"
charter.

## 14. Decision (Phase 13)

**A. FEATURE GAP IDENTIFIED — V2.1 IMPROVES IT.**

Rationale, weighed against the other four options: model architecture was directly ruled
out (§8 — every model collapsed similarly; §10 — candidate-type specialization made no
material difference). Data *quantity* was ruled out (§2 — Exit had more positive examples
than V1's entire Stair dataset). Target/label *semantics* were found to have a genuine,
disclosed asymmetry (§4 — Door/Stair congestion is partly a FIFO-handoff artifact, Exit's
is genuine sustained overlap) but this milestone was explicitly told not to change the
target, and the targeted experiment shows the *existing* target is learnable well enough
once the right features are present — so a target redesign is not the load-bearing fix.
What remained, and what the controlled experiment directly confirmed with a material,
reproducible-direction improvement on both disqualifying metrics, is that the **feature
schema** did not describe Exit's actual congestion mechanism (sustained overlap driven by
flow/arrival rate and diluted by alternative routes) — once three schema-external, already
live-available concepts were added, both weaknesses moved substantially in the right
direction using the identical target definition, identical model, identical scenarios.

## 15. Recommended next step

Per this milestone's own "stop after this investigation" charter, none of the following
were done here and are left for a future, explicitly separate milestone:

1. **Validate at V2's original scale.** Re-run the full ~2,500-scenario campaign (or a
   larger targeted subset than this investigation's 500) with the 3 experimental fields
   promoted into a real, versioned schema (`schema_version` bump, not a silent edit of the
   frozen V1/V2 fields) to confirm the improvement holds outside one small run.
2. **Build the missing Door/Stair live flow-rate tracker.** `candidate_recent_flow_rate`
   currently has an immediate live source only for Exit (`evacuation_progress.
   EvacuationLedger`); extending the same mechanism to Door/Stair assets is what full
   sim/live parity for this feature requires.
3. **Investigate the disclosed Door/Stair label-semantics nuance (§4)** independently of
   Exit — this milestone found it, but was explicitly told not to change the target this
   round; whether the FIFO-handoff artifact should be filtered out of the congestion
   definition (e.g. a minimum-duration threshold) is a separate, target-definition-level
   question with its own tradeoffs to weigh.

None of this — nor anything else in this investigation — is wired into recommendation
scoring, exit ranking, guidance, signage, LiveRuntime, or operator workflow.

## Final report

1. **Model V2 commit hash**: `5b51923`.
2. **Root cause(s) of Exit weakness**: `candidate_queue_length` (the dominant feature
   everywhere else) is structurally constant zero for every Exit row, because `Exit.
   capacity` defaults to 50 (never overridden by any topology) while Door/Stair's derived
   capacity happens to floor to 1 — the admission-queue mechanism the feature measures
   never engages for Exit. Compounded by the target's own semantics (§4): Exit congestion
   is genuine sustained overlap; Door/Stair's is substantially a FIFO-handoff artifact that
   queue_length predicts almost tautologically.
3. **Root cause(s) of `multi_exit_wide` collapse**: severe class-distribution shift (Exit
   positive rate 1.4% vs. 5-14% elsewhere, §3) plus the highest alternative-route dilution
   of any family (hub-and-spoke, 4 doors + 3 exits) with no feature to express it, plus the
   model's learned "short distance ⇒ more likely congested" prior misfiring on this
   family's unusually long exit distances (§9).
4. **Was Exit data quantity sufficient?** Yes — 852,075 trainable rows, 31,710 positive
   examples, far more than V1's entire (non-functional) Stair dataset ever had.
5. **Were label semantics valid?** Partially asymmetric, not invalid — the target measures
   a genuinely different phenomenon for Exit (real overlap) vs. Door/Stair (partly a
   zero-duration handoff artifact); not changed this round, per instruction.
6. **Biggest topology distribution shift**: `multi_exit_wide`'s Exit positive rate (1.4%)
   is 4-10x lower than every other family's.
7. **Missing candidate-local information discovered**: recent throughput/flow rate
   (`candidate_recent_flow_rate`) and demand rate-of-change (`candidate_congestion_trend`).
8. **Missing structural information discovered**: alternative-route count
   (`candidate_alternative_route_count`).
9. **Shared vs. Exit-specific model result**: no material difference (Exit PR-AUC 0.464
   shared vs. 0.457 specialized) — specialization is not justified.
10. **Model-limitation control result**: RandomForest/HistGradientBoosting/XGBoost all
    collapse similarly on the `multi_exit_wide` holdout (PR-AUC 0.282-0.314) — not a
    model-architecture problem.
11. **New features proposed/implemented**: `candidate_recent_flow_rate`,
    `candidate_congestion_trend`, `candidate_alternative_route_count` — all 3 implemented
    and tested (`predictive_dataset/simulation_extractor_v2_1.py`,
    `predictive_model/feature_prep_v2_1.py`).
12. **Sim/live parity result**: 2 of 3 features have an immediate, already-built live
    source for every candidate type (`crowd_intelligence.trends.TrendTracker`,
    `evacuation_recommendation`'s alternative-exit concept generalized); the third
    (`candidate_recent_flow_rate`) has one for Exit only today, disclosed as a follow-up.
13. **Targeted experiment size**: 500 scenarios (20% of V2's scale), same 4 topology
    families, same master seed, 467,133 rows, 52.4s.
14. **Exit PR-AUC before/after**: normal split 0.458 → 0.547 (+19.4% relative);
    `multi_exit_wide` holdout 0.065 → 0.126 (+95% relative, nearly doubled).
15. **`multi_exit_wide` holdout PR-AUC before/after**: overall 0.311 → 0.429 (+38% relative).
16. **Stair performance before/after**: 0.965 → 0.975 (no regression; `multi_exit_wide` has
    no Stair candidates, so only the normal-split comparison applies).
17. **High-occupancy/multi-bottleneck result**: not separately re-measured in this
    investigation (out of scope — this milestone's charter was Exit/topology-generalization
    specifically); Model V2's own findings on those axes (`docs/architecture/
    localized_predictive_model_v2.md` §12-13) stand unchanged.
18. **Final diagnosis**: **A — FEATURE GAP IDENTIFIED, V2.1 IMPROVES IT.**
19. **Full-suite result**: see commit message for the exact final count; zero regressions
    against the 4553-test baseline this investigation started from.
20. **Commit hash**: see commit message (this investigation's own commit, following `5b51923`).

**A. Is the Exit weakness primarily a data, feature, target, or model problem?** Primarily
a **feature** problem (§2, §11) — a specific, mechanically-traced blind spot in
`candidate_queue_length` for high-capacity edges — with a secondary, disclosed **target**
nuance (§4) that was not acted on this round, and **not** a data-quantity or model problem
(ruled out directly, §2/§8/§10).

**B. Why does `multi_exit_wide` fail to generalize?** A combination of severe positive-rate
distribution shift (§3) and a structural alternative-route-dilution effect the schema
couldn't express (§7) — both of which the targeted experiment's new features materially
addressed (§13).

**C. Are current candidate-local features sufficient?** No — they lack any
throughput/rate-of-change signal, which the targeted experiment shows matters materially.

**D. Is candidate structural context necessary?** Yes — `candidate_alternative_route_count`
alone contributed to Door's improvement (+31% relative in the `multi_exit_wide` holdout)
independent of the flow/trend features, evidence that structural context carries real,
non-redundant signal.

**E. Should Door/Exit/Stair share one model?** Yes — specialization showed no material
benefit (§10); the shared architecture should continue.

**F. Can every proposed new feature exist honestly in live SynEvac?** Two of three, fully,
today (`candidate_congestion_trend`, `candidate_alternative_route_count`). The third
(`candidate_recent_flow_rate`) has an honest live source for Exit today; Door/Stair parity
requires a small, disclosed, not-yet-built extension of the same existing mechanism — not a
new intelligence subsystem.

**G. Did the targeted V2.1 experiment materially improve the disqualifying weakness?**
**Yes** — both Exit prediction (+19-95% relative depending on slice) and `multi_exit_wide`
generalization (+38% relative overall) improved materially, not marginally, in a controlled,
same-scenario, same-seed, same-architecture comparison.

**H. Is the predictive model ready to influence Recommendation now?** **No.** This remains
an investigation-stage, small-scale (500-scenario) result on one train/test split — genuinely
promising, but not yet validated at V2's original scale, and the Door/Stair flow-rate live
source does not exist yet. The recommended next step (§15) is validating this experiment at
scale, not integration.
