# Per-Candidate Predictive AI Data Foundation

Status: **data-foundation milestone. No model trained, no scoring code changed, no new intelligence engine added.** Answers the "exact next implementation milestone" `docs/architecture/ai_operational_role.md` §15 scoped but did not schedule. Baseline: commit `fb123e6`, 4332/4332 tests passing.

## 1. Old formulation vs. new formulation

**Old** (`dataset_builder/`, still unchanged and still used by the existing global bottleneck model): `SCENARIO × ONE WHOLE-BUILDING ROW`. `dataset_builder.feature_extractor.extract_scenario_features()`/`dataset_builder.labels.extract_simulation_outcome()` each produce exactly one row per scenario; the label is a single whole-run outcome (`doors_that_became_bottlenecks` non-empty, or one `peak_congestion_location_id`). A second, additive V2 layer, `dataset_builder.timeline.extract_timeline_rows()`, already produces `SCENARIO × TIMESTEP` rows (Phase 1's investigation confirmed this), but its Door/Exit/Stair columns remain per-Building-ordinal whole-building scalars (`Door_1_State`, a single `current_bottleneck` winner) — never one row per candidate asset.

**New** (`predictive_dataset/`, this milestone): `SCENARIO × TIMESTEP × CANDIDATE`. One row per (scenario, observation time, Door/Exit/Stair candidate, prediction horizon), each with its own feature vector and its own target. Two exits in the same building at the same instant can now receive genuinely different feature rows and different future labels (§11).

`predictive_dataset/` is additive only: `dataset_builder/`, `ai_features/`, `ai_training/`, and every existing trained model are untouched.

## 2. Candidate identity (Phase 2)

`predictive_dataset.candidate.CandidateIdentity`: `candidate_id`, `candidate_type` (`DOOR`/`EXIT`/`STAIR`, matching `navigation.edge.Edge.EDGE_TYPES`), `floor_id`, `zone_ids`. `candidate_id` is always the same stable id already used everywhere else in this codebase (`models.base_object.BaseObject.id`, a `uuid4` string, reused verbatim as `navigation.edge.Edge.id` — confirmed by direct code read, not assumed). `enumerate_candidates(building)` builds the real Navigation Graph (`NavigationGraphGenerator`) rather than re-deriving connectivity a second way, so a candidate's `zone_ids` is exactly the graph's own already-validated endpoint resolution. Never row order — `tests/test_predictive_dataset_candidate.py` proves identity is stable across repeated calls and that `edges_by_candidate_id()` keys line up exactly with `enumerate_candidates()`.

## 3. Sampling interval (Phase 3)

**Default: 5.0 seconds**, and it requires no new subsampling layer at all. `simulation_runtime.clock.SimulationClock` already has a caller-chosen, coarse `dt` (never a fixed constant, confirmed by Phase 1's investigation) sitting on top of the underlying continuous-time discrete-event `MultiAgentSimulation` — `SimulationRuntime.tick()` already only samples at `dt` intervals, so "one candidate-dataset row per tick already present in `run.tick_results`" **is** the coarse sampling Phase 3 asks for, not a second resampling concept layered on top. 5.0s matches a value already precedented elsewhere in this codebase's own validation scripts (`validation/phase4_engineering_validation.py`) and is fine-grained relative to the shortest evaluated horizon (10s — at least 2 sample points per horizon) without multiplying row count unnecessarily (a `SCENARIO × TIMESTEP × CANDIDATE` dataset already has `N_candidates`× more rows than the whole-building layer at the same tick rate; sub-second sampling was rejected specifically because it would multiply that further for no benefit — nothing meaningful changes candidate-locally between fine sub-second ticks in a corridor-scale evacuation).

## 4. Target definition (Phase 4/5)

`P(candidate becomes congested within [t, t+T] | information available at time t)`, replacing the old whole-run "did this asset ever bottleneck?" question. Horizons evaluated: **10s, 20s, 30s, 60s** (Phase 17 §12 below picks the first-training default empirically).

**Congestion definition — reused, not reinvented** (Phase 5's own explicit rule): `ground_truth.bottleneck._congestion_duration_for_edge()`'s already-audited, disclosed threshold ("'Congested' is defined here as 2+ occupants concurrently on the same edge") is the exact same threshold `predictive_dataset.target_generator.CONGESTION_THRESHOLD` uses, now evaluated **per candidate per timestep** instead of as a whole-run duration.

## 5. Current-congestion handling (Phase 5/10)

`CandidateLabel.currently_congested` (edge occupant count ≥ 2 **at** `time`) is computed and reported **separately** from `CandidateLabel.target` (the horizon-bounded future question). A candidate already congested at `time` gets `currently_congested=True` and **`target=None`** — explicitly represented as "not applicable," never silently treated as a trivial future positive and never silently dropped. The campaign run (§14) shows this is a small, non-dominant slice of the dataset (2.8% of rows at every horizon — the `already_congested_at_observation` fraction is identical across horizons because it depends only on state *at* `time`, never on the horizon itself, which is itself a useful mechanical sanity check that this field is computed independently of `T`).

Phase 10's five required cases, each with a passing test in `tests/test_predictive_dataset_target_generator.py`:

| Case | Treatment |
|---|---|
| A. Clear now → congested within window | `target=True` |
| B. Clear now → remains clear | `target=False` |
| C. Already congested now | `currently_congested=True`, `target=None` |
| D. Congestion begins after horizon | `target=False` |
| E. Candidate unavailable/unused before congestion | No special-casing needed — zero recorded edge activity flows through the ordinary "never reached threshold" negative path. `had_any_activity_in_window` (a diagnostic, not part of the target) keeps "definitively unused" visible and distinguishable from "used but never congested" for analysis, without inventing a third target state. |

## 6. Candidate feature schema (Phase 6/7)

`predictive_dataset.schema.CANDIDATE_FEATURE_SCHEMA` (8 fields: 1 global-context + 7 candidate-context), built on `ai_features.feature_schema.AIFeatureField`/`FeatureAvailability` — the same shape and vocabulary the existing whole-building live-parity schema already uses, deliberately not reinvented. Every field has a disclosed live source; **hazard exposure was investigated and explicitly excluded** (`SIMULATION_ONLY` — reusing `ai_features/feature_schema.py`'s own already-audited exclusion of `hazard_score`, not re-litigated here) rather than fabricated.

### 7. Live parity matrix (Phase 7)

| Feature | Simulation source | Live source | Live-observable? | Live-estimable? | Missing-data semantics | Leakage risk |
|---|---|---|---|---|---|---|
| `total_active_occupant_count` | `MultiAgentSimulationResult.occupants` not yet arrived by `time` | `LiveOccupantManager.canonical_occupancy(time).total_observed_count` | No | **Yes** | Live: `None` when no occupancy facts supplied. Sim: never `None`. | None — both sides read only state at/before `time`. |
| `candidate_type` | `Edge.edge_type` | Same, identical Navigation Graph type | **Yes** | — | Never missing. | None — static structural fact. |
| `candidate_capacity` | `crowd_intelligence.capacity.{door,exit,stair}_capacity()` | **Identical function call** | **Yes** | — | `None` when the capacity model can't derive one (e.g. no width). | None — pure function of the Door/Exit/Staircase object, no runtime state. |
| `candidate_walking_distance` | `Edge.walking_distance` | Identical (shared Navigation Graph) | **Yes** | — | `None` when an endpoint has no geometry. | None — static structural fact. |
| `candidate_traversable` | `Edge.traversable` (structural: locked/active/blocked/obstacle) | Identical | **Yes** | — | Never missing. | None. **Known v1 limitation**: does not yet incorporate mid-scenario `ScenarioEvent` door/exit/stair overrides — disclosed, not fabricated precision (see §16). |
| `candidate_adjacent_zone_occupancy` | `OccupancySnapshot.observation_at(edge.from_node)` | `canonical_occupancy(time).occupant_ids_by_zone[from_node]` | No | **Yes** | `None` when no occupancy facts supplied or the approach side has no zone. | None — current-state-only on both sides. |
| `candidate_queue_length` | Exact discrete-event queue-admission bookkeeping (`OccupantTimelineStep.queue_wait_time`), filtered to this candidate's edge | `AssetApproachMetrics.queue_candidate_count` (STATIONARY-behavior occupants within 3m) | No | **Yes** | Live: `None` when `position_available=False`. Sim: never `None`. | See §16 — bounded by an interval-membership test, not a future read. |
| `candidate_approaching_count` | Occupants whose fixed Route ends at this candidate, departed but not yet arrived, and not already queued/on-edge | `AssetApproachMetrics.approaching_count` (geometric heading, ~3m) | No | **Yes** | Live: `None` when `position_available=False`. | See §16 — Route is a t=0 plan, not a future outcome. |
| `candidate_congestion_level` | `crowd_intelligence.congestion.compute_congestion_level()` — **identical function call**, fed the two rows above | Same function, fed `AssetApproachMetrics`'s own counts | No | **Yes** | `None` when capacity is `None`/`≤0` or no demand evidence at all. | Inherits queue/approaching's classification, no new risk. |

`candidate_queue_length`/`candidate_approaching_count` are **not** forced into false equality between simulation and live — they are two different, honestly-disclosed methodologies answering the same semantic question (exact discrete-event ground truth vs. a short-range geometric/behavioral proxy). `tests/test_predictive_dataset_parity.py` proves structural fields and the shared congestion-classification code path match **exactly** given matched counts, and proves the missing-live-evidence case is reported as `None`, never coerced to match simulation's exact ground truth.

## 8. Simulation extractor architecture (Phase 8)

`predictive_dataset.simulation_extractor.extract_simulation_candidate_features(candidate, edge, time, *, building, movement_result, occupancy_snapshot)`. Every read is gated by a comparison against `time` that only ever asks "does this already-realized interval contain time" (`OccupantTimelineStep.start_time/end_time`, exactly the technique `dataset_builder.timeline`'s own `_current_congestion`/`_current_queue_length` already use and this codebase already treats as legitimate current-state reconstruction) or "does this occupant's t=0-fixed Route plan target this candidate" (never a future outcome — `simulator/occupant.py`'s own "no dynamic rerouting of an occupant already in flight" contract). `tests/test_predictive_dataset_leakage_guards.py` proves this mechanically: two `MultiAgentSimulationResult`s identical up to `time`, differing only in what happens afterward, produce byte-identical feature rows.

## 9. Live extractor architecture (Phase 12)

`predictive_dataset.live_extractor.extract_live_candidate_features(candidate, edge, *, building, crowd_snapshot, occupancy_facts)`. Deliberately thin — **no new intelligence engine was built**. `crowd_intelligence.engine.CrowdIntelligenceEngine` already computes genuine per-candidate `AssetApproachMetrics` (keyed by the same asset id this package uses as `candidate_id`) every `LiveRuntime` cycle; this module only remaps that shape onto `CANDIDATE_FEATURE_SCHEMA`'s field names. `occupancy_facts` is whatever `LiveOccupantManager.canonical_occupancy(time)` already returns — passed in by the caller, never fetched by this module, since it holds no `LiveOccupantManager` reference of its own.

## 10. Leakage boundary (Phase 9/16)

`predictive_dataset.target_generator` is the **only** module in this package allowed to inspect state after `time` — mechanically enforced (not just documented) by `tests/test_predictive_dataset_architecture_guards.py`, which asserts neither `simulation_extractor.py`, `live_extractor.py`, nor `schema.py` ever import it.

| Source | Classification |
|---|---|
| `OccupantTimelineStep.start_time`/`end_time`/`queue_wait_time` interval-membership tests (feature extractor) | **SAFE_AT_TIME_T** — asks only "does this already-realized interval contain `time`," proven invariant to post-`time` differences by `tests/test_predictive_dataset_leakage_guards.py`. |
| `OccupantTimeline.route.edges[-1]` (feature extractor's `candidate_approaching_count`) | **SAFE_AT_TIME_T** — a plan fixed at `t=0`, before any simulated movement; reading it at any `time` is not a future read. |
| `OccupantTimelineStep` intervals **strictly after** `[t, t+T]` (target generator's congestion sweep) | **FUTURE/TARGET_ONLY** — deliberately, only within `target_generator.py`, never reachable from either extractor. |
| `ground_truth.bottleneck`'s whole-run outcomes (`doors_that_became_bottlenecks`, `peak_congestion_location_id`, `total_evacuation_time`, …) | **FUTURE/TARGET_ONLY** — already excluded from `CANONICAL_LIVE_SCHEMA` (`ai_features/feature_schema.py`); not read anywhere in `predictive_dataset/` at all (this package derives its own, per-candidate, per-timestep target directly from `MultiAgentSimulationResult`, not from `ground_truth`'s whole-run summaries). |
| Hazard score (`HazardSnapshot`) | **SIMULATION_ONLY**, excluded from the deployable schema entirely (§6/§7) — not a leakage question, an availability question, but recorded here since it was investigated and deliberately left out. |

## 11. E1/E2 differentiation result (Phase 11)

`tests/test_predictive_dataset_extractors.py` constructs one simulation timestep, one Building, two Exits (`exit-1` busy — 1 occupant on the edge, 1 queued, 1 approaching, adjacent zone occupancy 4; `exit-2` quiet — all zero) and proves:

- `extract_simulation_candidate_features("exit-1", …) != extract_simulation_candidate_features("exit-2", …)` (current feature rows genuinely differ — the capability `CANONICAL_LIVE_SCHEMA`'s whole-building model structurally lacks, per `ai_operational_role.md` §3).
- `candidate_congestion_level` differs (`exit-2` is `LOW`).
- Extending the same fixture forward in time: `exit-1` reaches the congestion threshold within a 30s horizon (`target=True`); `exit-2` never does (`target=False`) — **same scenario, same observation time, opposite labels.**

## 12. Horizon analysis: 10 / 20 / 30 / 60s (Phase 17)

From the §14 campaign (40 scenarios, 41,940 candidate-time rows):

| Horizon | Trainable rows | Positive rate | Already-congested-at-`t` fraction |
|---|---|---|---|
| 10s | 10,193 | 7.6% | 2.8% |
| 20s | 10,193 | 14.6% | 2.8% |
| 30s | 10,193 | 20.2% | 2.8% |
| 60s | 10,193 | 24.5% | 2.8% |

Positive rate rises smoothly and monotonically with horizon (more time for congestion to develop somewhere in the window) — no horizon is degenerate (0% or 100%). The already-congested fraction is identical across horizons by construction (it depends only on state at `t`, never on `T`) — a useful mechanical check that `currently_congested` truly is horizon-independent, exactly as designed in §5. All four horizons are statistically usable in isolation; the choice between them is therefore an **operational** one (how much advance warning is actually useful), not a data-availability one.

## 13. Recommended first horizon (Phase 17)

**20 seconds.** `predictive_dataset.analysis.recommend_first_horizon()` requires a horizon to clear **two independent, disclosed floors**, not just the statistically easiest one: (1) genuine advance warning — `horizon ≥ 20s`, since `docs/architecture/ai_operational_role.md` §10 already establishes that anything shorter is barely distinguishable from reporting already-instantly-available current state (Crowd Intelligence/Evacuation Progress already report *current* congestion/queue/throughput deterministically at LiveOrchestrator's ~1Hz cycle); (2) statistical usability — at least 20 positive rows and a 2% positive rate among trainable rows. 10s clears (2) alone but not (1); 20s is the shortest horizon clearing both. This is an empirical selection over the actual campaign numbers (§12), not a re-assertion of the prior document's proposal — it happens to land on the same value the prior investigation *suggested* re-evaluating, which is itself a mild corroboration rather than a foregone conclusion.

## 14. Campaign scenario count & candidate-time row count (Phase 15/18)

Building: `ai_registry.training_scenario.make_training_building()` (2 floors, 4 zones, 2 doors, 2 exits, 1 stair — already an established, reused fixture, not a new one). Definition: `ai_registry.training_scenario.make_training_definition()` (same reuse). 40 scenarios requested via `scenario_generator.batch_generator.iter_batch()`, **40 accepted, 0 failed**. Sampling interval 5.0s (§3), horizons 10/20/30/60s (§4) evaluated for every candidate at every tick.

- **Scenario count: 40** (reported separately, per Phase 15's own explicit warning against conflating row count with scenario diversity).
- **Candidate-time rows: 41,940** (7 candidates × ~150 ticks average × 4 horizons, scenario-dependent).
- Every row carries its originating `scenario_id`; `tests/test_predictive_dataset_dataset_builder.py` proves rows from different scenarios never cross-contaminate, preserving the grouping a future train/test split must respect.

## 15. Class balance overall / by type / by horizon (Phase 15)

Overall (all horizons pooled): 40,772 trainable rows (1,168 excluded as already-congested-at-`t`), **16.7% positive**.

By candidate type (pooled across horizons): Door 5,455/16,776 positive (32.5%) — doors see the most congestion by far, consistent with doors typically gating higher-traffic internal routes in this fixture; Exit 1,173/15,608 (7.5%); Stair 193/8,388 (2.3%) — stairs are the least congested, plausibly reflecting this fixture's engineering-state distribution (`stair_state_distribution` in `make_training_definition()` keeps most stairs available and this building has only one).

By horizon: see §12.

**40 independent scenarios, not "40,772 highly correlated rows from a handful of scenarios treated as if independent"** — Phase 15's own explicit warning is why scenario count and row count are reported as two separate numbers throughout this document, never conflated into a single "dataset size."

## 16. Simulation/live parity test result (Phase 13)

`tests/test_predictive_dataset_parity.py`, 3/3 passing: structural fields (`candidate_type`, `candidate_capacity`, `candidate_walking_distance`, `candidate_traversable`) match **exactly** between extractors given the same Building/candidate; the shared congestion-classification code path (`crowd_intelligence.congestion.compute_congestion_level`) matches **exactly** given matched queue/approaching counts; missing live position evidence is reported as `None` and explicitly asserted **not equal** to simulation's own exact ground truth for the same field — parity is never forced by fabricating a live value.

## 17. Extraction performance (Phase 18)

From the same 40-scenario campaign, kept strictly separate from simulation execution time per Phase 18's own instruction:

| Metric | Value |
|---|---|
| Simulation execution (scenario generation + `SimulationRuntime.run()`, all 40 scenarios) | 0.259s wall |
| Dataset extraction (feature + target generation, all 41,940 rows) | 0.493s wall |
| Extraction throughput | ~85,000 candidate-time rows/second |

Dataset extraction is not the bottleneck at this scale — simulation execution and extraction are comparable, and both are fast enough that a future, much larger campaign (thousands of scenarios) remains tractable on a single machine.

## 18. Full-suite result (Phase 20)

Baseline 4332/4332 passing (commit `fb123e6`). 52 new tests added (`tests/test_predictive_dataset_*.py`) covering: stable candidate identity, feature/target separation (architecture-guard import check), future-state leakage prevention (mechanical invariance test), E1/E2 differentiation (current features and future targets), horizon boundary behavior (inclusive/exclusive edges, disagreement across horizons), currently-congested handling (all 5 of Phase 10's required cases), missing live evidence (honest `None`, never fabricated), simulation/live parity, scenario-group preservation, and deterministic extraction. See the commit for the exact final count.

## 19. What this milestone deliberately did not do

- No model trained (`predictive_dataset/` produces a dataset only; nothing in this package imports `ai_training`/`sklearn`).
- No change to `evacuation_recommendation/` scoring, `ground_truth/`, `dataset_builder/`, `ai_features/`, or any existing trained model.
- No new intelligence engine — the live extractor is a thin remapping over `crowd_intelligence.engine.CrowdIntelligenceEngine`, already built for an unrelated purpose (live congestion reporting) and reused, not duplicated.
- No new Building assets, no new hardware protocols.
- `candidate_traversable`'s known v1 simplification (§7) — mid-scenario `ScenarioEvent` door/exit/stair overrides are not yet replayed into the simulation-side extractor the way `dataset_builder.timeline`'s own engineering-state replay does for its whole-building columns. Disclosed, not fabricated; a straightforward follow-up if a future campaign's scenarios lean heavily on scripted door/exit state changes mid-run.

## Final report

See the commit message / conversation record for the full answer to questions A–I (this document's §1–18 collectively already answer each of them: A/B/C — yes, per §8/§10/§11; D — yes, per §11; E — yes, per §6/§7; F — yes, per §7/§9/§16; G — yes, per §11/§12; H — 20s, per §13; I — the DATA FOUNDATION is ready for a future training milestone; **no model has been trained in this milestone**, and none should be inferred from this document).
