# AI's Operational Role in SynEvac

Status: **investigation + architecture decision. No model trained, no scoring code changed.** Answers the question `docs/architecture/synevac_end_to_end_architecture_review.md` §18 priority 4 left open: *"Decide AI's actual role before investing further in it."* Baseline: commit `ee209ab`, 4320/4320 tests passing.

## 1. Current AI architecture

```
Live pipeline:
BuildingState (canonical, whole-building)
    -> ai_features.building_state_extractor.extract_canonical_features() -> feature row (CANONICAL_LIVE_SCHEMA, 20 fields, all whole-building aggregates)
    -> ai_registry.inference_service.LiveAIInferenceService.predict_bottleneck_occurrence() / predict_evacuation_time()
       -> model.predict_proba([row]) / model.predict([row])   -- ONE row, whole building, per cycle
    -> live_system.live_ai_gateway.RegistryLiveAIInferenceGateway.predict() -> LiveAIPredictionSnapshot
    -> LiveOrchestrator.run_cycle() (live_ai_gateway param -- optional, None by default)
    -> StateManager.ai_prediction_snapshot
        |
        +--> evacuation_recommendation.engine.EvacuationRecommendationEngine.compute(ai_prediction_snapshot=...)
        |        -> evidence.ai_bottleneck_probability() -> a single float
        |        -> ranking.rank_exits_for_zone(ai_bottleneck_probability=...) -- SAME value passed to EVERY candidate
        |        -> scoring.score_candidate() -- adds ai_support_weight * (1 - probability) IDENTICALLY to every candidate
        |
        +--> live_system.live_advisory_gateway (ai_decision_evidence_from_prediction_snapshot)
                 -> advisory_system.ai_evidence.AIDecisionEvidence (available, probability, predicted, threshold,
                    model_id/version/status -- NO location field of any kind)
                 -> AdvisoryReport (annotation only, per that module's own "AI only ever ANNOTATES an
                    already-decided output, never mutates or replaces one" docstring)
    -> Command Center: command_center/live_ai_panel.py ("Operational AI -- Bottleneck Occurrence" /
       "Experimental AI -- Evacuation-Time Estimate", visually and structurally separate groups)
```

**Training pipeline** (offline, separate from the above):

```
scripts/train_live_compatible_models.py
    -> ai_registry.generate_training_campaign() -> 5000 synthetic scenarios (legacy TrainingDataset)
    -> ai_registry.build_live_compatible_dataset(legacy_dataset, building)
           -- REPLACES each record's .features with ai_features.simulation_extractor.
              extract_canonical_training_row() (the SAME CANONICAL_LIVE_SCHEMA, whole-building)
           -- KEEPS each record's .ground_truth/.zone_results UNCHANGED (still rich, per-asset)
    -> ai_registry.train_bottleneck_occurrence_model(live_dataset)
           -> ai_training.models.bottleneck_model.BottleneckModel.build_table(live_dataset, target="occurrence")
              -- HARD-CODED target="occurrence" (ground_truth.doors_that_became_bottlenecks, non-empty -> True)
           -> ModelStatus.PRODUCTION_CANDIDATE (F1 0.511 vs. 0.487 baseline, ROC-AUC 0.793 vs. 0.500, despite 95% class imbalance)
    -> ai_registry.train_evacuation_time_model(live_dataset)
           -> ai_training.models.evacuation_time_model.EvacuationTimeModel.build_table(live_dataset)
           -> ModelStatus.EXPERIMENTAL (R² = 0.088, does not beat its own median-MAE baseline)
    -> ModelRegistry.register_model() x2 -- NOT wired into LiveOrchestrator/advisory_system by this script
```

`RegistryLiveAIInferenceGateway` is never constructed by `live_runtime.factory.build_live_runtime()` or `live_runtime_launcher.session.LiveRuntimeSession` — `live_ai_gateway` stays `None` in every production `LiveRuntime` today. **AI is not merely rank-inert; it is not even running in the shipped application.**

## 2. Current models and actual capabilities

| Model | Target | Input features | Output shape | Deployability | Actual influence |
|---|---|---|---|---|---|
| `BottleneckOccurrenceModel_LiveCompatible` | `ground_truth.doors_that_became_bottlenecks` non-empty → `True`/`False` (did **any** door become a bottleneck **anywhere**, over the **whole** simulated run) | 20 whole-building fields (`CANONICAL_LIVE_SCHEMA`) — occupancy, camera health, sensor health, detector alarms/faults, FACP state, building control state. **No zone/exit/door/stair field of any kind.** | `BottleneckOccurrencePrediction(probability: float, predicted_occurrence: bool, threshold, model_id, ...)` — **no location field exists on the type itself** | `PRODUCTION_CANDIDATE` (beats baseline) | Uniform additive term in `evacuation_recommendation.scoring.score_candidate()`; annotates `AdvisoryReport` |
| `EvacuationTimeModel_LiveCompatible` | Ground-truth `total_evacuation_time` (a completed-simulation outcome) | Same 20-field whole-building schema | `EvacuationTimePrediction(predicted_seconds, uncertainty_seconds, ...)` | `EXPERIMENTAL` (does not beat baseline) | **None** — surfaced only as `evacuation_time_experimental` on `LiveAIPredictionSnapshot`; no code path lets it reach Decision Policy, Advisory, or Recommendation (confirmed by direct code read of `live_ai_gateway.py`'s own module docstring and `evacuation_recommendation/`, which never references it) |

## 3. Why global bottleneck probability is currently rank-inert (verified in code, not merely asserted)

`evacuation_recommendation/scoring.py::score_candidate()`:

```python
if ai_bottleneck_probability is not None:
    ai_goodness = 1.0 - ai_bottleneck_probability
    score += weights.ai_support_weight * ai_goodness
```

`ai_bottleneck_probability` is a single float threaded unchanged through `ranking.rank_exits_for_zone()` into **every** candidate's `score_candidate()` call for a given zone this cycle (`evacuation_recommendation/ranking.py:196-201`). Since it is the identical value added to every candidate's score, it can change the **absolute** score and the reason-code explanation, but can **never** change which candidate has the **higher** score than another — this is a structural (mathematical) consequence of the code, proven by `tests/test_ai_operational_role.py::GlobalBottleneckCannotAffectRankingTests` (§16), not merely documented in a comment.

This is not a bug. It is the correct, honest behavior **given the model's own target and features**: the model was never given the information needed to prefer one exit over another (see §5).

## 4. Current live localized signals (Phase 3 matrix)

| Signal | Simulation available? | Live available? | Localized to zone? | Localized to exit/stair/door? | Historical/temporal? | Reliability/coverage info? |
|---|---|---|---|---|---|---|
| Canonical occupancy (`LiveOccupantManager.canonical_occupancy()`) | No (simulation uses `ground_truth`/`DynamicHumanState`, a different type) | **Yes** | **Yes** (`occupant_ids_by_zone`) | No (zone-level only) | No | Unlocalized count tracked separately |
| Zone density (`ZoneCrowdMetrics`) | No | **Yes** | **Yes** | No | **Yes** (`TrendDirection` per zone) | `position_coverage_fraction` |
| Door/Exit/Stair approach metrics (`AssetApproachMetrics`) | No | **Yes** | N/A | **Yes** (per asset id) | **Yes** (`TrendDirection` per asset) | `position_available` bool |
| Queue length / congestion level | No (simulator has its own internal, unexposed per-edge congestion state — see §9) | **Yes** | No | **Yes** | **Yes** (trend) | Same as above |
| Exit throughput (`ExitFlow.recent_flow_per_minute`) | No | **Yes** | No | **Yes** | **Yes** (trend) | `position_available` |
| Zone clearance (`ZoneClearance`) | No | **Yes** | **Yes** | No | **Yes** (trend) | `observable` (camera coverage) |
| Trajectory route-progress (`TrajectoryResult`) | No | **Yes** | **Yes** (`zone_id`) | Indirect (`nearest_safe_exit_id`, aggregatable into per-exit votes) | **Yes** (`route_distance_trend`) | `position_available`, `stale` |
| Emergency response priority (`ZoneResponsePriority`) | No | **Yes** | **Yes** | No (zone-level only) | No | `observability_fraction` |
| Hazard severity (`hazard_summary`) | **Yes** (`HazardSnapshot`, shared type) | **Yes** | **Yes** | No | No | N/A |
| Exit/door/stair traversability | **Yes** | **Yes** | N/A | **Yes** (`Edge.traversable`) | No | N/A |

**The central Phase 3 finding**: every one of Crowd Intelligence / Evacuation Progress / Trajectory Intelligence / Emergency Response / canonical occupancy is **live-only** today — none has a simulation-side equivalent producing the same shape. `ai_features/`, `ai_training/`, `dataset_builder/` were grepped directly and contain **zero** references to `crowd_intelligence`, `evacuation_progress`, or `trajectory_intelligence`. This is the direct, verified reason `CANONICAL_LIVE_SCHEMA` (§1) has no per-zone/per-asset fields: it was finalized before these packages existed, and nothing has updated it since. The old conclusion ("no live-compatible signal can localize congestion," `advisory_system/ai_evidence.py`'s own module docstring, written during the AI-Augmented Advisory milestone) is **now stale as a statement about live signals in general** — Crowd Intelligence/Evacuation Progress already solved live localization, deterministically, months of milestones ago — but remains **true as a statement about what reaches the trained model**, because nothing wires these newer packages into `ai_features`.

## 5. Simulation labels available (Phase 4)

`ground_truth/bottleneck.py` already computes, from an already-completed `movement_result` (occupant timelines, `peak_edge_occupancy`, `peak_node_occupancy` — nothing resimulated):

| Target | Computed from | Granularity | Temporal? |
|---|---|---|---|
| Bottleneck occurrence | `doors_that_became_bottlenecks` non-empty | Whole scenario, boolean | No — whole-run outcome |
| Bottleneck location | `peak_congestion_location_id`/`peak_congestion_location_type` (zone/door/exit/stair, tie-broken deterministically) | **Per-asset**, one winner per scenario | No — whole-run outcome |
| Per-door congestion | `doors_that_became_bottlenecks` (any queue_wait_time > 0 ever) | **Per-door**, boolean | No |
| Per-exit congestion/utilization | `exits_underutilized`, `exits_exceeding_capacity` | **Per-exit**, boolean | No |
| Per-stair congestion | `stairs_exceeding_capacity` | **Per-stair**, boolean | No |
| Exit/stair/door usage counts | `_exit_usage_counts()` et al. | Per-asset, count | No |
| Remaining evacuation time | `total_evacuation_time` | Whole scenario | No |
| Zone clearance time | `zone_route_stats` (`ground_truth/evacuation_metrics.py::compute_zone_route_stats`) | **Per-zone** | Partial (start/clear times) |
| Route failure/rerouting need | Not directly labeled; derivable from `zone_route_stats`/occupant route-change events | Per-zone/per-occupant | Partial |

**All of the above already exist as whole-scenario (post-hoc) labels**, computable today without any simulator change — `BottleneckModel` already supports `target="location"` (`ai_training/models/bottleneck_model.py:7,24,81`), it is simply never selected by `train_bottleneck_occurrence_model()` (`ai_registry/training.py:211`, hard-coded `target="occurrence"`).

**What does NOT exist today**: none of these labels are computed **per-timestep with a forward horizon** (e.g., "will Exit E1 be congested at T+30s, evaluated fresh at every T"). `_congestion_duration_for_edge()` already sweeps `OccupantTimelineStep.start_time`/`end_time` as an interval list — the raw data needed to derive a per-timestep, horizon-bounded label already exists in every recorded scenario; producing it is a **new offline analysis pass over already-recorded simulation output**, not a simulator redesign (see §9/§10). Retraining `target="location"` on the CURRENT whole-building `CANONICAL_LIVE_SCHEMA` features would nonetheless be **scientifically indefensible** — see §6.

## 6. Option comparison

**Option A — global AI advisory only (current de facto state).** *Pros*: zero risk, zero new work, already matches what the model can honestly support. *Cons*: provides no information Advisory doesn't already annotate faithfully; the "Bottleneck Risk Probability" number, while honest, answers a question ("will *something* bottleneck *somewhere*") that is one step removed from any actionable decision — an operator cannot act on "somewhere."

**Option B — localized bottleneck prediction.** *Pros*: directly answers the question that would let AI meaningfully differentiate exit candidates; ground-truth labels already exist (§5); live per-asset infrastructure to build matching features now exists (§4, though not yet wired to `ai_features`). *Cons*: requires new work at three layers — (1) extend `CANONICAL_LIVE_SCHEMA`/`ai_features` with per-candidate fields, (2) build a matching simulation-side extractor (Crowd Intelligence has no simulation equivalent today, §4/§9), (3) reformulate the model as one-row-per-candidate rather than one-row-per-building. Cannot be done by retraining the existing model on existing features — the features themselves are the limiting factor, not the algorithm.

**Option C — evacuation-time prediction as AI's primary role.** *Cons dominate*: the existing model (§2) already targets this and scores `EXPERIMENTAL`, below its own baseline, on whole-building features. Nothing about localizing the *time* target changes without solving the same feature-availability problem as Option B, and RSET/clearance-time prediction is arguably a **harder** regression problem than a binary occurrence classification already struggling to beat baseline. No evidence this should be AI's primary role.

**Option D — hybrid: deterministic safety → measured live intelligence → predictive localized AI → recommendation scoring → deterministic guidance.** This is not a new architecture — it is what `evacuation_recommendation/` **already implements today**, minus the "predictive localized AI" layer (currently a no-op placeholder occupying the `ai_bottleneck_probability` slot). Every other layer is real, live, deterministic, and already rank-affecting (`SafeExitDistanceCalculator`'s hazard exclusion = safety; `congestion_level`/`queue_count`/`throughput`/`trajectory_support` = measured intelligence, per-candidate, already in `score_candidate()`).

**Recommendation: Option D**, understood correctly as *completing* the architecture already in place, not building a new one. The predictive-AI layer is the **only missing piece**, and it is missing because of a feature-availability gap (§4/§9), not an architectural one.

## 7. Recommended AI role

AI's role should be **narrow, predictive, and localized**: forecast near-future congestion risk **per exit/stair/door candidate**, to be added as one more per-candidate additive term in `score_candidate()` — never as a substitute for measured intelligence (which stays authoritative for *current* state) and never with authority over safety exclusion. AI's unique value is answering a question **nothing else in the architecture can honestly answer today**: "this currently-clear exit is about to become congested" — a genuinely predictive claim, distinct from every deterministic signal already wired in (which all describe the *present*, not the *near future*).

## 8. Safety boundary (unchanged, verified, must remain unchanged)

- `evacuation_recommendation/` never imports `decision_policy` — mechanically enforced (`tests/test_evacuation_recommendation_architecture_guards.py::test_decision_policy_never_imports_evacuation_recommendation` and its sibling, both pre-existing, unmodified).
- `SafeExitDistanceCalculator._excluded_zone_ids()` (hazard-based exclusion) runs **before** any candidate is even constructed — an unsafe exit is never scored, never ranked, never reachable by an AI term at all, regardless of what any AI model outputs.
- This boundary must remain absolute for any future localized model: AI proposes a **score contribution only**, never a candidate-inclusion/exclusion decision.

## 9. Proposed future prediction target

`P(candidate becomes congested within horizon T | current + recent-trend evidence)`, one probability **per exit/stair/door candidate**, not one per building. Distinct from the current occurrence model in both grain (per-candidate vs. whole-building) and time reference (forward-looking-from-now vs. whole-scenario-outcome).

## 10. Proposed temporal horizon

Given `LiveOrchestrator`'s ~1 Hz cycle (`interval_seconds=1.0` default) and that Crowd Intelligence/Evacuation Progress already report **current** congestion/queue/throughput instantly and deterministically (so predicting T+0 adds nothing), the horizon must be far enough ahead to constitute genuine advance warning, yet short enough to remain learnable from realistic scenario lengths (typically a few minutes) and operationally meaningful (an operator needs to be able to act on the warning before it's moot). A **20–30 second horizon** is proposed as a starting point, re-evaluated empirically against label sparsity in the next milestone — not fabricated as a validated number here.

## 11. Proposed feature families (existing today vs. needed)

| Family | Exists today (live) | Exists today (simulation) | Needed |
|---|---|---|---|
| Whole-building (`CANONICAL_LIVE_SCHEMA`, unchanged) | Yes | Yes | Keep as global context features |
| Per-candidate current occupancy/density | Yes (`ZoneCrowdMetrics`, canonical occupancy) | **No** | Simulation-side extractor |
| Per-candidate queue/congestion/throughput | Yes (`AssetApproachMetrics`, `ExitFlow`) | **No** (simulator has internal per-edge state, never exposed as a snapshot — `simulator/congestion.py`) | Simulation-side extractor, or expose the simulator's own internal per-edge occupancy |
| Per-candidate trend (RISING/STABLE/FALLING) | Yes | **No** | Same as above |
| Per-candidate trajectory support | Yes (`TrajectoryResult` votes) | **No** | Same as above |
| Route-distance to candidate | Yes (`SafeExitDistanceCalculator`) | Yes (navigation graph, shared) | Already parity-safe |

## 12. Leakage boundary

| Field | Classification |
|---|---|
| `doors_that_became_bottlenecks`, `peak_congestion_location_id`, `exits_underutilized`, `exits_exceeding_capacity`, `stairs_exceeding_capacity`, `total_evacuation_time`, `zone_route_stats` (final values) | **TARGET/FUTURE-ONLY** — whole-scenario outcomes; must never appear as an input feature (already excluded from `CANONICAL_LIVE_SCHEMA`, confirmed by that module's own exclusion list) |
| `ignition_zone`, `fire_profile`, `growth_time`, behaviour-profile identity | **TARGET/FUTURE-ONLY** — `SIMULATION_ONLY`, already excluded |
| Current `ZoneCrowdMetrics`/`AssetApproachMetrics`/`ExitFlow` values **as of prediction time T** | **ALLOWED AT INFERENCE TIME** |
| Any future-timestep value of the same live snapshots (T+1 onward) | **TARGET/FUTURE-ONLY** — the exact same leakage class the existing schema already guards against, now extended to the new per-candidate fields |
| `route_distance_trend`/`TrendDirection` computed strictly from history up to and including T | **ALLOWED** (already causally prior to T) |

The existing schema's own discipline (`ai_features/feature_schema.py`'s `FUTURE_INFORMATION`/`OUTCOME_LEAKAGE` enum members, already defined but currently unused since no feature has ever needed them) extends directly to this new family — no new leakage-discipline concept is required, only new fields correctly classified under the existing vocabulary.

## 13. Expected integration point

Unchanged from today's architecture: `evacuation_recommendation/scoring.py::score_candidate()` gains a new, **per-candidate** `ai_localized_bottleneck_risk` parameter (name illustrative), scored with its own disclosed weight, additive alongside (never replacing) `congestion_weight`/`queue_weight`/`throughput_weight`/`trajectory_weight`. `evidence.py` gains a per-candidate lookup analogous to `_crowd_evidence()`/`_throughput_evidence()`, reading a **new**, per-candidate-shaped `LiveAIPredictionSnapshot` field — the current single `bottleneck` field remains for the (unrelated) whole-building occurrence signal Advisory still legitimately wants.

## 14. What existing AI should be retained/deprecated

| Item | Decision |
|---|---|
| `BottleneckOccurrenceModel_LiveCompatible` (occurrence target) | **KEEP** — genuine `PRODUCTION_CANDIDATE`, honest whole-building signal, legitimate Advisory-only use case (§7's new localized model is additive, not a replacement) |
| `EvacuationTimeModel_LiveCompatible` | **KEEP EXPERIMENTAL** — see Addendum A below; do not remove from the codebase, do not present as operational |
| `live_system.live_ai_gateway` (Protocol + `RegistryLiveAIInferenceGateway` + `ThrottledLiveAIInferenceGateway`) | **KEEP, unchanged** — the correct, already-built integration seam; a localized model would need a sibling gateway, not a replacement of this one |
| `evacuation_recommendation.scoring.score_candidate()`'s current `ai_bottleneck_probability` term | **KEEP, unchanged, for now** — honest and harmless (rank-inert by design, not by oversight); superseded only once a genuinely localized signal exists to add alongside it |
| `BottleneckModel(target="location")` | **KEEP** (already exists, already correct) — simply never selected for `LiveCompatible` training; becomes relevant once whole-building features are no longer the only ones fed to it |

## 15. Exact next implementation milestone

**Not this one.** The next ML-relevant milestone, if pursued, should be scoped narrowly to: (1) a simulation-side extractor producing Crowd-Intelligence-shaped per-candidate features from an already-completed scenario run (reusing `simulator`'s own internal per-edge occupancy state, or literally driving `CrowdIntelligenceEngine` against simulated occupant data), (2) a matching live-side `ai_features` extension (per-candidate fields, correctly classified `LIVE_ESTIMABLE`/`LIVE_OBSERVABLE`), (3) horizon-bounded per-timestep relabeling of already-recorded scenario timelines (an analysis pass, not a simulator redesign) — all three **before** any new model is trained. This document recommends but does not schedule that milestone.

---

## Addendum A — Evacuation-time model operational decision (Phase 12)

R² = 0.088, does not beat its own median-MAE baseline (`ai_registry/training.py::evacuation_time_model_status`'s own disclosed criterion). **Decision: KEEP EXPERIMENTAL.** Not removed from the codebase (it is a legitimate, honestly-labeled research artifact, and `ModelRegistry`/`LiveAIInferenceService` already handle an `EXPERIMENTAL` model correctly — never silently promoted). Not retrained in this milestone (explicitly out of scope). Not given any new operational surface — it already has none (§1), and none is proposed. `command_center/live_ai_panel.py` already labels it "EXPERIMENTAL -- not validated for operational use" in its own dedicated, visually separate group — this wording is already correct and needs no change.

## Addendum B — Command Center honesty audit (Phase 13)

Audited `command_center/live_ai_panel.py` and `command_center/recommendation_center.py` directly:

- `LiveAIPanel` shows "Bottleneck Risk Probability" (no exit name, no route implication) in a group explicitly titled "Operational AI -- Bottleneck Occurrence," structurally separate from the "Experimental AI -- Evacuation-Time Estimate" group (already labeled "not validated for operational use").
- `recommendation_center.py`'s confidence-provenance labeling (`_confidence_label`/`_prediction_source_line`) already states "no AI/RL/crowd signal supplied for this recommendation" whenever `confidence_source` is empty (the case in every current run, since AI is never wired — §1) — never implies AI drove a recommendation it didn't influence.
- `live_evacuation_recommendation_panel.py` shows only a single summary reason code per candidate — never singles out `AI_BOTTLENECK_RISK_LOW`/`AI_BOTTLENECK_RISK_ELEVATED` in a way that could misattribute ranking to AI.

**No misleading wording found. No UI changes made**, per Phase 13's own "do not make broad UI changes... unless necessary for factual correctness" instruction — none was necessary.

## Addendum C — AI failure semantics (Phase 14)

`RegistryLiveAIInferenceGateway.predict()` catches every exception from `predict_bottleneck_occurrence()`/`predict_evacuation_time()` and turns it into an honest `AISystemStatus` (`UNAVAILABLE`/`INCOMPATIBLE`/`ERROR`/`PARTIAL`), never propagating. `evacuation_recommendation.evidence.ai_bottleneck_probability()` returns `None` for a `None` snapshot, a snapshot with `bottleneck=None`, or a missing `probability` attribute — `scoring.score_candidate()`'s own `else: score += weights.ai_support_weight * 0.5` branch (neutral, never a rank distortion) already handles every one of these. `EvacuationRecommendationEngine.compute()`'s own `ai_prediction_snapshot=None` default means the entire engine runs correctly with **zero** AI wiring at all — proven in `tests/test_ai_operational_role.py`.
