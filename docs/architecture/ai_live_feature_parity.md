# Simulation-to-Live AI Feature Parity Framework

Status: **feature-parity framework established; zero trained models newly deployed.** This document is the record of the audit and the architecture it produced. It does not connect AI inference to `LiveOrchestrator` or `advisory_system` — that remains a later, explicitly deferred milestone (see `docs/architecture/live_system_integration_audit.md` §13.8).

## 1. Current AI models

Four trained models exist (`ai_training/experiment.py:25-30`'s `MODEL_REGISTRY` — confirmed exhaustive by grepping every `class \w+\(BaseModel\)` in the repo):

| Model | Target | Task |
|---|---|---|
| `EvacuationTimeModel` | `total_evacuation_time` | regression |
| `BottleneckModel` (`target="occurrence"`) | `bottleneck_occurrence` (`bool`) | classification |
| `BottleneckModel` (`target="location"`) | `bottleneck_location` (zone/door id or `"NONE"`) | classification |
| `ExitUsageModel` | `exit_usage_percentage` per exit | multi-output regression |
| `SmokePredictionModel` | `next_highest_smoke_zone` (t+1 label) | classification |

## 2. Exact feature audit

**`EvacuationTimeModel` / `BottleneckModel` / `ExitUsageModel`** all train on the identical feature set: every column `dataset_builder.feature_extractor.extract_scenario_features()` produces, minus `scenario_id`/`definition_id`/`seed` (`ai_training/dataset.py:24`, `NON_FEATURE_SCENARIO_COLUMNS`). That column set (`dataset_builder/feature_extractor.py:44-100`) is built exclusively from `run.scenario`/`run.building` — **pre-simulation configuration only**, never `run.movement_result`:

- Fire configuration: `ignition_zone`, `ignition_floor`, `fire_profile`, `growth_time`
- Population: `total_occupants`, `Adult_Count`/`Child_Count`/`Elderly_Count`/`Wheelchair_Count`/`Visitor_Count`/`Firefighter_Count`
- Occupant-attribute means: `Mean_Walking_Speed_Multiplier`, `Mean_Reaction_Speed`, `Mean_Stamina`, `Mean_Smoke_Tolerance`, `Mean_Visibility_Tolerance`, `Mean_Fatigue_Resistance`, `Mean_Mobility_Factor`, `Mean_Leadership`, `Mean_Risk_Aversion`, `Mean_Route_Familiarity`, `Mean_Compliance`, `Mean_Helping_Likelihood`, `Mean_Panic_Susceptibility`, `Mean_Crowd_Following_Tendency`
- Social-group aggregates: `Group_Count`, `Grouped_Occupant_Count`, `Mean_Group_Size`
- Per-zone/door/exit/stair/obstacle/detector/camera state: `Zone_<n>_Occupancy` (**pre-simulation starting placement**, not a live reading), `Door_<n>_State`, `Exit_<n>_State`, `Stair_<n>_State`, `Obstacle_<n>_State`, `Detector_<n>_State`, `Camera_<n>_State`

Targets: `total_evacuation_time` (`labels.py:64`, read off completed `MultiAgentSimulationResult`), `bottleneck_occurrence`/`bottleneck_location` (from `GroundTruth.doors_that_became_bottlenecks`/`peak_congestion_location_id`, both computed post-simulation in `ground_truth/bottleneck.py`), `exit_usage_percentage` (from `zone_results.csv`'s `exit_used`/`evacuated`, post-simulation). **No outcome/Ground-Truth value was found merged into any of these three models' `X_rows`** — every target is used exclusively as `y`.

**`SmokePredictionModel`** is architecturally different: it trains on **Timeline rows** (one row per simulation tick — `dataset_builder/timeline.py`), not Scenario Features. Its inputs (`smoke_prediction_model.py:65`) are the *current* tick's own columns — `simulation_time`, `current_fire_zones`, `current_hazard_score`, per-zone `Zone_<n>_Occupancy`/`Zone_<n>_Smoke`/`Zone_<n>_Temperature`/`Zone_<n>_Visibility`, live engineering state, and running counters (`people_evacuated`, `current_congestion`, etc.) — predicting the **next** tick's highest-smoke zone. No leakage was found (each row is built from `current_row` only, never `next_row`), but its inputs are fundamentally mid-simulation telemetry, not pre-simulation configuration like the other three.

## 3. Feature availability matrix

Classified against `building_state.models.BuildingState` and everything reachable from it (full observability audit; see §6 for the reference table below):

| Feature category | Classification | Why |
|---|---|---|
| `ignition_zone`/`ignition_floor`/`fire_profile`/`growth_time` | SIMULATION_ONLY | `ScenarioDefinition.fire`-only facts; no live equivalent |
| `Adult_Count`...`Visitor_Count`, `Mean_*` occupant attributes, `Group_*` | SIMULATION_ONLY | `behaviour_profile_resolver` ground-truth facts; not observable |
| `Zone_<n>_Occupancy` (Dataset 1) | SIMULATION_ONLY | pre-simulation starting placement, not a live reading |
| `Door/Exit/Stair/Obstacle/Detector/Camera_<n>_State` | SIMULATION_ONLY (as-columned) | pre-simulation config snapshot; `CameraStatus.active`/`SensorStatus.active` are the LIVE_OBSERVABLE analogues, but indexed differently (aggregate, not per-asset-ordinal) |
| `HazardSeverity`/`hazard_score`/`BuildingState.hazard_summary` | **SIMULATION_ONLY in every current code path** | every producer (`fire_growth`, `smoke_propagation`, `tenability`) is simulated physics; the codebase's own design (`perception/models/building_observation.py`) deliberately walls this off from the real-perception type. No real-sensor-derived `HazardSnapshot` exists anywhere. |
| `occupant_tracks[*].classification`/`.track_id` (current impl.) | SIMULATION_ONLY in current implementation | sourced from ground-truth behaviour-profile identity / `occupant_id` equality, not vision/re-ID (design is LIVE_ESTIMABLE, per `live_camera_pipeline`'s own seam, just unimplemented) |
| `BuildingState.zone_occupancy` (current impl.) | SIMULATION_ONLY in current implementation | design (`OccupancyEstimator`) is LIVE_ESTIMABLE, but never wired to it in any current code path |
| `occupant_tracks` COUNT + mean confidence | **LIVE_ESTIMABLE** | exactly the CCTV milestone's own proven multi-camera-fusion signal |
| `CameraStatus`/`SensorStatus` (active/offline/health) | **LIVE_OBSERVABLE** | genuine device-management facts |
| `FACPSnapshot` (panel_state, alarm/fault sources, ack/silence) | **LIVE_OBSERVABLE** | matches a real FACP integration's outputs exactly |
| `BuildingState.building_alarm_status` | **LIVE_OBSERVABLE** | aggregated from detector `alarm_active` bits (binary), not `hazard_score` |
| `ControlStateSnapshot` (pending/confirmed) | **LIVE_OBSERVABLE** | genuine provider-confirmed state |
| Timeline-only telemetry (`current_congestion`, `current_queue_length`, ...) | Not evaluated for live availability | `SmokePredictionModel`-specific; excluded from the v1 canonical schema (see §12) |

## 4. Leakage findings

**Zero outcome-into-feature leakage found** across all four models — verified by reading every `build_table()` in full; each model's target column is drawn exclusively from `outcome_rows()`/`ground_truth_rows()`/`zone_result_rows()`, never merged into `X_rows`. **INVALID_DUE_TO_LEAKAGE: 0 models.** `ai_features.compatibility` still carries an explicit `outcome_leakage` detection category (a reference set of known outcome/label column names) so a *future* model accidentally requiring one of these as an input fails loudly, not silently.

## 5. Canonical live-compatible schema

`ai_features/feature_schema.py` — 25 fields, deterministic order, schema version `1.0`. Every field is LIVE_OBSERVABLE or LIVE_ESTIMABLE (never SIMULATION_ONLY/FUTURE_INFORMATION/OUTCOME_LEAKAGE):

`total_occupant_count`, `occupancy_observed`, `mean_occupant_track_confidence`, `camera_total_count`, `camera_active_count`, `camera_offline_count`, `sensor_total_count`, `sensor_active_count`, `sensor_offline_count`, `smoke_detector_coverage_count`, `smoke_detector_alarm_count`, `smoke_detector_fault_count`, `heat_detector_coverage_count`, `heat_detector_alarm_count`, `heat_detector_fault_count`, `building_alarm_status`, `facp_available`, `facp_panel_state`, `facp_active_alarm_source_count`, `facp_active_fault_source_count`, `facp_acknowledged`, `facp_silenced`, `control_status_available`, `control_pending_request_count`, `control_confirmed_entry_count`.

Deliberately excluded, with reasons already given in §3: hazard/severity fields, occupant classification breakdowns, all `ScenarioDefinition`/`BehaviourProfile` facts, and every per-zone/per-door/per-exit-indexed column (a topology-alignment problem this v1 schema does not attempt — see §12).

## 6. BuildingState extraction

`ai_features/building_state_extractor.py::extract_canonical_features(state: BuildingState) -> Dict[str, Any]` — a pure function, reads only already-assembled `BuildingState` fields, asserts its own output key order matches the canonical schema on every call. This is the one function both a genuine live deployment and the simulation-side path (§7) call — the mechanism that makes "same feature names and semantics from both sources" true by construction.

## 7. Simulation training extraction

`ai_features/simulation_extractor.py::build_building_state_at_alarm_activation()` builds a **real** `BuildingState` from simulation-side facts, reusing `CameraManager`/`SensorManager`/`SimulatedFACP`/`BuildingStateEstimator` exactly as `designer/building_state_debug_runner.py` already composes them for the Designer — no camera/sensor/FACP aggregation logic is reimplemented. `extract_canonical_training_row()` then calls the same `extract_canonical_features()` from §6.

**Temporal framing:** all four existing models are single-snapshot, pre-outcome predictors (none operate on a per-tick trajectory, except `SmokePredictionModel`'s own Timeline-based design). The lowest-risk live-compatible reinterpretation, and the one this module implements, is **T = alarm activation**: the full starting population (nobody has evacuated yet, by definition, at the instant alarm activates), every device's own configured active/offline status, and only the detector(s) actually covering the ignition zone reporting ALARM — an honest *consequence* of where the fire started (exactly what a real FACP would show), never the ignition zone identity itself exposed as a feature, and never a fabricated continuous hazard reading. This matches the existing models' own "one static snapshot → one outcome" shape; no temporal-model redesign was implemented (Phase 7's own "do not implement a major temporal-model redesign unless necessary").

**Labels remain simulation Ground Truth**, per Phase 6's explicit rule — only input features are constrained to the canonical schema; `total_evacuation_time`/`bottleneck_occurrence` etc. are read from the same `outcome_rows()`/`ground_truth_rows()` the existing models already use.

## 8. Missing-data policy

- **Never silently zero.** `total_occupant_count` is `None` (not `0`) when `occupancy_observed` is `False` — no cameras configured is a different fact from "confirmed zero people," and the two must never be conflated. Same discipline for `mean_occupant_track_confidence` (`None` with no tracks), every `facp_*` field (`None`, never `"NORMAL"`/`False`, when `facp_available` is `False`), and every `control_*` field (`None` when `control_status_available` is `False`).
- **Coverage counts always accompany condition counts.** `smoke_detector_alarm_count`/`fault_count` are always paired with `smoke_detector_coverage_count`, so "0 alarms" is distinguishable from "0 detectors of this kind exist at all" (same for heat detectors) — directly answering Phase 10's "no smoke detector coverage in a zone" scenario.
- **`building_alarm_status` has a known, documented ambiguity inherited from `BuildingState` itself**: its dataclass default is `NORMAL` even with zero detectors configured (`_aggregate_alarm_status` returns `NORMAL` over an empty collection). This is not fixed in this milestone (`BuildingState`'s own shape is out of scope) — `sensor_total_count`/`smoke_detector_coverage_count`/`heat_detector_coverage_count` are the documented way a consumer must pair with it to tell "confirmed clear" apart from "no coverage at all."
- **Sentinel strategy chosen:** `None` at the Python/dict level, which becomes `NaN` when `ai_training.preprocessing._rows_to_table()` builds a numeric/categorical table (this is the *existing*, already-used-by-every-current-model behavior — `SimpleImputer(strategy="median"/"most_frequent")` — not a new fabrication introduced by this milestone). Explicit `_available`/`_observed` boolean companion fields exist specifically so a consumer can distinguish "imputed because absent" from "genuinely zero" even after that imputation runs, per Phase 10's explicit instruction not to conflate the two.

## 9. Model deployability classification

Strict, per Phase 5's own instruction ("do not label a model LIVE_READY merely because missing features can technically be filled with zeros"):

| Model | Classification | Why |
|---|---|---|
| `EvacuationTimeModel` | **RETRAIN_REQUIRED** | Target remains valuable; retrained on the canonical schema alone (§10) — real, non-trivial (if reduced) predictive signal (r² = 0.27) |
| `BottleneckModel` (occurrence) | **RETRAIN_REQUIRED** | Same — retrained result: accuracy 0.56, ROC-AUC 0.84 (better than chance, clearly degraded) |
| `BottleneckModel` (location) | **RESEARCH_ONLY** | Multi-class *location* prediction fundamentally needs per-zone/per-door-indexed features; the v1 canonical schema is aggregate-only (§12) |
| `ExitUsageModel` | **RESEARCH_ONLY** | Multi-output per-exit prediction needs per-exit-indexed features, same gap as above |
| `SmokePredictionModel` | **RESEARCH_ONLY** | Target (`next_highest_smoke_zone`) requires continuous per-zone smoke-level knowledge, which is SIMULATION_ONLY in every current code path (§3) — a real smoke detector reports only a binary alarm bit, never a continuous level; no live path to this target's own information exists in this codebase today |
| — | **LIVE_READY: 0** | No existing model's current feature set is already reproducible from `BuildingState` with identical semantics |
| — | **INVALID_DUE_TO_LEAKAGE: 0** | No leakage found (§4) |

## 10. Retraining results

Ran via `scripts/ai_feature_parity_experiment.py` — a real, deterministic, 80-scenario campaign (`tests/training_dataset_fixtures.py::make_campaign`, seed 2026, the same small fixed 2-zone Building the repo's own `ai_training`/`training_dataset` test suites already use), features built two ways for the identical scenarios/labels: (a) the existing `extract_scenario_features()` (37 columns, all classified SIMULATION_ONLY/OUTCOME_LEAKAGE-adjacent per §3), and (b) `ai_features.extract_canonical_training_row()` (25 canonical columns). Both trained via the unmodified `ExperimentRunner`/`EvacuationTimeModel`/`BottleneckModel`, `random_state=0`, 80/20 split.

**Caveat, stated plainly:** this is a small, fixed, near-deterministic test fixture (2 zones, `FixedValue(200.0)` fire growth), not a large diverse production campaign — the existing model's near-perfect scores below are an artifact of that fixture's simplicity, not a general claim about the existing models' real-world accuracy. The *relative* degradation (existing vs. live-compatible, same scenarios, same split) is the genuine, reproducible finding this experiment was run to produce.

## 11. Existing-model vs. live-compatible-model performance

**`evacuation_time` (regression, 64 train / 16 test):**

| | existing (37 cols) | live-compatible (25 cols) | Δ |
|---|---|---|---|
| MAE | 0.000 | 7.591 | +7.591 |
| RMSE | 0.000 | 9.700 | +9.700 |
| R² | 1.000 | 0.274 | −0.726 |

**`bottleneck_occurrence` (classification, 64 train / 16 test):**

| | existing (37 cols) | live-compatible (25 cols) |
|---|---|---|
| accuracy | 1.000 | 0.5625 |
| precision | 1.000 | 0.767 |
| recall | 1.000 | 0.5625 |
| F1 | 1.000 | 0.459 |
| ROC-AUC | 1.000 | 0.836 |

**Interpretation:** removing simulation-only information costs substantial predictive power for both models — expected, since 12 of 37 legacy columns are direct occupant-behavior/population-profile facts and another ~20 are pre-simulation engineering-state columns with no live equivalent in this schema. The live-compatible bottleneck-occurrence model still meaningfully beats chance (ROC-AUC 0.84), suggesting occupancy + device-health signal alone carries real predictive value for *this* target even without simulation-only features — a genuinely useful, honest finding, not a discouraging one.

## 12. Remaining limitations

- **No per-zone/per-door/per-exit granularity in the v1 canonical schema.** This blocks `bottleneck_location` and `exit_usage_percentage` from any retraining attempt in this milestone — building a topology-aligned canonical schema (indexed consistently against a `Building`'s own zone/door/exit ordering, the same problem `dataset_builder/schema.py`'s `ordered_zones()`/etc. already solve for the *legacy* schema) is future work, not attempted here.
- **`SmokePredictionModel` has no live path at all**, not merely a reduced one — its target requires continuous per-zone smoke-level knowledge this codebase has no real-sensor path to (§3, §9). Deploying it live would require either a genuinely different smoke-level-estimating sensor/model (not evaluated here) or redefining its target around only binary alarm information.
- **`occupant_tracks` count is currently only as good as whatever identity-resolution strategy feeds `BuildingState`** — the CCTV milestone proved deduplication works with `MappingIdentityResolver`; real cross-camera re-identification (`LiveReIDIdentityResolver`) remains unimplemented (frozen, per the CCTV milestone's own scope), so `total_occupant_count` accuracy in an actual live deployment depends on that still-unbuilt piece.
- **The retraining experiment used a small, fixed test fixture**, not a large/diverse production campaign (§10's own caveat) — the absolute performance numbers should not be read as a production accuracy estimate; only the relative comparison is the intended, load-bearing result.
- **`building_alarm_status`'s "no coverage" vs "confirmed clear" ambiguity is inherited, not fixed** (§8) — a future milestone touching `BuildingState` itself could resolve this at the source.
- **This framework is not yet wired to anything.** `ai_features/` has zero callers outside its own tests and `scripts/ai_feature_parity_experiment.py` — by this milestone's own explicit design (see §13 below).

## 13. Explicit answer

**"Can SynEvac now honestly run AI inference from `BuildingState` without fabricated inputs?"**

**Partially — for exactly two models, and only once each is actually retrained and its results reviewed; no model is being deployed by this milestone.** `EvacuationTimeModel` and `BottleneckModel` (occurrence target) are classified **RETRAIN_REQUIRED**: a live-compatible feature contract for them now exists, is test-proven, and this milestone's own retraining experiment shows they retain real (if substantially reduced) predictive signal on it. Neither has been wired into `LiveOrchestrator`/`advisory_system`, and neither should be until that wiring milestone explicitly re-evaluates whether the reduced accuracy shown in §11 is acceptable for a live deployment. `BottleneckModel` (location), `ExitUsageModel`, and `SmokePredictionModel` are **RESEARCH_ONLY** — none can honestly run from `BuildingState` today, and none should be attempted without first building the topology-indexed schema extension (or, for `SmokePredictionModel`, an entirely different live smoke-sensing capability) named in §12.
