# Live Occupancy, Crowd Density & Congestion Intelligence

Status as of this milestone: a new, deterministic runtime analytics package, `crowd_intelligence/`, derives per-zone density, per-asset (Door/Exit/Stair) approach demand, queue formation, and congestion classification directly from the SAME canonical live occupants every other live package already reads — never a new AI model, never a modification to the existing trained bottleneck model, never an evacuation decision.

## 1. Investigation findings (Phase 1)

Verified directly against the current source before writing anything:

1. **Simulation congestion metrics** (`simulator/congestion.py`): `DefaultCongestionModel`/`StairAwareCongestionModel` compute a `speed_factor` from `other_occupants`/`opposing_occupants` counts on a navigation `Edge` — these counts only exist inside a running `MultiAgentSimulation`'s occupant-routing loop.
2. **Ground Truth bottleneck metrics** (`ground_truth/bottleneck.py`, `risk_analysis.py`): `compute_congestion()`/`compute_engineering_findings()` are entirely post-hoc analysis of a completed `MultiAgentSimulationResult` (`peak_edge_occupancy`, `occupant.steps[].queue_wait_time`) — none of this exists, or can exist, in Live mode.
3/4. **Simulation-only vs. live-computable**: occupancy counts, `queue_wait_time`, `peak_*_occupancy` are simulation-only. `simulator.capacity.DefaultCapacityModel`/`StairCapacityModel`/`derive_stair_capacity()` are **pure functions of `Edge.width`/`capacity`/`walking_distance`** — safely reusable live, with no simulation dependency.
5. **Zone area**: `Zone.area` = `width * height` (gross rectangle; `polygon` is stored but unused for area).
6. **Door/Exit/Stair geometry**: sufficient for approach/queue estimation. Door/Exit carry `start_point`/`end_point` (a segment) + `width`; Staircase carries `from_position`/`to_position` (one point per floor side) + `width` + `travel_distance(building)`. All three carry explicit zone connectivity, never geometrically inferred.
7. **NavigationGraph**: could associate occupants/zones with nearby assets, but was **not used** — Door/Exit/Staircase already carry their own zone connectivity fields directly, and `live_occupants.lifecycle.is_near_exit()` already establishes the exact "distance from world_position to a segment" pattern this package reuses (as its own, separately-owned copy — see Sec 4). Using `NavigationGraph` would add an avoidable dependency this package's own architecture guard (Sec 8) would then have to special-case.
8. **Existing crowd-density model**: none found anywhere in the repository.
9. **BuildingState congestion/density fields**: none (confirmed by reading `building_state/models.py` in full).
10. **Command Center congestion display**: `command_center/occupancy_panel.py` already renders `current_congestion`/`current_queue_length`/`current_bottleneck`, but these originate from **Simulation/Replay incident playback** (`simulator/congestion.py`'s own tick-level state), never from a Live source. This milestone does not touch that panel — wiring a live display is a future consumer's job, per Sec 3's own architecture.

**Key design decision (Phase 11)**: investigated stuffing a `crowd_intelligence` field onto `BuildingState` itself, alongside `facp_status`/`control_status`. Rejected: `facp_status`/`control_status` are **passthrough parameters of `BuildingStateEstimator.estimate()` itself** — this milestone must not add a new `estimate()` parameter (equivalent to redesigning `BuildingState`). Instead, this package follows the *already-existing* precedent of `live_system.state_manager.LiveBuildingSnapshot.ai_prediction_snapshot`/`advisory_report` — a sibling, additive snapshot field with its own `StateManager.update_*()`/`latest_*()` pair, populated by an optional Protocol-shaped Gateway in `LiveOrchestrator.run_cycle()`. `BuildingState`/`BuildingStateEstimator` are completely unmodified by this milestone.

Also found: `LiveOccupant.world_position` is `None` exactly when no calibration succeeded for that sighting (Phase 13's coverage distinction falls out for free from data already collected — no new field was needed), and `LiveOccupant.world_velocity` is a **scalar speed, not a direction vector** — so "approach evidence" is derived from `OccupantHistory.position_samples` (distance-to-asset decreasing across the occupant's own last two recorded samples), never a velocity vector.

## 2. Package architecture

```
Building geometry (models.zone.Zone / models.door.Door / models.exit.Exit / models.staircase.Staircase)
        +
live_occupants.manager.LiveOccupantManager.active_occupants() / all_occupants()   (Phase 12's canonical occupant source)
        |
        v
crowd_intelligence.density        -- per-zone occupant_count/density/moving-stationary-running/mean_speed/position coverage
crowd_intelligence.flow           -- Door/Exit/Stair "side" geometry, distance-to-asset, approach evidence (position-history based)
crowd_intelligence.capacity       -- SIMULATION-STYLE capacity, reusing navigation.edge.Edge + simulator.capacity (Sec 1.4)
crowd_intelligence.queue          -- approach region occupants, queue candidates (STATIONARY behavior), estimated_queue_length
crowd_intelligence.congestion     -- demand-to-capacity ratio -> IntensityLevel, a LIVE-ONLY classification (Sec 5)
crowd_intelligence.trends         -- bounded, configurable-time-window RISING/STABLE/FALLING/UNKNOWN
        |
        v
crowd_intelligence.engine.CrowdIntelligenceEngine.compute(time) -> crowd_intelligence.models.CrowdIntelligenceSnapshot
        |
        v
live_system.crowd_intelligence_gateway.EngineCrowdIntelligenceGateway   (never raises -- a failure produces None, never crashes the live cycle)
        |
        v
live_system.orchestrator.LiveOrchestrator.run_cycle()  -- AFTER building_state_gateway, BEFORE live_ai_gateway/live_advisory_gateway
        |
        v
live_system.state_manager.LiveBuildingSnapshot.crowd_intelligence   (StateManager.update_crowd_intelligence()/latest_crowd_intelligence())
        |
        v
Future AI / Advisory / Command Center consumers (none wired by this milestone)
```

`crowd_intelligence/` itself never imports `live_system`, `live_runtime`, AI, Advisory, Command Center, RL, YOLO, RTSP, Voice Evacuation, Building Control, or FACP — the composition above is entirely `live_system.crowd_intelligence_gateway`'s and `live_runtime.factory.build_live_runtime()`'s job (Phase 18, mechanically enforced).

## 3. Density model (Phase 4)

`crowd_intelligence.density.compute_zone_density(occupant_count, zone_area)` = `occupant_count / zone_area`, `None` whenever `zone_area` is `None` or `<= 0` (never a fabricated infinite/zero density). `zone_area` is always `Zone.area` (gross rectangle) — obstacle-based usable-area reduction was investigated and **not implemented**: `models.obstacle.Obstacle` carries no `zone_id` of its own anywhere in this codebase (its own docstring: "used by future simulation/navigation movement-cost calculations. Never interpreted here"), and no existing code anywhere geometrically associates an `Obstacle` with a `Zone`. Inventing that association now would be exactly the "inventing precision" Phase 4 warns against — the honest choice is the documented gross area.

`crowd_intelligence.models.DensityThresholds` (LOW/MODERATE/HIGH/VERY_HIGH/CRITICAL via `IntensityLevel`) is a **configurable dataclass**, not a fixed classmethod the way `hazard.severity.HazardSeverity.from_score()` is — deliberately, since no density standard is represented anywhere in this repository (confirmed) and Phase 4 requires thresholds to be reconfigurable per deployment. Default cutoffs (1.0/2.0/3.0/4.0 people/m²) are an explicit **project assumption**, disclosed the same way `simulator.capacity.DefaultCapacityModel`/`simulator.congestion.DefaultCongestionModel` already disclose their own non-validated defaults.

A zone's own **already-modeled** `Zone.max_occupancy` (an existing Designer-authored field, already given exactly this "this zone's own capacity" meaning by `navigation.node.Node.capacity` — confirmed, not a new interpretation invented here) is reused as an *additional*, independent signal for `BuildingCrowdSummary.zones_above_configured_density_threshold`: a zone is flagged if either the global `IntensityLevel` scale classifies it HIGH-or-worse, **or** its own `occupant_count` exceeds its own `max_occupancy`.

## 4. Flow / approach model (Phase 5)

Each Door/Exit contributes one `AssetSide` (its own real segment, on one floor); a Staircase contributes **two** `AssetSide`s (a degenerate point-segment at `from_position`/`to_position`, on `from_floor_id`/`to_floor_id` respectively — Sec 8's own worked multi-floor example). `crowd_intelligence.flow._distance_point_to_segment()` is the same formula `live_occupants.lifecycle._distance_point_to_segment()` already uses, deliberately **duplicated, not imported** — that function is a private helper of a different package never meant to be imported across a boundary, and this package's own dependency surface stays minimal.

Approach evidence (`evaluate_approach()`) requires **distance decreasing across the occupant's own last two recorded position samples** (`OccupantHistory.position_samples`) — satisfying Phase 5's "distance decreasing over time and/or velocity vector" requirement via the distance-decrease branch alone, since `LiveOccupant.world_velocity` carries no direction. Only the last two samples are compared (not the full history window): a `PositionSample` carries no per-sample floor id, so comparing across a floor **transition** (a discrete stair-crossing event) could otherwise mix two floors' coordinate spaces; restricting to the last two samples means this is wrong for at most the single cycle a floor change itself occurs in, and self-corrects immediately after.

## 5. Capacity and congestion (Phase 6)

`crowd_intelligence.capacity` reuses `simulator.capacity.DefaultCapacityModel`/`StairCapacityModel` **directly**, via a thin, stateless `navigation.edge.Edge` reference wrapper (no `NavigationGraph`, no `MultiAgentSimulation` state constructed) — returned as `simulation_style_capacity` everywhere, named and documented specifically so it is never confused with a measured live flow rate.

`crowd_intelligence.congestion.compute_congestion_level()` computes `(queue_candidate_count + approaching_count) / simulation_style_capacity` and classifies it through a separate, also-configurable `CongestionThresholds` scale (ratio-based, distinct units from `DensityThresholds`'s people/m²). **This ratio is the live-observable signal; `simulation_style_capacity` is only its denominator.** SIMULATION CAPACITY (`simulator.capacity`, a design-time engineering estimate from static geometry) and LIVE ESTIMATED CONGESTION (`crowd_intelligence.congestion`, computed every cycle from live queue/approach evidence) are two related but never-conflated numbers — the former feeds the latter as one input, never the reverse, and neither is a validated life-safety flow-rate model (same disclosure `simulator.capacity`/`simulator.congestion` already carry).

## 6. Queue detection (Phase 7)

`crowd_intelligence.queue.compute_queue_metrics()`: a configurable `approach_region_depth` (default 3.0m, a documented project assumption) defines the region; `occupants_near_asset()` finds occupants inside it **with a known world_position** (an occupant with no position is honestly excluded, never assigned an arbitrary coordinate); `queue_candidate_count` reuses the **already-computed** `RecognizedBehavior.STATIONARY` classification from `behavior_recognition` (never a second, competing speed threshold invented here). A door/exit/stair with **zero** occupants nearby reports a genuine `estimated_queue_length=0` (a position-confirmed zero) — never fabricated from zone occupancy alone (proven directly in `tests/test_crowd_intelligence.py::QueueFormationTests::test_no_queue_fabricated_from_zone_occupancy_alone`).

## 7. Temporal trends (Phase 8)

`crowd_intelligence.trends.TrendTracker`, one shared instance per `CrowdIntelligenceEngine`: bounded history (`max_history_length`, default 20 samples per key — never unlimited growth) and a **separate, configurable time window** (`trend_window_seconds`, default 30s) that only compares the latest sample against the earliest sample still inside that window. A metric near zero never flickers RISING/FALLING from noise — both a relative (10%) and an absolute (0.05) tolerance apply, whichever is larger. A `None` reading (genuinely unavailable this cycle) is never recorded and never compared against, so a real reading immediately after one still compares honestly against the last *real* value.

## 8. Double-count prevention (Phase 12)

Every occupant-derived number in this package (`ZoneCrowdMetrics.occupant_count`/`density`, every asset's `approaching_count`/`queue_candidate_count`) is computed **exclusively** from `LiveOccupantManager.active_occupants()` — the identical canonical source `live_perception.providers.LiveOccupantObservationProvider` already uses for `BuildingState.zone_occupancy` (established by the prior Live Perception → BuildingState Integration Bridge milestone). Never raw per-camera detections, never an independent `multi_camera_fusion` recount.

Proven directly, extending that same milestone's own worked example one layer further: `tests/test_crowd_intelligence_double_counting.py` — 2 cameras, 3 physical occupants, 4 raw detections (one person visible in both cameras simultaneously, resolved via `MappingIdentityResolver`) → `CrowdIntelligenceEngine`'s own `zone.occupant_count == 3`, `density == 3/zone_area`, `building_summary.total_observed_occupants == 3` — never 4.

## 9. Missing-calibration behavior (Phase 13)

A "known occupant" (a resolved global identity, `current_zone_id` set) is distinguished from a "known precise position" (`world_position` set) throughout:

- **Zone-level** metrics (`occupant_count`, `density`) are computed from zone identity alone — usable even when `world_position` is `None` for every occupant in the zone (an uncalibrated camera whose own fixed zone assignment is still known). `ZoneCrowdMetrics.position_coverage_count`/`position_coverage_fraction` separately, honestly report how many of those occupants also have a usable position (`None` only when the zone has zero occupants at all — no honest denominator to compute a fraction of).
- **Asset-level** metrics (`approaching_count`, `queue_candidate_count`, `estimated_queue_length`, `mean_approach_speed`) require a real world position, since they are inherently geometric. `AssetApproachMetrics.position_available` is `False` **only** when there is at least one active occupant known to be on that asset's own floor(s) whose `world_position` is `None` (an honest "reduced coverage" case) — and `True` (vacuously) when nobody at all is currently known to be on those floors (a genuine, position-confirmed zero, never confused with a coverage gap). Never a fabricated queue/approach reading computed as if a missing position were "no one there."

## 10. Confidence / coverage (Phase 14)

Reported wherever appropriate:
- `ZoneCrowdMetrics.position_coverage_count` / `position_coverage_fraction`
- `BuildingCrowdSummary.calibrated_occupant_count` / `total_observed_occupants` / `position_coverage_fraction`
- `AssetApproachMetrics.position_available` — this package's own equivalent of "metric_available", named for asset-specific clarity (an asset's approach/queue fields are specifically position-dependent; "metric" alone doesn't say which)

## 11. Runtime ownership (Phase 10) and BuildingState/StateManager integration (Phase 11)

Exactly **one** `CrowdIntelligenceEngine` per `LiveRuntime` (`build_live_runtime()` default-constructs one — reading the SAME shared `LiveOccupantManager` and `Building` every other stage uses — only if a caller did not already supply one), exposed as `runtime.crowd_intelligence_engine`. Wired into `LiveOrchestrator` via `EngineCrowdIntelligenceGateway`, run in `run_cycle()` **after** `building_state_gateway` (this cycle's live occupant/perception state has already been updated by whichever `fusion_result_provider` ran) and **before** `live_ai_gateway`/`live_advisory_gateway` (so a future AI/Advisory/Command Center consumer can read this cycle's crowd intelligence the same cycle it was computed) — neither existing stage is reordered, this is a new, independent, optional stage inserted between two that already existed. The result lands on `LiveBuildingSnapshot.crowd_intelligence` via `StateManager.update_crowd_intelligence()`/`latest_crowd_intelligence()`, mirroring `ai_prediction_snapshot`/`advisory_report` exactly (Sec 1's own design decision). `EngineCrowdIntelligenceGateway.compute()` never raises — an unexpected failure returns `None`, leaving the previous snapshot in place, never crashing the live cycle.

## 12. Offline end-to-end results (Phase 16)

`tests/test_live_runtime_crowd_intelligence_e2e.py` drives the full production chain (`ReplayFrameSource` → `YOLOHumanDetector` w/ fake backend → `SimpleSingleCameraTracker` → `WorldProjector` → `RuleBasedBehaviorRecognizer` → `LiveOccupantManager` → `LivePerceptionFusionCoordinator` → `CrowdIntelligenceEngine` → `BuildingState`/`StateManager`) across 5 cycles: two occupants walk toward the same `Exit`, stop (forming a queue), then both leave. Proven directly: occupancy rises `1 → 1 → 2 → 2 → 0`; density rises and falls in lockstep; approach evidence is honestly absent at the very first cycle (fewer than 2 position samples) and present as both occupants close in; the queue reaches 2 candidates at its peak and clears to 0 once both leave; congestion peaks at the same cycle the queue peaks (proven via `IntensityLevel` ordering) and falls back to `LOW` once everyone has left; the trend reports `RISING` at some point during the build-up and `FALLING` on the final, clearing cycle (using a `trend_window_seconds` matched to this test's own 1-second cycle cadence — the default 30s window is sized for a real deployment's much longer runtime, not a 4-second test). Zero network, zero physical CCTV anywhere in this file.

## 13. Performance (Phase 17)

`scripts/benchmark_crowd_intelligence.py`, at the milestone's required scale (50 zones, 20 doors, 10 exits, 10 stairs, 100 occupants — a 20-camera-equivalent population), zero real YOLO/tracker/RTSP inference included:
- Zone aggregation: ~0.22 ms/call (mean).
- Asset approach calculation (1 door, 100 occupants): ~0.10 ms/call (mean).
- Queue detection (all 20 doors): ~2.07 ms/call (mean).
- Trend update (50 zone keys/call): ~0.06 ms/call (mean).
- Complete crowd-intelligence cycle (all zones/doors/exits/stairs): ~5.5 ms/call (mean) — well within a 1-second live cycle budget.

Real per-camera detector/tracker inference timing is reported separately in `scripts/benchmark_yolo_human_detector.py`/`scripts/benchmark_live_perception.py` — never conflated with the numbers above.

## 14. Architecture guards (Phase 18)

`tests/test_crowd_intelligence_architecture_guards.py` mechanically verifies: `crowd_intelligence/` never imports AI/Advisory/Command Center/RL/YOLO/RTSP/Voice Evacuation/Building Control/FACP; never calls an action-execution verb (`.evaluate(`/`.acknowledge(`/`.broadcast(`/`.execute_control(`/etc.); and only imports from its own documented allow-list (`live_occupants`, `models`, `navigation.edge`, `simulator.capacity`, `behavior_recognition.observation`, plus itself).

## 15. Data-source classification table (Phase 19)

| Metric | Classification | Notes |
|---|---|---|
| `ZoneCrowdMetrics.occupant_count`/`density_people_per_m2` | **LIVE-OBSERVABLE** | from `LiveOccupantManager.active_occupants()` + `Zone.area` |
| `ZoneCrowdMetrics.moving_count`/`stationary_count`/`running_count`/`mean_speed` | **LIVE-OBSERVABLE** | from `LiveOccupant.behavior`/`world_velocity` |
| `ZoneCrowdMetrics.temporarily_lost_count` | **LIVE-OBSERVABLE** | from `LiveOccupantManager.all_occupants()`, `OccupantStatus.TEMPORARILY_LOST` |
| `AssetApproachMetrics.approaching_count`/`queue_candidate_count`/`estimated_queue_length`/`mean_approach_speed` | **LIVE-OBSERVABLE, requires calibration** | needs `world_position` — see Sec 9 |
| `AssetApproachMetrics.simulation_style_capacity` | **SIMULATION-STYLE ESTIMATE** | `simulator.capacity`'s own design-time formula from static width/geometry, not a live measurement |
| `AssetApproachMetrics.congestion_level` | **ESTIMATED** | live demand (observable) ÷ simulation-style capacity (estimate) — a derived blend, never claimed as validated hydraulics |
| `*.trend` | **ESTIMATED** | derived from bounded live history, a configurable-window classification, not a measurement of its own |
| `simulator.congestion`/`simulator.capacity`'s own `other_occupants`/`opposing_occupants`/`queue_wait_time` | **SIMULATION-ONLY** | requires a running `MultiAgentSimulation`; not reachable or reused live |
| `ground_truth.bottleneck`'s own `peak_edge_occupancy`/`doors_that_became_bottlenecks` | **SIMULATION-ONLY** | post-hoc analysis of a completed `MultiAgentSimulationResult` |
| Any zone/asset metric for a floor/asset with zero known occupants nearby | **UNAVAILABLE WITHOUT CALIBRATION → genuine zero** | distinguished from "unavailable" via `position_available`/`position_coverage_fraction` — see Sec 9 |

## 16. Every threshold/engineering assumption in this milestone

- `DensityThresholds` defaults (1.0/2.0/3.0/4.0 people/m²) — Sec 3.
- `CongestionThresholds` defaults (0.5/1.0/1.5/2.0 demand-to-capacity ratio) — Sec 5.
- `DEFAULT_APPROACH_REGION_DEPTH = 3.0` meters — Sec 6.
- `TrendConfig` defaults (`max_history_length=20`, `trend_window_seconds=30.0`, `stable_relative_tolerance=0.10`, `stable_absolute_tolerance=0.05`) — Sec 7.
- Every `simulator.capacity` constant this package reuses (`PEOPLE_PER_METER_OF_WIDTH`, `VERTICAL_DISTANCE_STEP_M`, `MINIMUM_CAPACITY`) — pre-existing, unmodified, already disclosed by that module itself.

All five are configurable dataclass/constructor parameters, never hardcoded inline, and none claims to be a validated fire-egress or life-safety standard.

## 17. Files created / modified

**Created:**
- `crowd_intelligence/{__init__,models,density,flow,capacity,queue,congestion,trends,engine}.py`
- `live_system/crowd_intelligence_gateway.py`
- `tests/test_crowd_intelligence.py` — 43 unit tests (Phase 15)
- `tests/test_crowd_intelligence_double_counting.py` — the Phase 12 proof
- `tests/test_crowd_intelligence_calibration_coverage.py` — 8 tests (Phase 13/14)
- `tests/test_crowd_intelligence_architecture_guards.py` — 4 import/action-verb guard tests (Phase 18)
- `tests/test_live_runtime_crowd_intelligence_e2e.py` — 3 tests, full offline chain (Phase 16)
- `scripts/benchmark_crowd_intelligence.py` — performance benchmark (Phase 17)
- `docs/architecture/live_crowd_intelligence.md` — this document

**Modified:**
- `live_runtime/factory.py` — new optional `crowd_intelligence_engine` parameter; default-constructs and wires one into `LiveOrchestrator` via `EngineCrowdIntelligenceGateway`.
- `live_runtime/runtime.py` — stores `crowd_intelligence_engine` as a new, deliberately untyped attribute.
- `live_system/orchestrator.py` — new optional `crowd_intelligence_gateway` constructor parameter; a new, independent `run_cycle()` stage (Sec 11); a new `latest_crowd_intelligence` forwarding property.
- `live_system/state_manager.py` — new `LiveBuildingSnapshot.crowd_intelligence` field; `StateManager.update_crowd_intelligence()`/`latest_crowd_intelligence()`.
- `live_system/event_bus.py` — new `EventType.CROWD_INTELLIGENCE_UPDATED`.

**Unchanged (verified, not modified):** `building_state/estimator.py`, `building_state/models.py`, `ai_training/models/bottleneck_model.py` (the existing trained bottleneck AI model), `sensor_fusion/*`, `live_perception/*`, `simulator/congestion.py`, `simulator/capacity.py`, `ground_truth/bottleneck.py`, `ground_truth/risk_analysis.py`, `navigation/*`, every `models/*` geometry class, `command_center/occupancy_panel.py`.

## 18. Answers to this milestone's own closing questions

**A. Can SynEvac now calculate live zone crowd density without counting the same person twice?** Yes — proven directly by `tests/test_crowd_intelligence_double_counting.py` (Sec 8): 4 raw detections from 2 overlapping cameras resolve to `occupant_count == 3` and `density == 3/zone_area`, never 4.

**B. Can it identify crowd accumulation near doors/exits/stairs using world-space evidence?** Yes — `AssetApproachMetrics.approaching_count`/`queue_candidate_count`, computed from real `world_position`/`OccupantHistory` evidence, never from zone occupancy or geometric proximity alone (Sec 4/6), proven end-to-end in `tests/test_live_runtime_crowd_intelligence_e2e.py`.

**C. Can it detect an estimated queue forming and clearing?** Yes — `tests/test_crowd_intelligence.py::QueueFormationTests` and the Phase 16 end-to-end test both prove `estimated_queue_length` rising as occupants accumulate and stop, then falling to 0 once they leave.

**D. Does it distinguish genuinely observed congestion from simulation-only queue/capacity information?** Yes — Sec 5: `simulation_style_capacity` is explicitly named and documented as a reused, design-time estimate (never a live measurement); `congestion_level` is a separate, live-only classification combining real queue/approach evidence with that estimate as only one input; `simulator.congestion`/`ground_truth.bottleneck`'s own simulation-only signals (`queue_wait_time`, `peak_edge_occupancy`) are never read or reused by this package at all.

**E. Does it remain honest when camera calibration/world position coverage is incomplete?** Yes — Sec 9/10, proven by `tests/test_crowd_intelligence_calibration_coverage.py`: zone-level occupancy/density remain usable from zone identity alone; asset-level approach/queue metrics honestly report `position_available=False` (not a fabricated zero) whenever a known occupant on that asset's own floor lacks a position; coverage fractions are reported as real, computed numbers only when there is an honest denominator, `None` otherwise.
