# Live Perception → BuildingState Integration Bridge

Status as of this milestone: `live_runtime.factory.build_live_runtime()` now owns a real, production **perception → fusion → BuildingState** bridge — `live_perception/` — instead of that translation existing only inside a test (as the Sensor Fusion Engine milestone left it). `BuildingState`, `BuildingStateEstimator`, and `sensor_fusion/` itself are all **unmodified**.

## 1. Investigation findings (Phase 1)

Verified directly against the current source before writing anything:

1. `build_live_runtime()` previously wired only `human_detector`/`identity_resolver` into `LiveCameraPipeline` — `tracker`, `behavior_recognizer`, `cross_camera_identity_resolver`, `world_projector`, and `live_occupant_manager` were each a real, separately-tested package, but **none was reachable from the production composition root at all**.
2. `EstimatorBuildingStateGateway` never received a `hazard_snapshot_provider` in production — `BuildingState.hazard_summary` was always the gateway's own empty default. `occupancy_snapshot_provider` was accepted but never had a real production source wired either.
3. `perception.providers.smoke_detector_provider.SmokeDetectorProvider`/`heat_detector_provider.HeatDetectorProvider` are themselves still abstract (`raise NotImplementedError`) — their own docstrings name "future Sensor Fusion stage" as the intended consumer, confirming `sensor_fusion`/`live_perception` (not these classes) is where a real detector reading provider belongs.
4. `models.detector_migration.adapt_legacy_detector()`, invoked automatically inside `sensor_manager.manager.SensorManager.discover_sensors()`, already unifies canonical and legacy detector identity — confirmed directly, never reimplemented here.
5. `LiveOrchestrator.run_cycle()`'s existing sequencing (sensors → perception → building_state → live_ai → live_advisory → legacy ai_inference/decision_policy/recommendation → command_center) already places `BuildingState` construction before Live AI — no reordering was needed; this milestone's new work slots entirely inside the existing `building_state_gateway` seam.
6. `hazard_evolution.merge_strategy.HazardMergeStrategy` and `multi_camera_fusion.engine.MultiCameraFusionEngine` remain the two existing, narrower fusion mechanisms (concurrent-hazard-source fusion; per-camera Detection fusion by `occupant_id`) — neither is duplicated or bypassed by `live_perception/`.

## 2. Package design (Phase 2)

`live_perception/` — a new, small integration package, deliberately **outside** `sensor_fusion/` (which stays completely generic — re-verified by `tests/test_live_perception_architecture_guards.py::SensorFusionRemainsGenericTests`):

- `providers.py` — production `sensor_fusion.provider.ObservationProvider` implementations, duck-typed against SynEvac's own runtime objects.
- `building_state_adapter.py` — `BuildingStateInputAdapter`: `FusedObservation` → `HazardSnapshot`/`OccupancySnapshot`.
- `snapshot.py` — `FusedPerceptionSnapshot` (`timestamp`, `fused_observations`, `hazard_snapshot`, `occupancy_snapshot`).
- `coordinator.py` — `LivePerceptionFusionCoordinator`: the one object `build_live_runtime()` constructs and wires into `EstimatorBuildingStateGateway`.

## 3. Production graph

```
ReplayFrameSource/RTSPFrameSource
        |
        v
HumanDetector.detect()  -->  RawHumanDetection (per-camera, per-frame)
        |
        v
SingleCameraTracker.update()  -->  TrackedHuman (per-camera, frame-to-frame STABLE track_id)
        |
        +--> WorldProjector.project()        -->  world_position/floor_id/zone_id (if calibrated)
        +--> BehaviorRecognizer.recognize()  -->  RecognizedBehavior (+ world_velocity, if projected)
        +--> CrossCameraIdentityResolver.resolve()  -->  building-wide global id (sequential transitions only)
        |
        v
IdentityResolver.resolve()  -->  Detection (one per RawHumanDetection, same order, FINAL occupant_id)
        |
        +----------------------------------------------------------------+
        v                                                                 v
LiveOccupantManager.update(occupant_id, ...)                  CameraManager.all_detections()
        |  (keyed by the FINAL resolved occupant_id --                    |
        |   see Sec 8)                                                    v
        v                                                    MultiCameraFusionEngine.fuse()
LiveOccupantObservationProvider                                          |
LiveSmokeObservationProvider  (SensorManager + reading_provider)          v
LiveHeatObservationProvider   (SensorManager + reading_provider)   FusionResult
LiveFACPObservationProvider   (SimulatedFACP.current_snapshot)           |
        |                                                                 |
        v                                                                 |
sensor_fusion.engine.SensorFusionEngine.collect()/.fuse()                 |
        |                                                                 |
        v                                                                 |
live_perception.building_state_adapter.BuildingStateInputAdapter          |
        |                                                                 |
        v                                                                 |
HazardSnapshot / OccupancySnapshot  ---------------+                      |
                                                    v                      v
                              live_system.building_state_gateway.EstimatorBuildingStateGateway.collect()
                                                    |
                                                    v
                          building_state.estimator.BuildingStateEstimator.estimate()  (UNMODIFIED)
                                                    |
                                                    v
                                     building_state.models.BuildingState  (UNMODIFIED)
                                                    |
                                                    v
                            live_system.orchestrator.LiveOrchestrator  -->  Live AI  -->  Advisory
```

## 4. Data ownership map (Phase 8)

| Concept | Owned by | Identity space |
|---|---|---|
| **Raw Detection** | `human_detection`/`live_camera_pipeline` (`RawHumanDetection`) | per-camera, per-frame only — no identity persists across frames |
| **TrackedHuman** | `tracking.tracker.SingleCameraTracker` | per-camera, frame-to-frame STABLE `track_id` — still camera-local |
| **CrossCameraIdentity** | `cross_camera_identity` (`RuleBasedCrossCameraIdentityResolver`) | building-wide global id, but only for **sequential** departure/arrival transitions (see Sec 8's limitation) |
| **Detection.occupant_id** | `live_camera_pipeline.identity_resolver.IdentityResolver` | the FINAL, authoritative global identity every downstream consumer agrees on |
| **LiveOccupant** | `live_occupants.manager.LiveOccupantManager` | keyed by `Detection.occupant_id` (corrected this milestone — see Sec 8) |
| **FusedTrack** | `multi_camera_fusion.engine.MultiCameraFusionEngine` | keyed by `Detection.occupant_id` (unchanged, pre-existing) |
| **SensorFusionEngine observation** | `sensor_fusion.engine.SensorFusionEngine`, fed by `live_perception.providers.LiveOccupantObservationProvider` | one `OCCUPANCY` observation per zone, summed from `LiveOccupantManager.active_occupants()` |
| **BuildingState.occupant_tracks** | `MultiCameraFusionEngine` → `BuildingStateEstimator` (unchanged path) | keyed by `occupant_id` |
| **BuildingState.zone_occupancy** | `SensorFusionEngine` → `BuildingStateInputAdapter` → `BuildingStateEstimator` (new path this milestone) | keyed by zone |

Both `BuildingState.occupant_tracks` and `BuildingState.zone_occupancy` are downstream of the **same** `Detection.occupant_id` chain, so they agree by construction once wired consistently — proven, not just asserted, in `tests/test_live_perception_double_counting.py`:

> 2 cameras (`CAM-A`, `CAM-B`), 3 physical occupants, 4 raw detections (one person visible in both cameras simultaneously) → `BuildingState.occupant_tracks` has 3 entries, `LiveOccupantManager.active_occupants()` has 3 entries, `BuildingState.zone_occupancy.observation_at("zone-1").occupant_count == 3.0`, and the shared occupant's own `FusedTrack.source_camera_ids == {"CAM-A", "CAM-B"}` — never 4, 6, or 7 anywhere.

**Known, investigated limitation** (not redesigned this milestone, per its own "do NOT redesign SensorFusionEngine" scope): `RuleBasedCrossCameraIdentityResolver` only matches *sequential* departure/arrival transitions. It has no mechanism to reconcile two cameras seeing the *same* person *simultaneously* from the very first cycle (there is no "departure" from either camera for it to match an "arrival" against). `MappingIdentityResolver` is the existing, already-proven mechanism for that scenario (established precedent: `tests/test_rtsp_offline_e2e.py::RTSPOfflineEndToEndTests`), and is what the double-counting test above uses.

## 5. Canonical OCCUPANCY source (Phase 3.1/3.2/5)

Chosen: `live_occupants.manager.LiveOccupantManager`, **not** `multi_camera_fusion.engine.MultiCameraFusionEngine`, as the canonical source `LiveOccupantObservationProvider` reads from. Reasoning: `LiveOccupantManager` already tracks per-occupant lifecycle (NEW/ACTIVE/TEMPORARILY_LOST/EXITED/EXPIRED) — only occupants `active_occupants()` currently reports as genuinely present this cycle are counted, which `MultiCameraFusionEngine`'s own `FusionResult` has no equivalent distinction for. Both sources are downstream of the same `Detection.occupant_id`, so they cannot disagree once both are fed by the same `IdentityResolver` output.

Exactly ONE `LiveOccupantManager` exists per `LiveRuntime` (`build_live_runtime()` default-constructs one only if the caller did not supply one, and threads that single instance into the one `LiveCameraPipeline` — one pipeline serves every camera in `frame_sources`, so every camera updates the same manager). Exactly ONE `SensorFusionEngine` exists per `LiveRuntime` the same way, and `runtime.sensor_fusion_engine` is the identical instance `LivePerceptionFusionCoordinator` fuses through — proven in `tests/test_live_perception.py` and `tests/test_live_runtime_e2e.py`.

CAMERA/detector STATUS (online/offline bookkeeping) deliberately is **not** forced into a fake `Observation` — `ObservationKind` has no STATUS member, and status already reaches `BuildingStateEstimator` directly via `camera_status_provider`/`smoke_detector_status_provider`/`heat_detector_status_provider`, unchanged.

## 6. Never fabricate missing data (Phase 4/10)

`BuildingStateInputAdapter` never invents a value for missing information:
- A zone with **zero** real fused SMOKE/HEAT/FIRE/TEMPERATURE/VISIBILITY observations gets **no entry** in `HazardSnapshot.node_states` — `BuildingState.zone_severity()` already honestly resolves an absent zone to `HazardSeverity.NONE` via its own existing default accessor; this module never fabricates a `HazardNodeState(hazard_score=0.0)` for a zone with no evidence.
- A zone with **zero** real fused OCCUPANCY observations gets **no entry** in `OccupancySnapshot.observations` — `OccupancySnapshot.observation_at()` already honestly returns `occupant_count=None` ("no reading available, never zero people") for an absent zone.
- `LiveSmokeObservationProvider`/`LiveHeatObservationProvider` produce **nothing** when their `reading_provider` is `None` (no real reading source configured yet) — never a fabricated non-alarming reading.
- `LiveFACPObservationProvider` produces **nothing** when `snapshot_provider` is `None` or returns `None` — never a fabricated NORMAL alarm status.
- ALARM-kind `FusedObservation`s are deliberately **not** translated into `BuildingState` at all by `BuildingStateInputAdapter` — `BuildingState.facp_status` is already correctly derived from `FACPSnapshot` via `BuildingStateEstimator`'s own existing, unmodified logic; a second, indirect ALARM path risked silently disagreeing with FACP's own more authoritative view. Fused ALARM observations remain available on `FusedPerceptionSnapshot.fused_observations` for any future diagnostic consumer.

Failure/degradation behavior (`tests/test_live_perception_failure_modes.py`, 12 tests) confirms: camera offline, all cameras offline, one/all detectors unavailable, FACP unavailable, a raising `ObservationProvider`, stale camera/detector observations (confidence decays, the reading itself is never dropped), partial `BuildingState` (hazard present with no occupancy, or vice versa), no occupant observations, camera online with genuinely zero detections, and conflicting manual-vs-sensor observations (flagged via `FusedObservation.conflict`, never hidden) — the runtime never crashes and never fabricates a healthy/zero/NORMAL reading in place of missing data.

## 7. Detector/FACP identity consistency (Phase 9)

`tests/test_live_perception_detector_identity.py` (4 tests) proves ONE detector identity survives every stage, for both a canonical `SmokeDetector`/`HeatDetector` asset and a legacy `Detector` adapted via `adapt_legacy_detector()`:

```
Detector ID -> SensorManager.discover_sensors() -> SensorStatus.sensor_id/zone_ids
            -> LiveSmokeObservationProvider/LiveHeatObservationProvider (reading_provider)
            -> Observation(source=f"smoke-{detector_id}", location=zone_id)
            -> SensorFusionEngine.fuse() -> FusedObservation.contributing_sources
            -> DetectorConditionReport.asset_id (facp/models.py)
            -> SimulatedFACP.evaluate()/current_snapshot() -> active_alarm_source_ids
            -> LiveFACPObservationProvider -> Observation(location=zone_id)
            -> BuildingState
```

No second detector-migration layer was introduced — `SensorManager.discover_sensors()`'s own automatic adaptation is the only place legacy/canonical identity is ever unified.

## 8. Bugs found and fixed in existing code

Two non-obvious bugs surfaced while proving Sec 4's double-counting guarantee — both fixed with the smallest possible change, neither touching `BuildingState`/`BuildingStateEstimator`/`sensor_fusion/`:

1. **`live_camera_pipeline/pipeline.py`** — `LiveOccupantManager.update()` was previously called from inside `_process_camera_cycle()`, keyed by the tracker's own per-camera `track_id` (or `cross_camera_identity_resolver`'s global id, only if that specific seam was configured) — **not** the final `Detection.occupant_id` `IdentityResolver.resolve()` produces. A deployment using `MappingIdentityResolver` for cross-camera reconciliation (Sec 4's documented, correct choice for simultaneous multi-camera visibility) would silently disagree between `LiveOccupantManager` and `BuildingState.occupant_tracks`. Fixed: `_process_camera_cycle()` now returns a `pending_updates` list positionally aligned with its detections; `run_cycle()` calls `identity_resolver.resolve()` first, then zips the two to call `live_occupant_manager.update()` keyed by the FINAL `occupant_id`.
2. **`live_system/building_state_gateway.py`** — `EstimatorBuildingStateGateway.collect()` evaluates keyword arguments in left-to-right source order. `hazard_snapshot`/`occupancy_snapshot` (now backed by `LivePerceptionFusionCoordinator`, which reads `LiveOccupantManager` state) were previously evaluated *before* `fusion_result` — the one provider whose evaluation has the side effect of calling `LiveCameraPipeline.run_cycle()`, which is what actually populates `LiveOccupantManager` for that cycle. Fixed: `fusion_result` is now resolved in its own explicit statement, first, before the `estimate()` call is built.

Both fixes were re-verified against the full existing regression suite (all pre-existing pipeline/gateway/runtime tests, ~112 tests at the time) in addition to the new tests that exposed them.

## 9. Files created / modified

**Created:**
- `live_perception/{__init__,providers,building_state_adapter,snapshot,coordinator}.py`
- `tests/test_live_perception.py` — 14 unit tests (Phase 3/4/6)
- `tests/test_live_perception_architecture_guards.py` — import-guard tests (Phase 13)
- `tests/test_live_perception_double_counting.py` — the Phase 8 worked-example proof
- `tests/test_live_perception_detector_identity.py` — 4 tests (Phase 9)
- `tests/test_live_perception_failure_modes.py` — 12 tests (Phase 10)
- `tests/test_live_runtime_perception_e2e.py` — full offline replay → advisory chain (Phase 11)
- `scripts/benchmark_live_perception.py` — performance benchmark (Phase 12)
- `docs/architecture/live_perception_building_state_integration.md` — this document

**Modified:**
- `live_runtime/factory.py` — new optional `tracker`/`behavior_recognizer`/`cross_camera_identity_resolver`/`world_projector`/`live_occupant_manager`/`sensor_fusion_engine`/`smoke_detector_reading_provider`/`heat_detector_reading_provider` parameters; builds and wires `LivePerceptionFusionCoordinator`; wires `hazard_snapshot_provider` for the first time in production.
- `live_runtime/runtime.py` — stores `live_occupant_manager`/`sensor_fusion_engine`/`perception_fusion_coordinator` as new, deliberately untyped attributes (no new concrete-class imports — architecture guard unaffected).
- `live_camera_pipeline/pipeline.py` — the ordering fix described in Sec 8.
- `live_system/building_state_gateway.py` — the evaluation-order fix described in Sec 8.

**Unchanged (verified, not modified):** `building_state/estimator.py`, `building_state/models.py`, every file in `sensor_fusion/`, `multi_camera_fusion/*`, `models/detector_migration.py`, `facp/*`.

## 10. Performance (Phase 12)

`scripts/benchmark_live_perception.py`, at the milestone's required scale (20 cameras, 100 occupants, 50 smoke detectors, 50 heat detectors, 20 zones), zero real YOLO/tracker/RTSP inference included:
- Observation collection: ~0.28 ms/call (mean).
- Fusion: ~0.53 ms/call (mean).
- BuildingState input adaptation: ~0.09 ms/call (mean).
- BuildingState estimation (`BuildingStateEstimator.estimate()`, unmodified): ~0.02 ms/call (mean).
- Complete perception → BuildingState stage (collect + estimate): ~1.03 ms/call (mean).

Real per-camera detector/tracker inference timing is reported separately in `scripts/benchmark_yolo_human_detector.py` and `scripts/benchmark_live_camera_pipeline.py` — never conflated with the numbers above.

## 11. Answers to this milestone's own closing questions

**A. Does production `LiveRuntime` now actually use `SensorFusionEngine`?** Yes. `build_live_runtime()` default-constructs (or accepts) exactly one `SensorFusionEngine`, wires it into exactly one `LivePerceptionFusionCoordinator`, and wires that coordinator's `hazard_snapshot_provider`/`occupancy_snapshot_provider` directly into `EstimatorBuildingStateGateway`. `runtime.sensor_fusion_engine` is the identical instance used every cycle — proven in `tests/test_live_runtime_e2e.py` and `tests/test_live_runtime_perception_e2e.py`.

**B. Does `BuildingState` now receive fused live perception data rather than relying only on a test bridge?** Yes. `hazard_snapshot_provider` is wired into production for the first time this milestone (it was always the gateway's own empty default before). `occupancy_snapshot_provider` prefers a caller-supplied value (backward compatible) and falls back to the same coordinator otherwise.

**C. Can two cameras observing the same person cause occupancy double-counting?** No — proven directly, not just asserted, by `tests/test_live_perception_double_counting.py`'s worked example (Sec 4): 4 raw detections from 2 cameras resolve to exactly 3 everywhere occupancy is represented (`BuildingState.occupant_tracks`, `LiveOccupantManager.active_occupants()`, `BuildingState.zone_occupancy`), because every one of those three is downstream of the same `Detection.occupant_id`.

**D. Is missing sensor information ever silently converted into a healthy/zero reading?** No — see Sec 6. Every absence (no reading provider, no FACP snapshot, no detections, a raising provider, a camera never configured) produces a genuinely empty/absent entry, never a fabricated NORMAL/zero value, verified across all 12 tests in `tests/test_live_perception_failure_modes.py`.
