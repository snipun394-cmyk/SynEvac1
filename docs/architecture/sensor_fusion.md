# Sensor Fusion Engine

Status as of this milestone: **one canonical, cross-KIND fusion layer** — `sensor_fusion.engine.SensorFusionEngine` — that reconciles evidence from every perception source (YOLO/CCTV occupancy and behavior, smoke detectors, heat detectors, FACP, manual operator input, and future BLE/WiFi/UWB/firefighter-report sources) into one deterministic, confidence-scored result per `(location, kind)` pair. No ML, no learned weighting — every number here is plain, documented arithmetic.

## 1. Pipeline (current)

```
LiveOccupants (live_occupants.manager.LiveOccupantManager)  -> sensor_fusion.provider.CameraObservationProvider
Smoke Detectors (perception.models.smoke_detector_observation.SmokeDetectorReading) -> sensor_fusion.provider.SmokeObservationProvider
Heat Detectors (perception.models.heat_detector_observation.HeatDetectorReading)   -> sensor_fusion.provider.HeatObservationProvider
FACP (facp.models.FACPSnapshot)                                                     -> sensor_fusion.provider.FACPObservationProvider
Manual operator input                                                               -> sensor_fusion.provider.ManualObservationProvider
Future sensors                                                                      -> a new ObservationProvider implementation
        |
        v
sensor_fusion.engine.SensorFusionEngine.collect() -> sensor_fusion.observation.Observation
        |
        v
sensor_fusion.engine.SensorFusionEngine.fuse()    -> sensor_fusion.observation.FusedObservation
        |
        v
[a small, caller-owned bridge -- OUTSIDE sensor_fusion/, see Sec 7]
        |
        v
building_state.estimator.BuildingStateEstimator.estimate()   (UNMODIFIED)
        |
        v
building_state.models.BuildingState   (UNMODIFIED)
```

## 2. Investigation findings (Phase 1)

Verified directly against the current source:

1. **Current fusion responsibilities**: two genuine, existing, working fusion mechanisms already exist, each scoped to ONE kind of data:
   - `hazard_evolution.merge_strategy.HazardMergeStrategy`/`DefaultHazardMergeStrategy` — fuses *concurrent hazard sources* (Fire Growth, Smoke Propagation, Detector Activation, ...) proposing values for the *same* node/edge, worst-case-wins (max smoke/temperature, min visibility, max hazard_score).
   - `multi_camera_fusion.engine.MultiCameraFusionEngine` — fuses per-camera `Detection`s into one `FusedTrack` by `occupant_id` equality.
   - Neither does — or was ever meant to do — **cross-kind** reconciliation (e.g. "does the camera's occupancy report agree with the smoke detector's silence, or with a manual operator's claim?"). That is the genuine gap this milestone closes, not a duplicate of either mechanism.
2. **Existing duplicated fusion logic**: none found. `sensor_fusion/` operates one level above both existing mechanisms and never touches their internals.
3. **Existing provider interfaces**: `hazard_evolution.source.HazardSource.propose()`, `perception.providers.provider.PerceptionProvider.observation_at()` — the latter's own docstring **explicitly names "Sensor Fusion"** as a future implementer ("no Ground Truth adapter, no Sensor Fusion, and no Occupancy Estimation exist yet to assemble a BuildingObservation out of"). `sensor_fusion.provider.ObservationProvider.collect()` mirrors both conventions (one method, the engine never knows which kind of source it's talking to).
4. **Existing observation models**: `perception.models.human_observation.HumanObservation`, `perception.models.building_observation.ObservedNodeState`/`ObservedOccupancy`/`ObservedEdgeState`, `occupancy.observation.OccupancyObservation`, `hazard.node_state.HazardNodeState`, `perception.models.smoke_detector_observation.SmokeDetectorReading`/`heat_detector_observation.HeatDetectorReading` — every one of these is scoped to exactly one existing package's own pipeline; none is cross-kind. `sensor_fusion.observation.Observation` is a genuinely new, unified shape, not a duplicate.
5. **Current `BuildingState` creation path**: confirmed directly in `designer/building_state_debug_runner.py::BuildingStateDebugRunner.run()` — it independently queries `PerceptionDebugRunner` (hazard/occupancy ground truth), `CameraManager`/`SensorManager` (asset statuses), builds a `FusionResult` from Sandbox occupants directly, evaluates FACP, **then** calls `BuildingStateEstimator.estimate(...)` with all of these independently-collected pieces. This is exactly the "every perception source reaches BuildingState independently" pattern this milestone's own brief describes — confirmed, not assumed.

## 3. Unified observation model (Phase 3)

`sensor_fusion/observation.py`: `ObservationKind` (`OCCUPANCY, BEHAVIOR, SMOKE, HEAT, FIRE, TEMPERATURE, VISIBILITY, ALARM`, extensible — mirrors `live_system.event_bus.EventType`'s own "extend this list" convention) + `Observation` (`source, kind, location, timestamp, confidence, measurement` — exactly Phase 3's required shape) + `FusedObservation` (adds `contributing_sources` and `conflict`). `measurement` is deliberately typed `Any` (the same "typed `Any`, not imported for its own sake" convention `live_system.event_bus.Event.payload` already uses) since different kinds carry genuinely different shapes.

## 4. Provider interface (Phase 4)

`sensor_fusion.provider.ObservationProvider.collect(time) -> Tuple[Observation, ...]`. Concrete examples, each **duck-typed** against plain objects (never importing `perception`/`facp`/`live_occupants` directly — verified by `tests/test_sensor_fusion_architecture_guards.py`'s stronger "imports nothing from this repository at all" check):
- `CameraObservationProvider` — `set_occupants(occupants)` (any object exposing `occupant_id`/`current_zone_id`/`behavior`/`confidence`, i.e. `live_occupants.occupant.LiveOccupant`-shaped) → one `OCCUPANCY` observation per zone (headcount), one `BEHAVIOR` observation per occupant with known behavior.
- `SmokeObservationProvider`/`HeatObservationProvider` — `set_readings(readings)` (any `SmokeDetectorReading`/`HeatDetectorReading`-shaped object) + a `zone_by_detector_id` map (a raw reading carries no location of its own — confirmed directly from that type's own fields).
- `FACPObservationProvider` — `set_snapshot(snapshot)` (`FACPSnapshot`-shaped) + `zone_by_source_id` → `ALARM` observations for currently-alarming sources only.
- `ManualObservationProvider` — `report(kind, location, measurement, confidence, timestamp)`, queued and cleared on the next `collect()`.

## 5. Fusion engine and merge model (Phase 5)

`sensor_fusion.engine.SensorFusionEngine`: `collect()` gathers from every provider (a failing provider is caught and skipped — Phase 9's "provider failures" — never blocking the rest); `fuse()` groups strictly by `(location, kind)` — **never across kinds** (Phase 7's own worked example: a camera's occupancy report and a smoke detector's silence are two separate `FusedObservation`s, never compared against each other). Duplicate observations from the *same* source in one cycle are de-duplicated (keeping the best) before any agreement/conflict reasoning, so a buggy double-report can never fabricate corroboration.

`sensor_fusion/merge.py` — deterministic, life-safety-appropriate rules matching `DefaultHazardMergeStrategy`'s own existing "worst-case-wins" philosophy, applied consistently rather than reinvented: `OCCUPANCY`/`TEMPERATURE` → max, `VISIBILITY` → min, `SMOKE`/`HEAT`/`FIRE`/`ALARM` → any-true, `BEHAVIOR` → majority vote (deterministic tie-break). A kind with no dedicated rule (a genuinely future one) falls back to the single highest-confidence contributing observation's own value — proven directly in `tests/test_sensor_fusion.py::FutureProviderCompatibilityTests`.

## 6. Confidence model (Phase 6) and conflict resolution (Phase 7)

`sensor_fusion/confidence.py`, entirely deterministic arithmetic (no ML):
- **Source weighting** — `source_weights: Mapping[str, float]`, defaulting to 1.0.
- **Staleness decay** — exponential, halving every `staleness_half_life_seconds` (default 30s) of age; a same-instant or future-timestamped ("late") observation gets zero decay, never a fabricated boost above 1.0.
- **Agreement bonus** — applied only when ≥2 sources contributed *and* did not conflict.
- **Conflict penalty** — applied only when `sensor_fusion.conflict.detect_conflict()` finds genuine disagreement (boolean kinds: not all equal; numeric kinds: spread beyond a per-kind tolerance, e.g. 2 people for `OCCUPANCY`, 10°C for `TEMPERATURE`).
- **Missing data** — a `(location, kind)` pair with zero observations produces no `FusedObservation` at all (never fabricated).

Both of Phase 7's own worked examples are proven directly as tests:
- *"Camera says occupied, smoke detector silent, manual report occupied"* — camera+manual **agree** on `OCCUPANCY` (bonus applied); the smoke detector's silence is an independent, non-conflicting `SMOKE` observation (`test_camera_occupied_smoke_silent_manual_occupied_worked_example`).
- *"Heat detector alarm, camera unavailable"* — a **single** contributing source neither gets an agreement bonus (nothing corroborated it) nor a conflict penalty (nothing disagreed with it) — its own confidence passes through unchanged, "retain confidence honestly" (`test_5_heat_alarm_camera_unavailable_retains_confidence_honestly`).

## 7. Pipeline integration (Phase 8) — `BuildingState` kept unchanged

**`BuildingStateEstimator`/`BuildingState` were not modified at all this milestone** (Phase 8's own "keep BuildingState unchanged if possible," honored literally). Instead, `tests/test_sensor_fusion_building_state_integration.py` proves the composition: providers → `SensorFusionEngine.fuse()` → a small, explicit, **caller-owned** bridge (`_fused_to_hazard_snapshot()`/`_fused_to_occupancy_snapshot()`, living in the test file, never inside `sensor_fusion/` itself, which must depend on nothing but observation providers/geometry/time) → the exact same, **unmodified** `HazardSnapshot`/`OccupancySnapshot` parameters `BuildingStateEstimator.estimate()` already accepts. This is a pure re-shaping of already-fused values — the same discipline `designer.building_state_debug_runner.BuildingStateDebugRunner._reconstruct_hazard_snapshot()` already establishes for its own, differently-sourced inputs (Sec 2 finding 5) — proving fused, cross-kind-reconciled observations *can* replace "independently querying each provider and hand-assembling snapshots" without requiring `BuildingStateDebugRunner`, `BuildingStateEstimator`, or any other existing caller to change at all.

## 8. Files created / modified

**Created:**
- `sensor_fusion/{__init__,observation,provider,confidence,conflict,merge,engine}.py`
- `tests/test_sensor_fusion.py` — 19 unit tests (Phase 9)
- `tests/test_sensor_fusion_building_state_integration.py` — 2 pipeline-integration tests (Phase 8)
- `tests/test_sensor_fusion_architecture_guards.py` — 3 import-guard tests (Phase 12)
- `scripts/demo_sensor_fusion.py` — offline demo (Phase 10)
- `scripts/benchmark_sensor_fusion.py` — performance benchmark (Phase 11)
- `docs/architecture/sensor_fusion.md` — this document

**Modified:** none. `building_state/estimator.py`, `building_state/models.py`, `designer/building_state_debug_runner.py`, `hazard_evolution/*`, `multi_camera_fusion/*` are all untouched.

**Unchanged (verified, not modified):** every existing perception/hazard/occupancy/building_state/FACP type and estimator.

## 9. Performance

`scripts/benchmark_sensor_fusion.py`, zero real sensor/CCTV involvement:
- Provider collection (600 observations/call): ~0.003 ms/call.
- Merge: ~0.0003 ms/call.
- Conflict resolution: ~0.0005 ms/call.
- Confidence calculation: ~0.0006 ms/call.
- Full fusion (200 zones, 3 sources/zone): ~1.13 ms/call.

## 10. Future extensibility

A new sensor kind: add an `ObservationKind` member (no other code changes required — `merge.py`'s generic fallback and `conflict.py`'s generic numeric tolerance both honestly handle an unknown kind already). A new evidence source (BLE/WiFi/UWB positioning, firefighter reports): implement `ObservationProvider.collect()` — nothing else in `sensor_fusion/` needs to change, proven directly in `tests/test_sensor_fusion.py::FutureProviderCompatibilityTests`.
