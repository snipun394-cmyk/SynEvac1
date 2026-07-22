# Digital Twin Asset → Zone Assignment & Live FACP Runtime

Fixes the two concrete production gaps the Digital Twin engineering asset audit (`docs/architecture/digital_twin_engineering_asset_audit.md`, commit `012b848`) identified: (1) Designer-created Speaker/SmokeDetector/HeatDetector assets always had `zone_ids=()` because the Property Panel exposed no zone-assignment UI, and (2) production `LiveRuntime` never actually ticked a configured `SimulatedFACP`'s `evaluate()`.

## 1. Physical location vs. service zone — the semantics chosen

Two genuinely different engineering concepts share the same `zone_ids: Tuple[str, ...]` field (inherited from `EngineeringAsset`), and this milestone gives each asset type an honest, investigated answer rather than one uniform UI:

| Asset | What `zone_ids` means | Cardinality | UI |
|---|---|---|---|
| **SmokeDetector / HeatDetector** | **Physical location** — the one zone containing the device's position. A point sensor detects what's physically around it; it does not have a "coverage area" concept beyond that. | Single (by convention — the field is still a tuple, and downstream code honestly fans out over however many are present, but the Designer's own authoring UI only ever produces zero or one). | Single-select `QComboBox` ("Assigned Zone"), same convention `Camera`/`DynamicEvacuationSign` already established. |
| **Speaker** | **Service/broadcast coverage** — which zone(s) this speaker's output is considered to reach. A speaker mounted in one zone may legitimately be wired to serve others (a corridor speaker covering two adjoining rooms, for instance). | Genuinely multi-valued — confirmed by reading `speaker_manager.manager.SpeakerManager.active_speakers_in_zone()`/`voice_evacuation.controller.VoiceEvacuationController`, both already a plain tuple-membership test with no cardinality limit. | Multi-select checklist (`QListWidget` with checkable rows) — never a `QComboBox`, which would silently cap a Speaker at one served zone. |

This distinction was investigated, not assumed: `perception/live_perception/providers.py::_DetectorReadingObservationProvider` already documents "a detector assigned to MULTIPLE zones honestly contributes its one real reading to EACH assigned zone independently" — the *runtime* has always been willing to honor more than one zone for a detector too. The Designer's own authoring UI is deliberately narrower than what the model/runtime can technically represent, because a point detector genuinely only occupies one physical location; nothing here prevents a future caller (a script, a different tool) from setting more than one zone_id on a detector, and the runtime will keep handling it correctly if it ever happens.

## 2. Designer workflow

**SmokeDetector / HeatDetector:**
- On placement (`GraphicsScene`'s `smoke_detector`/`heat_detector` click-to-place branches), `GraphicsScene._find_unambiguous_zone_at(floor, x, y)` is checked: if the click point falls inside exactly one zone, that zone is auto-assigned (`model.zone_ids = (zone.id,)`). If it falls inside zero or more-than-one zone (ambiguous, overlapping zones), `zone_ids` stays `()` — never a guessed first/nearest match.
- The Property Panel's "Assigned Zone" combo (`smoke_detector_zone`/`heat_detector_zone`) shows the current assignment and allows full manual reassignment at any time, exactly mirroring Camera's own `camera_zone`/`update_camera_zone()`.
- A modest inline warning label ("Zone assignment required for live operation.") appears whenever the selected detector's `zone_ids` is empty.

**Speaker:**
- Never auto-assigned from position (explicit requirement — physical mounting location and acoustic/broadcast coverage are not the same fact).
- The Property Panel's "Covered Zone(s)" checklist (`speaker_zones`, a `QListWidget`) lists every zone on the speaker's floor with a checkbox; any combination — zero, one, or several — can be checked. `update_speaker_zones()` reads back every checked row into `model.zone_ids` on every change.
- The same inline warning label appears whenever no zone is checked.

**Validation:** `designer/validation.py::validate_building_authoring()` (the pre-existing "Validate Project" mechanism, reused rather than duplicated) now also reports `speaker_missing_zone`/`smoke_detector_missing_zone`/`heat_detector_missing_zone` as `WARNING`-severity issues (deliberately softer than Door/Exit/Stair's own `ERROR` — an unassigned device still exists and still functions, only its zone-scoped behavior is degraded, unlike an unconnected Door/Exit/Stair which produces no Navigation Graph edge at all).

## 3. Save / reload

Zone assignment is ordinary model state (`EngineeringAsset.zone_ids`), already fully covered by each asset's existing `to_dict()`/`from_dict()`. No serialization changes were needed. A `.syn` project predating this milestone (or predating Speaker/SmokeDetector/HeatDetector's own existence entirely) loads unchanged — `Floor.from_dict()` already defaults every asset list to `[]` via `data.get(...)`, and each asset's own `from_dict()` already defaults `zone_ids` to `()` when the key is absent.

## 4. Runtime graph

```
Designer (real toolbar + Property Panel)
        |
        v
Floor.speakers / Floor.smoke_detectors / Floor.heat_detectors  (zone_ids populated)
        |
        v
Project.to_dict() -> .syn -> Project.from_dict()   (round-trips zone_ids exactly)
        |
        v
SpeakerManager.discover_speakers(building)      SensorManager.discover_sensors(building)
        |                                               |
        v                                               v
speakers_in_zone(zone_id)                     sensors_in_zone(zone_id) / all_statuses()
        |                                               |
        v                                               v
VoiceEvacuationController.broadcast(          live_system.facp_gateway.EngineFACPGateway
  VoiceMessage(target_zone_ids=...))            .evaluate(time)
        |                                               |
        v                                               v
SimulationVoiceOutputProvider                  facp.evaluate(detector_conditions, time)
  (only the correctly zone-assigned                     |
   speaker(s) receive the instruction)                  v
                                                facp.current_snapshot(time)
                                                        |
                                                        v
                                        EstimatorBuildingStateGateway (READ-ONLY;
                                        never calls evaluate/acknowledge/silence/reset)
                                                        |
                                                        v
                                                BuildingState.facp_status
```

Proven end-to-end (not from hand-built fixtures) in `tests/test_zone_assignment_full_e2e.py`: real Designer authoring → save/reload → `SensorManager`/`SpeakerManager` discovery on the *reloaded* building → detector-condition-driven FACP evaluation → zone-targeted voice broadcast that correctly reaches only the assigned speaker(s).

## 5. FACP production lifecycle

**Before this milestone:** `live_runtime.factory.build_live_runtime()` wired `facp.current_snapshot()` as a strictly read-only provider into `EstimatorBuildingStateGateway`. Nothing in production ever called `facp.evaluate()` — only `designer/building_state_debug_runner.py`'s own debug loop did, for the Designer's Building State Debug Panel.

**After this milestone:** a new `live_system/facp_gateway.py::EngineFACPGateway` mirrors `BuildingStateDebugRunner`'s own detector-condition-report construction exactly (same `SensorManager.all_statuses()` split by `sensor_type`, same `DetectorConditionReport.from_status_and_reading()` pairing with the optional smoke/heat reading providers) and is wired into `LiveOrchestrator.run_cycle()`, called **once per cycle, before `building_state_gateway`** (so `BuildingState.facp_status` reflects this cycle's evaluation, not last cycle's). `EngineFACPGateway.evaluate()` calls **only** `facp.evaluate()` — never `acknowledge()`/`silence()`/`reset()`, which remain explicit operator actions (Designer's debug controls today; a future Command Center wiring later).

**Backward compatibility preserved deliberately:** `facp` remains `Optional[object] = None`, never auto-constructed by the factory (unlike `CameraManager`/`SensorManager`/`SpeakerManager`/`SignManager`, which are always default-constructed). A caller that never supplies a `facp` still gets the pre-existing, fully honest `FACPUnavailableTests` degraded state (`BuildingState.facp_status is None`) — `tests/test_live_runtime_failure_modes.py::FACPUnavailableTests` passes unchanged. Only when a caller *does* supply a `SimulatedFACP` does it now actually get ticked every cycle, using that exact same shared instance (never a second FACP anywhere).

## 6. Legacy detector behavior (unchanged)

`models/detector_migration.py::adapt_legacy_detector()` is untouched. A legacy `Detector` (Smoke/Heat-typed) is still adapted at `SensorManager.discover_sensors()` time, with `zone_ids` still derived geometrically via `Zone.contains()` — legacy `Detector` has no `zone_ids` field of its own to assign through a Property Panel combo, so no UI was added for it (there is nothing to add a UI for). Flame/Gas-typed legacy detectors remain outside `SensorManager`'s reach, documented but unchanged, exactly as the audit found.

## 7. Known remaining limitations

- Moving an already-placed SmokeDetector/HeatDetector to a different physical zone does **not** automatically re-derive its zone assignment — auto-assignment happens only at placement time (this milestone's own explicit scope). A user must manually reassign via the Property Panel after a cross-zone move.
- `SimulatedFACP`'s own `acknowledge()`/`silence()`/`reset()` are still reachable only through Designer's debug controls in this milestone; a full Command Center operator-facing FACP panel was out of scope.
- Camera's own zone-assignment field remains present but functionally cosmetic (its real coverage geometry is FOV-based, not `zone_ids`-based) — unchanged by this milestone, per its own explicit "do not modify Camera zone semantics unnecessarily" instruction.
