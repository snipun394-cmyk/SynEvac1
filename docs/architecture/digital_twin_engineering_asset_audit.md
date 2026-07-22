# SynEvac Building Designer — Digital Twin Engineering-Systems Audit

Audit-only milestone. No production code was changed. This document reports, as of commit `e684415`, exactly which physical/logical building systems exist, whether they can be placed/configured/saved/reloaded through the Designer, and whether the live runtime actually consumes what the Designer produces.

Method: static reading of `models/`, `designer/`, `navigation/`, `visibility/`, `camera_manager/`, `sensor_manager/`, `speaker_manager/`, `sign_manager/`, `facp/`, `voice_evacuation/`, `dynamic_signage/`, `building_control/`, `live_runtime/`, `live_system/`, `perception/`; **plus** an actual offscreen (`QT_QPA_PLATFORM=offscreen`) run of `designer.windows.main_window.MainWindow`, driven through the real toolbar-trigger → `GraphicsScene.mousePressEvent` code path (not direct model construction) to place one of every currently-placeable asset, followed by a real `Project.to_dict()`/`Project.from_dict()` save/reload round trip.

---

## 1. Complete Digital Twin asset inventory

| Model | File | Base | Placeable via Designer |
|---|---|---|---|
| `Zone` | `models/zone.py` | `BaseObject` | Yes |
| `Door` | `models/door.py` | `BaseObject` | Yes |
| `Exit` | `models/exit.py` | `BaseObject` | Yes |
| `Staircase` | `models/staircase.py` | `BaseObject` | Yes (needs ≥2 floors) |
| `Elevator` | `models/elevator.py` | `BaseObject` | **No** — model exists, zero authoring tool |
| `Obstacle` | `models/obstacle.py` | `BaseObject` | Yes |
| `AssemblyPoint` | `models/assembly_point.py` | `BaseObject` | Yes |
| `Camera` | `models/camera.py` | `EngineeringAsset` | Yes |
| `Detector` (legacy generic) | `models/detector.py` | `BaseObject` | Yes |
| `SmokeDetector` | `models/smoke_detector.py` | `SensorAsset`→`EngineeringAsset` | Yes |
| `HeatDetector` | `models/heat_detector.py` | `SensorAsset`→`EngineeringAsset` | Yes |
| `Speaker` | `models/speaker.py` | `SensorAsset`→`EngineeringAsset` | Yes |
| `DynamicEvacuationSign` | `models/dynamic_sign.py` | `EngineeringAsset` | Yes |
| FACP (`SimulatedFACP`) | `facp/engine.py` | plain class, no `BaseObject` | **No** — logical only, see §9 |
| Stair Pressurization / Smoke Exhaust / Deluge | `building_control/types.py` (`ControlSystemType`) | enum members, no asset class | **No** — abstract state only, see §10 |
| Manual Call Point | — | **does not exist** | N/A |

`Floor` (`models/floor.py:50-86`) carries every placeable asset above as its own `list[...]` field, each with matching `add_*`/`remove_*`/`*_count`/`to_dict()`/`from_dict()` entries — confirmed for all 12 lists (`zones`, `exits`, `stairs`, `elevators`, `cameras`, `detectors`, `smoke_detectors`, `heat_detectors`, `speakers`, `signs`, `assembly_points`, `obstacles`, `doors`).

## 2. Exact Designer toolbar inventory (`designer/widgets/toolbar.py`)

File · New, Open, Save · Undo *(disabled — no command stack exists)*, Redo *(disabled)* · Select · Zone, Exit, Door, Stair, Elevator *(disabled — no authoring tool)*, Obstacle · Camera, Detector, Smoke Detector, Heat Detector, Speaker, Dynamic Sign · Assembly Point · Occupant, Simulation · Zoom+, Zoom−, Reset · Coverage (checkable, camera-visibility overlay toggle).

Confirmed against `designer/scene/graphics_scene.py`'s own `mousePressEvent` dispatcher (`if self.current_tool == "..."`) — **14 functional click-to-place branches** exist: `select`, `zone`, `exit`, `stair`, `camera`, `detector`, `smoke_detector`, `heat_detector`, `speaker`, `sign`, `assembly_point`, `obstacle`, `door`, `occupant`. There is no `"elevator"` branch anywhere — the toolbar button being disabled is not cosmetic, the interaction path genuinely does not exist.

## 3. Designer support matrix

| System | Domain model | Physical/Logical/Geometry | On Floor | Serialized | Graphics item | Toolbar tool | Click-to-place | Selectable/Movable | Property Panel | Floor assoc. | Zone assoc. | Save/reload | Runtime consumer | Legacy/dup. | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zone | `Zone` | Geometry | ✅ | ✅ | `zone_rectangle.py` | ✅ | ✅ | ✅ | ✅ full | ✅ | n/a (is the zone) | ✅ | Navigation Graph, everything | — | **Fully represented** |
| Door | `Door` | Geometry | ✅ | ✅ | `door_item.py` | ✅ | ✅ | ✅ | ✅ full incl. Zone A/B | ✅ | ✅ (2 zones) | ✅ | Navigation Graph | — | **Fully represented** |
| Exit | `Exit` | Geometry | ✅ | ✅ | `exit_item.py` | ✅ | ✅ | ✅ | ✅ full incl. zone | ✅ | ✅ | ✅ | Navigation Graph, Recommendation/Guidance | — | **Fully represented** |
| Staircase | `Staircase` | Geometry | ✅ | ✅ | `stair_item.py` | ✅ | ✅ (needs 2nd floor + dialog) | ✅ | ✅ full incl. from/to zone | ✅ (spans 2) | ✅ | ✅ | Navigation Graph | — | **Fully represented** |
| Elevator | `Elevator` | Geometry (intended) | ✅ (empty list) | ✅ (empty) | **none** | present, **disabled** | **no** | n/a | n/a | n/a | n/a | ✅ (trivially, always empty) | **none** | — | **Not yet modeled (authoring)** |
| Obstacle | `Obstacle` | Geometry | ✅ | ✅ | `obstacle_item.py` | ✅ | ✅ | ✅ | ✅ full | ✅ | n/a | ✅ | Navigation/pathfinding cost | — | **Fully represented** |
| AssemblyPoint | `AssemblyPoint` | Geometry/Safety | ✅ | ✅ | `assembly_point_item.py` | ✅ | ✅ | ✅ | ✅ full | ✅ | n/a | ✅ | Occupant lifecycle (`is_near_exit`-style), Navigation Node | — | **Fully represented** |
| Camera | `Camera(EngineeringAsset)` | Physical asset | ✅ | ✅ | `camera_item.py` | ✅ | ✅ | ✅ | ✅ full (incl. FOV/range/mount/mode/RTSP/IP/user/pass) | ✅ | ✅ **UI exists** | ✅ | `CameraManager` → `LiveCameraPipeline`; coverage geometry drives `perception.providers.ground_truth_camera_provider` — **zone_ids itself is cosmetic/administrative only** (§7) | — | **Fully represented**, one cosmetic-field caveat |
| Detector (legacy) | `Detector` | Physical asset | ✅ | ✅ | `detector_item.py` | ✅ | ✅ | ✅ | ✅ (position, radius, mount, **type incl. Flame/Gas**, active) | ✅ | n/a — **model has no `zone_ids` field at all** | ✅ | `SensorManager` (Smoke/Heat only, via `adapt_legacy_detector`) — **Flame/Gas silently dropped** | **Legacy**, coexists with SmokeDetector/HeatDetector | **Legacy only** / partial dead-end (Flame, Gas) |
| SmokeDetector | `SmokeDetector(SensorAsset)` | Physical asset | ✅ | ✅ | `smoke_detector_item.py` | ✅ | ✅ | ✅ | ✅ (position, active, health, mode, threshold, install date, test level, state) — **no zone field** | ✅ | model field exists, **no Property Panel UI** | ✅ | `SensorManager` → BuildingState / FACP inputs | — | **Partially represented** (zone assignment gap) |
| HeatDetector | `HeatDetector(SensorAsset)` | Physical asset | ✅ | ✅ | `heat_detector_item.py` | ✅ | ✅ | ✅ | ✅ (same shape as Smoke) — **no zone field** | ✅ | model field exists, **no Property Panel UI** | ✅ | `SensorManager` → BuildingState / FACP inputs | — | **Partially represented** (zone assignment gap) |
| Speaker | `Speaker(SensorAsset)` | Physical asset | ✅ | ✅ | `speaker_item.py` | ✅ | ✅ | ✅ | ✅ (position, active, health, mode, type, volume, install date) — **no zone field** | ✅ | model field exists, **no Property Panel UI**, and this field is **load-bearing** (§7) | ✅ | `SpeakerManager` → `VoiceEvacuationController.active_speakers_in_zone()` | — | **Partially represented** — most consequential gap found |
| DynamicEvacuationSign | `DynamicEvacuationSign(EngineeringAsset)` | Physical asset | ✅ | ✅ | `sign_item.py` | ✅ | ✅ | ✅ | ✅ full incl. orientation + zone **UI exists** | ✅ | ✅ **UI exists** | ✅ | `SignManager` → `dynamic_signage.planner` | — | **Fully represented** |
| FACP | `SimulatedFACP` (not `BaseObject`) | Logical | n/a | n/a | none | none | n/a | n/a | Designer *debug panel* only (`building_state_debug_panel.py`) | n/a | n/a | n/a | Consumed by Designer debug runner; only read (`current_snapshot()`) in `live_runtime/factory.py`, never ticked | — | **Logical by design** |
| Stair Pressurization / Smoke Exhaust / Deluge | `ControlSystemType` enum members | State-only | n/a | n/a | none | none | n/a | n/a | none | n/a (target is an existing Stair/Zone id) | n/a | n/a | `BuildingControlController`/`SimulationControlProvider` — records confirmed status, **zero physical effect** | — | **Logical/state-only by design**, not yet honestly modelable as equipment |
| Manual Call Point | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **Not yet modeled** |

## 4. Designer live-run results (real toolbar + click path, `QT_QPA_PLATFORM=offscreen`)

Placed via `action.trigger()` then `GraphicsScene.mousePressEvent()` (the exact production code path, not direct model construction), on the default new-project single floor:

```
zone: 1        camera: 1          detector: 1
smoke_detector: 1                 heat_detector: 1
speaker: 1     sign: 1            assembly_point: 1
obstacle: 1    exit: 1            door: 1
stair: 0 (expected — requires ≥2 floors; the default project has one)
elevator_action.isEnabled(): False (confirmed disabled)
```

Every placeable type genuinely appears on `Floor` through the real UI path — Phase 4's own "do not merely instantiate model classes" requirement is satisfied. Stair returned 0 only because `GraphicsScene`'s own stair branch (`designer/scene/graphics_scene.py:626-644`) requires clicking inside an existing `Zone` **and** at least one other unlocked floor to exist as a destination candidate — correct, expected gating, not a defect (the pre-existing test suite already exercises the 2-floor case).

## 5. Property Panel audit

**Fully exposed, matching the model:**
- **Zone** — origin, length/width, type, all four corners, area.
- **Door** — start/end, length, width, type, normally-open, locked, active, **Zone A / Zone B**.
- **Exit** — start/end, length, width, capacity, blocked, **zone**.
- **Stair** — from-floor, from position, **from-zone**, to-floor combo, to position, **to-zone**, width, vertical height, travel distance.
- **AssemblyPoint** — position, length/width, capacity, description, active.
- **Obstacle** — length/width, type, traversability, traversal cost, active.
- **Camera** — position, rotation, FOV, range, mount height, active, **zone**, resolution, fps, mode, RTSP address, IP, username, password.
- **DynamicEvacuationSign** — position, orientation, active, **zone** (added this milestone).

**Missing UI exposure (confirmed by direct read of `designer/widgets/property_panel.py`'s own `*_fields` lists):**
- **SmokeDetector / HeatDetector** — expose position, active, health status, mode, activation threshold, installation date, test-reading control, current state. **No zone-assignment field of any kind.** The model's own `zone_ids` (inherited from `EngineeringAsset`) can never be set through the Designer.
- **Speaker** — expose position, active, health status, mode, speaker type, volume, installation date. **Same gap: no zone-assignment field.**
- **Legacy Detector** — position, coverage radius, mount height, `detector_type` (Smoke/Heat/**Flame/Gas**), active. (No zone field is *correct* here — the model itself has no `zone_ids`; see §10.)

Only **Camera** and **DynamicEvacuationSign** have an "Assigned Zone" / "Covered Zone" combo box (`self.camera_zone`, `self.sign_zone`) wired to `model.zone_ids = (zone_id,) if zone_id else ()`. Nothing else does.

## 6. Save/reload test

Built one of every currently-placeable physical asset (per §4) on a fresh project, called `Project.to_dict()` then `Project.from_dict()`:

```
zone_count: before=1 after_reload=1 OK
camera_count: before=1 after_reload=1 OK
detector_count: before=1 after_reload=1 OK
smoke_detector_count: before=1 after_reload=1 OK
heat_detector_count: before=1 after_reload=1 OK
speaker_count: before=1 after_reload=1 OK
sign_count: before=1 after_reload=1 OK
assembly_point_count: before=1 after_reload=1 OK
obstacle_count: before=1 after_reload=1 OK
exit_count: before=1 after_reload=1 OK
door_count: before=1 after_reload=1 OK
stair_count: before=0 after_reload=0 OK
```

IDs, names, floor association, geometry, and asset-specific properties all survive (every `to_dict()`/`from_dict()` pair round-trips `id` verbatim and defaults every optional key — confirmed by code reading in the previous milestone's own Floor/asset serialization work, re-confirmed here structurally). **Legacy project loading remains functional**: `Floor.from_dict()` calls `data.get("signs", [])`/`data.get("speakers", [])`/etc. — a `.syn` file predating any of these asset types simply yields empty lists, never an error (already covered by `tests/test_dynamic_sign_model.py::test_3_legacy_project_without_signs_loads` and equivalent prior-milestone tests for Speaker/SmokeDetector/HeatDetector).

Zone association after a **fresh** placement (no manual Property Panel zone edit) — confirms §5's gap end-to-end:

```
camera zone_ids=()          detector <no zone_ids field>
smoke_detector zone_ids=()  heat_detector zone_ids=()
speaker zone_ids=()         sign zone_ids=()
```

Every device defaults to no zone assignment on placement; only Camera/Sign can subsequently be given one through the GUI.

## 7. Runtime connection audit

| Asset | Manager | Discovery wiring (`live_runtime/factory.py`) | Actually consumed by |
|---|---|---|---|
| Camera | `CameraManager` | `camera_manager.discover_cameras(building)` (unconditional) | `LiveCameraPipeline`/detection routing (Live mode, only if `frame_sources`+`human_detector`+`identity_resolver` supplied); `perception.providers.ground_truth_camera_provider` reads `Camera.coverage_polygon()` geometry directly — **`zone_ids` itself is never read by any of this**, it is purely administrative/display metadata (`command_center/live_status_panel.py` shows it in a table; `CameraManager.cameras_in_zone()` has no other caller anywhere in the repo) |
| Legacy Detector | `SensorManager` | `sensor_manager.discover_sensors(building)` → `adapt_legacy_detector()` for Smoke/Heat only | Smoke/Heat-typed ones reach BuildingState/FACP inputs with a geometrically-derived zone (`Zone.contains()`); **Flame/Gas-typed ones are silently dropped** (`adapt_legacy_detector` returns `None`) |
| SmokeDetector / HeatDetector | `SensorManager` | `sensor_manager.discover_sensors(building)` registers them **as-is**, trusting `sensor.zone_ids` verbatim — **no geometric fallback** (unlike the legacy-Detector path one line above it) | `sensor_manager.sensors_in_zone()` (used by `building_state/consistency.py`'s FACP-vs-zone cross-checks) will find **zero** Designer-placed Smoke/Heat detectors for any zone, because `zone_ids` is always `()` (§5/§6) |
| Speaker | `SpeakerManager` | `speaker_manager.discover_speakers(building)` — same as-is trust, no geometric fallback | `VoiceEvacuationController.active_speakers_in_zone()` — **every zone's voice broadcast reports `NO_SPEAKERS_AVAILABLE`** for a purely-Designer-authored building, because no speaker ever has a non-empty `zone_ids` |
| DynamicEvacuationSign | `SignManager` | `sign_manager.discover_signs(building)` | `DynamicSignagePlanner` — works correctly, since Sign has a Property Panel zone-assignment UI |
| Door/Exit/Stair | — (no manager; read directly) | `NavigationGraphGenerator().build(building)` | Pathfinding/Guidance/Recommendation — fully connected, these were never gated behind an asset-manager zone-assignment step to begin with (connectivity is via `zone_a_id`/`zone_b_id`/`from_zone_id`/`to_zone_id`, all of which **do** have Property Panel UI, §5) |
| FACP | none (caller-owned) | `facp` param is **caller-supplied only**, never default-constructed (contrast with the four managers above, all default-constructed if omitted); factory only builds a **read-only** `facp_snapshot_provider` (`lambda time: facp.current_snapshot(time)`) | Production `build_live_runtime()` **never calls `facp.evaluate()`** — only `designer/building_state_debug_runner.py`'s own debug loop ticks the FACP state machine. In Live mode today, unless some other caller independently advances FACP, its alarm state machine never moves. |
| Stair Pressurization/Smoke Exhaust/Deluge | `BuildingControlController` | `building_control_provider` param, caller-supplied | `SimulationControlProvider._execute_state_only()` records a confirmed status keyed by `(system_type, target_id)`; **no hazard/physical state is ever mutated** — by design (§10), not a wiring gap |

**Net conclusion for Phase 7's two named failure modes:**
- *Asset drawable but effectively ignored by runtime:* **Speaker** (zone-based voice routing is dead for any GUI-authored building) and **SmokeDetector/HeatDetector** (zone-based FACP consistency checks are dead the same way) — both for the identical reason (§5's missing Property Panel field), and **legacy Detector of type Flame/Gas** (dropped entirely by `adapt_legacy_detector`).
- *Backend/runtime model exists but cannot be created through Designer:* **none found** for physical assets — every `SensorAsset`/`EngineeringAsset` subclass that exists has a matching Designer tool. The only backend concepts with no Designer representation are the three **logical** systems (FACP panel, Stair Pressurization/Smoke Exhaust/Deluge), which is the correct outcome per §9/§10, not a gap.

## 8. FACP decision

**FACP remains a building-wide logical coordinator, not a Designer-placed asset, and this audit recommends it stay that way.**

- `SimulatedFACP` (`facp/engine.py`) has exactly one identity field, `panel_id: str = "FACP-1"` — no position, no `floor_id`, no geometry anywhere in `facp/models.py`. It does not subclass `BaseObject`/`EngineeringAsset`.
- It is explicitly documented and implemented as *"an aggregation/coordination layer over SensorManager-managed assets"* — it never discovers, enables, or looks up a Camera/Detector/Speaker itself; it only consumes already-built `DetectorConditionReport`s.
- Designer's only reference is a debug concept: `designer/building_state_debug_runner.py` owns one `SimulatedFACP` instance and exposes `acknowledge_facp()`/`silence_facp()`/`reset_facp()`; `designer/widgets/building_state_debug_panel.py` renders a "FACP tab" against those wrappers. Nothing in `toolbar.py`, `designer/items/`, or `main_window.py`'s placement logic references it.
- `live_runtime/factory.py` treats it as strictly read-only passthrough (documented explicitly at the call site) and never advances its state machine in production.

**Does physical panel location matter anywhere?** No — nothing in the codebase reads a FACP position, and no report/consistency check is keyed by "which physical panel is this alarm on." A real building may have one FACP (sometimes with annunciator repeaters), but SynEvac models exactly one coordinator per building already, with no multi-panel concept to justify placement.

**Would a placeable asset add engineering value right now?** No. Recommended outcome: **KEEP FACP LOGICAL.** (If a future milestone adds true multi-panel buildings or physical-panel siting for evacuation drills, that would be the trigger to revisit — not before.)

## 9. Smoke Exhaust / Stair Pressurization / Deluge representation

All three are **abstract confirmed-control status only** — no equipment model of any kind.

- `building_control/types.py`'s own `ControlSystemType` comment states this outright: they *"have no backing physics anywhere in this codebase... their control state is tracked honestly as state-only, never claimed to have a physical hazard effect."*
- `ControlRequest`/`ControlInstruction`/`ControlStateEntry` key everything by a bare `target_id` string, validated only against an *existing* Stair id (`STAIR_PRESSURIZATION`) or Zone id (`SMOKE_EXHAUST`, `DELUGE`) via `_categorize_target()` — there is no fan, damper, pump, valve, or sprinkler-head object; the "target" is just an existing Stair/Zone being annotated with a state.
- `SimulationControlProvider._execute_state_only()` writes `self._state_only[(system_type, target_id)] = action` and returns a confirmed result whose own message text is deliberately narrow, explicitly never claiming a physical/hazard effect.
- Zero Designer representation: no toolbar tool, no graphics item, no Property Panel field anywhere references `STAIR_PRESSURIZATION`/`SMOKE_EXHAUST`/`DELUGE`.

**What would need to exist before these could honestly become Digital Twin assets:** a real equipment model per system (e.g. a `SmokeExhaustFan`/`PressurizationFan` asset with `serves_zone_id`/`serves_stair_id`, capacity/CFM, on/off state; a `DelugeValve`/sprinkler-head asset with coverage zone and flow rate) **and**, more importantly, a physical simulation layer that actually changes hazard/smoke/visibility state when activated (today's `hazard`/`smoke_propagation`/`fire_growth` packages were not investigated in this audit for exhaust-interaction hooks — that is the real prerequisite, not the equipment model alone). Fabricating the equipment model without the physical effect would be worse than the current honest state-only representation, per this milestone's own explicit instruction not to do so.

## 10. Legacy Detector status

- `Floor` still carries all three independently: `detectors: list[Detector]` (legacy), `smoke_detectors: list[SmokeDetector]`, `heat_detectors: list[HeatDetector]` — confirmed current, not historical-only.
- The Designer still exposes the legacy generic "Detector" tool **fully wired and independently active** alongside the two newer tools (`toolbar.py:76,81-82`; all three connected to `change_tool(...)` in `main_window.py`) — unlike Elevator, this is not a disabled leftover.
- **A new project can genuinely create both representations for the same physical device today**: nothing prevents a user from placing a "Detector" (type=Smoke) and a "Smoke Detector" at the same spot; `adapt_legacy_detector()` only unifies identity for a *single* legacy object at `SensorManager` discovery time, it does not deduplicate or merge two independently-placed assets — they would register as two entirely distinct sensors with two distinct ids.
- The legacy tool's own `detector_type` combo still includes **Flame** and **Gas**, neither of which has any canonical `SensorAsset` counterpart — `adapt_legacy_detector()` returns `None` for both, so they are placeable, save/reload cleanly, but are **completely invisible to `SensorManager`/Perception/FACP** (confirmed by reading `models/detector_migration.py` directly).
- **Is the canonical model clean for new projects?** Not automatically — a new project *can* be built using only `SmokeDetector`/`HeatDetector` and never touch the legacy tool, which would be clean. But the legacy tool remains fully available and produces working (Smoke/Heat) or silently-inert (Flame/Gas) assets with no in-UI warning either way, so "clean" depends entirely on user discipline, not on any structural guard.

## 11. Digital Twin completeness classification

**FULLY REPRESENTED** — Zone, Door, Exit, Staircase, Obstacle, AssemblyPoint, DynamicEvacuationSign, Camera (with the zone-field caveat that the field itself turned out to be cosmetic, not a functional gap).

**PARTIALLY REPRESENTED** — SmokeDetector, HeatDetector, Speaker (model + tool + save/reload all work; a load-bearing property — zone assignment — has no Property Panel exposure, silently defeating downstream FACP-consistency/voice-routing logic for every Designer-authored building).

**LOGICAL BY DESIGN** — FACP (§8), Stair Pressurization / Smoke Exhaust / Deluge (§9).

**NOT YET MODELED** — Elevator authoring tool (model + Floor list exist, zero placement path), Manual Call Point (no model at all), physical exhaust/pressurization/deluge equipment (fan/damper/pump/valve/sprinkler), any hazard-simulation hook for those three control systems.

**LEGACY ONLY** — generic `Detector` for Flame/Gas specifically (Smoke/Heat-typed legacy Detectors are migrated live and remain genuinely functional, just superseded).

## 12. Missing physical fire-safety assets (observed, not to be built this milestone)

Manual Call Point, sprinklers, hydrants, hose reels, fire extinguishers, fire pumps, water tanks, smoke dampers, fire dampers, AHUs, smoke exhaust fans, pressurization fans, emergency lights — none exist anywhere in the codebase today. (Per Phase 12, none of these are being added now.)

## 13. Recommended Digital Twin additions, ranked by engineering usefulness

1. **Add a "Covered Zone" Property Panel field for Speaker, SmokeDetector, and HeatDetector**, mirroring Camera/Sign's existing `_populate_zone_combo()`/`update_*_zone()` pattern exactly. This is not a new asset — it is closing the single highest-impact gap this audit found: today, no Designer-authored building can produce a working zone-scoped voice broadcast or a FACP zone-consistency check at all. (Explicitly not implemented this milestone per the audit's own scope.)
2. **Warn or guard against legacy Detector / SmokeDetector-HeatDetector duplication** — even a passive Property Panel or Designer-status hint would reduce the confirmed redundancy risk in §10, short of removing the legacy tool (which must stay per backward-compatibility instructions).
3. **Elevator authoring tool** — the model and Floor list already exist; this is the smallest gap between "model exists" and "genuinely usable," on the same tier as Door/Stair authoring already solved.
4. Everything in §12/§9 (Manual Call Point, physical exhaust/pressurization/deluge equipment) — appropriately last, since each requires new engineering models and, for the control systems, a physical-effect simulation layer that does not exist yet.

## 14. Things that should deliberately NOT become floor-plan assets

- **FACP** — no spatial meaning to model; see §8.
- **Stair Pressurization / Smoke Exhaust / Deluge**, *as currently understood* — until a real equipment model AND a physical hazard-interaction layer exist, adding graphics-item placeholders for these would misrepresent simulation fidelity that does not exist (exactly the failure mode Phase 12 warns against).
- Any "confirmed control status" abstraction in general — these are building-control/Command-Center concepts, not Digital Twin geometry, and should stay off the floor plan even once (if ever) real equipment models are added; the equipment itself (a fan, a valve) would be the floor-plan asset, not the control-state abstraction.

## 15. Full test-suite result

`python -m unittest discover -s tests` → **3630 tests, OK, 257.5s.** No production code was modified during this milestone. (The previous milestone's run had shown 1 failure in `tests/test_zone_usability.py`'s clipboard test; it passed in isolation then and passes in this full run now, confirming it was environmental flakiness, not a real regression.)

## 16. Git status

Clean relative to this milestone: no production files were modified. `git status --short` shows only this new documentation file (once added) plus pre-existing, unrelated, already-uncommitted Human Detection milestone work (`human_detection/`, `docs/architecture/human_detection.md`, `scripts/benchmark_yolo_human_detector.py`, `scripts/demo_yolo_human_detection.py`, `tests/test_yolo_human_detector.py`, `tests/test_yolo_rtsp_live_runtime_compatibility.py`, `tests/human_detection_fixtures.py`) and a pre-existing modified `requirements.txt` — none of which this audit touched or commits.

---

## Explicit answers

**A. Can I currently build a SynEvac Digital Twin containing cameras, smoke detectors, heat detectors, speakers and dynamic evacuation signs entirely through the GUI?**
Yes — all five are placeable, selectable, movable, and editable through the real Designer UI (confirmed by an actual offscreen toolbar-and-click run, not just class instantiation).

**B. Are all of those assets saved in the .syn project?**
Yes — confirmed by an actual `to_dict()`/`from_dict()` round trip; every count, id, and property survived.

**C. Does the live runtime discover the same assets created in the Designer?**
Partially. `CameraManager`/`SensorManager`/`SpeakerManager`/`SignManager` all discover every one of these assets from `Floor` without exception. But **discovery ≠ usable**: Smoke/Heat Detectors and Speakers are discovered with an always-empty `zone_ids` (no Property Panel UI ever sets it), so zone-scoped consumers (`sensors_in_zone()`, `active_speakers_in_zone()`) never find them for a purely GUI-authored building. Cameras and Signs are fully functional because their zone field is either cosmetic (Camera) or has a working UI (Sign).

**D. Is FACP currently a physical Digital Twin asset or a logical system?**
Logical system only — no position, no floor association, no placeable graphics item; a debug-only concept in the Designer, and read-only passthrough in the production live runtime (which never itself advances FACP's own state machine).

**E. Should FACP become placeable now, based on actual architecture?**
No. Nothing in the architecture reads or would benefit from a panel position; recommend KEEP FACP LOGICAL.

**F. Are smoke exhaust, stair pressurization and deluge physically modeled or only logical/state-only?**
Only logical/state-only — `SimulationControlProvider` records a confirmed status against an existing Stair/Zone id; no equipment model exists and no physical/hazard effect is ever produced.

**G. Are there backend systems we built that the user currently cannot represent in the building?**
No physical asset case was found — every `SensorAsset`/`EngineeringAsset` subclass has a matching Designer tool. The only unrepresentable backend concepts are the deliberately-logical ones (FACP, the three state-only control systems), which is correct, not a gap. The Elevator model is the one physical asset that exists but has no placement path at all.

**H. What are the most important missing physical Digital Twin tools?**
In order: (1) not a new tool at all, but closing the Speaker/SmokeDetector/HeatDetector zone-assignment Property-Panel gap — the single highest-impact, lowest-cost fix found; (2) Elevator authoring; (3) a Manual Call Point asset (named directly in the milestone brief as a real gap, though explicitly deferred); (4) real equipment models for exhaust/pressurization/deluge, gated on a physical hazard-interaction layer that does not exist yet.
