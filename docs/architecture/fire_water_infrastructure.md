# Fire Water Supply & Suppression Infrastructure

Adds the infrastructure that supplies fire-water systems — **FireWaterTank**, **FirePump**, **JockeyPump**, **FireServiceInlet** (breeching inlet) — plus **FireWaterSystem**, a lightweight logical model connecting that infrastructure to the suppression assets it serves (Sprinkler/FireHydrant/HoseReel, from `docs/architecture/fire_suppression_safety_assets.md`). Builds on that milestone and on the zone-assignment/live-runtime foundation without modifying either.

**The single most important fact about this milestone:** **INFRASTRUCTURE AVAILABILITY ≠ HYDRAULIC PERFORMANCE.** This milestone implements physical infrastructure (A) and operational state (B) — never hydraulic performance (C). `FirePump RUNNING` never means "Hydrant HYD-1 definitely receives 7 bar." `FireWaterTank AVAILABLE` never proves adequate duration or flow. No pipe network, pump curve, pressure/flow calculation, K-factor, Hazen-Williams/Darcy-Weisbach, NPSH, or water-hammer simulation exists anywhere in this milestone's code.

## 1. Asset taxonomy

| Asset | Base class | Concept |
|---|---|---|
| **FireWaterTank** | `EngineeringAsset` + `health_status` | Static reservoir; capacity/level in explicit **liters**, `TankOperationalState` (`AVAILABLE`/`LOW_LEVEL`/`EMPTY`/`FAULT`/`UNAVAILABLE`) |
| **FirePump** | `PumpAsset` (shared base) | Primary fire pump; `PumpOperationalState` (`STOPPED`/`RUNNING`/`FAULT`/`UNAVAILABLE`) |
| **JockeyPump** | `PumpAsset` (shared base) | Pressure-maintenance pump; same state machine as FirePump |
| **FireServiceInlet** | `EngineeringAsset` + `health_status` | Breeching inlet a fire engine connects to; passive, `PassiveFireSafetyAvailability` (reused from `models/fire_safety_asset.py`) |
| **FireWaterSystem** | plain dataclass on `Building` | Logical grouping of the above asset ids — **not** pipe geometry |

## 2. Core architecture — A/B, never C

Per this milestone's own instruction, three things are kept mechanically separate:

- **A. Physical infrastructure** — the four asset models above: identity, location, and (for the tank) capacity/level.
- **B. Operational state** — `FirePump.compute_state()`, `FireWaterTank.compute_state()`, `FireServiceInlet.compute_availability()`, and `FireWaterInfrastructureManager.system_status()`'s own deterministic, conservative rollup.
- **C. Hydraulic performance** — never implemented. No pressure, flow, or pump-curve calculation exists; `FireWaterSystemStatus`'s own four values (`SYSTEM_AVAILABLE`/`SYSTEM_DEGRADED`/`SYSTEM_UNAVAILABLE`/`UNKNOWN`) contain no "PRESSURE"/"FLOW"/"ADEQUATE"/"CONFIRMED" vocabulary at all (mechanically checked, `tests/test_fire_water_infrastructure_architecture_guards.py`).

## 3. FirePump / JockeyPump — a real shared base, by design

`models/pump_asset.py::PumpAsset` (extends `EngineeringAsset`) is the shared foundation for both `FirePump` and `JockeyPump` — a deliberate, investigated choice (this milestone's own Phase 5): a jockey pump genuinely **is** a (smaller-duty, pressure-maintenance) fire pump, the same "is-a" relationship `SmokeDetector`/`HeatDetector`/`ManualCallPoint` already share through `SensorAsset`. This is different from `FireExtinguisher`/`FireHydrant`/`HoseReel` (from the prior milestone), which only share a small *computation* (`compute_passive_availability()`), not a type — those three are different *kinds* of device that happen to have identical availability logic, while FirePump/JockeyPump are the same *kind* of device.

`PumpAsset` fields: `health_status`, `control_mode` (`Automatic`/`Manual` — purely descriptive; nothing in this codebase senses pressure and flips it automatically), and `running: bool` — a direct, caller-reported intrinsic fact, mirroring `ManualCallPoint.activated` exactly (no continuous external reading this framework has). `compute_state()`: `FAULT` (health) > `UNAVAILABLE` (inactive) > `RUNNING`/`STOPPED` (the `running` flag). **`STOPPED` is a normal, healthy condition** — a correctly-functioning automatic fire pump sits stopped in standby between demand events; this is never itself a degradation (see §7).

The two Designer graphics items (`FirePumpItem`/`JockeyPumpItem`) are deliberately **not** subclassed from one another, despite sharing an octagon shape at different sizes/colors — every graphics item in this codebase extends `QGraphicsItem` directly, specifically so `isinstance(item, FirePumpItem)` in `MainWindow`'s own selection dispatch can never accidentally match a `JockeyPumpItem`.

## 4. FireWaterTank

`models/fire_water_tank.py` — `capacity_liters`/`current_level_liters`, both explicit **liters** (Phase 3's own "make units explicit" instruction). `current_level_liters: Optional[float] = None` means "not measured/unknown" — the honest default; a tank with no measurement configured reports `AVAILABLE` (unless health/active says otherwise), never a fabricated `LOW_LEVEL`/`EMPTY` with no evidence. `LOW_LEVEL_FRACTION = 0.2` is a documented placeholder (same convention as `HeatDetector.activation_threshold`'s own 57.0°C). No refill simulation, no consumption-rate modeling, no duration calculation.

## 5. FireServiceInlet

A passive resource — the external connection point a fire engine pumps into. `inlet_type` (Wet Riser / Dry Riser / Sprinkler System Inlet) is descriptive only. No external fire-engine simulation, no assumed pressure or flow — this model only represents that the connection point exists, where, and whether it's currently usable.

## 6. FireWaterSystem — the logical relationship model

`models/fire_water_system.py::FireWaterSystem` lives on **Building**, not Floor — a real system routinely spans multiple floors (a basement pump room feeding hydrants throughout). It is a plain dataclass with `id`/`name` and seven `Tuple[str, ...]` id-list fields: `tank_ids`, `pump_ids`, `jockey_pump_ids`, `inlet_ids`, `sprinkler_ids`, `hydrant_ids`, `hose_reel_ids`. This is **explicitly not pipe geometry** — no coordinates, no connectivity graph, just plain id references, the same "never resolved or validated here" convention `EngineeringAsset.zone_ids` already establishes.

**Relationships are never inferred from physical proximity.** Every association is authored/configured by an operator through the Designer (see §9/§10). Two module-level helpers do all the membership bookkeeping generically across all seven fields:

- `system_containing_asset(building, asset_id, field_name)` — which system (if any) currently contains this asset.
- `assign_asset_to_system(building, asset_id, field_name, target_system_id)` — reassignment: removes the id from every system's own field first, then adds it to the target (or none) — an asset belongs to at most one system per relationship type at a time.

Answering "which system serves HYD-3" or "which sprinklers depend on Pump P-1" is a direct read of these tuples via `FireWaterInfrastructureManager`/`FireWaterSystemStatusReport` (§8) — no separate query engine was needed.

## 7. Status intelligence — deterministic and conservative

`fire_water_manager/manager.py::FireWaterInfrastructureManager.system_status(system)` computes one of four `FireWaterSystemStatus` values:

- **`UNKNOWN`** — no supply assets configured at all (empty `tank_ids`/`pump_ids`/`jockey_pump_ids`/`inlet_ids`).
- **`SYSTEM_UNAVAILABLE`** — every configured supply component (across all four types) is unhealthy.
- **`SYSTEM_DEGRADED`** — at least one configured supply component is unhealthy, or a configured id is dangling (references a deleted asset), but not all.
- **`SYSTEM_AVAILABLE`** — every configured supply component is healthy.

"Healthy" is deliberately generous for pumps: `STOPPED` **and** `RUNNING` both count (§3) — only `FAULT`/`UNAVAILABLE` degrade. For tanks, `AVAILABLE` **and** `LOW_LEVEL` both count — only `EMPTY`/`FAULT`/`UNAVAILABLE` degrade. `reasons: Tuple[str, ...]` names exactly which asset and why (`"pump FP-1 fault"`, `"tank TANK-1 unavailable"`, `"pump GHOST-PUMP referenced but not found"`) — never a bare status code with no explanation (Phase 11's own instruction).

Dependent suppression assets (`sprinkler_ids`/`hydrant_ids`/`hose_reel_ids`) are copied onto the report for traceability **only** — their own health is never assessed here; that remains `fire_safety_manager`'s own, separate concern (§13).

## 8. Designer integration

Each of the four new asset types follows the identical established pattern: toolbar action → `GraphicsScene` click-to-place branch (unambiguous-zone auto-assignment) → graphics item (`FireWaterTankItem` wide rectangle, `FirePumpItem`/`JockeyPumpItem` octagons, `FireServiceInletItem` a plus/cross shape) → Property Panel section (zone combo, active/health, type-specific fields, and either an "Operational State" label or "Availability" label) → `MainWindow` selection dispatch. `designer/validation.py` reports `*_missing_zone` warnings for all four, plus a new `fire_water_system_dangling_reference` warning when a `FireWaterSystem`'s own id list references an asset that no longer exists.

## 9. Fire Water System authoring UI

**No JSON editing is ever required.** Two pieces, both reusing existing Designer precedent rather than inventing a new one:

- **`designer/widgets/fire_water_system_list.py::FireWaterSystemList`** — a dockable list panel (Add/Rename/Delete, `QInputDialog`/`QMessageBox`) mirroring `designer/widgets/floor_list.py::FloorList` exactly, the cleanest existing precedent for creating/renaming/deleting a small set of named, non-spatial entities (a `FireWaterSystem`, like a `Floor`, is never drawn on the canvas). Delegates every operation to `Building.create_fire_water_system()`/`rename_fire_water_system()`/`remove_fire_water_system()`.
- **A "Fire Water System" combo on all seven relevant asset types'** Property Panel sections — FireWaterTank, FirePump, JockeyPump, FireServiceInlet, **and** Sprinkler, FireHydrant, HoseReel (their existing sections each gained one more combo). Single-select, mirroring the exact "Assigned Zone" combo convention: current selection is looked up via `system_containing_asset()` (never stored on the asset itself), and a change calls `assign_asset_to_system()`.

## 10. Save/reload

`Floor` gained four new lists (`fire_water_tanks`/`fire_pumps`/`jockey_pumps`/`fire_service_inlets`) with the identical `add_`/`remove_`/count/`to_dict`/`from_dict` shape every prior asset type already has. `Building` gained `fire_water_systems: list[FireWaterSystem]` with its own `to_dict`/`from_dict`. A `.syn` project predating this milestone loads unchanged — every new list defaults to `[]` via `data.get(..., [])`.

## 11. Runtime integration

`fire_water_manager/manager.py::FireWaterInfrastructureManager` is constructed exactly once by `live_runtime.factory.build_live_runtime()` (always constructed, mirroring `EmergencyLightManager`/`FireSafetyAssetManager`), discovers all four asset types from the same `Building`, and computes system status from that same `Building.fire_water_systems`. `BuildingState.fire_water_status: Optional[FireWaterInfrastructureSnapshot]` is a new additive field, wired through `BuildingStateEstimator`/`EstimatorBuildingStateGateway` with the identical pass-through-only discipline `facp_status`/`control_status`/`fire_safety_status` already establish — `BuildingStateEstimator` never computes it, only passes through whatever the manager's `snapshot()` produced. Canonical `BuildingState` was extended additively, not redesigned.

## 12. Command Center

Per this milestone's own "prefer extending the existing fire-safety/live-status UI rather than creating many panels" instruction, two more tables were added **inside the same** `LiveStatusPanel`'s "Fire Suppression & Safety Assets" group (no new dockable panel or window):

- **`fire_water_asset_table`** (Asset/Type/Zone/State) — Tank/Pump/Jockey Pump/Inlet, the same shape as the existing `fire_safety_table`, kept as its own table since it's a genuinely different asset family.
- **`fire_water_system_table`** (System/Operational State/Supply Assets/Dependent Assets/Degradation Reasons) — one row per `FireWaterSystem`, showing exactly what §7 computes: the status, which assets it traces to on both ends, and why it's degraded if it is. The column is labeled **"Operational State"**, never "Hydraulic Capability Confirmed" — hydraulics are not modeled here.

Both tables populate independently of `facp_status` (proven in tests) — none of these assets are FACP sources (§13).

## 13. Sprinkler relationship

A Sprinkler can be associated with a `FireWaterSystem` (`sprinkler_ids`) purely for traceability. This changes **nothing** about the existing, already-committed semantics from the prior milestone: `Sprinkler ACTIVATED` still never reduces hazard/fire-growth/smoke values, and `FireWaterSystem SYSTEM_AVAILABLE` never guarantees sprinkler discharge performance. Sprinkler status (`fire_safety_manager`) and Fire Water System status (`fire_water_manager`) are computed by two entirely separate managers reading two entirely separate `BuildingState` fields — proven independent in `tests/test_fire_water_infrastructure_full_e2e.py::test_sprinkler_and_hydrant_remain_independent_of_system_status` (driving a system to `SYSTEM_UNAVAILABLE` leaves the Sprinkler's own status byte-for-byte unchanged).

## 14. Hydrant/HoseReel relationship

Same as Sprinkler (§13) — `hydrant_ids`/`hose_reel_ids` on `FireWaterSystem` let SynEvac trace `HYD-1 → system FW-1 → pump FP-1 → tank TANK-1` (proven in the full E2E test), but this dependency graph never yields a claimed pressure, flow, water velocity, or discharge rate. That would require a real hydraulic model, deliberately not built here (§20 of the milestone brief; §2 above).

## 15. FACP / BuildingControl boundaries

Investigated directly, per this milestone's own explicit caution:

- **FACP** — no supervisory-device abstraction (e.g. a waterflow switch) exists in this codebase for any of these four assets, so none of them is wired into FACP at all. Mechanically proven: none of the four new models or `fire_water_manager/` imports `facp`, and `facp/` never references any of the four new asset types.
- **BuildingControl** — `building_control/types.py::ControlSystemType.DELUGE` remains exactly what it already was (a state-only, no-backing-physics remotely-commanded system) and is structurally unrelated to `FirePump`/`JockeyPump` — no `FIRE_PUMP`/`JOCKEY_PUMP` control system was added. None of the four new asset types became a BuildingControl action; status **observation** (this milestone) is never the same as remote **control authority** (out of scope, no abstraction exists for it here).

## 16. Failure/degradation semantics

| Case | Behavior |
|---|---|
| Healthy tank + healthy pump | `SYSTEM_AVAILABLE` |
| Pump fault | `SYSTEM_DEGRADED`, reason names the pump |
| Pump `STOPPED` (not faulted) | Still `SYSTEM_AVAILABLE` — a normal automatic-standby condition, never itself a degradation |
| Tank unavailable/empty/fault | `SYSTEM_DEGRADED` or `SYSTEM_UNAVAILABLE` depending on what else is configured, reason names the tank |
| Broken/dangling asset reference | `SYSTEM_DEGRADED`, reason names the missing id honestly (`"pump GHOST-PUMP referenced but not found"`) — never a crash, never silently dropped |
| No supply configured at all | `UNKNOWN`, reason `"no supply assets configured"` |
| All configured supply bad | `SYSTEM_UNAVAILABLE` |
| Component restored | Status recovers on the very next `system_status()` call — nothing is cached/latched |
| Unassigned/outside-zone/ambiguous-zone placement | `zone_ids = ()`, no crash, Property Panel warning shown |
| Deleted zone reference | Property Panel combo falls back to index 0 — no crash; state still honestly computed |
| Legacy project without any new list/system | Loads with empty defaults, no crash |

At no point does driving a system through every one of these states change `HazardSnapshot`, `FireGrowth`, or `SmokePropagation` output — mechanically proven (no import path exists between this milestone's code and any of those three packages) and behaviorally proven (`test_degradation_then_recovery_never_touches_hazard_or_fire_growth` re-runs the identical `HazardSnapshot` through `BuildingStateEstimator` before and after a full degrade/recover cycle and asserts byte-for-byte identical `hazard_summary`).

## 17. Architecture guards

Mechanically proven in `tests/test_fire_water_infrastructure_architecture_guards.py`:

- Fire-water infrastructure cannot directly modify hazard/fire-growth/smoke (no import either direction).
- Cannot automatically execute evacuation actions (no import of `evacuation_recommendation`/`evacuation_guidance`/`emergency_response`/`dynamic_signage`/`trajectory_intelligence`/`evacuation_progress`).
- Cannot automatically broadcast voice (no `voice_evacuation` import).
- Cannot modify Decision Policy (no `decision_policy` import).
- Cannot grant AI/RL execution authority (no AI/RL package imports `fire_water_manager`, and none of the new models imports any AI/RL package).
- Cannot fabricate hydraulic values (no Hazen-Williams/Darcy-Weisbach/K-factor/CFD/flow-rate/pressure-loss/water-hammer vocabulary anywhere in the new code; `FireWaterSystemStatus`'s own vocabulary contains no pressure/flow/adequacy claim).
- No Modbus/BACnet/MQTT/vendor protocol imports anywhere in the new code.

## 18. Future expansion path

A real hydraulic model (pipe network, pump curves, pressure/flow calculation) is a legitimate, separate future milestone — this one deliberately stops at honest asset representation and status observation so that a future hydraulic layer has real infrastructure and relationship data to consume, without this milestone having guessed at physics it cannot yet verify. A waterflow-switch abstraction bridging Sprinkler activation to FACP is likewise a plausible, separate future addition (§15) — not attempted here because no such device model exists yet and inventing one wasn't asked for.
