# Fire Suppression & Water-Based Safety Asset Digital Twin

Adds four new Digital Twin engineering assets — **Sprinkler**, **FireExtinguisher**, **FireHydrant** (landing valve), and **HoseReel** — end-to-end: models, Designer authoring, a shared asset manager, an additive runtime status snapshot, and one Command Center table. Builds on the zone-assignment/live-FACP-runtime foundation (`docs/architecture/digital_twin_zone_assignment_and_facp_runtime.md`) and the Manual Call Point / Emergency Lighting milestone (`docs/architecture/manual_call_point_and_emergency_lighting.md`) without modifying either.

**The single most important fact about this milestone:** **DIGITAL-TWIN ASSET STATE ≠ PHYSICAL FIRE-SUPPRESSION EFFECT.** Every asset here answers "does this equipment exist, where is it, and is it currently usable/activated" — never "will this equipment actually control or extinguish a fire." No hydraulic, pump, tank, pipe-network, or suppression-physics simulation was added or implied.

## 1. Asset taxonomy

| Asset | Base class | Concept | Zone semantics |
|---|---|---|---|
| **Sprinkler** | `SensorAsset` (like Smoke/Heat Detector/MCP — has `health_status`/`installation_date`/`last_activation_time`) | Has a genuine `ACTIVATED` condition, driven by a temperature threshold | Physical location, single zone, auto-assigned |
| **FireExtinguisher** | `EngineeringAsset` + `health_status` (like EmergencyLight — no continuous reading) | Passive, manually-operated resource; availability only | Physical location, single zone, auto-assigned |
| **FireHydrant** | `EngineeringAsset` + `health_status` | Internal hydrant / landing valve outlet point; availability only | Physical location, single zone, auto-assigned |
| **HoseReel** | `EngineeringAsset` + `health_status` | Fixed hose reel installation; availability only | Physical location, single zone, auto-assigned |

FireExtinguisher/FireHydrant/HoseReel share one small helper, `models/fire_safety_asset.py::compute_passive_availability()` (`AVAILABLE`/`UNAVAILABLE`/`FAULT`, `PassiveFireSafetyAvailability`) — factored out because these three siblings, introduced together in this milestone, have byte-for-byte identical availability logic. `EmergencyLight` (a prior, separate milestone) deliberately keeps its own independent copy rather than being retrofitted onto this helper.

## 2. Sprinkler semantics

`models/sprinkler.py::Sprinkler` reuses `SensorAsset` exactly as SmokeDetector/HeatDetector/ManualCallPoint do, adding one field: `activation_temperature: float = 68.0` (a commonly documented ordinary-hazard glass-bulb/fusible-link rating, restated independently rather than imported from `HeatDetector.activation_threshold` — two different physical devices that happen to share a threshold-comparison shape).

```python
def compute_state(self, temperature, time=None) -> SprinklerActivationState:
    if self.health_status != HealthStatus.OK:
        return SprinklerActivationState.FAULT
    if not self.active:
        return SprinklerActivationState.NORMAL
    if temperature is None:
        return SprinklerActivationState.NORMAL
    if temperature >= self.activation_temperature:
        return SprinklerActivationState.ACTIVATED
    return SprinklerActivationState.NORMAL
```

Same `FAULT > threshold > NORMAL` priority `HeatDetector.compute_state()` already establishes. The Designer's Property Panel exposes a "Test Temperature" manual-entry field, exactly mirroring Heat Detector's own "Test Temperature (°C)" convention (no live hazard simulation is wired into the Designer for either).

## 3. Activation semantics — why `SprinklerActivationState`, never `DetectorState`

`SprinklerActivationState` (`NORMAL`/`ACTIVATED`/`FAULT`) is its own enum in `models/sprinkler.py`, structurally identical in shape to `models.sensor_asset.DetectorState` but a genuinely different type — the same "structurally distinct concepts get structurally distinct types" precedent `facp.models.PanelState` already establishes relative to `DetectorState`. Reusing `DetectorState.ALARM` for a sprinkler discharge would silently invite the exact assumption this milestone was warned against: that a Sprinkler is a FACP-style alarm-reporting device. It is not (see §12).

## 4. FireExtinguisher / FireHydrant / HoseReel semantics

All three are passive, manually-operated resources with no automatic activation of any kind — an operator can mark one `active=False` (removed from service) or set `health_status` to `Fault`/`Offline` (failed inspection / out of order), and `compute_availability()` reports `AVAILABLE`/`UNAVAILABLE`/`FAULT` accordingly, exactly mirroring `EmergencyLight.compute_availability()`'s own priority (`FAULT` outranks inactive/offline). `FireExtinguisher.extinguisher_type` (Water/Foam/CO2/Dry Powder/Wet Chemical) and `FireHydrant.hydrant_type` (Wet Riser Landing Valve/Dry Riser Landing Valve/External Hydrant) are small, closed, purely descriptive sets (Property Panel combo boxes), never affecting availability. `HoseReel` deliberately has no type field — investigation found no standards data this codebase could meaningfully use for one.

## 5. Zone semantics

All four assets follow the identical Smoke/Heat Detector/ManualCallPoint/EmergencyLight convention: physical location, single zone, auto-assigned on placement via `GraphicsScene._find_unambiguous_zone_at()` (a click inside exactly one zone auto-assigns it; zero or multiple overlapping zones leave `zone_ids = ()`, never a guessed match), with full manual reassignment always available through the Property Panel's "Assigned Zone" combo.

## 6. Designer authoring workflow

Each asset follows the same established pattern:

- **Toolbar:** `sprinkler_action` / `fire_extinguisher_action` / `fire_hydrant_action` / `hose_reel_action` (`designer/widgets/toolbar.py`).
- **Placement:** `GraphicsScene.mousePressEvent`'s `sprinkler`/`fire_extinguisher`/`fire_hydrant`/`hose_reel` branches (`designer/scene/graphics_scene.py`), single click, same unambiguous-zone auto-assignment.
- **Graphics items:** `designer/items/sprinkler_item.py` (blue circle, cross-hair glyph), `fire_extinguisher_item.py` (red triangle), `fire_hydrant_item.py` (orange-red pentagon), `hose_reel_item.py` (orange hexagon) — every shape distinct from every sibling item already in the Designer (generic engineering symbols, no vendor reproductions).
- **Property Panel:** `show_sprinkler()` / `show_fire_extinguisher()` / `show_fire_hydrant()` / `show_hose_reel()` (`designer/widgets/property_panel.py`) — zone combo, active checkbox, health combo, type combo (where applicable), and either a live "Current State" label (Sprinkler, via its Test Temperature field) or an "Availability" label (the other three).
- **MainWindow:** selection dispatch (`isinstance(item, SprinklerItem/...)`) routes to the correct Property Panel view.
- **Validation:** `designer/validation.py::validate_building_authoring()` reports `sprinkler_missing_zone` / `fire_extinguisher_missing_zone` / `fire_hydrant_missing_zone` / `hose_reel_missing_zone` as `WARNING`-severity issues, same convention as every sibling zone-scoped asset.

## 7. Manager architecture

**One coherent manager**, not four: `fire_safety_manager/manager.py::FireSafetyAssetManager`. Investigated deliberately — none of the four fits an *existing* manager's own semantics (`SensorManager` exists specifically to aggregate ALARM/FAULT conditions for FACP, and Sprinkler is deliberately excluded from that aggregation, see §12; `EmergencyLightManager`'s own docstring commits to being specifically about lighting). All four new assets, in contrast, genuinely share one identity — physical fire-suppression/firefighting resources a building has installed — which is exactly the "coherent family" case that justifies one shared manager rather than four separate ones or a forced fit into an existing one.

`FireSafetyAssetManager` provides: `discover_assets(building)`, per-type accessors (`sprinklers()`/`fire_extinguishers()`/`fire_hydrants()`/`hose_reels()`), `get_asset(id)`, `assets_on_floor(floor_id)`, `assets_in_zone(zone_id)`, `enable_asset(id)`/`disable_asset(id)`, `status_of(id, ...)`, `all_statuses(...)`, and `snapshot(...)`.

`FireSafetyAssetStatus` (`fire_safety_manager/status.py`) is **one shared status shape** for all four asset types (`asset_id`/`asset_type`/`name`/`floor_id`/`zone_ids`/`active`/`health_status`/`state`) rather than four separate status classes — `state` carries whichever vocabulary the underlying asset actually produces (`SprinklerActivationState`'s name for a Sprinkler, `PassiveFireSafetyAvailability`'s value for the other three), the same Source/Type/Zone/State shape `command_center.live_status_panel.LiveStatusPanel.facp_sources_table` already established for a different asset family.

## 8. Runtime status graph

```
Designer (real toolbar + Property Panel)
        |
        v
Floor.sprinklers / .fire_extinguishers / .fire_hydrants / .hose_reels
        |
        v
Project.to_dict() -> .syn -> Project.from_dict()
        |
        v
FireSafetyAssetManager.discover_assets(building)   (live_runtime.factory.build_live_runtime()
        |                                            always constructs one, mirroring EmergencyLightManager)
        v
FireSafetyAssetManager.snapshot(sprinkler_temperatures=..., time=...)
        |   -- sprinkler_temperatures defaults to None: no hazard-to-sprinkler-temperature
        |      wiring exists in this codebase (Phase 2's own scope boundary), so every
        |      Sprinkler honestly reports NORMAL/FAULT only unless a caller explicitly
        |      supplies a reading -- never a fabricated ACTIVATED.
        v
FireSafetyStatusSnapshot  (fire_safety_manager/snapshot.py)
        |
        v
BuildingState.fire_safety_status  (additive; BuildingStateEstimator only ever passes it
        |                          through, via building_state/estimator.py's own
        |                          fire_safety_snapshot kwarg and live_system.
        |                          building_state_gateway's fire_safety_snapshot_provider --
        |                          never computed there)
        v
command_center.live_status_panel.LiveStatusPanel.fire_safety_table
  (Asset / Type / Zone / State -- ONE table, all four asset types, populated
   independently of facp_status, since none of these four is ever a FACP source)
```

Proven end-to-end from real Designer authoring in `tests/test_fire_safety_asset_full_e2e.py`, and through the full production `LiveOrchestrator` cycle in ad-hoc verification against `live_runtime.factory.build_live_runtime()`.

## 9. Command Center representation

**No new panels were created** (four asset types, one table) — `LiveStatusPanel.fire_safety_table` mirrors `facp_sources_table`'s own "one additive table, no hardcoded per-type knowledge beyond the state string it displays" shape. An operator can see which assets exist, where (zone), and whether they are available/activated/faulted — but the table shows exactly `AVAILABLE`/`ACTIVATED`/`UNAVAILABLE`/`FAULT` and nothing more; it never implies that an `AVAILABLE` extinguisher/hydrant/hose reel guarantees fire control (see §11).

## 10. Sprinkler vs FACP — investigated, deliberately kept separate

Phase 13 of this milestone explicitly warned against assuming Sprinkler activation belongs in FACP. Investigation confirmed: `facp/models.py::DetectorConditionReport` is built specifically from *initiating-device* alarm sources (Smoke/Heat Detector, Manual Call Point) — devices whose entire purpose is reporting a hazard condition to the panel. A sprinkler head is an *output* device (it discharges water); the real-world engineering concept that would legitimately bridge "a sprinkler discharged" back to a fire alarm panel is a **waterflow switch** — a genuinely separate physical device (detects flow in the pipe, not the sprinkler head itself) that this codebase has no model for and was not asked to invent. Fabricating one here would be exactly the kind of unearned engineering abstraction this milestone's own brief repeatedly warns against.

**Decision: Sprinkler is never registered with `SensorManager`, never produces a `DetectorConditionReport`, and never appears in `FACPSnapshot.active_alarm_source_ids`/`active_fault_source_ids`.** Its own `SprinklerActivationState` is exposed only through `FireSafetyAssetManager`/`fire_safety_status` — a separate, additive `BuildingState` field, never merged into `facp_status`. Mechanically and behaviorally proven in `tests/test_fire_safety_asset_full_e2e.py::test_sprinkler_activation_never_reduces_hazard_or_fire_growth` (an activated sprinkler produces no FACP alarm/fault source at all) and `tests/test_fire_safety_asset_architecture_guards.py`.

## 11. BuildingControl relationship

`building_control/types.py::ControlSystemType.DELUGE` already exists (a prior milestone) as a state-only, no-backing-physics remotely-commanded control system. **Sprinkler was NOT merged into Deluge** — they are different systems: Deluge is a remotely-commanded building control action (open/close a valve on operator/system command), while Sprinkler here is a passive, independently-activating physical device the Digital Twin merely *represents*, never commands. Likewise, FireExtinguisher/FireHydrant/HoseReel are **not** BuildingControl actions — they are physical resources retrieved/used by a person, not systems with a remote ACTIVATE/DEACTIVATE verb. `building_control/` contains no reference to any of the four new asset types (mechanically proven in the architecture guard tests).

## 12. Suppression-physics boundary

Investigated directly: `hazard/`, `hazard_evolution/`, `fire_growth/`, and `smoke_propagation/` have **no suppression concept anywhere** (`fire_growth/model.py`'s own docstring: "No CFD, no smoke physics, no detector logic, no suppression — all of that is deliberately absent"). Per Phase 2's explicit instruction, this milestone did **not** retrofit suppression physics into those systems to make Sprinkler activation "do" something — that would require redesigning already-committed, frozen hazard/fire-growth architecture, which was out of scope. Instead, Sprinkler/FireExtinguisher/FireHydrant/HoseReel expose only asset state/availability, so a **future** milestone that does design an honest suppression model has real asset data to consume. Mechanically proven: none of `hazard/`, `hazard_evolution/`, `fire_growth/`, `smoke_propagation/` references `Sprinkler` in any form, and none of the new models/manager imports any of those four packages.

## 13. Failure/degradation semantics

| Case | Behavior |
|---|---|
| Unassigned asset (any of the four) | `zone_ids = ()`, Property Panel warning shown, no crash. |
| Placed outside every zone | `zone_ids = ()` (never a guessed match). |
| Placed in overlapping/ambiguous zones | `zone_ids = ()`. |
| Zone deleted after assignment | Property Panel combo falls back to index 0 (`findData` returns -1) — no crash; `compute_state()`/`compute_availability()` still honestly reflect real state; `validate_building_authoring()` correctly does not warn (it only flags an *empty* `zone_ids`, a pre-existing limitation shared by every zone-scoped asset type, not something newly introduced here). |
| Sprinkler below/at/above threshold | `NORMAL` / `ACTIVATED` / `ACTIVATED` respectively. |
| Sprinkler fault | `FAULT`, outranking any temperature reading. |
| Sprinkler inactive | `NORMAL` regardless of temperature. |
| Passive asset inactive/offline/fault | `UNAVAILABLE`/`UNAVAILABLE`/`FAULT` respectively — never fabricated `AVAILABLE`. |
| Duplicate id across floors | `FireSafetyAssetManager`'s id-keyed dict registration means the last-discovered asset with that id wins — no crash (same behavior every sibling manager's own registry already has). |
| Legacy project without any of the four new lists | `Floor.from_dict()` defaults each to `[]` via `data.get(..., [])` — loads unchanged. |
| Mixed old + new project (e.g. SmokeDetector + Sprinkler together) | Both round-trip independently and correctly through save/reload. |
| Empty building | `FireSafetyAssetManager.snapshot()` returns an all-zero `FireSafetyStatusSnapshot`, never fabricated non-zero counts. |

No case fabricates a zone, an available/activated state, or a healthy state that isn't genuinely true.

## 14. Architecture boundaries

Mechanically proven in `tests/test_fire_safety_asset_architecture_guards.py`:

- **Sprinkler cannot directly modify HazardEvolution/FireGrowth** — `models/sprinkler.py` imports neither `hazard`/`hazard_evolution`/`fire_growth`/`smoke_propagation`, and none of those packages references `Sprinkler`.
- **Sprinkler cannot automatically execute BuildingControl** — `models/sprinkler.py`/`fire_safety_manager/` never import `building_control`; `building_control/` never references `Sprinkler`; `DELUGE` remains the only water-related BuildingControl system, structurally unrelated to Sprinkler.
- **Extinguisher/Hydrant/HoseReel cannot automatically alter fire state** — same import-absence guarantee; these three have no activation concept at all, only availability.
- **No asset automatically changes Decision Policy** — none of the four models or `fire_safety_manager/` imports `decision_policy`.
- **No asset automatically broadcasts voice instructions** — none imports `voice_evacuation`.
- **No AI/RL authority added** — no `ai_decision`/`ai_registry`/`ai_inference`/`ai_training`/`ai_explainability`/`advisory_system`/`rl_training` module imports `fire_safety_manager`, and none of the four new models imports any AI/RL package.
- **No hardware/network protocol code** — no `pymodbus`/`bacpypes`/`paho`/`opcua`/`serial`/`socket` imports anywhere in the new model layer or `fire_safety_manager/`.
- **No hydraulic-calculation vocabulary** — mechanically checked absent (Hazen-Williams, Darcy-Weisbach, K-factor, pump curves, CFD, flow rate, pressure loss) from every new model file.

## 15. Remaining limitations

- Sprinkler's `ACTIVATED` state is only ever driven by a caller-supplied temperature (Designer's manual "Test Temperature" field, or a future caller-supplied `sprinkler_temperature_provider` in `live_runtime.factory.build_live_runtime()`) — there is no automatic hazard-to-sprinkler-temperature wiring in production today, by design (§12).
- No waterflow-switch model exists, so Sprinkler activation genuinely cannot reach FACP even in principle yet (§10) — a legitimate future milestone, not attempted here.
- Same pre-existing "no dangling-zone-reference validation warning" limitation already documented for every zone-scoped asset type (`docs/architecture/manual_call_point_and_emergency_lighting.md` §16) — unchanged, not newly introduced.
- `FireSafetyAssetManager.enable_asset()`/`disable_asset()` exist as bookkeeping methods with no Command Center operator control wired to them in this milestone (display-only, mirroring the same "manager exists, operator UI is a future increment" state several sibling managers were already in).
- No lux/hydraulic/inspection-schedule physics of any kind — deliberately out of scope throughout (§§2, 4, 12).
