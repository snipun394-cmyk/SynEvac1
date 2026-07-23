# Manual Call Points & Emergency Lighting

Adds two new Digital Twin engineering assets — **ManualCallPoint** (a fire-alarm pull station, a genuine FACP alarm SOURCE) and **EmergencyLight** (a building safety OUTPUT asset, never a sensor) — end-to-end: model, Designer authoring, save/reload, FACP integration, Command Center display, and architecture guards. Builds directly on the zone-assignment/live-FACP-runtime foundation laid by `docs/architecture/digital_twin_zone_assignment_and_facp_runtime.md` (commit `e071606`) rather than replacing any of it.

## 1. ManualCallPoint model

`models/manual_call_point.py` — reuses `SensorAsset` exactly as `SmokeDetector`/`HeatDetector` do (`id`/`name`/`floor_id`/`zone_ids`/`position`/`mount_height`/`active`/`mode`/`connection`/`health_status`/`installation_date`/`last_activation_time`), adding one field: `activated: bool = False`.

**The one structural difference from Smoke/Heat Detector:** an MCP has no continuous external hazard reading to compare against a threshold. There is no "smoke level" or "temperature" a Ground Truth provider computes for it — activation is a direct, binary **human action** on the device itself. `activated` is therefore stored as the device's own intrinsic state (mirroring a real pull station, which stays visibly latched until a technician physically resets it) rather than passed into `compute_state()` as an external reading. This is why `compute_state()` takes no reading parameter, unlike its two siblings:

```python
def compute_state(self, time=None) -> DetectorState:
    if self.health_status != HealthStatus.OK:
        return DetectorState.FAULT
    if not self.active:
        return DetectorState.NORMAL
    if self.activated:
        if time is not None:
            self.last_activation_time = time
        return DetectorState.ALARM
    return DetectorState.NORMAL
```

Priority order is the same ALARM > FAULT > NORMAL... actually here it is **FAULT outranks ALARM** — a genuinely unhealthy device's report is never trusted over its own activation flag, the same "device fault outranks anything else" discipline every `SensorAsset`-based `compute_state()` already applies.

`activate()` sets `activated = True`. `restore()` sets it back to `False` — the physical device's own reset (a technician re-arming the pull station at the box itself), **deliberately separate** from `facp.engine.SimulatedFACP.reset()` (the panel's own operator reset). Resetting the panel while the device is still physically activated must never silently clear it (see §4, latching).

## 2. MCP physical-zone semantics

Like SmokeDetector/HeatDetector (not like Speaker), `zone_ids` means **physical location**: the single zone containing the device's position, auto-assigned the same way, single-select in the Property Panel. A pull station is a point device mounted at one physical spot — it has no "coverage area" concept the way a Speaker's broadcast does.

## 3. Automatic zone assignment / manual reassignment

`GraphicsScene`'s `manual_call_point`/`emergency_light` click-to-place branches call the same `_find_unambiguous_zone_at(floor, x, y)` helper Smoke/Heat Detector already use: a click landing inside exactly one zone auto-assigns it (`model.zone_ids = (zone.id,)`); landing inside zero or several overlapping zones leaves `zone_ids = ()` — never a guessed nearest/first match. The Property Panel's "Assigned Zone" combo (`mcp_zone` / `emergency_light_zone`) allows full manual reassignment at any time, and a warning label appears whenever the assignment is empty. Both are proven in `tests/test_manual_call_point_designer.py::ZoneAutoAssignmentTests` and `tests/test_emergency_light_designer.py::ZoneAutoAssignmentTests`.

## 4. MCP → FACP runtime graph

```
Designer (real toolbar + Property Panel)
        |
        v
Floor.manual_call_points  (zone_ids populated, activated toggled)
        |
        v
Project.to_dict() -> .syn -> Project.from_dict()   (round-trips activated + zone_ids exactly)
        |
        v
SensorManager.discover_sensors(building)   (registers MCPs alongside Smoke/Heat Detector --
        |                                    same registry, no new manager: models/sensor_asset.py's
        |                                    own docstring already named "Manual Call Point" as a
        |                                    future candidate fitting this shape unchanged)
        v
live_system.facp_gateway.EngineFACPGateway.evaluate(time)
        |   -- reads status.sensor_type == "ManualCallPoint" separately from smoke/heat, then
        |      calls SensorManager.get_sensor(id).compute_state(time) directly (MCP's own
        |      compute_state() IS the complete, authoritative answer -- no reading-provider
        |      seam needed the way smoke/heat have one, since there is no external reading)
        v
facp.evaluate({..., "MCP-1": DetectorConditionReport(asset_type="ManualCallPoint", state=ALARM, ...)}, time)
        |
        v
facp.current_snapshot(time)  -> FACPSnapshot.active_alarm_source_ids includes "MCP-1"
        |
        v
BuildingState.facp_status  (read-only; EstimatorBuildingStateGateway never calls
        |                    evaluate/acknowledge/silence/reset)
        v
command_center.live_status_panel.LiveStatusPanel.facp_sources_table
  (Source / Type / Zone / State columns, cross-referencing FACPSnapshot.recent_events
   for type/zone -- no new MCP-specific panel; the exact same table already shows
   SmokeDetector/HeatDetector alarm/fault sources)
```

Proven end-to-end from real Designer authoring (not hand-built fixtures) in `tests/test_manual_call_point_emergency_light_full_e2e.py`.

## 5. Mixed MCP/detector alarm behavior

`FACPSnapshot.active_alarm_source_ids`/`active_fault_source_ids` are plain identity sets — nothing in `SimulatedFACP` distinguishes an MCP's id from a SmokeDetector's or HeatDetector's. Consequences, proven in `tests/test_manual_call_point_facp_integration.py::MixedAlarmSourceTests`:

- Two MCPs activated simultaneously both appear (`{"M1", "M2"}`).
- An MCP and a HeatDetector alarming together both appear (`{"M1", "H1"}`).
- A SmokeDetector alarm followed by an MCP activation: both source identities are retained. When the smoke condition clears, the MCP's identity remains — a real, distinct alarm condition never gets silently absorbed into or hidden by another source's state.

## 6. FACP latching behavior (unchanged, MCP participates in it honestly)

`ManualCallPoint.restore()` (device-level) is not `SimulatedFACP.reset()` (panel-level). Restoring the physical MCP alone never auto-clears the panel — it stays `ALARM` (latched) until an explicit `facp.reset()`. Only once *every* active condition (detector and MCP alike) has cleared does `reset()` legally reach `NORMAL`. `acknowledge()`/`silence()` behave identically regardless of whether the alarm's source was a detector or an MCP — `PanelState.ALARM_ACKNOWLEDGED`/`ALARM_SILENCED` never mean the underlying condition disappeared; `active_alarm_source_ids` stays populated straight through both. A brand-new MCP activation while the panel is silenced correctly re-alerts (returns to `ALARM`). All proven in `tests/test_manual_call_point_facp_integration.py`.

## 7. EmergencyLight model

`models/emergency_light.py` — a building safety **output** asset, not a sensor: reuses `EngineeringAsset` exactly as `Camera`/`DynamicEvacuationSign` do (`id`/`name`/`floor_id`/`zone_ids`/`position`/`mount_height`/`active`/`mode`/`connection`), **not** `SensorAsset` — it detects nothing, so it must not inherit `SensorAsset`'s `installation_date`/`last_activation_time` fields (an alarm-activation timestamp has no meaning for a light). `health_status` reuses `models.sensor_asset.HealthStatus`'s existing `OK`/`Fault`/`Offline` vocabulary directly rather than inventing a parallel one. `light_type` is one of `"Wall Mounted"`/`"Ceiling Mounted"`/`"Recessed"` (Property Panel combo box, purely descriptive — never affects availability).

No lux/photometric calculation, no battery-runtime physics — only what this codebase can honestly represent today.

## 8. EmergencyLight zone semantics

Same physical-location semantics as ManualCallPoint/Smoke/Heat Detector: single zone, auto-assigned by position on placement, manually reassignable via the `emergency_light_zone` combo.

## 9. EmergencyLight availability semantics

`compute_availability()` is deliberately simple and total — no external reading, no hazard/perception input of any kind:

```python
def compute_availability(self) -> str:
    if self.health_status == HealthStatus.FAULT:
        return EmergencyLightAvailability.FAULT
    if not self.active or self.health_status == HealthStatus.OFFLINE:
        return EmergencyLightAvailability.UNAVAILABLE
    return EmergencyLightAvailability.AVAILABLE
```

`FAULT` (the device itself reports a problem) outranks a merely inactive/offline light — the same "device fault outranks anything else" priority `SensorAsset`-based `compute_state()` methods already establish. `EmergencyLightAvailability` has exactly three members: `AVAILABLE` / `UNAVAILABLE` / `FAULT` — deliberately no `ALARM` member (a light detects nothing) and no route-safety member of any kind (see §10).

## 10. Why EmergencyLight availability does NOT imply route safety

This is the single most important scope boundary in this milestone. `EmergencyLightAvailability` answers exactly one honest question — "is this light currently able to provide egress lighting" — and nothing else. It is deliberately **not** wired into `decision_policy/`, `pathfinding/`, or any route-safety judgment:

- `models/emergency_light.py` never imports `decision_policy` or `pathfinding`.
- `emergency_light_manager/` never imports `decision_policy` or `pathfinding`.
- `pathfinding/` and `decision_policy/` never reference `EmergencyLight` in any form.

All three are mechanically proven in `tests/test_manual_call_point_emergency_light_architecture_guards.py::EmergencyLightCannotInfluenceRoutingOrSafetyTests`.

**Why this boundary matters:** an available light is a maintenance/status fact about one fixture. A route being *safe* is a claim about smoke, heat, occupancy, structural integrity, and egress capacity along an entire path — an entirely different kind of evidence this codebase does not fabricate by inference. Treating "the light near this exit is available" as "this route is safe" would be exactly the kind of unearned confidence this codebase's existing hazard/route-safety machinery (`hazard/`, `pathfinding/`) is built to avoid elsewhere. `EmergencyLightManager` (mirroring every sibling manager's own discipline) performs pure bookkeeping — it "never decides what should happen, only reports what already exists."

## 11. Designer authoring workflow

Both assets follow the identical, already-established pattern (Toolbar action → `GraphicsScene` click-to-place branch → auto zone-assignment → Property Panel):

- **Toolbar:** `manual_call_point_action` / `emergency_light_action` (`designer/widgets/toolbar.py`), added alongside the existing device actions.
- **Placement:** `GraphicsScene.mousePressEvent`'s `manual_call_point`/`emergency_light` branches (`designer/scene/graphics_scene.py`), single click, same unambiguous-zone auto-assignment as Smoke/Heat Detector.
- **Graphics items:** `designer/items/manual_call_point_item.py::ManualCallPointItem`, `designer/items/emergency_light_item.py::EmergencyLightItem` — selectable and movable, `sync_to_model()` on move.
- **Property Panel:** `show_manual_call_point()`/`show_emergency_light()` (`designer/widgets/property_panel.py`) — zone combo, live state/availability label, MCP's `activated` checkbox, EmergencyLight's `active` checkbox / health combo / light-type combo, each wired to update the model and the displayed state immediately.
- **MainWindow:** selection dispatch (`isinstance(item, ManualCallPointItem/EmergencyLightItem)`) routes to the correct Property Panel view, mirroring every existing asset type.

## 12. Save/reload behavior

`models/floor.py` adds `manual_call_points: list[ManualCallPoint]` / `emergency_lights: list[EmergencyLight]` fields, `add_`/`remove_` methods, `manual_call_point_count`/`emergency_light_count` properties, and `to_dict()`/`from_dict()` entries — the identical additive shape every existing asset list (`signs`, `speakers`, ...) already has. A `.syn` project predating this milestone loads unchanged: `Floor.from_dict()` defaults both lists to `[]` via `data.get(..., [])`, and each asset's own `from_dict()` defaults every optional field (`activated`, `health_status`, `light_type`) to its honest default when the key is absent. Proven in `tests/test_manual_call_point_model.py`/`test_emergency_light_model.py::*SerializationTests` (including explicit legacy-project-without-the-key tests) and end-to-end in `tests/test_manual_call_point_emergency_light_full_e2e.py::test_save_reload_preserves_identity_floor_zone_position_and_config` (identity, floor, zone assignment, position, and `activated`/`light_type` configuration all survive exactly).

## 13. Command Center representation

**No new MCP-specific panel was created.** `command_center/live_status_panel.py::LiveStatusPanel.facp_sources_table` is one additive table over the exact same, already-existing `FACPSnapshot.active_alarm_source_ids`/`active_fault_source_ids` data — it has no hardcoded knowledge of any specific asset type, and cross-references `FACPSnapshot.recent_events` purely for display (source type / zone), never a second alarm representation. A source id with no matching event in the bounded `recent_events` window shows honestly as `"-"` for type/zone, never fabricated. Proven in `tests/test_command_center_facp_sources.py` (MCP alone, Smoke+Heat alone, and all three mixed in the same table) and end-to-end in the full E2E test's Command Center assertion.

EmergencyLight has no FACP/alarm representation at all (it is not a sensor and produces no `DetectorConditionReport`) — its own status is queryable through `EmergencyLightManager.all_statuses()`/`light_status()`, following the same pattern `SignManager`/`SpeakerManager` already established for their own asset types, with no Command Center UI wired to it in this milestone.

## 14. Failure/degradation semantics

Covered in `tests/test_manual_call_point_model.py`, `test_emergency_light_model.py`, `test_manual_call_point_designer.py`, `test_emergency_light_designer.py`, `test_manual_call_point_facp_integration.py`, and (the genuinely new cases) `tests/test_manual_call_point_emergency_light_failure_modes.py`:

| Case | Behavior |
|---|---|
| MCP outside every zone | `zone_ids = ()`, no crash, Property Panel warning shown. |
| MCP ambiguous between overlapping zones | `zone_ids = ()` (never a guessed match). |
| MCP referencing a zone since deleted from the floor | Property Panel combo falls back to index 0 (`findData` returns -1) — no crash; `compute_state()` still honestly reports `ALARM` if activated (the missing zone never fabricates a cleared condition). |
| MCP inactive (`active=False`) | `compute_state()` returns `NORMAL` even if `activated=True` — an inactive device never alarms. |
| MCP fault (no activation) | `compute_state()` returns `FAULT`, appears in `active_fault_source_ids`, panel goes `FAULT`. |
| MCP activated | `compute_state()` returns `ALARM`. |
| Multiple MCPs | Each retains its own distinct source identity in `active_alarm_source_ids`. |
| Legacy project without a `manual_call_points` key | Loads with an empty list, no crash. |
| EmergencyLight unassigned | `zone_ids = ()`, Property Panel warning shown. |
| EmergencyLight inactive | `compute_availability()` returns `UNAVAILABLE`. |
| EmergencyLight fault | `compute_availability()` returns `FAULT` (outranks inactive/offline). |
| EmergencyLight offline health | `compute_availability()` returns `UNAVAILABLE` — never fabricated as `AVAILABLE`. |
| EmergencyLight referencing a deleted zone | Property Panel combo falls back to index 0 — no crash; `compute_availability()` still honestly reflects health/active state. |
| EmergencyLight save/reload | `light_type`/`health_status`/`zone_ids` all round-trip exactly. |
| Legacy project without an `emergency_lights` key | Loads with an empty list, no crash. |

No case fabricates a zone, a healthy state, or an available state that isn't genuinely true.

## 15. Architecture boundaries

Mechanically proven in `tests/test_manual_call_point_emergency_light_architecture_guards.py`:

- **ManualCallPoint cannot directly invoke `VoiceEvacuationController` or `BuildingControlController`** — `models/manual_call_point.py` imports neither package, and its only mutation methods (`activate()`/`restore()`) touch nothing but `self`.
- **FACP cannot automatically broadcast voice messages or execute building controls** — neither `facp/` nor `live_system/facp_gateway.py` imports `voice_evacuation` or `building_control`. The full E2E test additionally proves this *behaviorally*: after driving an MCP activation all the way to a `FACPSnapshot`, a freshly constructed `SimulationVoiceOutputProvider` shows zero sent instructions and a freshly constructed `BuildingControlController` shows an empty, zero-pending snapshot — nothing was dispatched on their behalf.
- **AI cannot activate/reset/acknowledge/silence FACP** — no `ai_decision`/`ai_registry`/`ai_inference`/`ai_training`/`ai_explainability`/`advisory_system`/`rl_training` module imports `facp` at all.
- **EmergencyLight cannot modify `decision_policy` or `pathfinding`** — see §10.
- **EmergencyLight availability must not imply route safety** — see §10.
- **No hardware/network protocol code** — `models/manual_call_point.py`, `models/emergency_light.py`, `emergency_light_manager/`, and `live_system/facp_gateway.py` contain no `pymodbus`/`bacpypes`/`paho`/`opcua`/`serial`/`socket` imports (mirroring the pre-existing guard in `tests/test_facp.py::ArchitectureGuardTests` for the `facp` package itself). No Modbus/BACnet/MQTT/vendor FACP protocol of any kind exists anywhere in this milestone's code.

## 16. Remaining limitations

- No dangling-zone-reference *validation warning* exists for any zone-scoped asset type (MCP/EmergencyLight included) — `designer/validation.py::_check_zone_assignment` only flags an **empty** `zone_ids`, not a stale one pointing at a since-deleted zone. This is a pre-existing limitation shared by Speaker/SmokeDetector/HeatDetector, not something newly introduced or newly fixed here; the UI/model layer still never crashes or fabricates on a stale reference (§14), it simply doesn't proactively warn about it.
- `EmergencyLightManager.enable_light()`/`disable_light()` exist as bookkeeping methods but have no Command Center UI wired to them in this milestone (no operator toggle) — mirroring the same "manager exists, Command Center integration for it is a future increment" state several sibling managers were already in before their own respective milestones.
- No lux/photometric calculation or battery-runtime physics for EmergencyLight — deliberately out of scope (§7).
- EmergencyLight has no dedicated Command Center panel or table (§13) — only `EmergencyLightManager`'s own query surface.
