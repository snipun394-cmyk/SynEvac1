# Designer Asset Connectivity, Relevance & Simplification Audit

Status: **investigation only.** No production code was changed to produce this document (a handful of read-only greps were run to fill one gap the agents' briefs didn't explicitly cover — dataset/training column participation — nothing else). No asset was removed, no toolbar was redesigned, no new physics was added. Every finding below is a direct citation of real code (`file:line`), never a doc-based assumption, never an inference about what an asset "should" do.

Methodology: every Designer-placeable engineering asset in `models/` was traced for real production callers across `models/`, `designer/`, `navigation/`, `pathfinding/`, `simulator/`, `simulation_interactive/`, `simulation_runtime/`, `scenario_definition/`, `scenario_generator/`, `scenario_runner/`, `scenario_validator/`, `scenario_storage/`, `scenario_pipeline/`, `scenario_event_executor/`, `hazard/`, `hazard_evolution/`, `fire_growth/`, `smoke_propagation/`, `perception/`, `live_perception/`, `building_state/`, `crowd_intelligence/`, `trajectory_intelligence/`, `evacuation_progress/`, `emergency_response/`, `ai_decision/`, `ai_registry/`, `ai_inference/`, `ai_training/`, `ai_explainability/`, `ai_features/`, `decision_policy/`, `advisory_system/`, `evacuation_recommendation/`, `evacuation_guidance/`, `voice_evacuation/`, `dynamic_signage/`, `facp/`, `building_control/`, `live_runtime/`, `command_center/`, `camera_manager/`, `camera_calibration/`, `human_detection/`, `live_camera_pipeline/`, `tracking/`, `cross_camera_identity/`, `sensor_manager/`, `emergency_light_manager/`, `fire_safety_manager/`, `fire_water_manager/`, `speaker_manager/`, `sign_manager/`, `dataset_builder/`. "Displayed in a table" and "serialized to JSON" were never counted as decision connectivity on their own.

---

## 1. Asset Connectivity Matrix

Legend: **●** = real, verified connectivity (cited below) · **○** = exists but never read further (a dead value) · **—** = not found, zero matches.

| Asset | Geometry | Nav Graph | Simulation | Scenario Gen | Hazard | Live Perception | BuildingState | Crowd/Traj/EvacProg | Emerg. Response | AI/Features | Decision Policy | Advisory | Recommendation | Guidance | FACP | Voice | Signage | Bldg Control | Command Center | Dataset/Training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zone | ● | ● | ● | ● | ● | — | ● | ● | — | ● | ● | ● | ● | ● | — | — | — | — | ● | ● |
| Exit | ● | ● | ● | ● | ● | — | ○ | — | — | — | ● | — | ● | ● | — | — | — | ● | ● | ● |
| Door | ● | ● | ● | ● | — | — | — | — | — | — | — | — | ●(indirect) | ●(indirect) | — | — | — | ● | ● | ● |
| Stair | ● | ● | ● | ● | ● | — | ○ | — | — | — | ● | — | ●(indirect) | ●(indirect) | — | — | — | ●(state-only) | ● | ● |
| Obstacle | ● | — | — | ● | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | ● |
| Assembly Point | ● | ● | ●(sandbox) | — | — | — | — | — | — | — | — | ● (text only) | — | — | — | — | — | — | — | — |
| Elevator | ● | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Occupant | — (no model) | n/a | ● | ● | — | ● | ● | ● | ● | ● | — | ● | ● | ● | — | — | — | — | ● | ● |
| Camera | ● | n/a | ●(Designer only)/○(Campaign) | — | — | ● | ● | ● | ● | ● | — | — | — | — | — | — | — | — | ○(display only) | — |
| Detector (legacy) | ● | — | ●(Smoke/Heat only) | — | ●(Smoke/Heat only) | ●(Smoke/Heat only) | ●(Smoke/Heat only) | — | ●(Smoke/Heat only) | ●(Smoke/Heat only) | — | — | — | — | ●(Smoke/Heat only) | — | — | — | ●(Smoke/Heat only) | ● |
| Smoke Detector | ● | — | ● | ● | ●(read-only) | ● | ● | — | ● | ● | — | ○(evidence only) | ●(indirect) | ●(indirect) | ● | — | — | — | ● | — |
| Heat Detector | ● | — | ● | ● | ●(field never populated) | ● | ● | — | ● | ● | — | ○ | ●(indirect) | ●(indirect) | ● | — | — | — | ● | — |
| Manual Call Point | ● | — | — | — | — | — | ○(no field) | — | ○(dropped) | — | — | — | — | — | ● | — | — | — | ○(alarm table only) | — |
| Speaker | ● | — | — | — | — | — | — | — | — | — | — | — | ● (consumes) | ● (consumes) | — | ● | — | — | ● | — |
| Dynamic Sign | ● | — | — | — | — | — | — | — | — | — | — | — | ● (consumes) | ● (consumes) | — | — | ○(computed, never displayed) | — | — | — |
| Emergency Light | ● | — | — | — | — | — | — (no field) | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Sprinkler | ● | — | — | — | — | — | ○ | — | — | — | — | — | — | — | ○(explicitly excluded) | — | — | — | ○ | — |
| Fire Extinguisher | ● | — | — | — | — | — | ○ | — | — | — | — | — | — | — | — | — | — | — | ○ | — |
| Fire Hydrant | ● | — | — | — | — | — | ○ | — | — | — | — | — | — | — | — | — | — | — | ○ | — |
| Hose Reel | ● | — | — | — | — | — | ○ | — | — | — | — | — | — | — | — | — | — | — | ○ | — |
| Fire Water Tank | ● | — | — | — | — | — | ○ | — | — | — | — | — | — | — | — | — | — | — | ○ | — |
| Fire Pump | ● | — | — | — | — | — | ○ | — | — | — | — | — | — | — | — | — | — | — | ○ | — |
| Jockey Pump | ● | — | — | — | — | — | ○ | — | — | — | — | — | — | — | — | — | — | — | ○ | — |
| Fire Service Inlet | ● | — | — | — | — | — | ○ | — | — | — | — | — | — | — | — | — | — | — | ○ | — |

Notes on the table:
- **Door/Stair "●(indirect)" for Recommendation/Guidance** means the safe-route search silently traverses these edges (a locked door removes itself from every candidate route) but neither package ever names/ranks/avoids a Door or Stair the way Zone/Exit are explicitly ranked.
- **"○" everywhere in the fire-suppression/water rows** means the value is computed and stored but its one and only reader is the Command Center status table — never a second consumer.
- **Manual Call Point's "○(dropped)"** under Emergency Response is not a "never wired" case — it is a genuine wiring gap: MCP alarms reach `BuildingState.facp_status.active_alarm_source_ids`, but `emergency_response/engine.py`'s zone-alarm matcher only iterates `smoke_detector_states`/`heat_detector_states`, so an MCP id present in `active_alarm_source_ids` is silently never matched to a zone. See §7.

---

## 2. Simulation Connectivity Matrix

"Simulation" here means: does pressing **Simulation** in the Designer (or running the Campaign/Scenario pipeline) actually compute/use this asset's state each tick, as opposed to it sitting as an inert placed object?

**A critical, cross-cutting finding first:** SynEvac currently has **two separate, non-overlapping "simulation" pipelines**, and no asset is exercised by both:

1. **Designer's Simulation Panel** (`designer/windows/main_window.py:1502-1616`, ticking `PerceptionDebugRunner`/`BuildingStateDebugRunner`/`CameraManagerPanel` every step). Hazard values here are **100% hand-entered** via the panel's "Apply Hazard" spin boxes (`designer/widgets/perception_debug_panel.py:139,626-636`) — no fire-growth/smoke-propagation physics runs in this pipeline at all.
2. **The Campaign/Scenario pipeline** (`scenario_runner/`, `simulation_runtime/runtime.py`, launched via `designer/campaign/campaign_worker.py`), which runs real `FireGrowthModel`+`FireSpreadModel`+`SmokePropagationModel` physics (`scenario_runner/fire_initializer.py:41-82`) but **never constructs a `PerceptionProvider`** (`campaign_worker.py:79-84,494-495`, explicit comment: "no concrete PerceptionProvider composition exists anywhere in this codebase yet") — so Camera/Detector/SmokeDetector/HeatDetector/FACP/CameraManager/SensorManager are completely absent from it.

| Asset | Simulation Panel (Designer, hand-authored hazard) | Campaign/Scenario pipeline (real fire physics) |
|---|---|---|
| Zone | ● routing origin/destination, hazard-score-weighted route choice | ● same, real physics-driven hazard this time |
| Exit | ● `Edge.traversable`, load-balancing by capacity | ● same |
| Door | ● `Edge.traversable`, replan on state change | ● same |
| Stair | ● graph edge include/exclude, replan on state change | ● same |
| Obstacle | ○ presence sampled by scenario generator, never read by pathfinding/visibility during a tick loop (visibility is precomputed once) | ○ same |
| Assembly Point | ● (sandbox `nearest_assembly_point()` query) | — not the default evacuation target (`OccupantSimulator.evacuate()` defaults to `nearest_exit()`) |
| Elevator | — no graph node exists to simulate | — same |
| Occupant | ● full sandbox/perception-debug tick loop | ● full `OccupantSimulator`/`MultiAgentSimulationResult` tick loop |
| Camera | ● real per-tick `GroundTruthCameraProvider` + FOV/occupancy intersection | — perception provider never constructed |
| Detector (legacy Smoke/Heat) | ● ticked via `PerceptionDebugRunner`/`BuildingStateDebugRunner` | — same absence as Camera |
| Smoke Detector | ● reads the hand-entered hazard override, real `SimulatedFACP.evaluate()` call each tick | — absent (no perception provider) |
| Heat Detector | ● same mechanism (though `temperature` is never set by any real hazard source even where physics runs) | — absent |
| Manual Call Point | — NOT wired into `BuildingStateDebugRunner._sensor_statuses_by_type()` at all; excluded from every Designer-simulation FACP tick | — absent |
| Speaker / Dynamic Sign | — zero references in `simulator/`, `scenario_runner/`, `simulation_runtime/`, `simulation_interactive/` | — same |
| Emergency Light / Sprinkler / Extinguisher / Hydrant / Hose Reel / Tank / Pump / Jockey Pump / Inlet | — zero references in any simulation-tick code | — same |

**Conclusion: no fire-safety/perception asset added in the last several milestones (Smoke/Heat Detector aside) is ever exercised by pressing Simulation.** Smoke/Heat Detector only "work" in the hand-authored Designer debug pipeline, never against real computed fire physics — the one pipeline with real physics has no perception layer at all.

---

## 3. Decision Connectivity Matrix

"Decision effect" means: can changing this asset or its state change what `evacuation_recommendation`/`evacuation_guidance`/`advisory_system` actually outputs, traced through real intermediate state — not "could it in principle."

| Asset | Decision effect? | Real chain |
|---|---|---|
| Zone | **YES** | Zone hazard severity → `SafeExitDistanceCalculator._excluded_zone_ids()` / `route_planner.excluded_zone_ids()` → recommendation/guidance route selection |
| Exit | **YES** | `Edge.traversable`(`is_blocked`) + zone hazard → `SafeExitDistanceCalculator.compute()` → ranked exit list → guidance route |
| Door | **YES (structural, not ranked)** | `Edge.traversable`(`locked`/`active`) silently removes/restores a route segment from every recommendation/guidance path search |
| Stair | **YES (structural, not ranked)** | Same mechanism as Door, cross-floor |
| Obstacle | **NO CURRENT DECISION EFFECT** | `traversal_cost` field confirmed unused outside its own model; only affects camera visibility geometry |
| Assembly Point | **NO CURRENT DECISION EFFECT (content only)** | Named in the advisory announcement text ("Move to {names}"); never changes which exit/route is chosen |
| Elevator | **NO CURRENT DECISION EFFECT** | Never becomes a graph node; structurally cannot be routed through, avoided, or referenced |
| Occupant | **YES** | Is the subject of every decision — position/state drives crowd/trajectory/evacuation-progress evidence feeding recommendation/advisory |
| Camera | **YES (via occupant evidence only)** | Camera itself has no decision logic; it is the sensing path that produces the Occupant evidence recommendation/advisory consume |
| Detector (legacy, Smoke/Heat types) | **YES (same as canonical)** | Transparently adapted to SmokeDetector/HeatDetector before any read |
| Detector (legacy, Flame/Gas types) | **NO CURRENT DECISION EFFECT** | `adapt_legacy_detector()` returns `None`; never registered anywhere |
| Smoke Detector | **YES (indirect, via Emergency Response)** | Reading → FACP alarm → `emergency_response._zone_alarm_ids()` score → `evacuation_recommendation._emergency_response_elevated()` penalty/elevation. NOT read by `evacuation_recommendation`/`evacuation_guidance` directly — only via hazard-derived zone severity and via this FACP→emergency-response path |
| Heat Detector | **YES in architecture, NO in current practice** | Same path as Smoke, but `temperature` is never populated by any current hazard source — so this chain is real but currently inert in every configuration this codebase can produce today |
| Manual Call Point | **NO CURRENT DECISION EFFECT** | Reaches `facp.active_alarm_source_ids`, but the one consumer that would translate an alarm source into a zone score (`emergency_response._zone_alarm_ids()`) only checks `smoke_detector_states`/`heat_detector_states` — an MCP-only alarm is silently excluded from that computation |
| Speaker | **NO CURRENT DECISION EFFECT (consumer only)** | Consumes an already-finalized recommendation/guidance message; never influences what that message says |
| Dynamic Sign | **NO CURRENT DECISION EFFECT (consumer only, and currently unreachable)** | Same as Speaker, and its own operator-approval path is never invoked outside tests |
| Emergency Light | **NO CURRENT DECISION EFFECT** | Zero downstream reads of any kind |
| Sprinkler / Extinguisher / Hydrant / Hose Reel / Tank / Pump / Jockey Pump / Inlet | **NO CURRENT DECISION EFFECT** | Every one terminates at a Command Center status table; explicitly disclaimed in each model's own docstring |

---

## 4. Asset classifications

**CORE** — directly required for SynEvac's evacuation reasoning:
Zone, Exit, Door, Stair, Occupant, Camera.

**OPERATIONALLY RELEVANT** — genuinely participates in perception, alarm, or guidance, even if only indirectly or partially:
Smoke Detector, Heat Detector (architecture is real; the specific hazard signal it needs is not yet produced anywhere — see §12), Manual Call Point (reaches FACP, but a real gap keeps it from reaching Emergency Response — see §7), Speaker, Dynamic Sign (fully computed but never surfaced to an operator in production — see §7), FACP (not a placeable asset but the coordinating system these all funnel through), Building Control (operator-approved actuation over Door/Exit).

**SUPPORTING / OPTIONAL** — useful for realism/dataset completeness, limited current influence:
Assembly Point (navigation node + advisory text, no routing influence), Obstacle (dataset column + visibility geometry, no pathfinding/hazard/decision effect).

**RESEARCH / FUTURE** — legitimate future purpose, currently lacks the physics/decision integration to justify prominent placement:
Elevator (model + serialization only; no authoring tool, no graph representation, no simulation, no decision code — explicitly disabled in the Designer with a developer comment, not merely "not yet built").

**OUT OF CURRENT SCOPE** — adds engineering-inventory complexity without a currently-observable contribution to evacuation reasoning:
Emergency Light, Sprinkler, Fire Extinguisher, Fire Hydrant, Hose Reel, Fire Water Tank, Fire Pump, Jockey Pump, Fire Service Inlet, and the legacy generic Detector's Flame/Gas sub-types.

This is a classification of **current code**, not a judgment that any of these are poorly engineered — every one of the nine fire-suppression/water assets has clean, well-tested, honestly-scoped code with explicit docstrings disclaiming exactly the physical effects they don't model. The finding is that they are complete, correct, and currently inert with respect to evacuation decisions.

---

## 5. Recent fire-safety asset deep audit (Phase 4)

For each, per the milestone's own required question format:

### Emergency Light
1. Reads it? `EmergencyLightManager` registers it; its own status methods (`light_status`/`all_statuses`/`available_lights`) have **zero production callers** anywhere in the repository.
2. Changes on state change? Nothing observable — no `BuildingState` field even exists for it.
3. Hazard? No. 4. Evacuation? No. 5. Pathfinding? No — explicitly disclaimed in the model's own docstring ("deliberately NOT wired into route safety/Pathfinding"). 6. Decision Policy? No. 7. Advisory? No. 8. AI? No. 9. Emergency Response? No. 10. BuildingState? No field exists at all. 11. Command-Center-only? Not even that — zero references anywhere in `command_center/`.
12. **Would removing it change any evacuation recommendation? NO.**

### Sprinkler
1-11: read once/cycle by `FireSafetyAssetManager`; reaches `BuildingState.fire_safety_status`; consumed only by Command Center's status table. Explicitly excluded from hazard feedback and from FACP by its own docstring ("do NOT assume sprinkler activation belongs in FACP").
12. **Would removing it change any evacuation recommendation? NO.**

### Fire Extinguisher
Same pattern as Sprinkler — availability-only model, Command-Center-only consumer, explicit docstring disclaiming any automatic effect.
12. **NO.**

### Fire Hydrant
Same pattern; additionally referenced by id only (never health-checked) inside `FireWaterSystemStatusReport`.
12. **NO.**

### Hose Reel
Same pattern as Fire Hydrant, including the id-only, never-health-checked `FireWaterSystemStatusReport` reference. Docstring explicitly disclaims any coverage/zone-protection semantic.
12. **NO.**

### Fire Water Tank
Read by `FireWaterInfrastructureManager`; its level genuinely can move the computed `FireWaterSystem` rollup status (`SYSTEM_DEGRADED`/`SYSTEM_UNAVAILABLE`) — the one case among these nine where a state change computes something beyond a bare pass-through. That computed status has exactly one consumer: the Command Center table.
12. **NO** — the computed rollup terminates at display, never reaches evacuation logic.

### Fire Pump / Jockey Pump / Fire Service Inlet
Same pattern as Fire Water Tank — each can move the same rollup status, same single terminal consumer. `PumpAsset`'s own docstring: "operational state != hydraulic performance."
12. **NO** for all three.

---

## 6. Legacy/duplicate asset audit (Phase 5)

`Detector` (`models/detector.py`) still coexists with `SmokeDetector`/`HeatDetector`. `models/detector_migration.py::adapt_legacy_detector()` transparently converts a legacy `Detector(detector_type="Smoke")`/`"Heat"` into the canonical type (same id preserved) before any real read happens — for those two types, the generic `Detector` toolbar button is now purely a compatibility shim with identical real behavior to authoring a canonical asset directly.

`Detector.DETECTOR_TYPES` also includes `"Flame"` and `"Gas"` — `adapt_legacy_detector()` returns `None` for both (no canonical counterpart exists), so a Flame/Gas Detector is **never** registered with `SensorManager`, never reaches perception/FACP/hazard/any intelligence engine, and never appears in Command Center. Its only observable effect anywhere in the running system is a distinct icon color in the Designer (`designer/items/detector_item.py:16-19`).

The generic **"Detector" toolbar button is still present today** alongside the dedicated Smoke Detector/Heat Detector/Manual Call Point buttons (`designer/widgets/toolbar.py:76,81-82,97`) — a new author can still place a generic Detector of any of the four types, including the two that do nothing.

**Recommendation:** the generic Detector toolbar button no longer serves a legitimate purpose for *new* authoring — Smoke/Heat are strictly better served by their own dedicated tools (same real behavior, clearer intent), and Flame/Gas currently do nothing at all. Per the milestone's own instruction, backward compatibility must not break: `Floor.detectors`, `models/detector.py`, and `adapt_legacy_detector()` must all be kept exactly as they are so existing `.syn` files with legacy Detectors keep loading and keep working. Only the *authoring* affordance (the toolbar button / new-placement path) is a candidate for hiding — never the model or the migration logic.

---

## 7. Elevator audit (Phase 6)

- **Model status**: fully modeled — `models/elevator.py` has `position`, `capacity`, `speed`, `current_floor`, `is_operational`, and even an `evacuation_enabled: bool` flag that nothing anywhere reads.
- **Designer status**: `Floor.elevators` is a real, serializable list with full CRUD — but the toolbar's `"Elevator"` `QAction` is constructed **disabled**, with a developer comment stating outright that no drawing tool exists in `GraphicsScene.set_tool()` to back it. The only way an Elevator ever ends up in a project is hand-editing a `.syn` file or a direct API/test call — never through the UI.
- **Navigation status**: absent by construction — `Node.NODE_TYPES`/`Edge.EDGE_TYPES` have no ELEVATOR entry; `NavigationGraphGenerator` has no `_add_elevator_*` method at all.
- **Simulation status**: cannot participate — with no graph representation, `PathfindingEngine` structurally cannot route through, avoid, or reference an Elevator.
- **Decision-policy status**: zero references anywhere in `decision_policy/`, or in any of the ~28 other packages checked.

**Verdict:** Elevator does not belong in the active Designer toolbar today. This is not "a deliberate fire-safety exclusion rule" (there is no comment anywhere claiming elevators are excluded *because* using them during a fire evacuation is unsafe, which would be the standard real-world justification) — it is simply unfinished: a data model with a disabled UI action and no downstream consumer of any kind. It should stay exactly where it is (model + serialization preserved for any hand-authored/test project that already uses it) but has no case for re-enabling in the toolbar until an authoring tool, a graph representation, and at least one downstream consumer exist.

---

## 8. Dead-end chains (Phase 7)

**Designer → Model → Serialization → Manager → Command Center status, and stops** (the most important category):
- Sprinkler, Fire Extinguisher, Fire Hydrant, Hose Reel, Fire Water Tank, Fire Pump, Jockey Pump, Fire Service Inlet — all nine terminate identically at a Command Center table row.
- Manual Call Point — reaches FACP's alarm-source table specifically (not the generic sensor table), but goes no further (see the Emergency Response gap below).
- Dynamic Sign — computed every live cycle (further than the fire-safety assets — it reaches a real `DynamicSignageSnapshot` and a fully-built `DynamicSignageController` with operator approve/reject methods) but those methods are **never called outside tests**, and `CommandCenterSnapshot` has no `dynamic_signage` field at all — so, in production, it is currently just as much a dead end as the fire-safety assets, only one layer deeper.

**Designer → Model → nothing else:**
- Emergency Light — does not even reach Command Center. The only asset in this audit with genuinely zero downstream consumer of any kind.
- Obstacle — reaches dataset-generation columns and camera-visibility geometry, but nothing that could be called a "decision" consumer.

**Model exists → no Designer authoring:**
- Elevator — the only asset in this category. Fully modeled and serializable, structurally unreachable from the UI.

**A genuine wiring gap, distinct from the above (worth flagging separately):** Manual Call Point is not a simple dead end — it partially works. It reaches `BuildingState.facp_status.active_alarm_source_ids` correctly, and that table is genuinely displayed in Command Center's FACP panel. But `emergency_response/engine.py::_zone_alarm_ids()` — the one place an active alarm source becomes a zone-level score that reaches `evacuation_recommendation` — only iterates `smoke_detector_states`/`heat_detector_states`, which have no corresponding dict for Manual Call Points. An MCP-only alarm is therefore invisible to Emergency Response and Evacuation Recommendation even though it is a real, confirmed FACP alarm. This looks like an oversight (Emergency Response's own alarm-matching code was written before/without MCP in mind), not a deliberate design boundary — unlike Sprinkler's documented exclusion, no comment anywhere states this is intentional.

---

## 9. Real decision chains (Phase 8)

```
Camera
  -> RTSPFrameSource/ReplayFrameSource -> YOLOHumanDetector -> SingleCameraTracker
  -> WorldProjector (optional) -> IdentityResolver -> LiveCameraPipeline.run_cycle()
  -> LiveOccupantManager -> MultiCameraFusionEngine -> BuildingStateEstimator
  -> BuildingState.occupant_tracks
  -> CrowdIntelligenceEngine / TrajectoryIntelligenceEngine / EvacuationProgressEngine
     / EmergencyResponseIntelligenceEngine (each reads LiveOccupantManager/BuildingState directly)
  -> live_ai_gateway (ai_registry inference service)
  -> EvacuationRecommendationEngine -> EvacuationGuidanceEngine
  -> live_advisory_gateway -> AdvisoryReport
  -> (operator-approved) VoiceEvacuationController / DynamicSignageController

Smoke Detector / Heat Detector
  -> SensorManager.discover_sensors() -> GroundTruthSmokeDetectorProvider/GroundTruthHeatDetectorProvider
     (reads HazardProvider.snapshot_at().smoke_level / .temperature -- temperature never actually
      populated by any current HazardSource)
  -> EngineFACPGateway._build_detector_condition_reports() -> SimulatedFACP.evaluate()
  -> BuildingState.facp_status / smoke_detector_states / heat_detector_states
  -> EmergencyResponseIntelligenceEngine._zone_alarm_ids() -> zone priority score
  -> EvacuationRecommendationEngine._emergency_response_elevated() -> exit ranking penalty/elevation
  -> EvacuationGuidanceEngine -> live_advisory_gateway (evidence only, never an override)

Manual Call Point
  -> SensorManager.discover_sensors() -> ManualCallPoint.compute_state(time) (intrinsic, no hazard reading)
  -> EngineFACPGateway._mcp_state() -> SimulatedFACP.evaluate()
  -> BuildingState.facp_status.active_alarm_source_ids
  -> command_center/live_status_panel.py FACP-sources table (DEAD END HERE --
     emergency_response never matches this id to a zone, see §7/§8)

Zone / Exit / Door / Stair
  -> NavigationGraphGenerator.build() -> NavigationGraph (nodes/edges)
  -> PathfindingEngine (Dijkstra/A*/Yen's) via DefaultCostModel
  -> SafeExitDistanceCalculator (evacuation_recommendation) / route_planner (evacuation_guidance)
     -- both exclude hazardous zones and respect Edge.traversable (Door.locked, Exit.is_blocked)
  -> EvacuationRecommendationEngine -> EvacuationGuidanceEngine
  -> live_advisory_gateway / VoiceEvacuationController / DynamicSignageController (consumers)

Sprinkler / Extinguisher / Hydrant / Hose Reel / Tank / Pump / Jockey Pump / Inlet
  -> FireSafetyAssetManager / FireWaterInfrastructureManager .snapshot()
  -> BuildingState.fire_safety_status / fire_water_status
  -> command_center/live_status_panel.py (STOPS HERE)
```

---

## 10. SynEvac's core product boundary (Phase 9)

SynEvac is **an AI-enabled, human-behavior-centric dynamic fire evacuation system** — its research value is in perceiving where people actually are, reasoning about hazard and route safety, and producing/guiding an evacuation recommendation a human operator can act on. It is **not** a fire-protection BIM/asset-management platform, and the code itself agrees: every one of the nine fire-suppression/water assets carries an explicit docstring disclaiming any evacuation-relevant physical effect, written by whoever built them, not inferred here. That is honest engineering — but it also means, by the project's own stated intent as expressed in the code, these assets were never meant to influence evacuation reasoning, only to exist as inventory. Evaluating every asset against "does this help SynEvac perceive people, reason about hazard/routes, or communicate a decision to an operator" is the right lens; "is this thing normally part of a real building's fire-protection system" is not, and applying it is exactly how the toolbar grew past the product's own boundary.

---

## 11. Proposed simplified Designer toolbar (Phase 10 — NOT implemented)

Derived from the audit above, not the milestone's own suggested grouping (which is close but not exact):

**BUILDING**
Zone, Door, Exit, Stair, Assembly Point

*(Obstacle omitted from the default/main toolbar — SUPPORTING/OPTIONAL, keep available but not front-and-center; see §12.)*

**PERCEPTION & ALARM**
Camera, Smoke Detector, Heat Detector, Manual Call Point

*(Generic "Detector" omitted from new authoring per §6.)*

**GUIDANCE / OUTPUT**
Speaker, Dynamic Sign

**SIMULATION**
Occupant, Simulation controls

**ADVANCED / OPTIONAL** (present, but not in the default view)
Obstacle, Assembly Point *(if the team prefers it here instead of BUILDING — it has real navigation/advisory-text use but no routing influence, a defensible judgment call either way)*, Elevator *(kept purely for schema/API compatibility — no case for surfacing it prominently, per §7)*.

**ADVANCED FIRE-SAFETY TOOLS** (a separate, explicitly-labeled section, not deleted)
Emergency Light, Sprinkler, Fire Extinguisher, Fire Hydrant, Hose Reel, Fire Water Tank, Fire Pump, Jockey Pump, Fire Service Inlet.

---

## 12. Disposition recommendations for non-core assets (Phase 11)

| Asset | Recommendation |
|---|---|
| Obstacle | KEEP IN MAIN TOOLBAR (or move to Advanced — defensible either way; it has real dataset/visibility use, just no decision effect yet) |
| Assembly Point | KEEP IN MAIN TOOLBAR (real navigation node + advisory content) |
| Elevator | HIDE FROM NEW AUTHORING BUT KEEP MODEL/SERIALIZATION (toolbar button is already disabled — no change needed there; just don't invest in enabling it without the missing navigation/decision work) |
| Generic Detector | HIDE FROM NEW AUTHORING BUT KEEP MODEL/SERIALIZATION (Smoke/Heat fully superseded; Flame/Gas inert) |
| Speaker, Dynamic Sign | KEEP IN MAIN TOOLBAR (both operationally relevant; Dynamic Sign's Command Center gap is a "worth fixing" item, not a reason to hide the asset — see §13) |
| Manual Call Point | KEEP IN MAIN TOOLBAR (genuine FACP alarm source; the Emergency Response gap is worth fixing, not a reason to hide it) |
| Emergency Light | MOVE TO ADVANCED FIRE-SAFETY TOOLS (fully dead-ended; no case for deletion — the code is small, tested, and harmless to keep available) |
| Sprinkler, Fire Extinguisher, Fire Hydrant, Hose Reel, Fire Water Tank, Fire Pump, Jockey Pump, Fire Service Inlet | MOVE TO ADVANCED FIRE-SAFETY TOOLS (all nine — well-built, well-tested, currently inert; hide from the default toolbar rather than delete tested code, per the milestone's own stated preference) |

No asset in this audit is recommended for DEPRECATE or REMOVE COMPLETELY. Every one of them is working code with real test coverage; the finding is about *prominence and default visibility*, not correctness or worth keeping in the repository.

---

## 13. Missing connections worth implementing later (Phase 18 lookahead — not implemented here)

1. **Manual Call Point → Emergency Response.** `emergency_response/engine.py::_zone_alarm_ids()` should also check `BuildingState.facp_status.active_alarm_source_ids` against MCP ids (or `BuildingState` should gain a `manual_call_point_states` field mirroring smoke/heat). Small, well-scoped, closes a real gap rather than a cosmetic one.
2. **Dynamic Sign → Command Center.** `CommandCenterSnapshot` should carry `dynamic_signage: Optional[DynamicSignageSnapshot]`, and a Command Center panel should call the already-built `ingest_signage_instructions()`/`approve_signage_instruction()` the same way Voice's two panels already do. All the machinery exists and is tested; only the UI wiring is missing.
3. **Heat Detector's `temperature` field.** No current `HazardSource` populates it — until `FireGrowthModel`/`SmokePropagationModel` (or a successor) sets a real temperature value, Heat Detector's real-physics chain is architecturally complete but practically inert. Worth tracking as a fire-physics gap, not a Heat Detector gap.
4. **Obstacle → pathfinding cost.** `navigation/cost.py` already documents an unimplemented `CostModel` extension point for obstacle penalties — a real, scoped future addition if Obstacle is ever meant to influence routing rather than only visibility.

---

## Final answers

1. **Genuinely CORE**: Zone, Exit, Door, Stair, Occupant, Camera.
2. **Genuinely influence Simulation**: Zone, Exit, Door, Stair, Occupant (both pipelines); Camera, legacy Detector (Smoke/Heat types), Smoke Detector, Heat Detector (Designer Simulation Panel only, never the real-physics Campaign pipeline); Assembly Point (sandbox routing query only). Everything else: no.
3. **Genuinely influence live decision-making**: Zone, Exit, Door/Stair (structurally), Occupant, Camera (via occupant evidence), Smoke Detector (via Emergency Response), legacy Detector Smoke/Heat sub-types. Heat Detector is architecturally wired but practically inert (no hazard source ever sets temperature). Manual Call Point should but currently does not (a real gap, not a design choice).
4. **Only status/inventory objects**: Emergency Light, Sprinkler, Fire Extinguisher, Fire Hydrant, Hose Reel, Fire Water Tank, Fire Pump, Jockey Pump, Fire Service Inlet, Obstacle (dataset/visibility only), legacy Detector's Flame/Gas types.
5. **Currently dead ends**: all nine fire-suppression/water assets, Manual Call Point (partial — reaches FACP, not Emergency Response), Dynamic Sign (computed, never displayed to an operator in production), Emergency Light (dead-ends before even reaching Command Center), Elevator (dead-ends before reaching the Designer UI at all).
6. **Stay in main toolbar**: Zone, Door, Exit, Stair, Assembly Point, Camera, Smoke Detector, Heat Detector, Manual Call Point, Speaker, Dynamic Sign, Occupant, Obstacle.
7. **Move to Advanced**: Emergency Light, Sprinkler, Fire Extinguisher, Fire Hydrant, Hose Reel, Fire Water Tank, Fire Pump, Jockey Pump, Fire Service Inlet.
8. **Hide from new authoring**: generic Detector (keep for legacy project loading), Elevator (already effectively hidden — toolbar button already disabled).
9. **Is generic Detector still needed?** Not for new authoring — Smoke/Heat are strictly better served by their dedicated tools, Flame/Gas do nothing. Needed permanently for loading old projects; `adapt_legacy_detector()` must never be removed.
10. **Is Elevator functional enough to expose?** No — it has no navigation, simulation, or decision connectivity of any kind, and no authoring tool exists to place one anyway.
11. **Do Emergency Lights currently affect evacuation?** No — not even Command Center displays them.
12. **Do Sprinklers currently affect fire/hazard/evacuation?** No — explicitly and deliberately excluded by the code's own design.
13. **Do Extinguishers affect anything operational?** No.
14. **Do Hydrants/Hose Reels affect anything operational?** No.
15. **Do Fire Water Tanks/Pumps/Jockey Pumps/Inlets affect evacuation decisions?** No — one genuine computed effect exists (a tank/pump/inlet fault can degrade the `FireWaterSystem` rollup status), but that status has exactly one consumer, the Command Center table, and reaches no evacuation logic.
16. **Simplified Designer toolbar**: see §11 — Building (Zone/Door/Exit/Stair/Assembly Point), Perception & Alarm (Camera/Smoke/Heat/MCP), Guidance/Output (Speaker/Dynamic Sign), Simulation (Occupant), plus an Advanced/Optional section (Obstacle, Elevator) and a separate Advanced Fire-Safety Tools section (the nine fire-suppression/water assets).
17. **What functionality would we lose by hiding the non-core assets?** None today — every asset recommended for hiding is confirmed, by direct code trace, to have no current evacuation-decision effect. We would lose only the ability to casually place them from the default toolbar; the models, serialization, managers, and Command Center displays all stay exactly as they are.
18. **Missing connections worth implementing later**: MCP→Emergency Response (real gap, small fix), Dynamic Sign→Command Center UI (machinery already built and tested, just never wired to a panel), a real temperature-producing hazard source for Heat Detector, and Obstacle→pathfinding cost (already a named, unimplemented extension point).

**HAS SYNEVAC STARTED DRIFTING FROM AN EVACUATION INTELLIGENCE PLATFORM INTO A GENERAL FIRE-SAFETY ASSET MODELER?**

**YES**, with direct evidence: nine consecutive recent assets (Emergency Light, Sprinkler, Fire Extinguisher, Fire Hydrant, Hose Reel, Fire Water Tank, Fire Pump, Jockey Pump, Fire Service Inlet) were added with complete models, managers, snapshots, and Command Center displays, and **every single one of them terminates at a status table with zero influence on hazard, pathfinding, Decision Policy, Advisory, AI, or Evacuation Recommendation/Guidance** — confirmed by exhaustive grep across every relevant package, not by assumption. Several of their own docstrings explicitly and correctly disclaim the very physical effects a fire-protection asset inventory would normally need ("operational state != hydraulic performance," "never claims physical extinguishing effect," "isolated from route safety"). That is honest, well-engineered code — but it is also nine assets' worth of Designer surface area, serialization schema, and manager/snapshot machinery built for a category of asset the evacuation-intelligence pipeline was never designed to consume. The toolbar now has more fire-protection-inventory buttons than evacuation-decision buttons, which is the concrete, measurable symptom of the drift this milestone asked about.
