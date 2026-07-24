# SynEvac End-to-End Architecture & Connectivity Review

Status: **investigation only.** No production code was changed to produce this document. Baseline: 4271/4271 tests passing, commit `a5f1ca0`.

This document traces the ACTUAL production architecture (what the code does when run), not the intended one. Every claim below is a citation of real code (`file:line`), never an assumption about design intent.

---

## Actual system diagram

```
main.py -> core.app.SynEvacApp -> designer.windows.main_window.MainWindow (Designer GUI)
           ** never constructs a LiveRuntime, never opens Command Center **

command_center.main_window.MainWindow (a SEPARATE, standalone PyQt app)
           ** exposes enable_live_mode(data_source, gateway) but never calls it itself **

live_runtime.factory.build_live_runtime(building, **~30 optional kwargs)
           ** the ONLY place that assembles a full LiveRuntime -- called only from
              tests/*.py and scripts/run_physical_camera_validation.py /
              scripts/dry_run_physical_cctv.py, never from main.py or any GUI app **

  [if frame_sources+human_detector+identity_resolver supplied]
  RTSPFrameSource -> YOLOHumanDetector -> (tracker/behavior/cross-cam-id/world-proj,
  all individually optional) -> LiveOccupantManager -> MultiCameraFusionEngine
                                                     \-> SensorFusionEngine (live_perception)
  [if facp supplied] Smoke/Heat/MCP -> SensorManager -> EngineFACPGateway -> SimulatedFACP
       |                                                                        |
       v                                                                        v
  BuildingStateEstimator (via EstimatorBuildingStateGateway) <--------------------
       |
       v
  BuildingState  --(read by, each independently, same live_occupant_manager/Building)-->
       CrowdIntelligenceEngine, EvacuationProgressEngine, TrajectoryIntelligenceEngine,
       EmergencyResponseIntelligenceEngine
       |
       v
  [if live_ai_gateway supplied -- no default construction exists] LiveAIPredictionSnapshot
       |
       v
  EvacuationRecommendationEngine -> EvacuationGuidanceEngine -> DynamicSignagePlanner
       |
       v
  [if live_advisory_gateway supplied -- no default construction exists] AdvisoryReport
       |
       v
  StateManager (LiveBuildingSnapshot, one field per stage above)
       |
       v
  LiveCommandCenterDataSource -> CommandCenterSnapshot -> command_center.dashboard.Dashboard
       |
       v
  Human operator click -> LiveOperatorActionGateway -> VoiceEvacuationController /
                            DynamicSignageController / BuildingControlController
       |
       v
  SimulationVoiceOutputProvider / SimulationDynamicSignageProvider / SimulationControlProvider
  (no real hardware provider implemented for any of the three)
```

---

## 1. Production Runtime Graph / Ownership Table (Phase 1)

**Critical finding first:** `main.py` → `core/app.py::SynEvacApp.__init__()` constructs only `designer.windows.main_window.MainWindow` (`core/app.py:14`) and never imports or calls `live_runtime.factory.build_live_runtime`, `command_center.main_window.MainWindow`, or `command_center.dashboard.Dashboard` anywhere. `command_center/main_window.py::MainWindow.enable_live_mode()` exists and is fully functional, but nothing in the actual application calls it — a grep for `build_live_runtime` across the whole repository (excluding `tests/`) turns up exactly two call sites: `scripts/run_physical_camera_validation.py` and (indirectly, via the same pattern) `scripts/dry_run_physical_cctv.py`, both standalone validation scripts with no Command Center GUI attached. **The real, shippable application (`main.py`) is Designer-only; the live evacuation-intelligence runtime is currently reachable only from tests and CLI validation scripts, never from a GUI a real operator would use.**

| Component | Who creates it | Shared? | Who writes | Who reads | Reachable from production (`main.py`)? |
|---|---|---|---|---|---|
| `Building` | Caller of `build_live_runtime(building, ...)` — itself loaded via Designer's `.syn` project I/O | Yes — one instance passed to every engine constructor | Designer authoring only | Every engine (read-only geometry) | Designer: yes. Live runtime: no (never constructed) |
| `CameraManager` | `factory.py:281` (`camera_manager or CameraManager()`) | Yes — one per `LiveRuntime` | `discover_cameras()`, `register_detection_provider()`, `set_camera_mode()` (`factory.py:282,365-368`) | `fusion_result_provider` closure (`factory.py:373`) | No — only inside `build_live_runtime()`, never called by `main.py` |
| `SensorManager` | `factory.py:284` | Yes | `discover_sensors()` (`factory.py:285`) | FACP gateway, status providers (`factory.py:394-454`) | No |
| `SpeakerManager` | `factory.py:289` | Yes | `discover_speakers()` (`factory.py:290`) | `VoiceEvacuationController`, `EngineEvacuationGuidanceGateway` (`factory.py:576`) | No |
| `SignManager` | `factory.py:292` | Yes | `discover_signs()` (`factory.py:293`) | `EngineEvacuationSignageGateway` (`factory.py:577`) | No |
| `LiveOccupantManager` | `factory.py:181-184` (defaulted with `event_bus`+`exits` — a prior gap this codebase's own comments say was previously missing) | Yes — the ONE shared instance every crowd/progress/trajectory/emergency-response/recommendation engine reads (`factory.py:198,208,224,233,245`) | `LiveCameraPipeline`/`LiveOccupantObservationProvider` (only if camera pipeline configured) | 5 engines listed above | No |
| `MultiCameraFusionEngine` | `factory.py:287` | Yes | `camera_pipeline.run_cycle()` + `fusion_result_provider` closure (`factory.py:370-375`) | `EstimatorBuildingStateGateway` (indirectly, via `fusion_result_provider`) | No; also only meaningfully populated if camera pipeline is configured |
| `SensorFusionEngine` | `factory.py:185` | Yes | `LivePerceptionFusionCoordinator` (`factory.py:524-526`) | `EstimatorBuildingStateGateway.occupancy_snapshot_provider`/`hazard_snapshot_provider` | No |
| `FACP` (`SimulatedFACP`) | **Caller-supplied only — `facp: Optional[object] = None`, no default construction** (`factory.py:128`) | Yes, when supplied | `EngineFACPGateway.compute()` calls `facp.evaluate()` once/cycle (`factory.py:396-403`) | `facp_snapshot_provider` → `BuildingState.facp_status` | No; and even inside `build_live_runtime()`, FACP is entirely absent unless the caller supplies one |
| `BuildingStateEstimator` | Instantiated **inside** `EstimatorBuildingStateGateway`, not by the factory directly | One per gateway, one gateway per runtime | Called every cycle by `LiveOrchestrator` via `building_state_gateway.collect()` | `StateManager.update_building_state()` | No |
| `CrowdIntelligenceEngine` | `factory.py:196-199` (default: `CrowdIntelligenceEngine(building, live_occupant_manager)`) | Yes | `EngineCrowdIntelligenceGateway` each cycle | `StateManager.crowd_intelligence`; consumption by later stages audited separately below | No |
| `EvacuationProgressEngine` | `factory.py:206-209` | Yes | `EngineEvacuationProgressGateway`, subscribed to `event_bus` | `StateManager.evacuation_progress` | No |
| `TrajectoryIntelligenceEngine` | `factory.py:222-225` | Yes | `EngineTrajectoryIntelligenceGateway` | `StateManager.trajectory_intelligence` | No |
| `EmergencyResponseIntelligenceEngine` | `factory.py:231-234` | Yes | `EngineEmergencyResponseGateway` | `StateManager.emergency_response` | No |
| `LiveAIInferenceService` / `live_ai_gateway` | **Caller-supplied only — `live_ai_gateway: Optional[object] = None`; no default-construction helper exists anywhere in the repository** (confirmed: no `build_live_ai_gateway`/`build_default_*` function found by repo-wide grep) | N/A unless supplied | `LiveOrchestrator` calls it if present | `StateManager.ai_prediction_snapshot` | No — and note this means **AI is silently absent from every production LiveRuntime unless a caller writes brand-new wiring code that does not exist anywhere in this repository today** |
| `EvacuationRecommendationEngine` | `factory.py:243-246` (`EvacuationRecommendationEngine(building, navigation_graph, live_occupant_manager)`) — takes NO crowd/trajectory/emergency-response/AI dependency at construction time, only at its own `compute()` call each cycle as parameters | Yes | `EngineEvacuationRecommendationGateway` | `StateManager.evacuation_recommendation` | No |
| `EvacuationGuidanceEngine` | `factory.py:255-258` (`EvacuationGuidanceEngine(building, navigation_graph)`) | Yes | `EngineEvacuationGuidanceGateway` | `StateManager.evacuation_guidance` | No |
| `DynamicSignagePlanner` | `factory.py:267-270` | Yes | `EngineEvacuationSignageGateway` | `StateManager.dynamic_signage` | No |
| `Advisory` (`live_advisory_gateway`) | **Caller-supplied only — same as `live_ai_gateway`, no default construction anywhere** | N/A unless supplied | `LiveOrchestrator` | `StateManager.advisory_report` | No — same "silently absent" finding as AI |
| `VoiceEvacuationController` | `factory.py:468-471` — **only if `voice_output_provider` supplied** (defaults `None`) | Yes | `LiveOperatorActionGateway.approve_voice_message()` | Command Center Voice panels | No |
| `DynamicSignageController` | `factory.py:482-485` — **only if `dynamic_signage_provider` supplied** (defaults `None`; `build_offline_demo_runtime()` does default one in, see below) | Yes | `LiveOperatorActionGateway.approve_signage_instruction()` | `LiveDynamicSignagePanel` | No |
| `BuildingControlController` | `factory.py:473-476` — only if `building_control_provider` supplied | Yes | `LiveOperatorActionGateway.approve_control_request()` | `BuildingControlsPanel` | No |
| `StateManager` | `LiveOrchestrator.__init__` (owns one internally) | Yes — the one canonical `LiveBuildingSnapshot` holder | Every `update_*()` call from `LiveOrchestrator`'s cycle | `LiveCommandCenterDataSource` | No |
| `EventBus` | `factory.py:177` (`event_bus or EventBus()`) | Yes | `LiveOrchestrator`, `LiveOccupantManager`, gateways that emit transition events | `LiveEventsPanel`, `LiveCommandCenterDataSource._recent_events()` | No |
| Command Center datasource (`LiveCommandCenterDataSource`) | `factory.py:592-595` | Yes, one per `LiveRuntime` | Reads `StateManager.current()` only | `command_center.main_window.MainWindow` (if `enable_live_mode()` is ever called by an external caller) | No |

**`build_offline_demo_runtime()`** (`factory.py:632+`) is a second, narrower factory that defaults `dynamic_signage_provider`/others in for demo/test purposes (`factory.py:658`: `kwargs.setdefault("dynamic_signage_provider", SimulationDynamicSignageProvider())`) — still never called from `main.py`, only from tests/scripts.

**Conclusion for Phase 1:** every one of the ~20 components above is genuinely SHARED (exactly one instance per `LiveRuntime`, no accidental duplication found) — the internal wiring inside `build_live_runtime()` is disciplined and correct. The actual defect is one level up: **there is no production call site that ever invokes `build_live_runtime()` at all**, so the entire live pipeline, however well-wired internally, is currently inert from the perspective of a person launching `main.py`.

---

## 9. Simulation/Live Parity Matrix (Phase 9)

*(Investigated by a dedicated research pass.)*

| Concept | Simulation | Live | Shared type? | Semantics identical? | Known gap |
|---|---|---|---|---|---|
| Occupancy | `OccupancySnapshot`/`OccupancyObservation` | Same type | **Shared** | Identical | None at type level (but see Duplication Audit #1) |
| Hazard | `HazardSnapshot`/`HazardSeverity` | Same type | **Shared** | Identical scale | Fidelity differs by producer (physics vs sparse detector), not by type |
| Door/Exit state | `scenario.engineering_state.DoorState`/`ScenarioExitState` | `Door.locked/.active` bools, read by `Edge.traversable` | **Two representations** | Roughly equivalent | Explicitly documented as distinct in `scenario/engineering_state.py:41-46` |
| Obstacle state | `ScenarioObstacleState.presence` (ACTIVE/INACTIVE) | `Obstacle.active` + live `Edge.blocking_obstacles` | Two representations, same intent | Equivalent | No explicit divergence but genuinely separate code paths |
| Human behavior | `ground_truth.human_behavior.DynamicHumanState` (provable fact) | `perception.models.human_observation.HumanState` (noisy observation) | **Two types**, one explicit bridge | Labels overlap, meaning differs; bridge is **lossy** (`POSSIBLE_INJURY` collapses into `FALLEN`) | Explicitly documented as deliberate in both docstrings |
| Human classification | `behaviour_profile_id` → resolver → `HumanClassification` | Vision-model output → same enum | One shared enum, two production mechanisms (authored intent vs visual inference) | Same values, different epistemic basis | Not divergence-prone, by design |
| Congestion/crowd density | `simulator/congestion.py::CongestionModel` (internal walking-speed multiplier, never exposed as a snapshot) | `crowd_intelligence.models.ZoneCrowdMetrics` | **Two unrelated representations**, different purpose | Not meant to be the same computation | Explicitly named to avoid confusion (`crowd_intelligence/models.py:157-162`) |
| Evacuation progress | `ground_truth.evacuation_metrics` — exact count against known total population | `EvacuationProgressSnapshot` — observed-only count | Two independent types | **Explicitly, deliberately different epistemics** — live never claims to know the true population | Documented directly in `evacuation_progress/models.py:13-28` |
| Route choice | `decision_policy.DecisionPolicy` — authoritative, resolved once | `EvacuationRecommendationEngine` — advisory, recomputed every cycle | Two separate engines, structurally firewalled (`decision_policy/` never imported by the recommendation engine) | Related purpose, not the same computation | Intentional separation |
| FACP | `SimulatedFACP` | Same class, ticked every cycle | **Shared, single implementation** | Identical | Class name "SimulatedFACP" is a slight misnomer given it's the one shared production engine |
| AI features | `ai_features.simulation_extractor` builds a real `BuildingState`, calls the same `extract_canonical_features()` | Same function/schema (`CANONICAL_LIVE_SCHEMA`) | **One shared function/schema — the best-engineered parity case found** | Identical by design | None |
| Recommendation/Guidance engines | No sim caller found | Take generic `building_state`/`recommendation_snapshot`, no sim/live branch | Single code path by construction | N/A | Unclear whether Replay mode ever actually exercises these engines directly |

**Overall verdict**: parity is genuinely strong where it matters most for future AI work (AI feature extraction is one shared code path — the single most AI-relevant risk is fully closed) and FACP (fully shared). The real gaps are Door/Exit/Obstacle state representation (two independently-defined types per side, though semantically similar) and human behavior/state (an explicit, deliberate, and somewhat lossy translation layer). Evacuation progress and route choice are INTENTIONALLY different (sim knows ground truth, live never does) — not a gap to close, a correct epistemic boundary.

## 10. State Duplication Audit (Phase 10)

*(Investigated by a dedicated research pass.)*

1. **Zone occupant count — a genuine, unreconciled duplication.** `BuildingState.zone_occupancy` (fed by whatever occupancy provider a caller wires in) and `CrowdIntelligenceEngine`'s own `zone_metrics[zone_id].occupant_count` (computed independently and exclusively from `LiveOccupantManager.active_occupants()`, per its own docstring's explicit "never raw per-camera detections") are two independently-computed headcounts for the same zone at the same instant, with **no code found reconciling them**. `EvacuationProgressSnapshot`/`EmergencyResponseSnapshot` both correctly share `LiveOccupantManager` as their one authoritative owner — `BuildingState.zone_occupancy` sits outside that shared lineage as a second, potentially-disagreeing source.
2. **Human classification/state — a known, deliberately intentional duplication, not a bug.** `FusedTrack.classification`/`.human_state` (recomputed fresh every cycle, no cross-cycle memory) and `LiveOccupant.human_classification`/`.human_state` (persistent, cross-cycle-reconciled specifically so a FALLEN reading doesn't flicker to UNKNOWN) can legitimately disagree for the same person at the same instant — `LiveOccupant`'s own docstring documents this as intentional.
3. **Hazard severity — single owner, no risk.** `EmergencyResponseSnapshot`'s own `hazard_severity` is a direct pass-through of `BuildingState.zone_severity()`, never recomputed.
4. **Detection → FusedTrack → LiveOccupant — single owner, no risk.** Each layer feeds, never duplicates, the next; `Detection` is explicitly ephemeral.
5. **Dormant duplicate seam on `LiveBuildingSnapshot` — a real maintainability risk, not a runtime-disagreement risk today.** `LiveBuildingSnapshot` still carries an ORIGINAL seam (`ai_predictions`, `recommendations`, `decision_policy`) alongside the NEWER, actually-populated one (`ai_prediction_snapshot`, `evacuation_recommendation`, `advisory_report`) — the old fields' own docstrings call them "still-unimplemented-in-production." Harmless today (the old fields are simply never written), but a future maintainer could read/write the wrong one.
6. **Evacuation time — no risk.** `ground_truth.total_evacuation_time` (measured) vs `LiveAIPredictionSnapshot.evacuation_time_experimental` (explicitly named "experimental" ML prediction) are clearly distinguished by design.

---

## 13. Designer → Runtime Connectivity Matrix (Phase 13)

A full asset-by-asset connectivity audit already exists at `docs/architecture/designer_asset_connectivity_audit.md` (produced earlier this session) and is **reused, not re-derived**, per this phase's own instruction to re-audit only what changed. Three concrete deltas since that audit:

1. **Obstacle**: was "NO CURRENT DECISION EFFECT" (visibility/dataset only) → now genuinely **CORE-adjacent/OPERATIONALLY RELEVANT**: an active `Blocked` obstacle changes `Edge.traversable` for any intersecting Door/Exit, which changes `EvacuationRecommendationEngine`'s chosen exit and `EvacuationGuidanceEngine`'s route (commit `38081d6`, this session). Still SUPPORTING for pathfinding *cost* (only blocks outright, doesn't yet penalize — `navigation/cost.py`'s obstacle-cost extension remains unimplemented).
2. **Manual Call Point**: was "reaches FACP, dead-ends before Emergency Response" → now reaches Emergency Response/Recommendation/Advisory/Command Center as a structurally distinct, additive alarm source (commit `a6a4630`, this session) — genuinely OPERATIONALLY RELEVANT, no longer a partial dead end.
3. **Dynamic Sign**: was "computed every cycle, operator-approval machinery built but never wired to Command Center" → now fully wired end-to-end through `LiveDynamicSignagePanel` (commit `a5f1ca0`, this session) — genuinely OPERATIONALLY RELEVANT, no longer a dead end.

Every other classification in the original audit (CORE: Zone/Exit/Door/Stair/Occupant/Camera; OPERATIONALLY RELEVANT: Smoke/Heat Detector, Speaker, FACP, Building Control; SUPPORTING: Assembly Point; RESEARCH/FUTURE: Elevator; OUT OF CURRENT SCOPE: all nine fire-suppression/water assets and legacy Detector's Flame/Gas sub-types) remains accurate — nothing in this session's work touched the Designer toolbar, the fire-suppression/water assets, or Elevator/generic-Detector status. The advanced fire-safety assets (Emergency Light, Sprinkler, Extinguisher, Hydrant, Hose Reel, Tank, Pump, Jockey Pump, Inlet) are summarized as a group as instructed: all nine remain well-tested, Command-Center-display-only, zero evacuation-decision effect, unchanged since the last audit.

---

## 14. Over-Engineering Findings (Phase 14)

*(Investigated by a dedicated research pass.)*

| Finding | Rank | Evidence | Justification |
|---|---|---|---|
| `live_system/integration.py`'s Phase-7 gateways (`PerceptionGateway`, `AIInferenceGateway`, `DecisionPolicyGateway`, `CommandCenterGateway` + their sole adapters) are dead in production | **CRITICAL** | `live_runtime/factory.py:567-581` never passes these 4 params to `LiveOrchestrator`; referenced only by `tests/test_live_system.py` | A full Protocol+adapter layer with zero production callers — the real composition root bypasses it entirely in favor of the newer `Engine*Gateway` seams |
| `LiveAIInferenceGateway` (`RegistryLiveAIInferenceGateway`/`ThrottledLiveAIInferenceGateway`) never constructed by any production entry point | **HIGH** | `live_runtime/factory.py:133,578`; both concrete classes instantiated only inside test files | Confirms the Phase 1/6 finding independently: AI is not merely "missing a trained model," the entire gateway construction path has no production caller at all |
| Remaining `live_system/*_gateway.py` Protocol+single-impl pairs (crowd/progress/trajectory/emergency-response/recommendation/guidance/signage/FACP/building_state) | LOW | `factory.py:544-577,397` | Each has exactly one concrete implementation, but every one IS actually instantiated by the real factory — a live seam, not a dead one; not flagged higher |
| `dynamic_signage/consistency.py` vs `building_state/consistency.py` | Not a finding | Explicit non-overlap docstrings in both | Genuinely different domains, intentional separation |
| Unused manager/controller sweep | Not a finding | — | Every Manager/Controller class outside Designer/simulation is either instantiated in `factory.py` or default-constructed inside `LiveOrchestrator.__init__` |

**This independently corroborates Phase 1's own finding from a different angle**: it isn't only that `main.py` never calls `build_live_runtime()` — a whole PRIOR generation of gateway abstraction (`live_system/integration.py`) was already superseded by the current `Engine*Gateway` generation and left in place, unused, never deleted. Two dead layers stacked on top of each other (the superseded gateway generation, and the fact that even the current generation is never reached from `main.py`).

## 15. Full System Offline Proof (Phase 15)

*(Investigated by a dedicated research pass.)* **No single existing test drives the complete chain** (camera → detection → tracking → occupancy → smoke/heat/MCP → FACP → BuildingState → crowd intelligence → trajectory → evacuation progress → emergency response → AI → recommendation → guidance → advisory → Command Center → operator-approved Voice AND Dynamic Sign AND Building Control) in one place. The closest coverage is spread across five tests, each covering a different slice:

1. `tests/test_live_runtime_e2e.py::OfflineFullSystemDemonstrationTests` — real `build_live_runtime()`, camera→detection→BuildingState→AdvisoryReport, operator-approved Voice and Building Control in separate test methods — but never asserts crowd/trajectory/progress/recommendation/guidance objects, MCP, or Dynamic Signage approval (even though those engines run internally).
2. `tests/test_manual_call_point_emergency_response_integration.py::FullOfflineE2ETests` — real `build_live_runtime()`, MCP+Smoke→FACP→BuildingState→Emergency Response evidence — but starts from a direct `live_occupant_manager.update()` call (skips camera/detection/tracking) and explicitly asserts voice/control/signage controllers are `None`.
3. `tests/test_dynamic_signage_e2e.py::OfflineDynamicSignageE2ETest` — starts from a hand-built recommendation, never touches camera/FACP/BuildingState.
4. `tests/test_live_dynamic_signage_operator_workflow.py::ProductionWiringOfflineE2ETests` (this session's own new test) — starts from a hand-built `Building`/`StateManager`, bypasses perception and FACP entirely.
5. `tests/test_live_operator_action_routing.py::VoiceOperatorOfflineE2ETests`/`ControlOperatorOfflineE2ETests` — both start from a hand-built `AdvisoryReport`.

**No test currently exists that combines test #1's real perception ingress with Dynamic Signage's operator approval, or with explicit crowd/trajectory/progress assertions, in one place.** Per this audit's own explicit instruction, this gap is documented, not filled, here.

## 16. Test Architecture Audit (Phase 16)

*(Investigated by a dedicated research pass.)* Total: **4271 test methods**, 266 files, 1195 `TestCase` classes.

| Category | Approx. count | Share |
|---|---|---|
| Architecture guard tests (`*GuardTests`/`*architecture_guard*`) | ~452 methods, 33 classes / 22 files | ~10-11% |
| UI tests (QApplication-based) | ~840 methods, 42 files | ~20% |
| Integration/E2E (real factory wiring) | ~179 methods, 22 files (~135 of these explicitly `E2E`-named) | ~4% |
| Real-model opt-in (skip-decorated) | 2 methods, 1 file | <1% |
| Unit tests (remainder) | ~2700-2900 methods | ~65-68% |

**Verdict: the 4271-test count is not dominated by redundant guard-test proliferation.** Guard tests are a meaningful but modest ~10-11% of the suite; UI tests are actually the larger category, reflecting the many Designer/Command Center panel files. No near-identical test bodies were found duplicated across files — the recurring "offline E2E" pattern across 5+ files is conceptually repetitive but each one genuinely exercises a different entry point/slice, not a restated duplicate.

---

## 11. Timing / Staleness Audit (Phase 11)

*(Investigated by a dedicated research pass.)*

`LiveOrchestrator.run_cycle()` (`live_system/orchestrator.py:282`) executes, in exact order: Sensor read → (legacy Perception gateway, unused in production) → **FACP** → **BuildingState** → **Crowd Intelligence** → **Evacuation Progress** → **Trajectory Intelligence** → **Emergency Response** → **AI Inference** → **Evacuation Recommendation** → **Evacuation Guidance** → **Dynamic Signage** → **Advisory** → Command Center notify.

**No structural one-cycle-lag bug was found.** The call order is exactly the dependency order each stage needs — Evacuation Recommendation (step 10) runs strictly after Crowd/Progress/Trajectory/Emergency-Response/AI (steps 5-9), so it reads their CURRENT-cycle values, not stale ones; Guidance/Signage/Advisory likewise sit strictly downstream of Recommendation. The module's own inline comments repeatedly confirm this ordering was deliberate. Every "previous-cycle" read that does occur is a graceful-degradation fallback (StateManager never overwrites a field with `None` when a gateway is unconfigured/fails that cycle — downstream stages transparently see the last successful value), not a race condition. One lag was actively engineered OUT: FACP is evaluated BEFORE BuildingState specifically so `BuildingState.facp_status` never lags a cycle behind the detector conditions that produced it.

## 12. Failure Propagation (Phase 12)

*(Investigated by a dedicated research pass; every failure condition below resolves to an EXPLICIT degraded state — `None`, an `UNAVAILABLE`/`FAILED`/`NO_SAFE_EXIT_AVAILABLE`-style status, or an empty-but-present collection — never silently normal-looking data.)*

| Condition | Resulting state | Evidence |
|---|---|---|
| One camera fails/offline | Empty stream from that camera only; others unaffected, no crash | `tests/test_live_runtime_failure_modes.py:117-136`; `live_camera_pipeline/pipeline.py:239-242` |
| All cameras fail | `BuildingState` still generates; `occupant_tracks` empty | `tests/test_live_runtime_failure_modes.py:138-159` |
| YOLO detector raises internally | Caught at the detector boundary, returns `()`, never propagates | `human_detection/yolo_human_detector.py:107-115` |
| Calibration missing | `world_position=None`, `floor_id=None` — never fabricated | `tests/test_camera_calibration_failure_modes.py:43-52` |
| Detector fails | Hazard has no entry for that zone at all — never a fabricated `0.0` | `tests/test_live_perception_failure_modes.py:97-125` |
| FACP unavailable | `BuildingState.facp_status = None` | `live_runtime/factory.py:394-403` |
| AI model/registry unavailable | `ai_prediction_snapshot.system_status = AISystemStatus.UNAVAILABLE` (or `ERROR`/`INCOMPATIBLE` on internal exception) | `tests/test_live_runtime_failure_modes.py:198-271` |
| No safe exit exists | Explicit `RecommendationStatus.NO_SAFE_EXIT_AVAILABLE`; orchestrator emits `EventType.NO_SAFE_EXIT` | `evacuation_recommendation/engine.py:121-123`; `orchestrator.py:913-914` |
| Sign unavailable | Explicit `SignageStatus.UNAVAILABLE` / `SignIndication.UNAVAILABLE` | `dynamic_signage/planner.py` (multiple sites) |
| Speaker unavailable | Route stays valid but flagged `GuidanceInconsistency.NO_SPEAKER_COVERAGE` (degraded-but-present) | `evacuation_guidance/engine.py:94-100` |
| Building-control provider unavailable | `control_status=None`; operator approval raises `OperatorActionUnavailable` explicitly | `factory.py:559-562`; `tests/test_live_runtime_failure_modes.py:329-353` |

---

## 2. Person-tracing chain (Phase 2)

*(Investigated by a dedicated research pass; summarized here.)*

`build_live_runtime(building)` called with no camera kwargs produces `camera_pipeline=None` — the camera pipeline is entirely opt-in (`live_runtime/factory.py:97-138,349-376`). When wired, the chain is: `RTSPFrameSource` → `CameraFrame` → `YOLOHumanDetector` → `RawHumanDetection` (all CONNECTED when supplied) → `SingleCameraTracker`, `WorldProjector`, `BehaviorRecognizer`, `CrossCameraIdentityResolver` (all four individually OPTIONAL, and all four are no-ops unless `tracker` is *also* supplied — `live_camera_pipeline/pipeline.py:96-206,246-326`) → `IdentityResolver` → `LiveOccupantManager` (updates only flow if `tracker` was configured) → `MultiCameraFusionEngine`/`SensorFusionEngine` → `BuildingState`.

Two concrete, LOSSY transitions: (1) `FusedTrack` (`multi_camera_fusion/track.py:52-83`) carries `track_id/floor_id/zone_id/classification/human_state/confidence` but **no world position, no velocity, no calibration provenance, and no fine-grained `RecognizedBehavior`** (only a coarse WALKING/RUNNING/None `HumanState`) — so `BuildingState.occupant_tracks` never exposes a person's world coordinates or a FALLEN/HELPING behavior state, even though `LiveOccupant` (a sibling object) has both. (2) `LiveOccupantObservationProvider` (`live_perception/providers.py:59-100`) emits only per-zone occupancy counts and zone-keyed behavior observations — `building_state_adapter.py` has no translation branch for the `BEHAVIOR` observation kind at all, so it is silently dropped before reaching `BuildingState`. `BuildingState.camera_observations` is permanently empty in production because the factory never supplies a `camera_observation_provider` (`live_system/building_state_gateway.py:107,177` vs. `factory.py:544-565`, which omits it). `HumanClassification` is architecturally wired end-to-end but factually always `UNKNOWN` because no real classifier exists (`human_detection/yolo_human_detector.py:44-59` — YOLO detects "person," nothing more).

**Net finding: identity/zone survive to `BuildingState` reliably (when the camera pipeline is wired at all); world position, calibration provenance, and fine-grained behavior do not survive past `LiveOccupantManager` — anything reading only `BuildingState` (which is everything downstream) never sees them.**

---

## 3. Fire/Emergency input tracing (Phase 3)

Traced directly from this session's own prior FACP/MCP integration work (`docs/architecture/manual_call_point_emergency_response_integration.md`, `facp/engine.py`, `emergency_response/engine.py`) and re-verified.

**Smoke/Heat Detector** → `SensorManager.discover_sensors()` → (a reading provider, if the caller supplies one — `smoke_detector_reading_provider`/`heat_detector_reading_provider`, both `None` by default, `factory.py:123-124`) → `EngineFACPGateway._build_detector_condition_reports()` → `SimulatedFACP.evaluate()` (one call per cycle) → `BuildingState.facp_status` / `smoke_detector_states` / `heat_detector_states` → `EmergencyResponseIntelligenceEngine._alarm_sources_by_zone()` (now returning structured `AlarmSourceEvidence`, not a bare boolean, since commit `a6a4630`) → `ZoneResponsePriority.alarm_sources`/`automatic_alarm_active` → `EvacuationRecommendationEngine`'s emergency-response-elevated penalty (per exit) → Guidance (indirect, via the already-ranked recommendation) → Advisory (`manual_emergency_report_zone_ids`/alarm source evidence) → Command Center (`LiveEmergencyResponsePanel`'s "Alarm Sources" column, distinguishing `Auto: SD-1` from `Manual: MCP-1`).

**Manual Call Point** → same `SensorManager` → `ManualCallPoint.compute_state(time)` (self-contained, no external hazard reading needed) → `EngineFACPGateway._mcp_state()` → `SimulatedFACP.evaluate()` → `BuildingState.manual_call_point_states` (added in commit `a6a4630`, mirrors smoke/heat) → `EmergencyResponseIntelligenceEngine` treats it as an ADDITIVE, independent contribution (`ResponseWeights.manual_report_weight`) alongside, never overwriting, an automatic detector alarm in the same zone → reaches Recommendation/Advisory/Command Center exactly like a detector alarm, but explicitly labeled `MANUAL_EMERGENCY_REPORTED` throughout, never merged into `FACP_ALARM_ACTIVE`.

**FACP Panel State** (`PanelState.NORMAL/ALARM/FAULT/...`) is its own concept, computed by `SimulatedFACP.evaluate()` from whichever detector/MCP condition reports it was handed that cycle — `BuildingState.facp_status.panel_state` is a single building-wide value, deliberately distinct from any one zone's alarm status.

**Hazard State** (`HazardSummary.zone_severities`, from `hazard_snapshot_provider`) is a THIRD, independent concept — smoke/fire physics severity per zone, with **no code-level coupling** to FACP/detector-alarm state at all (confirmed: FACP evaluates purely from `DetectorConditionReport`s built from detector *readings*, not from `HazardSummary`; the two can and do diverge, e.g. a detector fault with zero real hazard, or real hazard with a detector that hasn't yet crossed its alarm threshold).

**Verdict: DETECTOR ALARM, MANUAL EMERGENCY REPORT, FACP PANEL STATE, and HAZARD STATE remain four structurally distinct, never-conflated concepts** — mechanically enforced by the MCP milestone's own architecture guards (`tests/test_manual_call_point_emergency_response_integration.py`) which verify zero code coupling between MCP activation and `hazard`/`fire_growth`/`smoke_propagation`/`decision_policy`/`voice_evacuation`/`building_control`/AI-RL.

---

## 4. Routing & evacuation decision tracing (Phase 4)

Zone/Door/Exit/Stair/Obstacle → `NavigationGraphGenerator.build()` → `NavigationGraph` (nodes/edges, `Edge.traversable` property) → `PathfindingEngine` (Dijkstra/A*/Yen's via `DefaultCostModel`) → deterministic safety logic (`SafeExitDistanceCalculator._excluded_zone_ids()` in `evacuation_recommendation/`, `route_planner.excluded_zone_ids()` in `evacuation_guidance/`) → `EvacuationRecommendationEngine` (WHICH EXIT, ranked) → `EvacuationGuidanceEngine` (HOW TO GET THERE, a concrete route) → `DynamicSignagePlanner` (per-sign instruction) → Voice Guidance (`voice_evacuation/adapter.py::guidance_plan_to_voice_message()`).

These distinct concepts are kept genuinely separate in code, never conflated:
- **TRAVERSABLE / NON-TRAVERSABLE** — purely structural, a boolean property of an `Edge` (`navigation/edge.py::Edge.traversable`): a locked Door, a blocked Exit, or (since the Obstacle→Navigation milestone, commit `38081d6`) an Obstacle whose geometry blocks a Door/Exit's own line segment (`Edge.blocking_obstacles`, `navigation/obstacle_geometry.py::segment_blocked_by_obstacles()`). This is checked BEFORE pathfinding even considers an edge — a non-traversable edge simply doesn't exist for routing purposes this cycle.
- **SAFE / UNSAFE** — a zone-level judgment from `hazard_summary.zone_severities` (`SafeExitDistanceCalculator._excluded_zone_ids()`), entirely independent of traversability — a zone can be perfectly traversable but excluded for being hazardous, or hazard-free but structurally unreachable (locked door).
- **PREFERRED / NOT PREFERRED** — a ranking outcome among the SAFE+TRAVERSABLE candidates only, produced by `evacuation_recommendation/ranking.py::rank_exits_for_zone()`'s multi-factor score (distance, congestion, throughput, trajectory support, emergency-response penalty, AI bottleneck term) — this is the ONLY one of the five concepts that is a comparative judgment rather than a boolean gate.

**Obstacle specifically**: since the Obstacle→Navigation milestone this session, an active `Blocked` obstacle genuinely changes `Edge.traversable` for any Door/Exit whose line segment it intersects, which genuinely changes `EvacuationRecommendationEngine`'s chosen exit and `EvacuationGuidanceEngine`'s route — a real decision effect, correcting the earlier Designer Asset Connectivity Audit's finding of "NO CURRENT DECISION EFFECT" for Obstacle (that finding predates this session's work and is now stale; `navigation/cost.py`'s obstacle-cost-penalty extension point remains unimplemented, so Obstacle still cannot make a route merely *more expensive*, only impossible).

---

## 5. Live Intelligence Consumption Audit (Phase 5)

*(Investigated by a dedicated research pass.)*

| Intelligence type | Inputs | Stored on | Reaches Recommendation? | Reaches Guidance? | Reaches Advisory? | Reaches Command Center? |
|---|---|---|---|---|---|---|
| Crowd Intelligence | `Building` + `LiveOccupantManager` | `StateManager.crowd_intelligence` | **Yes** — per-exit `congestion_level`/`queue_count` genuinely shift ranking (`evacuation_recommendation/scoring.py:43-69`) | No | Yes, awareness-only (`IncidentCommanderDashboard.crowd_highest_density_zone_id`, explicitly "never folded into critical_zones/available_exits") | No dedicated `CommandCenterSnapshot` field — only surfaces via confidence-explanation labels |
| Evacuation Progress | `BuildingState`, `CrowdIntelligenceSnapshot` | `StateManager.evacuation_progress` | **Yes** — per-exit `throughput` scoring term | No | Yes, awareness-only | Yes — own panel + `CommandCenterSnapshot` field |
| Trajectory Intelligence | `BuildingState`, `CrowdIntelligenceSnapshot`, `EvacuationProgressSnapshot` | `StateManager.trajectory_intelligence` | **Yes** — per-exit `trajectory_support` votes | No | Yes, awareness-only | Yes — own panel + field |
| Emergency Response Intelligence | `BuildingState`, `CrowdIntelligenceSnapshot`, `EvacuationProgressSnapshot`, `TrajectoryIntelligenceSnapshot` | `StateManager.emergency_response` | **Yes** — per-exit binary `emergency_response_elevated` penalty | No | Yes, awareness-only | Yes — own panel + field |
| AI bottleneck prediction | `BuildingState` | `StateManager.ai_prediction_snapshot` | **Technically yes, but rank-inert** — added *identically* to every candidate in a zone, so it changes the displayed score but can never, by construction, reorder candidates (`evacuation_recommendation/scoring.py:106-121`, sort key confirmed at `ranking.py:215`) | No | Yes, explicitly "never claims a specific stair/door/exit/zone" | Yes — own panel + field |

**Verdict: four of the five (Crowd, Progress, Trajectory, Emergency Response) are genuinely rank-affecting, not cosmetic** — each contributes its own per-exit scoring term that can change which exit wins. **`evacuation_guidance/engine.py` reads none of the five at all** — Guidance only ever consumes the already-ranked `EvacuationRecommendationSnapshot` plus `building_state`, never intelligence snapshots directly. Only the AI bottleneck signal is provably rank-inert by construction, making it the one output in this list that is effectively decision-neutral despite being "consumed."

## 6. AI Authority Audit (Phase 6)

*(Investigated by a dedicated research pass.)*

- **What AI actually changes today**: `ai_bottleneck_probability` is added uniformly to every exit candidate's score for a zone (`evacuation_recommendation/scoring.py:113-121`) — it changes the absolute displayed score and an explanatory reason code, but an identical additive term applied to every candidate can never reorder them under the engine's own `(-score, distance, exit_id)` sort key. AI cannot alter route safety (safety exclusion reads only `hazard_summary`, no AI input) and cannot alter Guidance at all (`evacuation_guidance/engine.py` takes no AI parameter).
- **`decision_policy/`** has zero references to AI prediction/crowd/trajectory/RL of any kind (repo-wide grep) — fully disconnected from live intelligence.
- **`confidence_source` "ai"/"rl" labels** (seen in `command_center/recommendation_center.py`'s own explainability text) are not evidence of a deployed model — that module's own comment states "every current run has `confidence_source == ()` ... so today every label reads as plain, honest, rule-based." `ai_registry.registry.ModelRegistry` starts with zero entries by default, and `live_ai_gateway` has no default-construction path in `build_live_runtime()` (confirmed independently by both the Phase 1 ownership trace and this pass, plus the Phase 14 over-engineering finding that its concrete gateway classes have zero production callers).

**Net verdict: AI is currently present in the codebase, structurally wired to receive `BuildingState` and to be read by `evacuation_recommendation/engine.py`, but genuinely inert for any actual decision outcome** — it can move a displayed number and an explanatory label, never which exit gets recommended or how guidance routes. This is "useful" only in the narrow sense of surfacing a bottleneck-risk number to the operator; it is not currently "AI-augmented decision-making" in any load-bearing sense.

---

## 7. Recommendation → Guidance → Signage/Voice consistency (Phase 7)

The invariant "Recommendation selects Exit E2 → Guidance terminates at E2 → Signs point toward E2 → Voice says E2" is enforced by construction, not by a single central validator:
- `EvacuationGuidanceEngine.compute()` takes the `EvacuationRecommendationSnapshot` as its one input and builds a route strictly TO the recommended exit — it structurally cannot compute a route to a different exit (`evacuation_guidance/engine.py`).
- `DynamicSignagePlanner` and `voice_evacuation/adapter.py::guidance_plan_to_voice_message()` both derive their own output SOLELY from the same `EvacuationGuidancePlan.recommended_exit_id` — there is exactly one upstream fact, read twice, never two independently-computed answers.
- An INDEPENDENT cross-check nonetheless exists and is exercised at the operator-approval boundary: `dynamic_signage/consistency.py::detect_inconsistencies()`/`instruction_inconsistencies()` re-verifies, at approval time, that a `SignageInstruction`'s `recommended_exit_id` still matches the CURRENT `EvacuationGuidancePlan` and the CURRENT voice plan for that zone — added originally for staleness detection, and (as of this session's Dynamic Sign Operator Approval milestone, commit `a5f1ca0`) now also gates `LiveOperatorActionGateway.approve_signage_instruction()` itself, so a genuinely stale/inconsistent sign instruction cannot be approved even if it was rendered before a same-cycle Guidance update.
- **Can any output channel disagree with another under a real production cycle?** Only transiently, across a revision boundary — e.g. Guidance retargets from E1→E2 on cycle N+1 before an operator has approved cycle N's sign instruction for E1; the consistency checker/gateway catches exactly this case and blocks approval (proven in `tests/test_live_dynamic_signage_operator_workflow.py::RouteInvalidationE2ETests`). No code path was found that could let an APPROVED (dispatched) sign or voice message silently continue pointing at a since-abandoned exit without the checker flagging it on the next approval attempt — though note there is no periodic re-validation of an ALREADY-CONFIRMED instruction; it is only re-checked at the moment of a NEW approval action.

---

## 8. Operator Authority Audit (Phase 8)

All three operator-controlled actions (Voice, Dynamic Signage, Building Control) share one execution boundary: `command_center/live_operator_action_gateway.py::LiveOperatorActionGateway` — the only module in `command_center/` permitted to import `voice_evacuation.controller`, `building_control.controller`, or `dynamic_signage.controller` directly (mechanically enforced by `tests/test_live_command_center.py::CommandCenterLiveIntegrationGuardTests` and `tests/test_dynamic_signage_architecture_guards.py::AIAdvisoryCannotDispatchSignageTests`).

| Channel | RECOMMENDED | PENDING | APPROVED/REJECTED | DISPATCHED | CONFIRMED/FAILED |
|---|---|---|---|---|---|
| Voice (civilian) | `AdvisoryReport.civilian_announcements` | gateway-tracked decision (`_voice_decisions`) | `approve_voice_message()`/`reject_voice_message()` | `VoiceEvacuationController.broadcast()` | `BroadcastStatus.BROADCAST`/`NO_SPEAKERS_AVAILABLE` |
| Voice (guidance) | `EvacuationGuidancePlan.voice_plan()` | same gateway pattern, keyed on `zone_id::guidance_revision` | same | same | same |
| Dynamic Signage | `DynamicSignagePlanner` output (`SignageStatus.ACTIVE`) | `DynamicSignageController`-tracked (`SignageRequestStatus.PENDING_APPROVAL`) | `approve_signage_instruction()`/`reject_signage_instruction()` | `DynamicSignageController._dispatch()` | `CONFIRMED`/`FAILED` via `SimulationDynamicSignageProvider` |
| Building Control | `AdvisoryReport.building_recommendations` (translated via `building_control/advisory_adapter.py`) | `BuildingControlController`-tracked (`RequestStatus.PENDING_APPROVAL`) | `approve_control_request()`/`reject_control_request()` | `BuildingControlController._dispatch()` | `CONFIRMED`/`FAILED` via `SimulationControlProvider` |

**Does Command Center genuinely expose the workflow?** Yes for Voice and Building Control (`command_center/recommendation_center.py::VoiceEvacuationPanel`/`BuildingControlsPanel`, wired into `dashboard.py` since the Live Operator Action Routing milestone) and, as of this session's own Dynamic Signage milestone, yes for Dynamic Signage too (`command_center/live_dynamic_signage_panel.py`, wired into `dashboard.py`). All three panels call ONLY the gateway's public methods, never the underlying controller/provider directly.

**Verified — nothing executes merely because AI/Advisory/FACP/MCP/Crowd/Emergency-Response computed something.** Every recommendation/instruction sits in a RECOMMENDED or PENDING_APPROVAL state until an explicit operator click calls one of the gateway's `approve_*` methods; the gateway itself has zero callers anywhere in `advisory_system/`, `live_system/live_ai_gateway.py`, `live_system/live_advisory_gateway.py`, `decision_policy/`, or `live_system/orchestrator.py` (mechanically enforced, `tests/test_live_operator_action_routing.py::AIAuthorityGuardTests`, `tests/test_dynamic_signage_architecture_guards.py::AIAdvisoryCannotDispatchSignageTests`). This was directly proven for FACP alarms (`FACPSeparationTests.test_facp_alarm_alone_produces_zero_voice_dispatches_and_zero_control_executions`) and is architecturally impossible for Crowd/Emergency-Response/AI by the same import-boundary guards.

---

## 17. Table Index

This document's eight requested tables/sections, for reference: (1) Production Runtime Graph — §1's diagram + ownership table; (2) Data Ownership — §1's ownership table (same table, "who creates/writes/reads" columns); (3) Intelligence Consumption — §5's table; (4) Simulation/Live Parity — §9's table; (5) Designer Asset Connectivity — §13 (delta against the pre-existing full audit at `docs/architecture/designer_asset_connectivity_audit.md`); (6) Operator Authority — §8's table; (7) Failure Propagation — §12's table; (8) Technical Debt/Over-Engineering — §14's table.

---

## 18. Recommended Next Development Priorities

Based only on the findings above, ranked by the audit's own stated priority order (broken core connectivity > correctness/safety gaps > state/timing problems > sim/live parity > physical CCTV readiness > new features):

1. ~~**Wire `build_live_runtime()` into an actual runnable application entry point.**~~ **RESOLVED** — see `docs/architecture/application_live_runtime_integration.md` (Application Live Runtime Launcher milestone). `designer/windows/main_window.py` now owns an explicit, opt-in `LiveRuntimePanel`/`LiveRuntimeController` (backed by the new `live_runtime_launcher/` package) that constructs a `Building` from the currently loaded `.syn` project, calls `build_live_runtime()`/`build_offline_demo_runtime()`, starts/stops the runtime, and opens `command_center.main_window.MainWindow` against the SAME `orchestrator`/`StateManager`/operator-action gateway — reachable from the real application, not merely a script, with zero change to `main.py` or default Designer behavior. 4303/4303 tests passing (4271 baseline + 32 new).
2. ~~**Reconcile or explicitly document the `BuildingState.zone_occupancy` vs. `CrowdIntelligenceSnapshot.occupant_count` duplication**~~ **RESOLVED** — see `docs/architecture/canonical_live_occupancy.md` (Canonical Live Occupancy Source of Truth milestone). `live_occupants.manager.LiveOccupantManager.canonical_occupancy()` is now the ONE per-cycle, memoized computation of "who currently occupies which zone"; `BuildingState.zone_occupancy` (via `live_perception.providers.LiveOccupantObservationProvider`), `CrowdIntelligenceSnapshot`, `EvacuationProgressSnapshot`, and `EmergencyResponseSnapshot` all read it, none independently re-derive it. Proven, in one test, on the same worked multi-camera example this audit's own §10 referenced: all four (plus AI's `total_occupant_count` and Command Center's live frame conversion) agree on the same headcount with the same occupant ids. 4318/4318 tests passing (4303 baseline + 15 new).
3. **Delete or explicitly mark deprecated the two dead gateway generations** found in §14 (`live_system/integration.py`'s Phase-7 gateways, and confirm whether `LiveAIInferenceGateway`'s concrete classes should gain a real default-construction path or be left fully caller-supplied) — reduces the maintenance surface a future contributor has to understand before touching AI/perception wiring, and removes the "two generations of the same seam" confusion.
4. **Decide AI's actual role before investing further in it.** §6 found AI is structurally wired but rank-inert by construction (an identical additive term across every candidate in a zone). Either (a) give AI a genuinely differentiating per-candidate signal if it's meant to influence ranking, or (b) explicitly re-scope it as "operator awareness only" in its own docstrings/UI labeling to match what it actually does today — continuing to build AI features on top of a rank-inert integration without deciding this first risks wasted effort.
5. **Close the offline full-chain E2E gap identified in §15** (as a future milestone, not this one) — no single test currently proves camera-to-operator-approval end to end; the closest test (`OfflineFullSystemDemonstrationTests`) is the natural place to extend, adding assertions for crowd/trajectory/progress/recommendation/guidance/signage rather than building a sixth parallel E2E test file.

**What SynEvac should STOP building for now:**
- **New Designer assets or fire-safety inventory items.** §13 confirms the toolbar is already asset-heavy relative to its evacuation-decision core; the Designer Asset Connectivity Audit's own verdict ("drifting toward a fire-safety asset modeler") stands unchanged.
- **New intelligence subsystems.** Crowd/Progress/Trajectory/Emergency-Response are already genuinely rank-affecting and well-consumed (§5) — a sixth intelligence engine would add more per-exit scoring terms to an already-multi-factor ranking function without first confirming the application that would ever display/act on it exists (see priority 1).
- **AI model retraining or new AI features**, until priority 4 above is resolved — retraining a rank-inert integration doesn't change its decision authority.
- **New hardware provider protocols** (voice/signage/building-control) — all three remain `SIMULATION`-only by design; this is correct given priority 1 is unresolved (no live application exists yet for a hardware provider to serve).

---

## 19. Regression

Baseline 4271/4271 confirmed at commit `a5f1ca0`. No production source was modified during this audit — only this new documentation file was added. Full-suite re-run recorded in the final report below.


