# Production Live Runtime Composition Root

Status: **implemented, offline-tested. No production entry point previously assembled the live SynEvac system; `live_runtime/` is the first one.** This document records Phase 1's investigation findings, the composition root built on top of them, and the four graphs Phase 10 requires.

## 1. Phase 1 investigation findings

**The one-sentence answer, re-confirmed:** `main.py` → `core.app.SynEvacApp.__init__()` → `designer.windows.main_window.MainWindow` is the *only* production entry point in this repository, and it launches the Designer — nothing downstream of it imports `live_system`, constructs a `LiveOrchestrator`, or assembles `CameraManager`+`SensorManager`+`MultiCameraFusionEngine`+`SimulatedFACP`+`LiveOperatorActionGateway` together. `designer/windows/main_window.py` owns one `CameraManagerPanel` (Digital Twin *configuration*, not a running pipeline) and nothing else live-shaped. `command_center/main_window.py` is a second, separate `QMainWindow` with zero production callers of its own (only `tests/test_command_center.py`/`tests/test_live_command_center.py`).

1. **What happens when SynEvac starts?** `SynEvacApp.__init__()` builds a `QApplication` and the Designer's `MainWindow`; `.run()` shows it and enters the Qt event loop. No live runtime of any kind starts.
2. **Which object owns application lifetime?** `core.app.SynEvacApp` (`self.app = QApplication(...)`, `sys.exit(self.app.exec())`).
3. **How are the Designer and Command Center launched?** Designer: via `main.py` as above. Command Center: **not launched by any production code path at all** — only ever constructed inside tests.
4. **Does `LiveOrchestrator` have a real production construction site?** No — confirmed by grep, `LiveOrchestrator(` appears only in `tests/test_live_system.py` and `tests/test_live_command_center.py::_LiveChain` (a test harness).
5. **Are `CameraManager`, `SensorManager`, `BuildingStateEstimator`, `MultiCameraFusionEngine`, `SimulatedFACP`, `RegistryLiveAIInferenceGateway`, `ReplayCompatibleAdvisoryGateway`, `LiveCommandCenterDataSource`, and `LiveOperatorActionGateway` ever assembled together outside tests?** No. `_LiveChain` (test-only) was the closest existing precedent, and even it bypasses `CameraManager`'s own detection-routing (`register_detection_provider`/`set_camera_mode`/`all_detections`) in favor of calling the detection provider directly — a shortcut appropriate for a test fixture, not a production composition root.
6. **Which components require Building/Project data?** `CameraManager.discover_cameras(building)`, `SensorManager.discover_sensors(building)`, `SpeakerManager.discover_speakers(building)`, `BuildingControlController(building, provider)`, `LiveCommandCenterDataSource(..., building=building)` — all Digital Twin–asset-driven, all take the *same* `Building` instance in `live_runtime`.
7. **Which components require trained AI model registry access?** Only `ai_registry.LiveAIInferenceService`/`ModelRegistry`, wrapped by `RegistryLiveAIInferenceGateway` — entirely optional and constructed by the *caller*, never by `live_runtime` itself (§5 below).
8. **Which components require optional providers?** Voice (`VoiceOutputProvider` → `VoiceEvacuationController`), Building Control (`BuildingControlProvider` → `BuildingControlController`), AI (`live_ai_gateway`), Advisory (`live_advisory_gateway`) — all `None`-able; `NO_PROVIDER`/absent-gateway is a valid, fully-supported configuration everywhere.
9. **Which components currently create their own duplicate manager/controller instances?** None in production (none exist in production at all yet) — the risk `live_runtime` guards against is a *future* one: two different call sites each constructing their own `CameraManager`, or Command Center holding a different `VoiceEvacuationController` than the one operator actions actually dispatch through. See §4.
10. **Which objects must have exactly one shared instance during a live session?** `CameraManager`, `SensorManager`, `MultiCameraFusionEngine` (holds cross-cycle `TrackHistory` state — reconstructing it per cycle would silently break handover tracking), `SimulatedFACP`, `StateManager` (owned by `LiveOrchestrator`), `VoiceEvacuationController`, `BuildingControlController`. All seven are constructed exactly once per `LiveRuntime` and threaded everywhere they are needed — see §4/Phase 4 tests.

## 2. The composition root: `live_runtime/`

Two files, ~380 lines combined, zero new subsystem logic:

- **`live_runtime/runtime.py` — `LiveRuntime`.** A plain container holding references to already-existing components (`building`, `camera_manager`, `sensor_manager`, `fusion_engine`, `facp`, `speaker_manager`, `frame_sources`, `camera_pipeline`, `orchestrator`, `command_center_data_source`, `operator_action_gateway`, `voice_evacuation_controller`, `building_control_controller`) plus `start()`/`stop()`/`is_running`/`run_cycle(time)`. Imports nothing but `typing` — mechanically enforced (`tests/test_live_runtime_architecture_guards.py::GatewayIsTheOnlyExecutionSeamTests`).
- **`live_runtime/factory.py` — `build_live_runtime(building, **collaborators)` / `build_offline_demo_runtime(building, **collaborators)`.** The one place every component above gets constructed and wired together, reusing each package's own existing constructor unchanged. `build_offline_demo_runtime` is Phase 3's named "OFFLINE LIVE DEMO MODE" entry point — the same function with `SimulationVoiceOutputProvider`/`SimulationControlProvider` defaulted in.

**Nothing is duplicated.** `EstimatorBuildingStateGateway`, `LiveOrchestrator`, `LiveCameraPipeline`, `VoiceEvacuationController`, `BuildingControlController`, `LiveCommandCenterDataSource`, `LiveOperatorActionGateway` are all reused verbatim; `live_runtime`'s only original code is the wiring closures (`fusion_result_provider`, `facp_snapshot_provider`, `smoke_detector_status_provider`, `heat_detector_status_provider`) and the lifecycle methods.

## 3. Runtime ownership graph (production)

```
build_live_runtime(building, frame_sources=..., human_detector=..., identity_resolver=...,
                    voice_output_provider=..., building_control_provider=...,
                    live_ai_gateway=..., live_advisory_gateway=...)
    |
    v
LiveRuntime
    .building                          <- ONE Building, shared everywhere below
    .camera_manager      (CameraManager)            \
    .sensor_manager       (SensorManager)             \  Digital Twin / asset layer
    .fusion_engine  (MultiCameraFusionEngine)          /  (Phase 4: exactly one each)
    .facp                (SimulatedFACP, optional)    /
    .speaker_manager      (SpeakerManager)            /
    .frame_sources     {camera_id: CameraFrameSource} <- caller-supplied, never auto-created
    .camera_pipeline      (LiveCameraPipeline, optional)
    .orchestrator          (LiveOrchestrator)  ---owns--->  StateManager (the ONE canonical
    |                                                        LiveBuildingSnapshot/BuildingState)
    .command_center_data_source (LiveCommandCenterDataSource)  <- reads orchestrator.state_manager
    .operator_action_gateway    (LiveOperatorActionGateway)
    .voice_evacuation_controller (VoiceEvacuationController, optional)   \ same instances as
    .building_control_controller (BuildingControlController, optional)  / operator_action_gateway
```

`runtime.command_center_data_source` is handed to `command_center.main_window.MainWindow.enable_live_mode(data_source, operator_action_gateway=runtime.operator_action_gateway)` — the *same* two objects a real deployment's UI layer would use, never reconstructed.

## 4. Shared-instance ownership (Phase 4, test-proven)

`tests/test_live_runtime.py::SharedInstanceOwnershipTests` proves, by `assertIs`:

- `operator_action_gateway.voice_controller is runtime.voice_evacuation_controller`
- `operator_action_gateway.control_controller is runtime.building_control_controller`
- `command_center_data_source._state_manager is orchestrator.state_manager`
- `command_center_data_source._building is runtime.building`
- `voice_evacuation_controller._speaker_manager is runtime.speaker_manager`
- `building_control_controller._building is runtime.building`
- `camera_pipeline.frame_sources[camera_id] is runtime.frame_sources[camera_id]` for every camera
- `MultiCameraFusionEngine` (which holds cross-cycle `TrackHistory`) is the *same* object across multiple `run_cycle()` calls, never reconstructed per cycle
- A caller-supplied `CameraManager`/`SensorManager`/`MultiCameraFusionEngine` is reused as-is, never silently duplicated

`LiveOperatorActionGateway` gained two small, read-only accessors (`voice_controller`, `control_controller`) to make this provable — the same "expose what is already stored, add no new logic" precedent its own `VoiceEvacuationController.provider`/`BuildingControlController.provider` properties already established.

## 5. Lifecycle (Phase 5, test-proven)

`LiveRuntime.start()`:
1. Starts every configured `CameraFrameSource` individually, each wrapped in its own `try/except` — one failed camera never blocks another or the orchestrator (`RTSPFrameSource`/`ReplayFrameSource` already convert connection failures into honest status internally and never raise; this is defense in depth).
2. Starts `LiveOrchestrator` (which itself raises `LiveSystemAlreadyRunningError` on a double call — `LiveRuntime` is never modified to change this; instead `LiveRuntime`'s own `_running` flag makes `LiveRuntime.start()` idempotent one layer up).
3. Starts `LiveCommandCenterDataSource`, if configured.
4. If step 2 or 3 raises unexpectedly, every camera source started in step 1 is rolled back (stopped) and the orchestrator is stopped if it managed to start, before re-raising — never a half-started runtime.

`LiveRuntime.stop()` always runs through every component's own `stop()`, each individually wrapped so one failing component never blocks the others, and unconditionally ends with `is_running = False`.

**Proven:** repeated `start()` is a safe no-op; repeated `stop()` is safe, including before any `start()`; one failed camera does not prevent others or the orchestrator from starting; a core-infrastructure startup failure rolls back cleanly; a component failing during shutdown does not block the rest.

## 6. Command Center connection (Phase 6)

`Command Center never constructs AI models, `BuildingStateEstimator`, `CameraManager`, `SensorManager`, `FACP`, or `AdvisoryOrchestrator`` — mechanically unchanged from the prior milestone's `CommandCenterLiveIntegrationGuardTests`, re-verified here (`tests/test_live_runtime_architecture_guards.py::CommandCenterPanelsStayCleanTests`) to additionally confirm no Command Center panel imports `live_runtime` either — a panel only ever needs the already-constructed `LiveCommandCenterDataSource`/`LiveOperatorActionGateway` objects handed to it.

## 7. Offline live-demo graph (Phase 7, test-proven — `tests/test_live_runtime_e2e.py`)

```
Demo Building (2 zones, 2 cameras, 1 smoke + 1 heat detector, 2 speakers, 1 door, 1 exit, 1 stair)
    -> CameraManager.discover_cameras()                                   [IMPLEMENTED, OFFLINE TESTED]
    -> ReplayFrameSource x2 -> LiveCameraPipeline -> MockHumanDetector
       -> MappingIdentityResolver -> Detection                            [IMPLEMENTED, OFFLINE TESTED]
    -> CameraManager.all_detections() -> MultiCameraFusionEngine.fuse()   [IMPLEMENTED, OFFLINE TESTED]
    -> SensorManager status (bookkeeping) + SimulatedFACP (optional)      [IMPLEMENTED, OFFLINE TESTED]
    -> EstimatorBuildingStateGateway -> BuildingState                     [IMPLEMENTED, OFFLINE TESTED]
    -> Live AI (unconfigured in this demo -- optional)                    [IMPLEMENTED, OFFLINE TESTED
                                                                             elsewhere; omittable]
    -> ReplayCompatibleAdvisoryGateway -> AdvisoryReport                  [IMPLEMENTED, OFFLINE TESTED]
    -> LiveCommandCenterDataSource -> CommandCenterSnapshot               [IMPLEMENTED, OFFLINE TESTED]
    -> Operator reviews civilian_announcements/building_recommendations  [IMPLEMENTED, OFFLINE TESTED]
    -> Operator approves ONE voice message
       -> LiveOperatorActionGateway.approve_voice_message()
       -> VoiceEvacuationController.broadcast()
       -> SimulationVoiceOutputProvider                                  [IMPLEMENTED, OFFLINE TESTED]
    -> Operator approves ONE building-control request
       -> LiveOperatorActionGateway.approve_control_request()
       -> BuildingControlController.approve()
       -> SimulationControlProvider                                      [IMPLEMENTED, OFFLINE TESTED]
```

Zero automatic execution before either operator action; zero network access; zero hardware access anywhere in this chain (structurally proven — every collaborator is `ReplayFrameSource`/`MockHumanDetector`/`MappingIdentityResolver`/`SimulationVoiceOutputProvider`/`SimulationControlProvider`, none of which import a networking or computer-vision library, `tests/test_no_cv_dependencies.py`).

## 8. Future physical CCTV graph (not implemented this milestone)

```
Digital Twin Camera (Camera.id, unchanged identity)
    -> CameraManager                                                     [IMPLEMENTED]
    -> RTSPFrameSource                                                   [IMPLEMENTED, OFFLINE TESTED
                                                                             via FakeRTSPBackend only]
    -> FrameDecoderBackend (real implementation)                         [REQUIRES REAL CV MODEL /
                                                                             REQUIRES PHYSICAL CCTV]
    -> HumanDetector (real YOLO/tracking implementation)                 [REQUIRES REAL CV MODEL]
    -> IdentityResolver / ReID (real cross-camera implementation)        [REQUIRES REAL CV MODEL]
    -> MultiCameraFusionEngine                                           [IMPLEMENTED, unchanged]
    -> BuildingState                                                     [IMPLEMENTED, unchanged]
```

`live_runtime.factory.build_live_runtime()` accepts `frame_sources`/`human_detector`/`identity_resolver` as plain, duck-typed constructor arguments — swapping `ReplayFrameSource` for a real `RTSPFrameSource` (with a real `FrameDecoderBackend`), or `MockHumanDetector`/`MappingIdentityResolver` for real implementations, requires **zero changes to `live_runtime/` itself**. `CameraManager.register_detection_provider(DeviceMode.LIVE, ...)` + `set_camera_mode(camera_id, DeviceMode.LIVE)` is exactly the production routing path documented in `docs/architecture/cctv_integration_readiness.md` — `live_runtime` uses it directly, not a test-only shortcut, so the Digital Twin `Camera.id` remains the permanent identity across any future physical camera replacement, exactly as that document's own camera-replacement test already proves at the `CameraManager` layer.

## 9. Output graph (Phase 10D)

```
BuildingState                                                            [IMPLEMENTED]
    -> Live AI (RegistryLiveAIInferenceGateway, optional)                [IMPLEMENTED, OFFLINE TESTED]
    -> Advisory (ReplayCompatibleAdvisoryGateway, optional)              [IMPLEMENTED, OFFLINE TESTED]
    -> Live Command Center (LiveCommandCenterDataSource)                 [IMPLEMENTED, OFFLINE TESTED]
    -> Human Approval (VoiceEvacuationPanel/BuildingControlsPanel)       [IMPLEMENTED, OFFLINE TESTED]
    -> LiveOperatorActionGateway                                        [IMPLEMENTED, OFFLINE TESTED]
    -> VoiceEvacuationController / BuildingControlController             [IMPLEMENTED, unchanged]
    -> SimulationVoiceOutputProvider / SimulationControlProvider         [IMPLEMENTED, OFFLINE TESTED]
    -> Future Physical System (real VoiceOutputProvider/                 [REQUIRES HARDWARE PROVIDER]
       BuildingControlProvider)
```

## 10. Architecture guards (Phase 9, `tests/test_live_runtime_architecture_guards.py`)

Mechanically proven, by source-text regex scan:

- `advisory_system/`, `decision_policy/`, `ai_registry/`, `ai_inference/`, `live_system/live_ai_gateway.py`, `live_system/live_advisory_gateway.py`, and (new for this milestone) `facp/` never import `voice_evacuation`, `speaker_manager`, `building_control.controller`, `building_control.providers`, `command_center.live_operator_action_gateway`, or `live_runtime` itself.
- `live_system/orchestrator.py` never imports any of those either — re-stated explicitly on top of the pre-existing, unmodified `LiveSystemPackageDependencyDirectionTests` package-wide sweep.
- Every Live Command Center file (`building_controls_panel.py`, `recommendation_center.py`, `dashboard.py`, `main_window.py`, `data_source.py`) never imports an execution-capable module directly, and never imports `live_runtime` either.
- `live_runtime/factory.py` never imports `advisory_system.orchestrator`, `ai_registry`, `ai_inference`, `ai_training`, or `decision_policy` — every AI/Advisory collaborator is accepted as an opaque, already-constructed gateway object.
- `live_runtime/runtime.py` imports no concrete collaborator class at all, only `typing` — composition and lifecycle only, per Phase 2.

## 11. What remains out of scope (unchanged, not attempted this milestone)

Real `FrameDecoderBackend`, real `HumanDetector` (YOLO), real `IdentityResolver`/ReID, real PA/BMS/FACP hardware protocols, AI/RL retraining, a second `BuildingState`, duplicate managers/controllers, autonomous AI execution. `live_runtime` makes zero network calls and touches zero hardware anywhere in its own code or tests.
