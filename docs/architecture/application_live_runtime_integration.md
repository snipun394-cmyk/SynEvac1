# Application Live Runtime Launcher

Status: **implemented, offline-tested.** Closes the highest-priority gap `docs/architecture/synevac_end_to_end_architecture_review.md` §18 identified: `live_runtime.factory.build_live_runtime()` existed, was fully tested, and was never reachable from any application a real operator would launch. Baseline before this milestone: 4271/4271 tests passing, commit `6243e24`.

## 1. Phase 1 — the BEFORE graph (re-confirmed, unchanged from the prior audit)

```
main.py -> core.app.SynEvacApp.__init__() -> designer.windows.main_window.MainWindow
           ** never imports live_runtime, never constructs a LiveRuntime,
              never opens Command Center **

command_center.main_window.MainWindow      -- a second, standalone QMainWindow,
           ** exposes enable_live_mode(data_source, gateway) but nothing in
              production ever calls it **

live_runtime.factory.build_live_runtime(building, **~30 optional kwargs)
           ** the only place a full LiveRuntime gets assembled -- called only
              from tests/*.py and scripts/run_physical_camera_validation.py /
              scripts/dry_run_physical_cctv.py **
```

`main.py` → `core/app.py::SynEvacApp.__init__()` constructs only `designer.windows.main_window.MainWindow` (`core/app.py:14`) and never imports `live_runtime`, `command_center.main_window`, or `command_center.dashboard` anywhere. `command_center/main_window.py::MainWindow.enable_live_mode()` is fully functional and already test-proven (`tests/test_live_command_center.py`), but has zero production callers. No existing application-mode abstraction, menu action, launcher, or CLI argument selects between "Designer" and "something live" anywhere in the codebase — there was nothing to reuse beyond the composition root itself and the Designer's own existing "lazily-constructed secondary tool window" convention (`designer/windows/main_window.py::open_campaign_studio()` / `CampaignController`, used for Scenario Campaign Studio).

## 2. Design decision: where "Live mode" lives

Two shapes were possible: (a) a second top-level application main.py could launch instead of Designer, selected by a CLI flag, or (b) an explicit, opt-in capability added to the *same* running Designer session. **(b) was chosen.** Reasons:

- Phase 11's own required test sequence (*launch Designer, load project, enter Live mode, start, run several cycles, open Command Center, stop, start again, stop again, close application*) describes one continuous application session, not a re-launch with a different CLI flag.
- Phase 2's own requirement that "DESIGNER: Current behavior. Must remain unchanged" is satisfied most literally by leaving `main.py`/`core/app.py` untouched — every existing Designer behavior, test, and script continues to work with zero modification.
- Phase 1's own explicit instruction — "do not create a parallel application framework if one already exists" — pointed directly at `designer/windows/main_window.py`'s existing lazy-secondary-window pattern (`CampaignController`/`CampaignWindow`) as the one to reuse, not invent a second one.

`main.py` is therefore **unchanged**. Application mode selection happens *inside* the running Designer session, through a new, hidden-by-default dock panel — never automatically, never merely by launching the app or opening a project.

## 3. The AFTER graph

```
main.py -> core.app.SynEvacApp -> designer.windows.main_window.MainWindow  (unchanged)
                                        |
                                        +-- self.live_runtime_panel     (designer/widgets/live_runtime_panel.py)
                                        +-- self.live_runtime_controller (designer/live_runtime_controller.py)
                                                  |
                                                  |  Start Live Runtime (explicit button click)
                                                  v
                                        live_runtime_launcher.session.LiveRuntimeSession(mode)
                                                  |
                                                  |  .construct(building)  -- building() OR build_offline_demo_runtime()
                                                  v
                                        live_runtime.runtime.LiveRuntime          <- SAME composition root,
                                              .orchestrator -> StateManager           unchanged, reused verbatim
                                              .command_center_data_source
                                              .operator_action_gateway
                                                  |
                                                  |  .start() / .stop()  (explicit button clicks)
                                                  |
                                                  |  Open Command Center (explicit button click)
                                                  v
                                        command_center.main_window.MainWindow.enable_live_mode(
                                            runtime.command_center_data_source,
                                            runtime.operator_action_gateway,
                                        )
```

`ApplicationMode` (`live_runtime_launcher/modes.py`) is `DESIGNER | LIVE | OFFLINE_DEMO`. `DESIGNER` is implicit and unchanged (the Designer canvas itself never constructs a `LiveRuntimeSession`). `LIVE`/`OFFLINE_DEMO` are selected via `LiveRuntimePanel.mode_combo`, defaulting to `LIVE`.

### New files

| File | Role |
|---|---|
| `live_runtime_launcher/modes.py` | `ApplicationMode`, `RuntimeLifecycleState` enums — vocabulary only, no logic. |
| `live_runtime_launcher/session.py` | `LiveRuntimeSession` — owns exactly one `LiveRuntime` (and, once opened, exactly one Command Center window) for its lifetime. `construct()`/`start()`/`stop()`/`open_command_center()`/`ai_status()`/`provider_capabilities()`/`shutdown()`. Calls `build_live_runtime()`/`build_offline_demo_runtime()` unchanged — no new composition logic, no new subsystem. |
| `designer/widgets/live_runtime_panel.py` | `LiveRuntimePanel` — dumb `QWidget` (mode combo, status/AI/capability labels, Start/Stop/Open Command Center buttons). Same "dumb widget, controller pushes updates in" convention as `CameraManagerPanel`/`SpeakerManagerPanel`. |
| `designer/live_runtime_controller.py` | `LiveRuntimeController` — mediator between the panel and `LiveRuntimeSession`, same role `CampaignController` already plays for Campaign Studio. The only class that touches `LiveRuntimeSession`'s public methods. |

### Changed files

`designer/windows/main_window.py`: constructs `live_runtime_panel`/`live_runtime_controller`, adds a hidden-by-default "Live Runtime" dock (View menu → "Live Runtime Panel", tabified alongside Camera Manager/Speaker Manager/Camera Validation — same convention, same default-hidden state), and adds three small integration points:

- `new_project()` / `open_project()` now call `live_runtime_controller.stop_and_reset()` before replacing `self.canvas.scene_obj.project` — the same "stop the loop before the project disappears" discipline `stop_simulation()` already applies to the Manual Simulation Sandbox, extended to the Live Runtime session.
- `closeEvent()` now calls `live_runtime_controller.shutdown()` before accepting the close — stops any running `LiveRuntime` and closes any open Command Center window, so neither survives as a zombie background timer/orchestrator after Designer itself closes.

No other file changed. `live_runtime/factory.py`, `live_runtime/runtime.py`, `live_system/*`, `command_center/*` are all reused verbatim — this milestone adds zero lines to any of them.

## 4. Safe startup semantics (Phase 3)

Constructing `MainWindow` constructs `LiveRuntimePanel`/`LiveRuntimeController` but **never** a `LiveRuntimeSession` — `LiveRuntimeController.session` is `None` until the operator clicks **Start Live Runtime** (`tests/test_application_live_runtime_launcher.py::PanelAndControllerWiringTests::test_constructing_the_controller_starts_nothing`). Loading a project (`open_project()`/`new_project()`) never touches the Live Runtime panel at all except to reset an already-running session — it never constructs or starts one. `LiveRuntimeSession.construct(building)` itself never calls `.start()` (mirrors `build_live_runtime()`'s own "construction performs zero network I/O" discipline one layer up); `CameraManager.discover_cameras()` only ever reads Building geometry.

This milestone wires **zero `frame_sources`** into either `LIVE` or `OFFLINE_DEMO` mode — no RTSP-from-Camera-asset builder exists anywhere in the codebase yet (confirmed by repo-wide search; building one would mean implementing a physical-hardware wiring path, explicitly out of scope). `runtime.frame_sources == {}` and `runtime.camera_pipeline is None` in every session this launcher constructs today, in both modes (`LiveRuntimeSessionLifecycleTests::test_zero_network_zero_hardware_by_default`). Camera capability therefore honestly reports `NO_PROVIDER` today regardless of mode — see §6.

## 5. `.syn` project → LiveRuntime (Phase 4)

`LiveRuntimeSession.construct(building)` takes whatever `Building` the Designer already has loaded — `self.canvas.scene_obj.project.building`, populated by the **existing**, unmodified `Serializer.load()` project loader (`designer/windows/main_window.py::open_project()`). No second serializer/parser was written. Every authored asset the loader already reconstructs (Zones, Doors, Exits, Stairs, Obstacles, Cameras, Smoke/Heat Detectors, MCPs, Speakers, Dynamic Signs, and the nine advanced fire-safety assets) survives unchanged into `build_live_runtime()`/`build_offline_demo_runtime()`, which already discover every one of these from a `Building` (`CameraManager.discover_cameras()`, `SensorManager.discover_sensors()`, `SpeakerManager.discover_speakers()`, `SignManager.discover_signs()`, etc. — all pre-existing, unmodified). The advanced fire-safety assets load and are visible in the resulting `LiveRuntime`'s own managers exactly as before; they remain Command-Center-display-only and irrelevant to this milestone, per the existing Designer Asset Connectivity Audit.

## 6. Runtime lifecycle & honest status (Phase 5/8/9)

`RuntimeLifecycleState`: `STOPPED | STARTING | RUNNING | DEGRADED | FAILED`. `STOPPED`/`RUNNING` mirror `LiveRuntime.is_running` directly — never a second, independently-tracked notion of "running." `DEGRADED` is reported only when a **configured** `CameraFrameSource` genuinely failed to reach `Online` (`CameraManager.connection_status(camera_id) != ONLINE`, checked per camera in `runtime.frame_sources`) — mechanically unreachable today given §4 (zero frame sources), proven correct in isolation against a stand-in runtime (`DegradedStateDetectionTests`), and forward-compatible the day a real camera wiring path exists. `FAILED` covers every other honest failure: no `Building` loaded, an exception during `factory(building)`, an exception during `runtime.start()`, or calling `start()` before `construct()` — none of these raise out of `LiveRuntimeSession`; the panel shows the error text instead (`LiveRuntimeSessionConstructionTests`).

**AI** (`session.ai_status()`): always `"NOT CONFIGURED"` — this launcher never constructs a `live_ai_gateway` (no default-construction helper exists anywhere in the repository, confirmed by the prior end-to-end audit). Honest and forward-compatible, never fabricated (`LiveRuntimeSessionCapabilityHonestyTests::test_ai_is_always_not_configured`).

**Provider capability** (`session.provider_capabilities()`), per channel:

| Channel | LIVE mode | OFFLINE_DEMO mode | Why |
|---|---|---|---|
| Camera | `NO_PROVIDER` | `NO_PROVIDER` | Zero `frame_sources` wired by this launcher in either mode (§4). |
| Voice | `NO_PROVIDER` | `SIMULATION` | `build_live_runtime()` supplies no `voice_output_provider`; `build_offline_demo_runtime()` defaults `SimulationVoiceOutputProvider()` in. |
| Dynamic Signage | `NO_PROVIDER` | `SIMULATION` | Same pattern, `SimulationDynamicSignageProvider()`. |
| Building Control | `NO_PROVIDER` | `SIMULATION` | Same pattern, `SimulationControlProvider()`. |

No physical Voice/Signage/Building-Control provider implementation exists anywhere in this codebase — `LIVE` mode's `NO_PROVIDER` result for all three is the accurate, undecorated truth, never labeled `LIVE` merely because a controller class exists (Phase 9's own explicit instruction).

## 7. Command Center shares the SAME runtime (Phase 6)

`LiveRuntimeSession.open_command_center()` constructs `command_center.main_window.MainWindow` **once** per session (lazily, on first Open Command Center click) and calls `enable_live_mode(self.runtime.command_center_data_source, self.runtime.operator_action_gateway)` — the exact two objects `self.runtime` already owns, never reconstructed. Proven by identity assertion through the real application entry path (not a hand-built test chain):

```python
command_center_window.live_data_source._state_manager is session.runtime.orchestrator.state_manager   # True
command_center_window.live_data_source is session.runtime.command_center_data_source                   # True
session.runtime.operator_action_gateway.voice_controller is session.runtime.voice_evacuation_controller # True
session.runtime.operator_action_gateway.control_controller is session.runtime.building_control_controller # True
```
(`tests/test_application_live_runtime_launcher.py::ApplicationLevelOfflineFullChainE2ETests::test_full_chain_reaches_command_center_through_the_same_instances`)

## 8. Failure isolation (Phase 10)

| Condition | Result |
|---|---|
| No project loaded (`building=None`) | `construct()` sets `state=FAILED`, honest `last_error`, `runtime=None` — no exception raised. |
| Empty Building (0 floors) | Constructs successfully — every manager's own `discover_*()` simply finds nothing (pre-existing `build_live_runtime()` behavior, unchanged). |
| `start()` called before `construct()` | `state=FAILED`, honest `last_error`, no exception. |
| Exception during `factory(building)` | Caught, `state=FAILED`, `runtime=None`, `last_error=str(exc)`. |
| Exception during `runtime.start()` | Caught (also handled one layer down by `LiveRuntime.start()`'s own rollback), `state=FAILED`, `last_error=str(exc)`. |
| No cameras / camera configured but no frame source / one or all cameras failing / no FACP / no AI / no speakers / no signs / no building-control provider | All pre-existing `build_live_runtime()` honest-degradation behavior (`docs/architecture/synevac_end_to_end_architecture_review.md` §12), unchanged and untouched by this launcher — this milestone adds no new failure path here, only a caller that can actually reach the existing ones. |

The application itself (Designer, the panel, the controller) never crashes under any of the above — every case resolves to a `FAILED`/error-labeled state, never an unhandled exception surfacing to the GUI.

## 9. Start/stop lifecycle (Phase 11)

`start()`/`stop()` never reconstruct the underlying `LiveRuntime` — the *same* instance is reused across a `start → stop → start → stop` cycle (`LiveRuntimeSessionLifecycleTests::test_start_stop_restart_reuses_the_same_runtime_instance`), so Command Center keeps looking at the same `orchestrator`/`StateManager` throughout. Double-`start()` and double-`stop()` (including `stop()` before any `start()`) are both safe no-ops, mirroring `LiveRuntime`'s own existing guarantees one layer down. Switching `ApplicationMode` while **stopped** discards the old session (forcing a fresh `construct()` on the next Start, so an `OFFLINE_DEMO` session's `Simulation*` providers are never silently reused under a `LIVE` label); switching while **running** is a no-op on the session (the operator must Stop first) — both proven in `PanelAndControllerWiringTests`. Closing Designer (`closeEvent`) or loading a different project (`new_project()`/`open_project()`) both call through to `LiveRuntimeSession.shutdown()`, which stops the runtime and closes any open Command Center window — no zombie frame source, orchestrator, or Command Center refresh timer survives either event (`ProjectLifecycleResetTests`).

## 10. Application-level offline E2E (Phase 12) & operator output E2E (Phase 13)

`tests/test_application_live_runtime_launcher.py::ApplicationLevelOfflineFullChainE2ETests` exercises the **application entry path** — a real `MainWindow`, its real `live_runtime_panel`/`live_runtime_controller` — never `build_live_runtime()` called directly:

1. `MainWindow()` constructed, a Designer-shaped `Project`/`Building` assigned (the same shape `Serializer.load()` would produce).
2. Offline Demo mode selected, **Start Live Runtime** clicked → `LiveRuntimeSession.construct()` + `.start()`.
3. `run_cycle()` called directly on `session.runtime` (standing in for the orchestrator's own timer-driven cadence, unchanged) → `BuildingState`, Crowd/Evacuation-Progress/Trajectory/Emergency-Response intelligence, Recommendation, Guidance, Dynamic Signage all populate `StateManager`, reachable through `command_center_data_source.current_snapshot()`.
4. **Open Command Center** clicked → real `command_center.main_window.MainWindow`, `enable_live_mode()` called with the session's own objects (§7).
5. Nothing dispatches automatically: `voice_evacuation_controller.broadcast_log`, `building_control_controller.all_requests()`, and `operator_action_gateway.all_signage_instructions()` are all empty immediately after start + several cycles (`test_nothing_auto_dispatches_before_operator_approval`).
6. One Voice announcement, one Building Control recommendation, and one Dynamic Signage instruction are each approved through `session.runtime.operator_action_gateway` — the SAME gateway `open_command_center()` handed to the Command Center window, never a test-only replacement controller (`test_operator_approves_voice_signage_and_building_control_through_the_same_controllers`).
7. Under `LIVE` mode (no Simulation providers), the same gateway's `voice_controller`/`control_controller`/`signage_controller` are all `None` — approval is honestly unavailable, never silently routed anywhere (`test_live_mode_leaves_every_output_channel_at_no_provider`).

## 11. Authority guards (Phase 14) & network guards (Phase 15)

- `live_runtime_launcher/session.py` never imports `advisory_system.orchestrator`, `ai_registry`, `ai_inference`, `ai_training`, or `decision_policy` (regex-scanned, mirroring the pre-existing `tests/test_live_runtime_architecture_guards.py::GatewayIsTheOnlyExecutionSeamTests` convention).
- `LiveRuntimeSession` never supplies a `live_ai_gateway`/`live_advisory_gateway` — `runtime.orchestrator.live_ai_gateway`/`.live_advisory_gateway` are `None` in every session this launcher constructs, structurally, not merely by test luck.
- Every `approve_voice_message()`/`approve_control_request()`/`approve_signage_instruction()` call in §10 required an explicit call in test code standing in for an explicit operator click — nothing this launcher added can auto-approve, auto-broadcast, auto-dispatch, auto-execute, auto-reset FACP, or alter Decision Policy/Pathfinding. The launcher composes (`construct`/`start`/`stop`/`open_command_center`); it decides nothing.
- **Network**: Designer mode → zero network (unchanged — no camera/RTSP code runs at all outside a `LiveRuntimeSession`). `LiveRuntimeController`/`LiveRuntimePanel` construction (i.e., simply launching Designer) → zero network (`session is None` until Start is clicked). Offline Demo mode → zero network (`ReplayFrameSource`/`Simulation*` providers only, and in fact zero frame sources at all this milestone, §4). Loading a project containing Live cameras → zero network, because this launcher does not yet wire any camera into either mode's `frame_sources` at all (§4) — the existing CCTV safety behavior (`config != connection`, `RTSPFrameSource.start()` is the one explicit network step) is untouched, not weakened.

## 12. Full regression (Phase 17)

Baseline 4271/4271 (commit `6243e24`). This milestone added one new test module, `tests/test_application_live_runtime_launcher.py` (32 tests). Full suite after this milestone: **4303/4303 passing**, zero regressions.

## 13. End-to-end audit gap status

`docs/architecture/synevac_end_to_end_architecture_review.md` §18 priority 1 — *"Wire `build_live_runtime()` into an actual runnable application entry point"* — is **RESOLVED**. `main.py` → Designer → an explicit, opt-in Live Runtime panel now reaches a real `LiveRuntime`, a real Command Center sharing that same runtime, with the operator-approval boundary structurally intact. What remains open (unchanged, explicitly out of scope this milestone, tracked separately): real camera/RTSP wiring from Designer-authored Camera assets into `frame_sources` (§18 priority 5's physical-CCTV-readiness item), the `BuildingState.zone_occupancy` vs. `CrowdIntelligenceSnapshot` duplication (§18 priority 2), the two dead gateway generations in `live_system/integration.py` (§18 priority 3), and AI's rank-inert integration (§18 priority 4) — none of these were touched by this milestone, per its own explicit scope boundary.
