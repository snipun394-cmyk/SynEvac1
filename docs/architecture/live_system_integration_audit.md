# Live SynEvac System — End-to-End Integration Audit

Status: **investigation only, no production code changed.** This document answers one question: if SynEvac were connected to a real building tomorrow, does one coherent live runtime actually connect the platform's subsystems together? It does not. Every subsystem below is individually implemented and (mostly) individually well-tested. What does not exist is a production composition root that wires them into a running cycle. This audit traces exactly where that breaks down, subsystem by subsystem, with file:line evidence, then proposes the smallest integration path that does not require redesigning any already-working package.

**The one-sentence answer:** `live_system.orchestrator.LiveOrchestrator` — the class whose own docstring says it is "the one object a real deployment... constructs and owns" — is constructed **nowhere in production code**. Grep confirms `LiveOrchestrator(` appears only in `tests/test_live_system.py`. `main.py` → `core/app.py` → `designer/windows/main_window.py` (the actual application entry point) never imports `live_system` at all. The live runtime is a fully-built, fully-tested library that nothing calls.

## 1. The exact current live runtime graph

There is not one graph — there are **two independent, non-intersecting graphs**, plus a third replay-only path Command Center actually uses. None of the three feed each other.

### Graph A — the proven offline CCTV chain (from the CCTV Pipeline milestone)

```
Camera(id) → CameraManager → ReplayFrameSource → CameraFrame                     ✅ proven (offline)
    → HumanDetector → IdentityResolver → Detection                               ✅ proven (offline)
    → CameraManager.detections_for_camera()                                      ✅ proven (offline)
    → MultiCameraFusionEngine.fuse()                                             ✅ proven (offline)
    → BuildingStateEstimator.estimate() → BuildingState                          ✅ proven (offline)
    → AI Inference                                                               ❌ nothing consumes BuildingState
    → Advisory System                                                            ❌ nothing consumes BuildingState
    → Command Center                                                             ❌ nothing consumes BuildingState
```

This chain is real, tested (2577/2577 suite), and exactly what the CCTV milestone proved. It simply **terminates at `BuildingState`** — nothing downstream reads that object in production code. `grep -r building_state` inside `live_system/`, `advisory_system/`, `ai_inference/`, `decision_policy/`, and `command_center/` returns **zero matches in all five packages**.

### Graph B — `live_system`'s own orchestrator (architecturally complete, never constructed)

```
LIVE INPUTS (SensorRegistry: CCTVSensor/SmokeDetectorSensor/HeatDetectorSensor/
             FireAlarmControlPanelSensor — all abstract, read() raises
             NotImplementedError; zero concrete subclasses outside tests)        ❌ no concrete sensor exists
    ↓
SensorRegistry.read_all() → SensorFusionPerceptionGateway.collect()
    → perception.fusion.sensor_fusion.SensorFusion.fuse() → BuildingObservation  ✅ wiring exists, real, tested
    ↓
StateManager.update_perception() → LiveBuildingSnapshot                          ✅ wiring exists, real, tested
    ↓
PredictorAIInferenceGateway.predict() → ai_inference.Predictor                   ⚠️ interface real; feature_row_builder
                                                                                     never implemented outside tests
    ↓
GeneratePolicyDecisionPolicyGateway.evaluate() → decision_policy.generate_policy ⚠️ interface real; decision_inputs_builder
                                                                                     never implemented outside tests
    ↓
RecommendationBuilder (a plain injected callable)                                ⚠️ slot exists; no production implementation
    ↓
advisory_system                                                                  ❌ live_system never imports advisory_system
    ↓
Voice Evacuation / Building Controls                                             ❌ never invoked from live_system
    ↓
DashboardCommandCenterGateway.notify() → command_center.dashboard.Dashboard      ✅ adapter is real and tested —
                                                                                     but its only caller (LiveOrchestrator)
                                                                                     is never constructed in production   ❌
```

Every ✅ in Graph B is proven **only under `tests/test_live_system.py`'s own stub gateways and fixed sensors** (`_FixedCCTVSensor`, `_StubAIInferenceGateway`, etc. — `tests/test_live_system.py:93-129, 822-860`). No concrete `Sensor` subclass, `feature_row_builder`, or `decision_inputs_builder` exists anywhere outside that test file.

### Graph C — what Command Center actually runs today (replay only)

```
Saved .syn project + Scenario + optional GroundTruth/DecisionPolicy/Timeline    (files on disk, offline pipeline output)
    ↓
command_center.incident_data.load_incident() → IncidentData                     ✅ real, tested
    → AdvisoryOrchestrator (fed AdvisoryInputs built from Building/Scenario/
      GroundTruth/DecisionPolicy — all offline artifacts)                       ✅ real, tested, replay-only
    → AdvisoryReport per frame → _build_voice_broadcasts()/_build_control_
      requests() (command_center/incident_data.py)                             ✅ real, tested, replay-only
    ↓
command_center.main_window.MainWindow._on_playback_tick() (a QTimer, 200ms)
    → Dashboard.set_frame_index()/show_frame(IncidentFrame)                     ✅ real, tested — but driven
                                                                                     exclusively by the replay clock
```

`command_center/main_window.py`'s own docstring states this plainly: *"This is a visualization layer only — it never runs a simulation, never generates a scenario, and never recomputes GroundTruth/DecisionPolicy. Loading an incident only ever reads already-written files."* Additionally, **`command_center.main_window.MainWindow` itself has zero production callers anywhere in the repo** — not even `main.py` launches it. It is a second, standalone Qt application, distinct from the Designer's `MainWindow`, invoked only by `tests/test_command_center.py`.

**Graphs A, B, and C never intersect.** No code path carries a value from Graph A's `BuildingState`, or Graph B's `LiveBuildingSnapshot`, into Graph C's `AdvisoryInputs`/`IncidentData`.

## 2. Is `BuildingState` the canonical live state, or isolated?

**Isolated.** Concretely:

1. **Who constructs `BuildingState` in production code?** `building_state/estimator.py` (the estimator's own `.estimate()`), `designer/building_state_debug_runner.py:112,182`, and `scripts/benchmark_live_camera_pipeline.py:77` (a manual benchmark script). Every other constructor site (11+) is inside `tests/`.
2. **How frequently is it updated?** `BuildingStateDebugRunner.run()` is invoked from `designer/windows/main_window.py`'s `_refresh_building_state_debug_panel()`, itself called from `on_simulation_tick()` — driven by `self.simulation_timer`, a 50ms `QTimer` (`main_window.py:130-132`). So it *is* continuously recomputed — but only while the Designer's own occupant-simulation sandbox is running, and only to refresh a debug panel. Nothing downstream consumes that recomputed value.
3. **Does `live_system.StateManager` know about `BuildingState`?** No. `StateManager` (`live_system/state_manager.py`) only ever constructs `LiveBuildingSnapshot`, wrapping `BuildingObservation` — a different type from a different package (`perception/`), never `building_state.models.BuildingState`.
4. **Does Advisory System consume `BuildingState`?** No — zero references anywhere in `advisory_system/`.
5. **Does AI Inference consume `BuildingState`?** No — zero references anywhere in `ai_inference/`.
6. **Does Decision Policy consume `BuildingState`?** No — zero references anywhere in `decision_policy/`.
7. **Does Command Center consume `BuildingState`?** No — zero references anywhere in `command_center/`. Command Center consumes `IncidentData`/`IncidentFrame`, built from `GroundTruth`/`Scenario`/`DecisionPolicy` instead.
8. **One canonical state object, or multiple parallel ones?** **Four parallel, non-communicating representations of "current building status" exist:**
   - `BuildingState` (`building_state/estimator.py`) — built from Camera+Sensor+FACP fusion; consumed only by the Designer's own debug panel.
   - `LiveBuildingSnapshot` (`live_system/state_manager.py`) — built from `SensorFusion`'s `BuildingObservation`; consumed only by `LiveOrchestrator`, itself never constructed in production.
   - `BuildingObservation` (`perception/`) — the fusion output that feeds `LiveBuildingSnapshot`; a different fusion pipeline from the one that feeds `BuildingState`.
   - `GroundTruth`/`Scenario`/`DecisionPolicy` (offline simulation artifacts) — feed `AdvisoryInputs` in Command Center's replay path; the only "state" Advisory/Voice/Building-Control/Command-Center actually see.

None of the four ever appears as an input to the construction of any of the other three.

## 3. Camera → Live System

The offline CCTV chain (Camera → CameraFrame → Detection → `MultiCameraFusionEngine` → `BuildingState`) is real and proven (§1, Graph A). Whether `live_system.Orchestrator` invokes it:

- **Is `CameraManager` instantiated by Live System?** No. `grep CameraManager\(` across the repo, excluding tests/docs/scripts, hits only `designer/widgets/camera_manager_panel.py` (the Designer's own management UI) and `designer/building_state_debug_runner.py`. `live_system/*.py` never imports `camera_manager` (also enforced by `live_system`'s own dependency-direction test).
- **Are detections collected each live cycle?** No — `live_system`'s `SensorFusionPerceptionGateway` reads from `SensorRegistry`/`CCTVSensor` (Graph B), never from `CameraManager`/`live_camera_pipeline` (Graph A). These are two structurally different "camera integration" designs that were never reconciled.
- **Does fusion run each live cycle?** `MultiCameraFusionEngine` runs only inside the offline-proven chain (tests + benchmark script), never inside `LiveOrchestrator.run_cycle()`.
- **Do fused tracks reach the live state?** No — `LiveBuildingSnapshot` has no field populated from `MultiCameraFusionEngine`'s `FusedTrack` output.
- **Does camera online/offline status reach the live state?** No — `CameraManager.connection_status()` is never read by `live_system` or `BuildingState`.

**Verdict: ⚠️ both halves exist (a proven CCTV pipeline, and a live orchestrator with a camera-sensor seam) but no caller connects them.** This is a pre-existing architectural duplication, already flagged in `docs/architecture/cctv_integration_readiness.md` §2 as out of scope for the CCTV milestone — this audit confirms its exact shape.

## 4. Detectors → FACP → Live System

```
SmokeDetector/HeatDetector → SensorManager        ✅ (Designer-only: sensor_manager/manager.py, discover_sensors())
    → DetectorState → SimulatedFACP                ✅ (Designer-only: facp/engine.py, constructed only at
                                                        designer/building_state_debug_runner.py:113)
    → BuildingState (facp_status field)             ✅ (additive, proven)
    → Live System                                   ❌ (never reaches live_system/, advisory_system/,
                                                        ai_inference/, or command_center/)
```

- `sensor_manager/manager.py`'s `SensorManager.discover_sensors()` walks canonical `Floor.smoke_detectors`/`heat_detectors` plus legacy `Floor.detectors`, converting legacy ones via `models/detector_migration.py:adapt_legacy_detector()`. It's pure bookkeeping/routing — never itself computes hazard state — and its only non-test constructor is inside `BuildingStateDebugRunner`.
- The FACP class is `SimulatedFACP` (`facp/engine.py`), with `PanelState`/event log/alarm/fault/acknowledge/silence/reset all implemented and tested (`tests/test_facp.py`). Its **only** non-test constructor anywhere in the repo is `designer/building_state_debug_runner.py:113` — the same Designer-sandbox-driven 50ms `QTimer` loop as `BuildingState` itself (§2, question 2).
- `live_system.sensor_registry.FireAlarmControlPanelSensor`/`FACPReading` (`live_system/sensor_registry.py:73,93`) are a **completely separate, never-implemented abstraction** — `live_system/` never imports the real `facp/` package at all. Two independent "FACP integration" concepts exist in the codebase, exactly mirroring the two independent "camera integration" concepts in §3.
- FACP/detector alarm state is unreachable from `advisory_system/`, `ai_inference/`, and `command_center/` — confirmed by grep for `facp_status`/`detector_alarm_state` inside each, zero hits.

**Verdict: the FACP chain up to `BuildingState` is ✅ genuinely wired (unlike the camera chain, which needs an extra fusion-engine hop) — but ❌ that whole chain, `BuildingState` included, never reaches Live System, AI, Advisory, or Command Center.**

## 5. BuildingState → AI (the critical check)

This is the one finding that most changes what "the next milestone" should be, so it is stated plainly: **the AI models trained on synthetic simulation data cannot currently receive equivalent features from any live observation — not because a translation function is merely unwritten, but because the two feature schemas are structurally different.**

- `ai_inference.predictor.Predictor.predict_all(X_row)` expects a single `Dict[str, Any]` feature row whose keys exactly match the column names each model's `FeatureSchema`/`Preprocessor` was fit on (`ai_training/models/base.py:160-168`). Returns `Dict[str, Prediction]`.
- Training features come from `dataset_builder/feature_extractor.py:extract_scenario_features()`, built entirely from `Scenario`/`Building` — offline simulation objects: `ignition_zone`, `ignition_floor`, `fire_profile`, `growth_time`, `total_occupants`, per-behavior-profile occupant counts (`Adult_Count`…`Visitor_Count`, `Firefighter_Count`), mean occupant-attribute features (`Mean_Walking_Speed_Multiplier`, `Mean_Reaction_Speed`, `Mean_Smoke_Tolerance`), group features (`Group_Count`, `Mean_Group_Size`), and full building-enumeration `Zone_N_Occupancy`/door/exit/stair/obstacle/detector/camera state columns.
- `LiveBuildingSnapshot`/`BuildingObservation` exposes only categorical node observations (`ObservationState`/`PerceptionSeverity`, alarm flags), estimated zone occupancy counts, and human observations. **No live source produces `ignition_zone`/`fire_profile`/`growth_time` (nothing is "ignited" in a live building), no per-occupant behavior-profile category counts (those are synthetic-population fields with no sensor analogue), no `Group_Count`.** At best, zone occupancy is partially derivable; the majority of trained features have no live equivalent at all — not "missing," structurally absent.
- No production `feature_row_builder` implementation exists anywhere — `grep feature_row_builder=` across the repo hits only `live_system/integration.py` (the parameter's own definition) and two test files.

**Per-model audit:**

| Model | Input features | Available from live state? | Transformation exists? | Live caller exists? | Output consumed by |
|---|---|---|---|---|---|
| `evacuation_time` (regression) | Full `scenario_features.csv` schema (see above) | Partial (occupancy only) | No | No | `ai_inference/recommendation.py:build_recommendation` — itself only called from tests |
| `bottleneck_occurrence`/`bottleneck_location` (classification) | Same, + Ground Truth-derived labels for training | Partial | No | No | same |
| `smoke_prediction` (`next_highest_smoke_zone`) | Same | Partial | No | No | same |
| `exit_usage` (`exit_usage_percentage`) | Same | Partial | No | No | same |

- `decision_policy.policy.DecisionInputs` has the identical problem: it requires `building: Building`, `scenario: Scenario`, `ground_truth: Any` — offline-simulation-shaped. `GeneratePolicyDecisionPolicyGateway`'s required `decision_inputs_builder` has zero non-test implementations anywhere; its own docstring in `live_system/integration.py:159-172` already names this exact gap ("synthesizing a GroundTruth-compatible view... is a deployment-specific decision this orchestration layer must not make unilaterally").

**Verdict: ❌ both AI Inference and Decision Policy are architecturally wired into `live_system` (real Protocol interfaces, real adapters) but functionally inert in production.** Per this audit's own instruction, no generic feature row should be fabricated to paper over this — the correct next step is a deliberate, human-made decision about which features a live deployment can honestly supply (see Milestone 3, §11).

## 6. AI / Decision Policy → Advisory System

- `AdvisoryInputs` (`advisory_system/recommendation_models.py:24-80`) requires `building: Building`, `scenario: Scenario`, `ground_truth: Any`, `decision_policy: Any` as non-optional fields — all four offline-simulation-shaped. `human_observations`/`ai_predictions`/`ai_confidence`/`rl_action`/`rl_confidence`/`building_system_state` are all optional, defaulting to empty.
- **Who creates `AdvisoryInputs` in production?** Exactly two call sites, both replay/validation, never a live loop: `command_center/incident_data.py:652` (inside `_build_advisory_reports()`, fed by already-replayed `IncidentFrame`s) and `validation_framework/recommendation_validator.py:194` (a per-tick validation pipeline over stored `timeline_rows`).
- **Does a live `BuildingState`/`LiveBuildingSnapshot` ever become `AdvisoryInputs`?** No — confirmed by grep across the whole repo; the `live_system`/`building_state`/`advisory_system` clusters never intersect. `live_system/` never imports `advisory_system` at all.
- **Are AI confidence values genuinely included?** The field exists and is read defensively (`advisory_system/advisory_engine.py:117-145`, `_ai_signal_for`), but **neither real call site ever populates `ai_predictions`/`ai_confidence`** — both leave them at their empty defaults.
- **Is RL involved in live advisory generation?** No — `tests/test_rl_training.py` never touches `advisory_system`; no call site sets `rl_action`/`rl_confidence` anywhere, tests included.
- **Are deterministic safety rules available if AI/RL is unavailable?** **Yes — this is the one genuinely reusable piece here.** `advisory_engine.py`'s rule-based/threshold logic (derived from `decision_policy`) runs unconditionally; `_ai_signal_for` simply returns `(None, None)` when absent, and confidence-source labeling only appends `"ai"`/`"rl"` when actually present. Advisory generation already degrades gracefully with zero AI/RL input — it just has never been fed anything but replay data at all.

**Verdict: ❌ Advisory System has never been wired to any live state, in any form — but its own internal design is already tolerant of partial/absent AI, which is a real asset for future live wiring.**

## 7. Advisory → Output Systems

**A. Voice Evacuation:** `CivilianAnnouncement` → `VoiceMessage` → `VoiceEvacuationController` → `SpeakerManager` → `VoiceOutputProvider` all exist and are wired correctly as a chain. The only non-test constructor of `VoiceEvacuationController` is `command_center/incident_data.py:718` (`_build_voice_broadcasts()`), fed by already-computed replay `frames` using `SimulationVoiceOutputProvider` — its own file explicitly documents this as *"never a live simulation, purely a deterministic reconstruction."* **❌ no live trigger exists anywhere.**

**B. Building Control:** `BuildingRecommendation` → `ControlRequest` → `BuildingControlController` → approval → `BuildingControlProvider` also exists correctly, **and the human-approval gate is genuinely enforced in code** — `submit()` sets `PENDING_APPROVAL`; auto-dispatch only happens under `ApprovalMode.AUTO_APPROVE_SIMULATION`, itself gated to require `provider.is_simulation_only`; the default mode is `REQUIRES_APPROVAL`; `approve()` is the sole path to actual dispatch. **✅ the safety gate is real.** But its only non-test constructor (`command_center/incident_data.py:243`) uses `SimulationControlProvider` (`is_simulation_only=True`) fed from replay data — **❌ no live/production caller.**

**C. Firefighter Intelligence:** `FirefighterIntelligenceReport` reaches Command Center via `RecommendationCenter.show_frame()` → a dedicated `FirefighterIntelligencePanel`, explicitly documented in its own code as *"intelligence only, never a directive."* No code path connects it to `BuildingControlProvider`/`VoiceOutputProvider` — those are only reachable via the separate chains in A/B above. **✅ confirmed to remain intelligence-only, never automatic command authority.**

## 8. Live Command Center

Command Center supports **replayed completed incidents only.** Every panel (`IncidentPanel`, `OccupancyPanel`, `HazardPanel`, `HumanPanel`, `IncidentStatusBar`, `BuildingView`, `TimelinePanel`, `RecommendationTimelinePanel`, and `RecommendationCenter`'s tabs — `CivilianAnnouncementsPanel`, `VoiceEvacuationPanel`, `BuildingRecommendationsPanel`, `FirefighterIntelligencePanel`, `CommanderSummaryPanel`, `BuildingControlsPanel`) is wired through `Dashboard.show_frame()`/`set_frame_index()`, called **only** from `MainWindow._on_playback_tick()` — a 200ms `QTimer` replay clock reading a pre-loaded `IncidentData` — or manual timeline scrubbing. `BuildingControlsPanel` has the one interactive approve/reject affordance, but it still only mutates the pre-built replay `IncidentData`'s own controller; no external/live input path exists. **No panel has an independent live-update path.** This confirms `show_frame()`'s existence is not evidence of live wiring — its only real caller anywhere in production is the replay clock.

The plumbing to change this already exists and is tested: `live_system.integration.DashboardCommandCenterGateway` composes `Dashboard.set_incident()`/`show_frame()` directly from a `LiveBuildingSnapshot`, exactly matching `MainWindow`'s established "owns the clock, widgets just render a frame index" convention (`tests/test_live_system.py:759-814` proves this against the real `Dashboard` widget, not a mock). What's missing is a caller: nothing ever constructs a `LiveOrchestrator` with a `DashboardCommandCenterGateway` wired in, outside that one test file.

## 9. Live Integration Matrix

| Subsystem | Implemented | Live Input | Live Runtime Wiring | BuildingState | AI/Advisory | Command Center |
|---|---|---|---|---|---|---|
| Camera Manager | ✅ | ❌ (frozen, no RTSP) | ❌ never constructed by `live_system` | ✅ (offline-proven) | ❌ | ❌ |
| Multi-Camera Fusion | ✅ | ❌ | ❌ same gap as above | ✅ (offline-proven) | ❌ | ❌ |
| Sensor Manager | ✅ | ❌ (no real hardware) | ⚠️ Designer-sandbox-only (50ms QTimer, not a live loop) | ✅ (via debug runner) | ❌ | ❌ |
| FACP | ✅ | ❌ | ⚠️ Designer-sandbox-only, same runner | ✅ (`facp_status`) | ❌ | ❌ |
| BuildingState | ✅ | N/A | ⚠️ continuously recomputed, but only inside Designer's own debug tooling; nothing downstream reads it | (self) | ❌ zero imports in any of the 5 downstream packages | ❌ |
| AI Inference | ✅ (4 trained models) | N/A | ⚠️ real `Predictor`/gateway interface; `feature_row_builder` never implemented; feature schemas structurally incompatible with live data | ❌ | ⚠️ fields exist on `AdvisoryInputs`, never populated by either real call site | ❌ |
| Decision Policy | ✅ | N/A | ⚠️ real `generate_policy`/gateway interface; `decision_inputs_builder` never implemented; `DecisionInputs` requires offline `Scenario`/`GroundTruth` | ❌ | ✅ (required field of `AdvisoryInputs`, but only ever populated from offline/replay `DecisionPolicy`) | ❌ |
| Advisory System | ✅ | ❌ | ❌ `live_system` never imports `advisory_system`; only 2 production callers, both replay/validation | ❌ | (self) | ✅ (via replay `IncidentData`, not live) |
| Voice Evacuation | ✅ | ❌ | ❌ only built as replay reconstruction (`SimulationVoiceOutputProvider`) | ❌ | ⚠️ consumes `CivilianAnnouncement`, replay-derived only | ✅ (`VoiceEvacuationPanel`, replay-driven) |
| Building Control | ✅ (incl. real approval gate) | ❌ | ❌ only built with `SimulationControlProvider` (`is_simulation_only=True`) | ❌ | ⚠️ consumes `BuildingRecommendation`, replay-derived only | ✅ (`BuildingControlsPanel`, replay-driven) |
| Command Center | ✅ | ❌ | ❌ every panel driven exclusively by the replay `QTimer`; `DashboardCommandCenterGateway` is real but has zero production callers | ❌ | ✅ (via replay `IncidentData`) | (self) |

For every ⚠️/❌ above, the missing caller/adapter and required composition are the same one thing: **a production entry point that constructs `LiveOrchestrator` (or a redesigned equivalent) wired to real gateways**, which does not exist anywhere in the repo outside `tests/test_live_system.py`. No individual subsystem needs new code to be *individually* more correct — every subsystem's own package is already complete and tested for what it claims to do.

## 10. Canonical live pipeline — compatibility with existing architecture

The requested target shape:

```
Physical Building Inputs → Camera/Sensor Providers → CameraManager/SensorManager
    → Multi-Camera Fusion + Detector State → FACP → BuildingStateEstimator
    → Canonical BuildingState → AI Inference → Decision Policy → AdvisoryOrchestrator
    → AdvisoryReport → {Command Center, Voice Evacuation, Building Control Requests, Firefighter Intelligence}
```

**This is compatible with the existing architecture up through `BuildingState`** — Graph A (§1) and §4's FACP chain already build exactly this far, offline-proven. It is **not compatible, as written, with `live_system`'s current implementation**, because `live_system` was built around a parallel `SensorRegistry`/`SensorFusion`/`LiveBuildingSnapshot` stack that never reaches `BuildingState` at all (§1 Graph B, §2). Reaching the diagram above requires an explicit choice: either (a) retire `live_system`'s own `SensorRegistry`/`SensorFusion`/`LiveBuildingSnapshot` path in favor of the already-proven `CameraManager`+`SensorManager`+`FACP`+`BuildingStateEstimator`+`BuildingState` path, or (b) keep both and add a translation layer between them. Per this audit's own instruction not to redesign working packages merely to fit a diagram, **this choice is a decision for the user, not something this audit resolves** — see Milestone 1 below.

Downstream of `BuildingState`, the diagram's ordering (AI → Decision Policy → Advisory → outputs) matches `live_system.orchestrator.LiveOrchestrator.run_cycle()`'s existing, already-correct sequencing (`live_system/orchestrator.py:128-180`: perception → AI → decision policy → recommendations → command center, in that exact order). That part of the architecture does not need reordering — it needs a real state source feeding it and real builder callables filling its two documented gaps (§5).

## 11. Prioritized integration plan

Derived from the actual dependency structure found above, not the illustrative example in the task brief. Every milestone below explicitly notes whether physical CCTV access is required — **none of them are**, since every gap found is an integration/composition gap, not a "needs a real camera" gap. All can be built and proven against `ReplayFrameSource` + simulated `SensorManager`/`FACP` readings, exactly as the CCTV milestone already demonstrated offline.

**Milestone 1 — Reconcile canonical live state (decision required, not pure integration).**
Decide whether `live_system.state_manager.LiveBuildingSnapshot` is retired in favor of `building_state.models.BuildingState` (recommended: `BuildingState` is the one already proven end-to-end with real multi-camera dedup and FACP integration), or whether both are kept with an explicit translation layer. Packages touched: `live_system/state_manager.py`, `live_system/orchestrator.py`, `live_system/integration.py` (this is a redesign of `live_system`'s internals, not simple wiring — flag to the user before starting). APIs reused: `BuildingStateEstimator`, `CameraManager`, `SensorManager`, `SimulatedFACP` — all already proven, zero changes needed to any of them. Tests required: new `live_system` tests replacing the `SensorFusion`-based ones. Risk: **high** — this is the one milestone that is genuinely architectural, not additive. CCTV access required: **no**.

**Milestone 2 — Build a real, continuous live composition root.**
A production (non-test) construction that owns `CameraManager` + `SensorManager` + `SimulatedFACP` + `MultiCameraFusionEngine` + `BuildingStateEstimator` on a driven cycle — whether that's a rebuilt `LiveOrchestrator` per Milestone 1, or a new adapter module. This is what `designer/building_state_debug_runner.py` already does *inside the Designer's sandbox loop*; the work here is extracting/generalizing that into something a genuine live deployment (not just a debug panel) can own. APIs reused: everything from Milestone 1, plus `BuildingStateDebugRunner` as a working reference implementation. Tests required: an integration test proving a continuous cycle assembles `BuildingState` from `ReplayFrameSource` + simulated detector/FACP readings, end-to-end. Risk: medium. CCTV access required: **no**.

**Milestone 3 — AI feature bridge (requires a feature-engineering decision, flagged, not silently built).**
Per §5's finding that the incompatibility is structural, not just unimplemented: either (a) retrain the four existing models on a reduced, honestly-live-derivable feature subset, (b) build an explicit, documented best-effort `feature_row_builder` with clearly-labeled defaults for the synthetic-only fields it cannot honestly fill (accepting known-reduced prediction quality), or (c) defer live AI inference entirely until Milestone 2 produces enough real live `BuildingState` history to inform which approach is worth it. **Do not silently fabricate a generic feature row** — this repeats the instruction already given for this exact audit. Risk: medium-high (an ML/product decision, not an engineering one). CCTV access required: no for the decision itself; real detection data would improve any eventual live-derived occupancy features, but is not a blocker to making the decision.

**Milestone 4 — Decision Policy live bridge.**
Same category of problem as Milestone 3 (`DecisionInputs` needs `Scenario`/`GroundTruth`-shaped data with no live equivalent) — defer concrete design until Milestone 1/2 clarify what a live `GroundTruth`-compatible view would even mean. Risk: medium-high. CCTV access required: no.

**Milestone 5 — Advisory System live bridge.**
Once Milestone 2 produces a canonical live `BuildingState`, wire a live `AdvisoryInputs` builder. Per §6's finding, `advisory_engine.py`'s rule-based path already degrades gracefully with absent AI/RL — so Advisory could plausibly go live with just `BuildingState` plus an honest placeholder/empty `DecisionPolicy`, **ahead of** Milestones 3/4 fully landing, if the `DecisionInputs` requirement is relaxed for a live deployment (itself part of Milestone 4's scope). Packages touched: `advisory_system` (a live `AdvisoryInputs` builder, additive), `live_system/integration.py`. Tests required: a live-fed `AdvisoryReport` test parallel to the existing replay-fed one. Risk: medium. CCTV access required: no.

**Milestone 6 — Command Center live mode.**
`DashboardCommandCenterGateway` already works and is tested (§8) — the work here is a live-mode entry point, since the existing `command_center.main_window.MainWindow` owns its own replay `QTimer` clock and is architecturally a *replay viewer*, not a live subscriber. This likely means a new, small live-mode window/controller that constructs `Dashboard` directly and pushes `LiveOrchestrator`-produced snapshots into it via the existing gateway, rather than retrofitting the replay `MainWindow`. Risk: low-medium (the hard adapter work is done; this is mostly a new thin entry point). CCTV access required: no.

**Milestone 7 — Voice Evacuation / Building Control live triggering.**
Once Milestone 5 lands, swap `SimulationVoiceOutputProvider`/`SimulationControlProvider` for real triggers reading a live `AdvisoryReport`, preserving the already-correct, already-enforced human-approval gate in `BuildingControlController` (§7B) unchanged. Risk: low (the safety-critical logic is already correct and tested; this is a provider swap). CCTV access required: no for the wiring; a real `BuildingControlProvider`/`VoiceOutputProvider` execution against real hardware is a separate physical dependency (building control/speaker hardware), distinct from and not blocked by the CCTV freeze.

**Explicitly not proposed as a milestone:** implementing `RTSPFrameSource`, a real `HumanDetector`, or `LiveReIDIdentityResolver` — those remain correctly frozen per the CCTV milestone's own scope and are not on the critical path to any of the seven milestones above; every one of them is buildable and testable against `ReplayFrameSource`/simulated sensor readings today.

## 12. Validation

No production code was modified during this audit. Repository consistency verified after adding this document (see final git status below). Since only documentation was added, the full test suite was not re-run — no source file changed.

## 13. Milestone 1 — Canonical Live BuildingState Runtime Assembly (COMPLETED)

Status: **implemented and test-proven.** `building_state.models.BuildingState` is now the canonical integrated state snapshot the live runtime assembles each cycle. `live_system` was extended, not redesigned — `LiveOrchestrator` keeps owning lifecycle/start-stop/timing/incident-lifecycle exactly as before; it now additionally coordinates one new optional stage.

### 13.1 What was built

- **`live_system/building_state_gateway.py`** (new file) — `BuildingStateGateway` (a `Protocol`, one method: `collect(time) -> BuildingState`, mirroring `PerceptionGateway`/`AIInferenceGateway`'s own established shape) and `EstimatorBuildingStateGateway`, a real adapter composing `building_state.estimator.BuildingStateEstimator` (reused verbatim — zero aggregation logic duplicated) from eleven independently-optional provider callables, one per `BuildingStateEstimator.estimate()` keyword argument (`hazard_snapshot_provider`, `occupancy_snapshot_provider`, `camera_status_provider`, `camera_observation_provider`, `fusion_result_provider`, `smoke_detector_status_provider`, `smoke_detector_reading_provider`, `heat_detector_status_provider`, `heat_detector_reading_provider`, `facp_snapshot_provider`, `control_snapshot_provider`). Every provider left `None` resolves to the same honest empty default `BuildingStateEstimator.estimate()` itself already accepts — a gateway with zero providers configured still returns a valid, non-crashing `BuildingState`.
- **`live_system/state_manager.py`** — `LiveBuildingSnapshot` gained one new field, `building_state: Optional[BuildingState] = None` — the canonical state field. `StateManager.update_building_state(building_state, time)` (mirrors `update_perception()` exactly) and `StateManager.latest_building_state()` (a thin accessor, the "existing appropriate accessor" this milestone's own Phase 2 asked to determine) were added. The pre-existing `building_observation`/`ai_predictions`/`decision_policy`/`recommendations` fields and their own `update_*()` methods are untouched — proven by `StateManagerBuildingStateTests.test_existing_perception_based_consumers_are_unaffected` and `.test_no_parallel_independently_computed_state_exists` (`tests/test_live_building_state_integration.py`).
- **`live_system/orchestrator.py`** — `LiveOrchestrator` gained one new optional constructor parameter, `building_state_gateway`, and one new `latest_building_state` property (a thin forward onto `state_manager.latest_building_state()`). `run_cycle()` gained exactly one new conditional stage, inserted after Live Perception and before AI Inference — the existing stage order is otherwise byte-for-byte unchanged. `LiveOrchestrator` still does not import or construct `CameraManager`, `SensorManager`, `MultiCameraFusionEngine`, `SimulatedFACP`, `ReplayFrameSource`, or any future `RTSPFrameSource` anywhere — enforced by a new `LiveSystemPackageDependencyDirectionTests` guard.
- **`live_system/event_bus.py`** — added `EventType.BUILDING_STATE_UPDATED`, published (payload = the `BuildingState`) exactly when `building_state_gateway` is configured, on every cycle, mirroring `OCCUPANCY_UPDATED`/`HAZARD_UPDATED`'s own unconditional-when-configured publication style. This is the seam Phase 8 of the prior milestone named as future work for AI Inference/Advisory System/Command Center — added, not yet subscribed to by anything.
- **Two new architecture guards** (Phase 10): `tests/test_live_system.py::LiveSystemPackageDependencyDirectionTests` (forbids `live_system` from importing `camera_manager.manager`/`sensor_manager.manager`/`multi_camera_fusion.engine`/`facp.engine`/`facp.provider`/`building_control.controller`/`building_control.providers`/`live_camera_pipeline.replay_frame_source`/`live_camera_pipeline.frame_source`/`cv2`/`torch`/`ultralytics`/`onvif` — while still permitting the read-only `.status`/`.track`/`.models`/`.snapshot` value-type submodules `building_state_gateway.py` legitimately imports) and `tests/test_building_state.py::BuildingStatePackageDependencyDirectionTests` (forbids `building_state` from importing `live_system`/`ai_inference`/`ai_decision`/`advisory_system`/`command_center`/`decision_policy`/`rl_training`, and separately forbids it from importing the control-capable `facp.engine`/`facp.provider`/`building_control.controller`/`building_control.providers` — `building_state` stays observational, never acknowledging/silencing/resetting/controlling anything). `tests/test_no_cv_dependencies.py`'s existing `TARGETS` tuple was extended to include `live_system` and `building_state`.

### 13.2 Before/after runtime graph

Before (§1 of this document):

```
Graph A (offline CCTV): Camera -> ... -> MultiCameraFusionEngine -> BuildingState -> [nothing downstream]
Graph B (live_system):  SensorRegistry -> SensorFusion -> BuildingObservation -> LiveBuildingSnapshot -> [never constructed in production]
```

After this milestone:

```
Live Inputs (CameraManager.all_statuses() / MultiCameraFusionEngine.fuse() / SensorManager.all_statuses() /
             SimulatedFACP.current_snapshot() / BuildingControlController.snapshot() -- each supplied by a
             caller-composed provider, never owned by live_system itself)
    ↓  [NEW, ✅ wired]
EstimatorBuildingStateGateway.collect(time)
    ↓  [✅ reused verbatim, zero duplication]
BuildingStateEstimator.estimate(...)
    ↓  [✅ wired]
BuildingState
    ↓  [NEW, ✅ wired]
StateManager.update_building_state() -> LiveBuildingSnapshot.building_state
    ↓  [✅ wired]
LiveOrchestrator.run_cycle() / LiveOrchestrator.latest_building_state
    ↓  [NEW event seam, ✅ published; ❌ not yet subscribed to by anything]
EventType.BUILDING_STATE_UPDATED
    ↓  [❌ still not built -- next milestones]
AI Inference / Decision Policy / Advisory System / Live Command Center / Voice Evacuation / Building Control dispatch
```

The `live_system` → `building_state` dependency this graph now has is exactly the one direction §10 of this document's canonical-pipeline analysis called for; `building_state` → `live_system` remains, and is now mechanically enforced to remain, absent.

### 13.3 Exact treatment of `LiveBuildingSnapshot`

Not deleted, not turned into a compatibility view over `BuildingState` — kept as-is, plus one additive field. `LiveBuildingSnapshot.building_observation`/`ai_predictions`/`decision_policy`/`recommendations`/`engineering_state` remain exactly what they were; `SensorFusionPerceptionGateway`, `PredictorAIInferenceGateway`, `GeneratePolicyDecisionPolicyGateway`, and every existing test using them are untouched (2577/2577 pre-existing tests still pass unmodified). This shape was chosen over "`LiveBuildingSnapshot` becomes a view over `BuildingState`" because `BuildingState`'s own docstring is explicit that it "deliberately carries no AI/RL/decision-policy field of any kind" — collapsing the two would have required either violating that contract or discarding fields real, passing tests already depend on. `building_state` is documented as the field new work should read; `building_observation` remains for existing callers only.

### 13.4 Exact treatment of `StateManager`

`StateManager` now maintains exactly one additional field, `building_state`, alongside its pre-existing `LiveBuildingSnapshot`. There are not two independently-mutated "current state" objects: `update_building_state()` only ever touches the `building_state` field (via `replace()`'s "carry every field forward except the one just changed" discipline, proved by `test_no_parallel_independently_computed_state_exists`), and nothing recomputes a second, competing `BuildingState` anywhere.

### 13.5 Input seams now supported

Camera/fusion (§ Phase 4): `camera_status_provider`/`camera_observation_provider`/`fusion_result_provider` accept already-produced values — `live_system` never imports `CameraManager`, `MultiCameraFusionEngine`, `ReplayFrameSource`, or any future `RTSPFrameSource`. Sensor/FACP (§ Phase 5): `smoke_detector_status_provider`/`smoke_detector_reading_provider`/`heat_detector_status_provider`/`heat_detector_reading_provider`/`facp_snapshot_provider` accept already-produced values; `facp_snapshot_provider`'s only allowed shape is a read-only `current_snapshot()`-returning callable — `BuildingStateEstimator`/the gateway never call `acknowledge()`/`silence()`/`reset()`/`evaluate()` themselves. Building Control (§ Phase 3): `control_snapshot_provider` accepts an already-computed `ControlStateSnapshot`, proven against a real `BuildingControlController`.

### 13.6 Integration smoke-test results

`tests/test_live_building_state_integration.py::LiveBuildingStateSmokeTest` (Phase 12): a real `Building` (2 cameras, 1 smoke detector, 1 zone) → real `CameraManager` discovery → the offline-proven `ReplayFrameSource`/`LiveCameraPipeline`/`MappingIdentityResolver` chain → real `MultiCameraFusionEngine` → real `SimulatedFACP` → `EstimatorBuildingStateGateway` → a real, started `LiveOrchestrator`, run for 3 cycles. Result: **4 raw detections fuse to exactly 3 `BuildingState.occupant_tracks` every cycle** (the same "2 cameras, 3 physical people" scenario the CCTV milestone proved, now reached through `LiveOrchestrator` instead of a direct `BuildingStateEstimator.estimate()` call), camera asset status for both cameras reaches `BuildingState.camera_observations`, `FACPSnapshot.panel_state == ALARM` with `SMOKE-1` in `active_alarm_source_ids` reaches `BuildingState.facp_status`, and `stop()` cleanly halts the orchestrator (`run_cycle()` raises `LiveSystemNotRunningError` afterward, exactly as before this milestone). Zero network access, zero real CCTV, zero AI inference, zero advisory generation anywhere in the test.

### 13.7 Full test-suite result

**2606/2606 passing** (2577 pre-existing + 26 new tests in `tests/test_live_building_state_integration.py` + 1 new `LiveSystemPackageDependencyDirectionTests` + 2 new `BuildingStatePackageDependencyDirectionTests`), `python -m unittest discover -s tests`. No pre-existing test was modified.

### 13.8 Still NOT connected (unchanged from this document's earlier sections — do not overclaim)

- `BuildingState` → AI Inference (§5 of this document's findings stand unchanged — `feature_row_builder` remains unimplemented, and the feature-schema incompatibility remains structural, not merely unwired)
- `BuildingState` → Decision Policy (`decision_inputs_builder` remains unimplemented)
- `BuildingState` → Advisory System (`live_system` still does not import `advisory_system`)
- `BuildingState`/`EventType.BUILDING_STATE_UPDATED` → Live Command Center (`DashboardCommandCenterGateway` still only ever notified from `LiveBuildingSnapshot`'s pre-existing `building_observation`-derived path in tests; nothing yet builds an `IncidentFrame` from `building_state`)
- Advisory → Live Voice Evacuation (still only replay-reconstructed, per §7A)
- Advisory → Live Building Controls (still only replay-reconstructed with `SimulationControlProvider`, per §7B — though this milestone's own smoke test proves a real `BuildingControlController`/`SimulationControlProvider` pairing composes cleanly into the new gateway, which the *next* milestone touching this seam can build on)

Milestone 2 onward (§11 of this document) remain exactly as previously planned and are explicitly out of scope for this milestone.

## 14. Milestone 2 — Simulation-to-Live AI Feature Parity Framework (COMPLETED, NOT WIRED)

Status: **feature-parity framework established; AI inference still NOT wired into `LiveOrchestrator` or `advisory_system`.** Full detail: `docs/architecture/ai_live_feature_parity.md`.

This milestone answered §5's central question — whether the AI trained on synthetic simulation data can honestly receive equivalent features from live observations — and found, precisely: **no existing model's current feature set is reproducible from `BuildingState` (0 LIVE_READY)**; two of four models (`EvacuationTimeModel`, `BottleneckModel`-occurrence) are **RETRAIN_REQUIRED** and now have a real, test-proven live-compatible feature contract (`ai_features/`, 25 canonical fields) plus an actual retraining experiment showing real-but-reduced predictive power (evacuation-time R² drops from 1.00 to 0.27 on a small fixed test fixture; bottleneck-occurrence ROC-AUC drops from 1.00 to 0.84); the remaining two (`BottleneckModel`-location, `ExitUsageModel`) plus `SmokePredictionModel` are **RESEARCH_ONLY** — none can honestly run from `BuildingState` today, for reasons specific to each (no per-zone/per-exit-indexed schema; `SmokePredictionModel`'s target needs continuous smoke-level data this codebase has no real-sensor path to at all, not merely a reduced one).

**What this milestone deliberately did NOT do**, exactly as instructed: it did not wire `ai_features`, `ai_inference`, or any retrained model into `LiveOrchestrator.run_cycle()`, `live_system.integration`, or `advisory_system` — `ai_features/` has zero callers outside its own tests and the standalone `scripts/ai_feature_parity_experiment.py` report script (mechanically enforced by a new dependency-direction guard in `tests/test_ai_features.py`: `live_system` never imports `ai_features`, and `ai_features` never imports `live_system`/`advisory_system`). §5/§6 of this document's earlier findings (the feature-bridge gap, `feature_row_builder` never implemented) remain otherwise unchanged and still block any live AI inference — this milestone only establishes what a future, separate wiring milestone would need to build on top of, and explicitly does not attempt that wiring itself.

## 15. Milestone 3 — Live-Compatible AI Model Training & Model Registry (COMPLETED, NOT WIRED)

Status: **two versioned, metadata-tagged, registry-served model artifacts exist; still zero AI wiring into `LiveOrchestrator` or `advisory_system`.** Full detail: `docs/architecture/live_ai_model_training.md`.

On top of Milestone 2's feature contract, this milestone added `ai_registry/`: a `ModelMetadata` contract (deployability explicitly `LIVE_COMPATIBLE`/`RESEARCH_ONLY`, never inferred from a filename), a reproducible training pipeline trained at real production scale (5000 diverse synthetic scenarios, not the earlier 80-scenario experiment), a `ModelRegistry` that mechanically refuses to serve a `RESEARCH_ONLY` model or a schema-incompatible one, and a `LiveAIInferenceService` (`BuildingState → canonical features → compatibility check → registry → model → structured prediction`) that fails loudly (never a fabricated zero) on every failure mode named in that milestone's own brief.

**Result:** `EvacuationTimeModel_LiveCompatible` is `EXPERIMENTAL` (R²=0.088 on held-out data, does not beat its own MAE baseline — an honest, un-softened negative result). `BottleneckOccurrenceModel_LiveCompatible` is a genuine `PRODUCTION_CANDIDATE` (F1 0.511 vs. 0.487 baseline, ROC-AUC 0.793 vs. 0.500 baseline, despite 95% class imbalance). Neither is wired into the live runtime — `ai_registry` has zero callers from `live_system` or `advisory_system`, mechanically enforced by `tests/test_ai_registry.py::NoLiveWiringGuardTests`.

**Still explicitly unconnected, same boundary as Milestone 2:** `BuildingState → AI Inference` now has a real, working, safe-failing implementation (`LiveAIInferenceService`) — but nothing calls it from `LiveOrchestrator.run_cycle()`. `AI Inference → Decision Policy → Advisory System → Command Center` remain entirely unbuilt. AI predictions, when this next milestone eventually wires them in, are designed to remain advisory input only — no path from a prediction directly to a building control action was built or implied.
