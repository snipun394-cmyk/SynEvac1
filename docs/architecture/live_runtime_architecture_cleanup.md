# Live Runtime Architecture Cleanup

Status: **implemented, tested.** Closes architecture-audit finding §14 row 1 (`docs/architecture/synevac_end_to_end_architecture_review.md`): *"`live_system/integration.py`'s Phase-7 gateways... are dead in production."* Baseline before this milestone: 4318/4318 tests passing, commits `c62c2ce` (launcher) / `c6da250` (canonical occupancy).

This is a cleanup milestone. No functionality was added; nothing about `LiveOrchestrator`, `BuildingState`, or passive fire-protection modeling was redesigned.

## 1. Phase 1 — complete usage audit

Every symbol `live_system/integration.py` defined, and its classification:

| Symbol | Kind | Non-test production callers | Classification |
|---|---|---|---|
| `PerceptionGateway` | Protocol | `live_system/orchestrator.py` (type annotation on `perception_gateway` param) | **A** — active, but only as unused-in-production extension-point typing (see §2) |
| `AIInferenceGateway` | Protocol | `live_system/orchestrator.py` (type annotation on `ai_inference_gateway` param) | **A** (same caveat) |
| `DecisionPolicyGateway` | Protocol | `live_system/orchestrator.py` (type annotation on `decision_policy_gateway` param) | **A** (same caveat) |
| `CommandCenterGateway` | Protocol | `live_system/orchestrator.py` (type annotation on `command_center_gateway` param) | **A** (same caveat) |
| `RecommendationBuilder` | type alias | `live_system/orchestrator.py` (type annotation on `recommendation_builder` param) | **A** (same caveat) |
| `SensorFusionPerceptionGateway` | concrete class | **none** | **B/D** — test-only (`tests/test_live_system.py`), zero production callers |
| `PredictorAIInferenceGateway` | concrete class | **none** | **B/D** |
| `GeneratePolicyDecisionPolicyGateway` | concrete class | **none** | **B/D** |
| `DashboardCommandCenterGateway` | concrete class | **none** | **B/D** |
| `FeatureRowBuilder` | type alias | only `PredictorAIInferenceGateway`'s own constructor | **D** (dies with its owner) |
| `DecisionInputsBuilder` | type alias | only `GeneratePolicyDecisionPolicyGateway`'s own constructor | **D** (dies with its owner) |

**Method**: grepped the entire repository for every symbol name, then read every hit. Six files outside `integration.py`/tests mention one of the deleted classes' names (`command_center/data_source.py`, `command_center/incident_data.py`, `human_decision_engine/view.py`, `live_system/live_advisory_gateway.py`, `live_system/state_manager.py`, `perception/providers/human_observation_provider.py`) — every one confirmed, by reading the surrounding code, to be a **comment-only analogical cross-reference** ("the same pattern `DashboardCommandCenterGateway` already established"), never an import or a live dependency. `live_system/__init__.py` (public export) and `live_system/orchestrator.py` (real import, for type annotations only) were the only two files with an actual `from live_system.integration import ...` statement outside `tests/test_live_system.py`.

**Critical finding inside `LiveOrchestrator.run_cycle()` itself**: the five constructor parameters typed against these symbols (`perception_gateway`, `ai_inference_gateway`, `decision_policy_gateway`, `command_center_gateway`, `recommendation_builder`) are genuinely *called* when configured (`if self.perception_gateway is not None: ... self.perception_gateway.collect(...)`, etc. — `live_system/orchestrator.py:347-373,643-662`) — this is not dead code inside `LiveOrchestrator` itself. What makes them dead **in production** is one layer up: `live_runtime/factory.py` — the actual, only production composition root — never passes any of the five to `LiveOrchestrator(...)` (confirmed by reading its full ~580-line construction call). Every production `LiveOrchestrator` instance therefore has all five at their `None` default, permanently, and the four concrete adapter classes have no reason to ever be constructed outside a test.

**The underlying packages the four dead adapters wrapped are themselves very much alive**, just via other callers, unrelated to `live_system.integration`:

- `perception.fusion.sensor_fusion.SensorFusion` / `perception.fusion.occupancy_estimation.OccupancyEstimator` — used by Designer's **Perception Debug Panel** (`designer/perception_debug_runner.py`, `designer/widgets/perception_debug_panel.py`), a genuinely active, unrelated simulation-side debugging tool. **Not touched by this milestone.**
- `ai_inference.predictor.Predictor` — used by `advisory_system/recommendation_models.py`, `ai_features/compatibility.py`. **Not touched.**
- `decision_policy.generate_policy` — used extensively, including by the **current** `live_system.live_advisory_gateway.ReplayCompatibleAdvisoryGateway` (a different composition of the same underlying function). **Not touched.**
- `perception.providers.human_observation_provider.HumanObservationProvider` (the Protocol) — implemented by simulation-side callers (`virtual_camera/provider.py`, `simulation_runtime/human_observation_bridge.py`), entirely independent of `live_system.integration`. **Not touched.**

Only the *adapter/glue* classes gluing these into the OLD orchestration shape were dead; the packages themselves were never at risk.

## 2. Phase 2 — old graph vs. current graph

**OLD graph** (`live_system/integration.py`'s own Phase-7 design):

```
SensorRegistry.read_all() -> SensorReadings
    -> PerceptionGateway.collect() [SensorFusionPerceptionGateway]
       -> perception.fusion.sensor_fusion.SensorFusion.fuse() -> BuildingObservation
    -> StateManager.update_perception() -> LiveBuildingSnapshot.building_observation
    -> AIInferenceGateway.predict() [PredictorAIInferenceGateway]
       -> ai_inference.predictor.Predictor.predict_all() -> {name: Prediction}
    -> StateManager.update_ai_predictions() -> LiveBuildingSnapshot.ai_predictions
    -> DecisionPolicyGateway.evaluate() [GeneratePolicyDecisionPolicyGateway]
       -> decision_policy.generate_policy() -> DecisionPolicy
    -> StateManager.update_decision_policy() -> LiveBuildingSnapshot.decision_policy
    -> RecommendationBuilder(snapshot) -> (Recommendation, ...)
    -> StateManager.update_recommendations() -> LiveBuildingSnapshot.recommendations
    -> CommandCenterGateway.notify() [DashboardCommandCenterGateway]
       -> command_center.dashboard.Dashboard.show_frame()  (PUSH model)
```

**CURRENT graph** (`live_runtime/factory.py::build_live_runtime()`, unchanged by this milestone, reused verbatim):

```
Designer -> LiveRuntimeSession -> build_live_runtime()
    -> CameraManager/SensorManager/MultiCameraFusionEngine/FACP (Digital Twin asset layer)
    -> LivePerceptionFusionCoordinator (live_perception/ + sensor_fusion/, NOT perception.fusion.*)
    -> EstimatorBuildingStateGateway -> BuildingState  (via live_system.building_state_gateway)
    -> LiveOrchestrator.run_cycle()
       -> CrowdIntelligenceGateway / EvacuationProgressGateway / TrajectoryIntelligenceGateway /
          EmergencyResponseGateway / EvacuationRecommendationGateway / EvacuationGuidanceGateway /
          EvacuationSignageGateway / FACPGateway  (the Engine*Gateway generation, all actually
          instantiated by the real factory)
       -> live_ai_gateway.LiveAIInferenceGateway (optional; None in production today)
       -> live_advisory_gateway.LiveAdvisoryGateway (optional; None in production today --
          a live incident has no live DecisionPolicy equivalent at all, not a like-for-like
          replacement of the old DecisionPolicyGateway)
    -> StateManager (live_system.state_manager -- SAME class both generations always shared)
    -> LiveCommandCenterDataSource -- Command Center reads StateManager.current() on its own
       timer (PULL model), never pushed a frame by the orchestrator
    -> LiveOperatorActionGateway -> VoiceEvacuationController / DynamicSignageController /
       BuildingControlController
```

**Responsibilities replaced, precisely:**

| Old responsibility | Replaced by |
|---|---|
| `PerceptionGateway` (SensorReadings → BuildingObservation) | `live_perception/` providers + `sensor_fusion.engine.SensorFusionEngine` + `EstimatorBuildingStateGateway` → canonical `BuildingState` |
| `AIInferenceGateway` (predict from `LiveBuildingSnapshot`) | `live_system.live_ai_gateway.LiveAIInferenceGateway` (predicts from `BuildingState` directly) |
| `DecisionPolicyGateway` (live `DecisionPolicy`) | **Not replaced 1:1** — `live_system.live_advisory_gateway.LiveAdvisoryGateway` produces an `AdvisoryReport` instead, a deliberately different, live-shaped concept; live incidents have no `DecisionPolicy` equivalent (needs a `Scenario`/`GroundTruth`, both offline-simulation artifacts) |
| `CommandCenterGateway` (push a frame) | `LiveCommandCenterDataSource` (Command Center pulls `StateManager.current()` on its own refresh timer) |
| `RecommendationBuilder` | `EvacuationRecommendationEngine`/`EngineEvacuationRecommendationGateway` |

`StateManager`/`LiveBuildingSnapshot` (`live_system/state_manager.py`) is the one piece of Phase-1/7 infrastructure **both generations genuinely share, unchanged** — `LiveOrchestrator.__init__` still constructs it as its own canonical snapshot holder either way.

## 3. Phase 3 — other superseded/dead architecture discovered (inventoried, NOT deleted)

Per this milestone's own explicit instruction ("Do NOT delete them automatically... produce a small inventory"), the following were found but deliberately left untouched — they live inside `live_system/orchestrator.py`/`live_system/state_manager.py` themselves, which are out of scope ("do not redesign `LiveOrchestrator`"):

1. **`LiveBuildingSnapshot`'s dormant original fields** (`building_observation`, `ai_predictions`, `decision_policy`, `recommendations`) sit alongside the newer, actually-populated fields (`ai_prediction_snapshot`, `evacuation_recommendation`, `advisory_report`, etc.) on the same dataclass. Already identified by the prior end-to-end audit (§10 finding 5) as "harmless today... a future maintainer could read/write the wrong one." Still true; not resolved by this milestone.
2. **`live_system.sensor_registry.SensorRegistry`/`Sensor`/`CCTVSensor`/`SmokeDetectorSensor`/`HeatDetectorSensor`/`FireAlarmControlPanelSensor`** — `LiveOrchestrator.__init__` always constructs a fresh, empty `SensorRegistry()` when none is injected (same "always constructed, never Optional" pattern as `incident_manager`), but `build_live_runtime()` never injects one either — production `LiveOrchestrator.sensor_registry` is always empty. Same "dead-in-production, alive-in-signature" shape as the four deleted adapters, but living inside `orchestrator.py`'s own constructor body, not `integration.py` — out of this milestone's scope.
3. **`live_system.incident_manager.IncidentManager`** — same pattern: always constructed, never wired to anything a real operator currently triggers in production (the operator workflow goes entirely through `LiveOperatorActionGateway.approve_*()`, never `orchestrator.transition_incident()`). Out of scope, same reasoning.
4. **`perception.fusion.sensor_fusion.SensorFusion` vs. `sensor_fusion.engine.SensorFusionEngine`** — confusingly similar names, genuinely different, both alive: the former is Designer's simulation-side Perception Debug Panel tool; the latter is the live production pipeline's own fusion engine. **Not a duplication to resolve** — flagged here only to prevent a future contributor's confusion, exactly this milestone's own stated primary goal.

None of these four were acted on. They are recommended as separate, future, narrowly-scoped cleanup targets — each would require touching `LiveOrchestrator`/`StateManager` themselves, explicitly out of bounds here.

## 4-5. Phases 4-5 — cleanup strategy applied

| Symbol(s) | Decision | Why |
|---|---|---|
| `SensorFusionPerceptionGateway`, `PredictorAIInferenceGateway`, `GeneratePolicyDecisionPolicyGateway`, `DashboardCommandCenterGateway` | **DELETE** | Zero legitimate (non-test) callers; no serialization dependency; no test documented an intentional compatibility contract for the adapters themselves (only for `LiveOrchestrator`'s own generic dispatch, covered by hand-written stubs that never depended on these classes) |
| `FeatureRowBuilder`, `DecisionInputsBuilder` | **DELETE** | Dead the moment their sole owner (above) is deleted |
| `PerceptionGateway`, `AIInferenceGateway`, `DecisionPolicyGateway`, `CommandCenterGateway`, `RecommendationBuilder` | **KEEP FOR COMPATIBILITY, explicitly labeled legacy** | `LiveOrchestrator.__init__`'s own unmodified constructor signature still type-annotates its five optional extension-point parameters against them — deleting these types would require modifying `LiveOrchestrator`, explicitly out of scope. `live_system/integration.py`'s module docstring and `live_system/__init__.py`'s `__all__` both now say, in plain language, that these are legacy-only and never populated by `build_live_runtime()` in production. |

No replacement abstraction was created. The replacement (`live_runtime` + `LiveOrchestrator`'s current `Engine*Gateway`/`live_ai_gateway`/`live_advisory_gateway` wiring) already existed before this milestone, unchanged.

## 6. Phase 6 — public export cleanup

`live_system/__init__.py`: removed `SensorFusionPerceptionGateway`, `PredictorAIInferenceGateway`, `FeatureRowBuilder`, `GeneratePolicyDecisionPolicyGateway`, `DecisionInputsBuilder`, `DashboardCommandCenterGateway` from both the import block and `__all__`. Kept `PerceptionGateway`, `AIInferenceGateway`, `DecisionPolicyGateway`, `CommandCenterGateway`, `RecommendationBuilder`, now under an explicit "LEGACY extension-point typing only" comment in `__all__` pointing at `live_runtime.factory.build_live_runtime()` as the real composition path.

## 7. Phase 7 — test cleanup

`tests/test_live_system.py`: deleted four obsolete implementation-test classes (`SensorFusionPerceptionGatewayTests`, `PredictorAIInferenceGatewayTests`, `GeneratePolicyDecisionPolicyGatewayTests`, `DashboardCommandCenterGatewayTests`, plus the `_FakeRegressionModel` fixture only `PredictorAIInferenceGatewayTests` used) — these tested the deleted adapters' own internal wiring, not a still-needed guarantee. Removed the now-orphaned `make_building()`/`make_scenario()` helpers and their imports (`models.building.Building`, `models.exit.Exit`, `models.floor.Floor`, `models.zone.Zone`, `scenario.metadata.ScenarioMetadata`, `scenario.scenario.Scenario`, `ground_truth.labels.GroundTruth`, `decision_policy.policy.DecisionInputs`, `ai_inference.loader.LoadedModel`/`ModelProvenance`, `ai_inference.predictor.Predictor`, `perception.fusion.occupancy_estimation.OccupancyEstimator`, `perception.fusion.sensor_fusion.SensorFusion`, `perception.models.building_observation.PerceptionSeverity`) — all confirmed used only by the deleted test classes.

**Preserved, reexpressed**: `LiveOrchestratorContinuousUpdateLoopTests::test_repeated_runs_over_identical_mocked_input_are_deterministic` used a real `SensorFusionPerceptionGateway(sensor_fusion, occupancy_estimator)` to prove `UpdateLoop`/`LiveOrchestrator` produce identical output across repeated runs — a genuinely still-valid `LiveOrchestrator` behavioral guarantee, unrelated to which gateway implementation is used. Rewritten to use the pre-existing `_StubPerceptionGateway` (deterministic by construction) instead — same claim, same coverage, zero dependency on the deleted class.

**Preserved, unmodified**: every `LiveOrchestrator*Tests` class exercising the five optional-gateway constructor parameters (`LiveOrchestratorSensorRegistrationTests`, `LiveOrchestratorEventFlowTests`, `LiveOrchestratorStateUpdatesTests`, `LiveOrchestratorAIAndDecisionPolicyInvocationTests`) — these already used hand-written duck-typed stubs (`_StubPerceptionGateway`, `_StubAIInferenceGateway`, `_StubDecisionPolicyGateway`, `_StubCommandCenterGateway`), never the deleted concrete classes, so they needed zero changes. This is `LiveOrchestrator`'s own still-current, still-tested "if configured, dispatch; if not, skip" contract — exactly what "do not redesign `LiveOrchestrator`" requires be preserved.

Test count: `tests/test_live_system.py` went from covering the four now-deleted classes plus everything else, to 63 tests (all passing), none referencing the deleted symbols.

## 8. Phase 8 — architecture guards

New `tests/test_live_runtime_architecture_cleanup.py`:

- `DeletedSymbolsHaveNoRemainingImportsTests` — regex-scans every `.py` file in the repository for an `import`/`from ... import` of any of the six deleted symbol names; confirms they no longer exist on `live_system.integration`/`live_system`; confirms the five kept symbols still do.
- `NoNonHistoricalMentionsOutsideAllowedFilesTests` — a stricter, name-only (not import-specific) scan proving every remaining mention of a deleted symbol's name anywhere in `.py`/`.md` sits inside one of the files this milestone's own Phase 5/11 decisions explicitly kept it in (the slimmed `integration.py` itself, the two historical docs, this milestone's own new doc, and `tests/test_live_system.py`'s own removal-notice comment) — a future accidental reintroduction anywhere else fails this guard.
- `LiveRuntimeNeverDependsOnIntegrationConcretesTests` — proves `live_runtime/` and `live_runtime_launcher/` never import `live_system.integration` at all, confirming the retired generation has no path back into the current production composition root.

## 9-10. Phases 9-10 — regression proof

`CurrentRuntimeE2ERegressionTests`/`OperatorAuthorityRegressionTests` (same new test module) construct a real `build_offline_demo_runtime()` runtime, run cycles, and prove: every current stage still reaches `CommandCenterSnapshot` (`BuildingState`, Evacuation Progress, Trajectory, Emergency Response, Recommendation, Guidance, Dynamic Signage all populated; AI/Advisory honestly `None`, unconfigured — unchanged by this cleanup); Command Center and the runtime share the same `StateManager`; and — critically — nothing dispatches automatically (`voice_evacuation_controller.broadcast_log`, `building_control_controller.all_requests()`, `operator_action_gateway.all_signage_instructions()` all empty after several cycles), with every operator-action-gateway provider still exactly the `Simulation*` ones `build_offline_demo_runtime()` has always defaulted in — proving cleanup removed dead code, not authority guarantees or provider ownership.

## 11. Documentation

- `docs/architecture/live_system_integration_audit.md` — marked **HISTORICAL / SUPERSEDED** at the top (its own central finding, "no production composition root exists," is now resolved); body left unmodified as an accurate record of the investigation that led here.
- `docs/architecture/live_command_center_integration.md` — left unmodified; its content (Command Center's Replay/Live dual-mode capability) remains accurate today, the one passing mention of `live_system.integration` in it is still literally true (the module still exists, just slimmed).
- `docs/architecture/synevac_end_to_end_architecture_review.md` — §14's dead-gateway finding marked RESOLVED; §18 priority 3 marked PARTIALLY RESOLVED (the `LiveAIInferenceGateway` default-construction question is explicitly a separate, still-open, out-of-scope question, not a superseded-architecture one).
- This document.

## 12. Regression

Baseline 4318/4318 (commits `c62c2ce`, `c6da250`). This milestone added one new test module (`tests/test_live_runtime_architecture_cleanup.py`, 11 tests) and removed 9 obsolete implementation tests (4 test classes) from `tests/test_live_system.py`. Full suite after this milestone: **4320/4320 passing** (4318 − 9 + 11), zero regressions.
