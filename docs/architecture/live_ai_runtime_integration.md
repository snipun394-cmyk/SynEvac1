# Live AI Inference Runtime Integration

Status: **`LiveOrchestrator` now produces real, structured AI predictions from canonical `BuildingState` every cycle it is configured to. Nothing downstream of that prediction (`decision_policy`, `advisory_system`, building controls, voice broadcast) is wired or changed.**

## 1. Before/after live runtime graph

**Before** (after commit `c6103ce`):

```
Sensors -> [perception_gateway] -> [building_state_gateway] -> BuildingState -> StateManager
                                                                       |
                                                    [ai_inference_gateway]  <- OLD seam, operates on
                                                                                LiveBuildingSnapshot via a
                                                                                feature_row_builder never
                                                                                implemented in production
```

**After** (this milestone):

```
Sensors -> [perception_gateway] -> [building_state_gateway] -> BuildingState -> StateManager
                                                                       |
                                                          [live_ai_gateway]  <- NEW: BuildingState -> LiveAIPredictionSnapshot
                                                                       |               via ai_registry.LiveAIInferenceService
                                                                       v
                                                            StateManager.ai_prediction_snapshot
                                                                       |
                                                    [ai_inference_gateway]  <- OLD seam, still untouched, still unimplemented
                                                                       |
                                                          decision_policy_gateway (unchanged)
                                                                       |
                                                          recommendation_builder (unchanged)
                                                                       |
                                                          command_center_gateway (unchanged)
```

The new `live_ai_gateway` stage is inserted immediately after `building_state_gateway` and is entirely independent of the pre-existing `ai_inference_gateway` slot, which remains exactly as unimplemented-in-production as before (see `docs/architecture/live_system_integration_audit.md` §5).

## 2. AI injection boundary

`live_system.live_ai_gateway.LiveAIInferenceGateway` — a `Protocol`, one method: `predict(state: Optional[BuildingState], time: float) -> Optional[LiveAIPredictionSnapshot]`. `LiveOrchestrator` holds only this gateway — it never constructs a `ModelRegistry`, loads a model artifact, builds a `TrainingDataset`, or calls a feature extractor directly (Phase 3's own explicit requirement); composing an `ai_registry.LiveAIInferenceService` (and the `ModelRegistry` it wraps) is a caller's job, mirroring exactly how `live_system.building_state_gateway` never owns `CameraManager`/`SensorManager` itself.

`predict()` returning `None` is this Protocol's own documented "skip this cycle" signal (used by `ThrottledLiveAIInferenceGateway`, §9) — never "error." The real adapter, `RegistryLiveAIInferenceGateway`, always returns a real `LiveAIPredictionSnapshot` when called with a non-`None` `state`, even on total failure (an `UNAVAILABLE`/`INCOMPATIBLE`/`ERROR` one, never `None`).

## 3. `LiveAIPredictionSnapshot`

`live_system.live_ai_gateway.LiveAIPredictionSnapshot` — immutable, one per cycle:

- `timestamp`, `building_state_timestamp` (the `BuildingState.timestamp` this prediction was actually computed from — `None` if no `BuildingState` existed this cycle), `feature_schema_version`
- `system_status: AISystemStatus` — `AVAILABLE`, `PARTIAL`, `UNAVAILABLE`, `INCOMPATIBLE`, `ERROR`
- `bottleneck: Optional[ai_registry.BottleneckOccurrencePrediction]` — reused verbatim from `ai_registry.inference_service`, never re-wrapped; carries `probability`, `predicted_occurrence`, `threshold`, `model_id`, `model_version`
- `evacuation_time_experimental: Optional[ai_registry.EvacuationTimePrediction]` — same reuse; the field is named `evacuation_time_experimental`, not `evacuation_time` or `rset`, deliberately (§6)
- `errors`/`warnings: Tuple[str, ...]`

**`bottleneck.probability` is never renamed, rescaled, or reinterpreted as "recommendation confidence" anywhere in this module** — `tests/test_live_ai_runtime_integration.py::ProbabilityAndExperimentalLabelingTests` proves the field is literally absent a `confidence` name.

## 4. Model eligibility rules

Only `bottleneck_occurrence` is treated as producing an operational signal — `system_status` only ever reaches `AVAILABLE`/`PARTIAL` based on whether the bottleneck prediction succeeded; the experimental evacuation-time prediction succeeding or failing never changes `system_status` away from what the bottleneck result alone determined (failure there only ever appends a `warning`, never an `error`).

## 5. Bottleneck production-candidate usage

`RegistryLiveAIInferenceGateway._predict_bottleneck()` calls `ai_registry.LiveAIInferenceService.predict_bottleneck_occurrence()` unmodified. Every value on the returned `BottleneckOccurrencePrediction` (`probability`, `predicted_occurrence`, `threshold`, `model_id`, `model_version`) is preserved exactly. No recommendation is generated from it anywhere in this milestone — `decision_policy`/`advisory_system` are never imported by `live_system.live_ai_gateway` (mechanically enforced, §11).

## 6. Evacuation-time experimental restriction

Enforced structurally, not just by convention:

- The field is named `evacuation_time_experimental` on `LiveAIPredictionSnapshot` — there is no field named `evacuation_time`, `predicted_rset`, or `rset` anywhere in this module (`tests/test_live_ai_runtime_integration.py` asserts this directly against the dataclass's own field names).
- Its failure/absence can only ever add a `warning`, never an `error`, and never changes `system_status` on its own (§4).
- `decision_policy`/`advisory_system` are mechanically proven to never import `live_system.live_ai_gateway`, `ai_registry`, or `ai_features` at all (§11) — there is no code path by which this value could reach either package in this milestone.
- `RegistryLiveAIInferenceGateway.__init__(include_evacuation_time: bool = True)` lets a caller omit it entirely.

## 7. Failure behavior

Every failure mode is caught inside `RegistryLiveAIInferenceGateway.predict()` and turned into an honest `LiveAIPredictionSnapshot` field — nothing raises out of it into `LiveOrchestrator.run_cycle()`:

| Scenario | `system_status` | Notes |
|---|---|---|
| No `live_ai_gateway` configured | (no snapshot produced at all) | existing "optional stage" behavior, unchanged |
| No `BuildingState` yet this cycle | `UNAVAILABLE` | `predict(None, time)` still returns a real, honestly-labeled snapshot |
| No compatible bottleneck model registered | `UNAVAILABLE` | `ModelRegistry.get_latest_compatible_model()` returns `None` and no model of that type/deployability is registered at all |
| A model is registered but fails its own schema/dtype compatibility check | `INCOMPATIBLE` | distinguished from `UNAVAILABLE` by querying `LiveAIInferenceService.registry.list_models(...)` directly (§ new `registry` property added to `ai_registry.inference_service.LiveAIInferenceService` this milestone) |
| Missing/partial `BuildingState` fields (camera offline, sensor unavailable, ...) | `AVAILABLE`/`PARTIAL` | `extract_canonical_features()` always returns the full 25-key row with honest `None`s — this is not a failure, just reduced-confidence input, exactly as the AI Feature Parity milestone designed |
| Corrupted model artifact (raises inside `predict_proba`) | `ERROR` | caught, never crashes the cycle |
| Experimental evacuation model unavailable/incompatible/erroring | `system_status` unaffected (bottleneck alone decides); a `warning` is recorded | §4/§6 |

`tests/test_live_ai_runtime_integration.py::SafeFailureModeTests` and `StalePredictionProtectionTests` prove every row above directly, including that Camera/Sensor/FACP/`BuildingState`/`LiveOrchestrator` itself are never affected by an AI failure (`test_inference_exception_does_not_stop_the_live_cycle` runs a full `LiveOrchestrator.run_cycle()` against a deliberately-broken model and asserts `orchestrator.is_running` stays `True`).

## 8. Stale-prediction handling

The simplest architecture-compatible solution, per this milestone's own instruction: **every cycle `live_ai_gateway` is configured, a fresh `LiveAIPredictionSnapshot` — tied to that cycle's own `time` and whatever `BuildingState` existed this cycle — unconditionally replaces the previous one** (`StateManager.update_ai_prediction()`, mirroring `update_building_state()` exactly). A cycle where inference fails produces an honest `UNAVAILABLE`/`ERROR` snapshot, not a re-presentation of the last success. `component_timestamps["ai_prediction_snapshot"]` (the pre-existing per-field freshness-tracking mechanism every other optional stage already uses) is only bumped when a real update happens — this is what makes `ThrottledLiveAIInferenceGateway` (§9) safe: returning `None` to skip a cycle leaves the *previous* snapshot in place under its own, honest, un-bumped timestamp, never disguised as fresher than it is.

## 9. Inference frequency

Measured (`scripts/train_live_compatible_models.py`'s own benchmark, `docs/architecture/live_ai_model_training.md` §10): bottleneck-only inference ≈32 ms, combined (both models) ≈108 ms. `LiveOrchestrator`'s default cycle interval is 1.0 second (1 Hz). **Decision: run every cycle, no throttling, by default** — 108 ms is roughly 11% of a 1-second cycle budget even in the worst case (both models, every cycle), comfortably reasonable, and bottleneck-only (the actually-operational model) is under 3%. `ThrottledLiveAIInferenceGateway(inner, min_interval_seconds=N)` exists as a simple, optional wrapper (no threads, no async, no scheduler) for a future deployment that wants a lower refresh rate, but is not the default and was not needed to make this milestone's own numbers work.

## 10. Performance

See §9's numbers. Registry lookup itself is ~0.015 ms (cached, §21 below) — inference cost is entirely the estimator's own prediction time (200-tree `RandomForestClassifier`/`Regressor`), not anything this milestone's own gateway/registry plumbing adds.

## 11. Simulation/Replay/Future-Live compatibility

The AI boundary is mechanically enforced to stay `BuildingState -> AI`, never `CameraFrame -> AI` or `Scenario -> AI`:

- `tests/test_live_system.py::LiveSystemPackageDependencyDirectionTests::test_never_imports_scenario_ground_truth_control_or_voice_modules` — `live_system/*.py` (including the new `live_ai_gateway.py`) never imports `scenario_definition`/`scenario_generator`/`scenario_runner`/`ground_truth`/`voice_evacuation`/`speaker_manager`.
- `tests/test_live_ai_runtime_integration.py::OriginIndependenceTests` proves the identical `RegistryLiveAIInferenceGateway`/`LiveAIInferenceService` answers correctly for a `BuildingState` built via `ai_features.simulation_extractor` (simulation-origin) and one built via the real `ReplayFrameSource`/`CameraManager`/`MultiCameraFusionEngine` chain (replay-origin) — neither code path is special-cased anywhere in `live_ai_gateway.py`. A future `RTSPFrameSource`-derived `BuildingState` would work identically, with zero changes to this module, for the same reason the CCTV milestone's own Simulation/Replay/Live mode-independence already held for `CameraManager`/`MultiCameraFusionEngine`.

## 12. Remaining disconnected systems

Exactly as instructed, this milestone stops at `BuildingState -> AI Inference -> LiveAIPredictionSnapshot`:

- `LiveAIPredictionSnapshot -> DecisionPolicy` — **not built.** `decision_policy/` never imports `live_system.live_ai_gateway`, `ai_registry`, or `ai_features` (mechanically enforced).
- `LiveAIPredictionSnapshot -> AdvisoryInputs/AdvisoryOrchestrator` — **not built.** Same guard, against `advisory_system/`.
- `AdvisoryReport -> Live Command Center` — **still not built** (unchanged from `docs/architecture/live_system_integration_audit.md` §8/§15; Command Center remains entirely replay-driven).
- **Command Center / Designer debug visibility (Phase 12) — investigated, deliberately deferred, not built.** The Designer's own `BuildingStateDebugPanel` is driven by `BuildingStateDebugRunner`/the Sandbox simulation loop, structurally independent of `LiveOrchestrator`; Command Center itself has no live data path at all (confirmed in the original Live Integration Audit). Wiring "AI Runtime Status" into either would require fabricating a `ModelRegistry`/trained-model-loading path inside Designer/Command Center that does not exist yet (this milestone's own trained models are produced fresh by `scripts/train_live_compatible_models.py` into a temp directory, never persisted to a stable, discoverable location) — exactly the kind of premature plumbing this phase's own instructions warn against ("do not wire replay Command Center panels to fake live data"). Deferred to the later Live Command Center milestone, per this phase's own explicit permission to do so.
- No building control execution, no automatic voice broadcast, and no Decision Policy/Advisory System behavior change were introduced — mechanically enforced by dependency-direction guards in both `tests/test_live_system.py` and `tests/test_live_ai_runtime_integration.py`.
