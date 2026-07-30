# Shadow-Mode Predictive AI Integration

Status: implemented. Phase 1's two Freeze Review cleanup items are done; the prediction pipeline itself
was found, on inspection, to already exist end-to-end (see "What already existed" below) -- this
milestone's own new work is therefore narrower than its own brief assumed: one small, additive
instrumentation field, and a large amount of tracing/proving/documenting. **One important, pre-existing
finding is disclosed prominently in its own section below** -- read it before assuming "Predictions
NEVER influence Recommendation" is 100% true today.

---

## Phase 1: the two Freeze Review cleanup items

### 1. EventBus moved out of `live_system`

The Freeze Review's Finding 1: `live_system/event_bus.py` had zero project imports of its own (a
genuinely leaf, dependency-free pub/sub type), but lived inside `live_system/`, whose `__init__.py`
eagerly imports the entire package (`ai_registry`, `orchestrator`, `sensor_registry`, ...). Every
consumer of the trivial `EventBus`/`EventType` type (`live_occupants`, `evacuation_progress`,
`command_center`) was transitively forced to load the whole orchestration stack.

**Fix**: the real implementation moved, byte-for-byte, to a new `event_bus/` package (`event_bus/bus.py`,
re-exported from `event_bus/__init__.py`). `live_system/event_bus.py` is now a thin, three-line
re-export shim -- `from live_system.event_bus import EventBus` keeps working unchanged for every
existing caller (tests, scripts, `live_system`'s own internal modules). The production packages that
actually caused the package-level cycle (`live_occupants`, `evacuation_progress`, `command_center`,
`live_runtime`) now import `event_bus.bus` directly instead.

**Verified**: an 88-package AST-based circular-dependency scan, re-run after this change, no longer
shows `live_occupants -> live_system -> evacuation_progress -> live_occupants` (or any of that family) at
all. Two unrelated, pre-existing cycles remain (`command_center <-> live_system`, genuinely deeper
coupling unrelated to `event_bus`; and `behavior_library <-> human_decision_engine`, confined to the
Human Behavior Layer) -- both were already disclosed as separate findings by the Freeze Review and are
explicitly out of this milestone's scope.

### 2. Legacy `AIInferenceGateway` retired

The Freeze Review's Finding (Phase 9/11): `live_system.integration.AIInferenceGateway` (a Phase-7,
generation-1 inference seam, `predict(snapshot) -> Mapping[str, Prediction]`) coexisted with the current,
actually-used `live_system.live_ai_gateway.LiveAIInferenceGateway` in the same public API surface --
"which one do I use" risk for a future contributor.

**Fix**: `AIInferenceGateway` deleted from `live_system/integration.py`; `LiveOrchestrator`'s own
`ai_inference_gateway` constructor parameter, attribute, and `run_cycle()` dispatch block removed;
`live_system/__init__.py`'s exports updated. `live_system.state_manager.StateManager.
update_ai_predictions()`/`LiveBuildingSnapshot.ai_predictions` (the storage that legacy seam used to
populate) were deliberately left untouched -- passive data storage, not a gateway; removing a snapshot
field is a materially larger, different change than retiring an unused seam. `live_system.live_ai_gateway.
LiveAIInferenceGateway` is now the ONLY inference-gateway-shaped thing `LiveOrchestrator` accepts.

**Verified**: `tests/test_live_runtime_architecture_cleanup.py` extended (its own established
`_DELETED_SYMBOLS`/`_ALLOWED_MENTIONS` pattern) to prove `AIInferenceGateway` no longer exists, is not
exported, and is mentioned nowhere outside its own retirement notice and historical documentation.

## Phase 2: what already existed (traced, not built)

Tracing `ai_registry`/`live_system.live_ai_gateway` found the ENTIRE pipeline the milestone's own brief
describes already built, already tested, and already wired at the library level, by an earlier "Live AI
Inference Runtime Integration" milestone that predates this one:

- **Feature extraction**: `ai_features.building_state_extractor.extract_canonical_features(BuildingState)
  -> row`. A whole-building, aggregate feature schema (`ai_features.feature_schema.
  CANONICAL_LIVE_FEATURE_NAMES` -- `total_occupant_count`, `camera_active_count`, `facp_panel_state`, ...
  25 fields total), building-shape-independent (no per-zone/per-stair columns) -- **not** the same schema
  as `predictive_dataset`'s own per-candidate Door/Exit/Stair schema from the prior Stair Predictive-
  Feature Live Parity milestone; a genuinely different, coarser-grained model type.
- **Model discovery/loading**: `ai_registry.registry.ModelRegistry` -- `register_model()`/
  `register_model_directory()` load once, cache in memory (Phase 14's own "models should not be reloaded
  from disk every cycle" requirement); `get_latest_compatible_model(model_type, feature_schema_version)`
  is the one discovery query, returning either a valid model or an honest "no compatible model," never a
  silent RESEARCH_ONLY fallback.
- **Prediction execution**: `ai_registry.inference_service.LiveAIInferenceService.
  predict_bottleneck_occurrence(state, timestamp)` / `.predict_evacuation_time(...)` -- extract, validate
  compatibility, predict, wrap every failure mode in `InferenceUnavailableError`, never a fabricated
  result.
- **Gateway / snapshot**: `live_system.live_ai_gateway.RegistryLiveAIInferenceGateway` (the real adapter,
  catches every exception, never crashes the live cycle) producing `LiveAIPredictionSnapshot`
  (`AISystemStatus` AVAILABLE/PARTIAL/UNAVAILABLE/INCOMPATIBLE/ERROR, `bottleneck`,
  `evacuation_time_experimental`, `errors`, `warnings`). `ThrottledLiveAIInferenceGateway` wraps any
  gateway for configurable-interval-only inference.
- **Orchestrator wiring**: `LiveOrchestrator.__init__(live_ai_gateway=...)` + `run_cycle()`'s own
  `if self.live_ai_gateway is not None: ai_prediction_snapshot = self.live_ai_gateway.predict(snapshot.
  building_state, time); ... snapshot = self.state_manager.update_ai_prediction(ai_prediction_snapshot,
  time); self.event_bus.emit(EventType.AI_PREDICTION_UPDATED, ...)` -- already present, unmodified.
- **Factory wiring**: `live_runtime.factory.build_live_runtime(..., live_ai_gateway=...)` already accepts
  an opaque, already-constructed gateway object and threads it straight through to `LiveOrchestrator`.
  **This is a deliberate, tested architectural boundary**
  (`tests/test_live_runtime_architecture_guards.py::GatewayIsTheOnlyExecutionSeamTests`): the factory must
  never import `ai_registry`/`ai_inference`/`ai_training`/`decision_policy` itself to construct a gateway,
  mirroring `live_advisory_gateway`'s own identical "opaque object in" contract. An earlier draft of this
  milestone's own work added a `model_registry=` convenience parameter that imported `ai_registry`
  directly into `factory.py` -- this VIOLATED that boundary (caught immediately by the existing guard
  test) and was reverted. **The correct way to enable Shadow Mode is exactly what the factory already
  supports**: a caller constructs `RegistryLiveAIInferenceGateway(LiveAIInferenceService(model_registry))`
  itself and passes the result as `live_ai_gateway=` to `build_live_runtime()` -- see
  `tests/test_shadow_mode_prediction.py` for a complete worked example.

## Phase 3: pipeline (confirmed, not extended)

```
Camera -> Detection -> Tracking -> Calibration -> Localization -> Occupancy
    -> Crowd Intelligence -> BuildingStateEstimator -> BuildingState
    -> LiveAIInferenceGateway.predict(building_state, time)
         -> extract_canonical_features(building_state)
         -> ModelRegistry.get_latest_compatible_model(...)
         -> model.predict_proba(...) / model.predict(...)
    -> LiveAIPredictionSnapshot
    -> StateManager.update_ai_prediction() -> LiveBuildingSnapshot.ai_prediction_snapshot
    -> EventType.AI_PREDICTION_UPDATED published
```

No downstream consumer of `ai_prediction_snapshot` exists in `Voice`, `Signage`, `Building Control`, or
`Simulation` (verified, Phase 8). **One downstream consumer already exists in Recommendation** -- see the
dedicated section below.

## Phase 4/5: feature extraction and model execution (reused, not modified)

`extract_canonical_features()` is called exactly once per `predict()` call, from inside `ai_registry.
inference_service.LiveAIInferenceService` -- never duplicated, never re-implemented in `live_system`.
UNKNOWN handling, optional features, and missing-observation semantics are exactly what that function
already establishes (unchanged by this milestone). Timestamp consistency: `LiveAIPredictionSnapshot.
building_state_timestamp` records the SOURCE `BuildingState.timestamp` this prediction was computed
from, independent of `LiveAIPredictionSnapshot.timestamp` (the cycle time the prediction was produced) --
both are always set, never conflated. This milestone neither interprets, ranks, nor recommends based on
model output -- it only traces and enables the existing, unmodified execution path.

## Phase 6: prediction snapshot

`LiveAIPredictionSnapshot` already carried every field the milestone's own brief asked for except one:

| Requested field | Existing field | Notes |
|---|---|---|
| timestamp | `timestamp` | cycle time |
| candidate id | *(not applicable)* | this model type predicts a single, whole-building outcome, not a per-Door/Exit/Stair candidate value -- see Phase 2's own schema note. A future per-candidate model would need its own, separate snapshot shape; not built here (would be a new model type, out of this milestone's "do not redesign" scope) |
| prediction values | `bottleneck: BottleneckOccurrencePrediction` / `evacuation_time_experimental: EvacuationTimePrediction` | typed, structured |
| confidence if available | `bottleneck.probability` / `evacuation_time_experimental.uncertainty_seconds` | model probability/uncertainty, never recommendation confidence -- deliberately never renamed |
| model version | `bottleneck.model_id` / `.model_version` | nested per prediction |
| feature version | `feature_schema_version` | top-level AND nested per prediction |
| **prediction latency** | **`inference_duration_seconds`** | **NEW, added by this milestone** -- see below |

**The one genuine addition**: `inference_duration_seconds: Optional[float] = None`, a new, defaulted
field on the (frozen) `LiveAIPredictionSnapshot` dataclass -- every existing construction of this type
keeps working unchanged. Populated by `RegistryLiveAIInferenceGateway.predict()` timing its own body with
`time.perf_counter()` -- pure instrumentation, never a change to WHAT is predicted or HOW. `None` means
"not measured" (e.g. a hand-built test snapshot, or the `state=None` early-return path, which has no
inference work to time), never a fabricated `0.0`.

No decision fields exist anywhere on this type -- no rank, no recommendation, no action, no "what should
happen next."

## Phase 7: shadow logging

Predictions are stored on `LiveBuildingSnapshot.ai_prediction_snapshot`, a field entirely separate from
`recommendations`, `evacuation_guidance`, `crowd_intelligence`, or any simulation-facing state.
Queryable via `StateManager.latest_ai_prediction()` / `LiveOrchestrator.latest_ai_prediction` (a
convenience forwarding property, unchanged), and via `EventBus.history_of(EventType.
AI_PREDICTION_UPDATED)` for the full cycle-by-cycle history. None of these reads have any side effect on
runtime behavior -- querying prediction history can never change what happens next.

## Phase 8: no decision integration -- mechanically proven, with one important, disclosed exception

**Guidance, Voice, Signage, Building Control, Simulation**: mechanically proven clean.
`tests/test_shadow_mode_prediction.py::NoDecisionIntegrationTests` proves, by source-text scan, that none
of `evacuation_guidance/`, `voice_evacuation/`, `dynamic_signage/`, `building_control/`, `simulator/`, or
`simulation_runtime/` import `ai_registry`/`ai_inference`/`ai_training`/`ai_features`/
`live_system.live_ai_gateway` anywhere, and that the orchestrator's own call into
`evacuation_guidance_gateway.compute()` never even mentions `ai_prediction_snapshot` as an argument.
Enforced by dependency direction, exactly as required.

**Recommendation: a genuine, pre-existing, important nuance.** `evacuation_recommendation.engine.
EvacuationRecommendationEngine.compute()` DOES accept `ai_prediction_snapshot` as a parameter (threaded
through from `LiveOrchestrator.run_cycle()`'s own unconditional call --
`self.evacuation_recommendation_gateway.compute(time, snapshot.building_state, ..., snapshot.
ai_prediction_snapshot)` -- this argument is passed whenever the Recommendation gateway is configured at
all, regardless of whether Shadow-Mode AI is separately enabled). This is **not** something this milestone
introduced -- it is a deliberate design from the prior "Live Dynamic Evacuation Recommendation Engine"
milestone, whose own code comments name it explicitly: `evacuation_recommendation/scoring.py`'s own
`score_candidate()` applies `ai_bottleneck_probability` (a single, BUILDING-WIDE value, never per-exit)
identically to every candidate exit for a zone, with the comment "it can never, by construction, change
which candidate ranks higher than another, only the absolute score/explanation/confidence context."

**This milestone verified that claim directly, empirically, against the real scoring code** (`tests/
test_shadow_mode_prediction.py::NoDecisionIntegrationTests::test_recommendation_ranking_is_
mathematically_invariant_to_ai_evidence`): the exit ranking order is proven identical across
`ai_bottleneck_probability` values of 0.0, 0.1, 0.5, 0.9, and 1.0, and against `None`. A second,
end-to-end test (`test_recommendation_decision_fields_identical_before_and_after_shadow_mode`) runs the
SAME building/occupancy through two independent orchestrators -- one with no AI configured, one with a
real trained model producing real predictions every cycle -- and proves `recommended_exit_id`,
`ranked_exit_ids`, `status`, and `confidence` are byte-identical, and that downstream `Guidance`
(`recommended_exit_id`, `ordered_door_ids`, `ordered_stair_ids`) is identical too.

**The honest, precise statement of Phase 8's guarantee is therefore**: enabling Shadow-Mode AI can never
change WHICH exit is recommended, WHICH route Guidance plans, or any OPERATIONAL evacuation-behavior
field -- proven mathematically (a uniform additive constant cannot change a ranking's relative order) and
empirically (the tests above). It is **not** true, in the strictest possible sense, that "the prediction
path is completely passive all the way through Recommendation's own code" -- raw prediction data does
reach `evacuation_recommendation`'s scoring function, and DOES influence purely diagnostic/explanatory
output on the (unaffected) top-ranked candidate: `reason_codes` gains an `AI_BOTTLENECK_RISK_LOW`/
`AI_BOTTLENECK_RISK_ELEVATED` entry, and each candidate's raw internal `score` value shifts (though
`confidence`, computed independently of `score`, does not). This milestone's own explicit instruction was
"do not modify Recommendation" -- this pre-existing, already-reviewed, already-bounded channel was left
completely untouched, and is disclosed here in full rather than glossed over. **If a future milestone
wants Recommendation's own code to be reached by zero AI-shaped data at all during a Shadow-Mode-only
deployment phase (not just decision-invariant to it), that would require a deliberate, separate decision
to change `live_system/orchestrator.py`'s own call site (stop passing `ai_prediction_snapshot` into
`evacuation_recommendation_gateway.compute()`) -- not attempted here, and explicitly flagged as an open
question for a future milestone, not a defect this one silently accepted.**

## Phase 9: validation

`tests/test_shadow_mode_prediction.py::ShadowModeValidationTests` -- against a real, small, trained
`bottleneck_occurrence` model (mirrors `tests/test_live_ai_runtime_integration.py`'s own `setUpModule`
convention):

- A feature vector is genuinely produced every cycle the gateway runs (`AISystemStatus.AVAILABLE`,
  `bottleneck` populated, `feature_schema_version` set).
- A prediction is executed and stored every orchestrator cycle Shadow Mode is configured
  (`result.ai_prediction_snapshot` populated; `orchestrator.latest_ai_prediction` reflects it).
- The runtime continues normally across multiple cycles with Shadow Mode active -- no exception, no
  behavior change to cycle progression.
- Recommendation's decision fields are identical before/after Shadow Mode (the central proof, detailed
  in Phase 8 above).

## Phase 10: performance

`tests/test_shadow_mode_prediction.py::ShadowModePerformanceTests` -- ~20 cameras, ~100 occupants, ~20
stairs (`BuildingState` assembled via the real `BuildingStateEstimator`, a real trained model): feature
extraction + model inference + snapshot assembly measured at **~11.4 ms** total (gateway-internal
`inference_duration_seconds` measured the same ~11.4 ms, confirming feature extraction itself is a small
fraction of that). This is incremental cost on top of the perception/tracking pipeline, which was already
separately benchmarked at ~1-2 ms in the Stair Flow Intelligence and Stair Predictive-Feature Live Parity
milestones -- Shadow-Mode AI inference is the dominant, but still small (order-of-10-ms), addition to a
single live cycle at this scale.

## Phase 11 note: dependency graph

```
ai_registry (LiveAIInferenceService, ModelRegistry)
ai_features (extract_canonical_features, feature_schema)
        │  imported by
        ▼
live_system.live_ai_gateway (RegistryLiveAIInferenceGateway, LiveAIPredictionSnapshot)
        │  passed in as an opaque object (never constructed internally)
        ▼
live_runtime.factory.build_live_runtime(live_ai_gateway=...) -- unmodified boundary
        │
        ▼
live_system.orchestrator.LiveOrchestrator.run_cycle()
        │  writes
        ▼
live_system.state_manager.LiveBuildingSnapshot.ai_prediction_snapshot
        │
        ├──> evacuation_recommendation.engine (see Phase 8's own disclosed exception --
        │     decision-invariant, not zero-coupling)
        ├──> Guidance / Voice / Signage / Building Control / Simulation: NO edge (verified)
        └──> queryable via StateManager/EventBus for future Command Center display (not built here)
```

## Future activation path

1. Train and register a real `bottleneck_occurrence` (and, optionally, `evacuation_time`) model via the
   existing `ai_training`/`ai_registry` pipeline (unchanged, out of this milestone's scope).
2. A caller (future `main.py`/`live_runtime_launcher` wiring, or an operator-facing configuration step --
   neither built here) constructs `RegistryLiveAIInferenceGateway(LiveAIInferenceService(model_registry))`
   (optionally `ThrottledLiveAIInferenceGateway`-wrapped for a configurable inference interval) and passes
   it as `live_ai_gateway=` to `build_live_runtime()`.
3. Shadow Mode is now active: predictions run, are logged, and are queryable -- with zero change to
   evacuation behavior, per Phase 8/9's own proofs.
4. A FUTURE, deliberate milestone would be required to let predictions influence any operational decision
   -- not started, not assumed, not implied by this one.

## Known limitations

- The disclosed Recommendation coupling (Phase 8) -- decision-invariant, not zero-coupling. Worth a
  future, deliberate decision, not silently left ambiguous.
- Only `bottleneck_occurrence` (`PRODUCTION_CANDIDATE`) is treated as an operational-shape signal;
  `evacuation_time_experimental` (`EXPERIMENTAL`) is deliberately named to prevent misreading it as an
  authoritative RSET, per the pre-existing `live_ai_gateway.py` design this milestone did not change.
- No production entry point (`main.py`, `live_runtime_launcher/`) currently constructs and passes a real
  `live_ai_gateway` by default -- Shadow Mode remains fully opt-in, requiring a caller to explicitly
  compose one (Phase 3's own "no downstream consumer yet" requirement, satisfied).
- `inference_duration_seconds` measures wall-clock time inside `RegistryLiveAIInferenceGateway.predict()`
  only (feature extraction + model inference + result assembly) -- it does not include time spent inside
  `ThrottledLiveAIInferenceGateway`'s own throttle check (deliberately near-zero) or any upstream
  perception/Crowd-Intelligence/BuildingState-assembly cost, which remains separately benchmarked
  elsewhere.
