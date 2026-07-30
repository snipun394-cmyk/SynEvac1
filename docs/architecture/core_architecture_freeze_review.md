# Core Architecture Freeze Review & Readiness Audit

Status: **review only — zero code changes**. As instructed, no functionality was implemented or
redesigned; two genuine architectural findings were discovered (Phase 1) and are recorded as technical
debt (Phase 11), not fixed, because neither causes an active correctness failure and both would require
a multi-file reorganization out of proportion to a freeze review. Baseline: commit `02b958b`, 5040 tests
passing, unchanged by this review.

---

## Phase 1 — Global architecture audit

### Architecture diagram (verified against actual imports, not documentation)

```
 Designer (models/*, designer/)
   │  authors Building/Floor/Zone/Door/Exit/Staircase/Camera/Detector/... — the ONE source
   │  of engineering-object truth, saved/loaded via Project/serialization
   ▼
 Building Model (models.building.Building)
   │  read-only from here down; Building is never mutated by anything below it except Designer/live_runtime_launcher (project load)
   ├──────────────────────────────┬─────────────────────────────┐
   ▼                              ▼                             ▼
 Navigation                    Perception                   Scenario Generator
 (navigation/, pathfinding/)   (camera_calibration/,        (scenario_generator/ — takes an
   │  NavigationGraphGenerator  human_detection/,             ALREADY-BUILT Building as input,
   │  builds Node/Edge from     tracking/, live_camera_       never constructs Stair/Door/Zone
   │  Building; PathfindingEngine  pipeline/, cross_camera_    geometry itself)
   │  is the ONE routing entry    identity/, multi_camera_
   │  point every subsystem       fusion/)
   │  below reuses                 │  Camera → YOLO → Tracking →
   ▼                                │  Calibration → WorldProjection →
 Simulation                         │  Identity Resolution → LiveOccupant
 (simulator/, simulation_runtime/)  ▼
   │  MultiAgentSimulation walks   Live Occupants
   │  a FIXED, pre-planned Route   (live_occupants/) — the ONE
   │  one edge at a time; reuses   canonical runtime occupant
   │  PathfindingEngine, never     registry (LiveOccupantManager),
   │  re-plans                    canonical_occupancy() is the
   ▼                              single source every consumer below reads
 Hazard / Fire / Tenability          │
 (hazard/, fire_growth/,             ▼
  smoke_propagation/, tenability/)  Observable Assets / Camera Coverage / Stair Flow
   │                                (observable_assets/, camera_coverage/, stair_flow/)
   ▼                                │  pure evidence layers, no occupancy computation
 Building State                     │  of their own — read LiveOccupantManager + calibration
 (building_state/)                  ▼
   │  BuildingStateEstimator      Crowd Intelligence
   │  FUSES hazard+occupancy+     (crowd_intelligence/)
   │  camera/sensor status +      │  reads canonical_occupancy() once per cycle,
   │  FACP/control/fire-safety/   │  computes zone/door/exit/stair metrics +
   │  fire-water/observable-      │  stair_flow_metrics — reporting only, never decides
   │  assets/camera-coverage      ▼
   │  into ONE immutable         Feature Extraction
   │  BuildingState snapshot      (predictive_dataset/{simulation,live}_extractor*.py)
   ▼                              │  SIM side reads MultiAgentSimulationResult;
 Recommendation                   │  LIVE side reads CrowdIntelligenceSnapshot/
 (evacuation_recommendation/)     │  stair_flow/evacuation_progress — same schema both ways
   │  SafeExitDistanceCalculator  ▼
   │  wraps PathfindingEngine    Predictive AI (offline)
   │  directly — Stair-aware      (predictive_model/, ai_registry/, ai_training/,
   │  by construction, never      ai_inference/, ai_explainability/)
   │  floor-local Euclidean       │  trains/evaluates OFFLINE; ai_registry.
   ▼                              │  LiveAIInferenceService + live_system.live_ai_gateway.
 Guidance                         │  LiveAIInferenceGateway ALREADY EXIST as a gateway
 (evacuation_guidance/)           │  abstraction — built, tested, NOT wired into
   │  route_planner reuses        │  live_runtime.factory.build_live_runtime() in production
   │  Recommendation's own        ▼
   │  route, never re-plans      Live Runtime / live_system
   ▼                              (live_runtime/, live_system/, live_runtime_launcher/)
 Voice / Signage / Command Center  orchestrates one cycle: perception → BuildingState →
 (voice_evacuation/,               CrowdIntelligence → Recommendation → Guidance →
  dynamic_signage/, command_center/) Voice/Signage/Command Center, via EventBus pub-sub
```

### Circular dependencies — mechanically detected, then verified by hand

A package-level import-graph scan (AST-based, 88 top-level packages) found **9 cycle paths reducing to
2 genuinely distinct root causes** (the rest are transitive restatements of the same two):

**Finding 1 (real, but not a runtime bug today) — `live_system.event_bus` is architecturally
foundational but physically nested inside the top-of-stack orchestration package.**
`event_bus.py` itself has **zero** project imports (verified: only `dataclasses`/`enum`/`typing`/
`uuid`) — a genuinely leaf, dependency-free pub-sub type. But `live_occupants`, `evacuation_progress`,
and others import it as `live_system.event_bus`, and **`live_system/__init__.py` eagerly imports the
ENTIRE package** (`live_ai_gateway` → `ai_registry`, `orchestrator`, `sensor_registry`, `incident_manager`,
...) on any submodule import. This means `from live_system.event_bus import EventBus` transitively loads
`ai_registry` and the full orchestrator stack — a real hidden coupling and a textbook dependency-
inversion violation (a foundational type should not require its own top-level consumer package to fully
initialize first). It has not caused a failure (5040/5040 tests pass, 222/222 architecture-guard tests
pass) only because nothing in the cycle accesses a not-yet-initialized attribute during import — this is
fragile, not broken. **Recommendation: extract `event_bus.py` into its own zero-dependency package
(e.g. a new `event_bus/` package) in a future cleanup pass.** Not fixed here — the blast radius
(~10+ import sites across `live_occupants`, `evacuation_progress`, `crowd_intelligence`, `live_system`
itself) is out of proportion to a review milestone, and it does not block Shadow-Mode readiness (nothing
in the predictive-feature path touches `event_bus`).

**Finding 2 (real, pre-existing, simulation-only) — `behavior_library` ↔ `human_decision_engine` is a
genuine bidirectional package coupling.** `behavior_library/dynamic_human_strategies.py` imports
`human_decision_engine.engine`; `human_decision_engine/engine.py` imports
`behavior_library.assistance_strategies`. Both packages depend on each other's actual logic, not just a
misplaced leaf type. This is Human Behavior Layer machinery (simulation-side decision strategies),
entirely outside the evacuation/perception/predictive pipeline this review is gating. **Not fixed** —
same reasoning as Finding 1: real, pre-existing, non-blocking, out of scope for a conservative freeze
review.

No other cycles were found. Every other package-level edge in the diagram above is one-directional.

### Hidden coupling / duplicated responsibility / drift

- **No duplicated occupancy computation** — verified via `docs/architecture/canonical_live_occupancy.md`'s
  own history (four independent zone-grouping implementations existed before that milestone; all four
  now read `LiveOccupantManager.canonical_occupancy()`). Still true today (unchanged by any of the six
  Stair-focused milestones in this session).
- **No duplicated pathfinding** — every routing consumer (`Simulation`, `Recommendation`, `Guidance`,
  Scenario/Building Analysis) goes through the single `pathfinding.engine.PathfindingEngine` — verified
  directly for `evacuation_recommendation.ranking.SafeExitDistanceCalculator` this session (Stair
  Simulation Reliability Audit milestone).
- **Deliberate, disclosed dual-mechanism, not accidental duplication**: `candidate_recent_flow_rate`
  (predictive feature) has a genuinely different LIVE mechanism per candidate type (Exit reuses
  `evacuation_progress.ExitFlow`, Stair now reuses `stair_flow.StairFlowMetrics.exits`, Door keeps a
  zone-transition proxy) — each disclosed in `predictive_dataset/schema_v4.py`'s own field
  documentation, not silently inconsistent.
- **`live_system.integration.AIInferenceGateway`** (an older Phase-7-era seam) and
  **`live_system.live_ai_gateway.LiveAIInferenceGateway`** (newer, proven `ai_registry`-backed) coexist
  in the same package — the OLDER one is explicitly documented as unused in production ("operating on
  LiveBuildingSnapshot via an injected feature_row_builder that was never implemented in production").
  This is disclosed drift, not silent duplication, but it is real dead-weight in the public API surface
  — flagged in Phase 9/11.

## Phase 2 — Module ownership

| Concept | Canonical owner | Read-only consumers | Writers |
|---|---|---|---|
| Building geometry | `models.building.Building` (+ Floor/Zone/Door/Exit/Staircase) | everything downstream | Designer only (+ project load) |
| Navigation graph | `navigation.graph_builder.NavigationGraphGenerator` → `NavigationGraph` | Pathfinding, Simulation, Recommendation, Guidance | rebuilt fresh from Building on every use, never mutated |
| Routing | `pathfinding.engine.PathfindingEngine` | Simulation, Recommendation, Guidance, Scenario/Building Analysis | none (pure query) |
| Occupancy (canonical) | `live_occupants.manager.LiveOccupantManager.canonical_occupancy()` | Crowd Intelligence, Evacuation Progress, Emergency Response, `ai_features`, Command Center | `LiveOccupantManager.update()`/`sweep_missing()` only |
| Hazards | `hazard.snapshot.HazardSnapshot` (sim) / real sensors (live, via `sensor_manager`/`facp`) | `building_state.BuildingStateEstimator` | Hazard Evolution engine (sim) / sensor fusion (live) |
| Stair flow | `stair_flow.compute.compute_stair_flow_snapshot()` | `crowd_intelligence.models.CrowdIntelligenceSnapshot.stair_flow_metrics`, `predictive_dataset.live_extractor_v2_1` | none downstream — pure derivation from `live_occupants` history |
| Camera coverage | `camera_coverage.compute.compute_camera_coverage_snapshot()` | `BuildingState.camera_coverage`, optionally `stair_flow` (provenance only) | none downstream |
| Observable assets (occupancy truth for Stair etc.) | `observable_assets.facts.compute_asset_occupancy_snapshot()` | Crowd Intelligence, `BuildingState.observable_assets` | none downstream |
| Recommendation | `evacuation_recommendation.engine.EvacuationRecommendationEngine` | Guidance, Voice, Signage, Command Center | none — pure computation each cycle |
| Guidance | `evacuation_guidance.engine.EvacuationGuidanceEngine` | Voice, Signage, Command Center | none |
| Predictions (offline) | `predictive_model` (training/evaluation), `ai_registry.ModelRegistry` (deployed artifacts) | `live_system.live_ai_gateway` (gateway, not yet wired to production `live_runtime.factory`) | trained offline only, by `predictive_model`/`ai_training` scripts |
| Live orchestration | `live_system.orchestrator.LiveOrchestrator` + `live_runtime.factory.build_live_runtime()` | Command Center, Voice/Signage dispatch | drives every cycle |

**Every major concept has exactly one canonical owner.** No ownership conflicts were found — this is the
single strongest finding of the whole review, and it is not an accident: it is the direct, repeated
product of the "canonical source of truth" discipline this codebase enforces at every milestone
(`canonical_live_occupancy.md`, the `AssetApproachMetrics.observed_occupant_count` reuse pattern,
`stair_flow` reusing `crowd_intelligence`'s already-computed occupancy rather than re-deriving it, this
session's own `build_stair_flow_snapshot_for_prediction()` reusing `stair_flow.compute` rather than a
new computation).

## Phase 3 — Boundary validation

Verified via the **222 passing architecture-guard tests** already in the suite (28 dedicated guard
files, one or more per major package), each mechanically scanning real import statements (AST or regex)
rather than trusting documentation. Representative, verified boundaries:

- `crowd_intelligence/` — cannot import `ai_*`, `advisory_system`, `command_center`, `voice_evacuation`,
  `building_control`, `facp`, raw YOLO/RTSP backends; cannot call FACP/Voice/Building-Control action
  verbs. Allow-list: `live_occupants`, `models`, `navigation.edge`, `simulator.capacity`,
  `behavior_recognition.observation`, `observable_assets`, `stair_flow`.
- `stair_flow/` — cannot import `navigation.graph`, `evacuation_recommendation`, `evacuation_guidance`,
  any `ai_*`/`predictive_*`, `simulator`/`simulation_runtime`, any Voice/Signage/Building-Control/FACP
  package (verified this session).
- `predictive_dataset.simulation_extractor_v2_1` — cannot import `live_occupants`, `crowd_intelligence`,
  `evacuation_progress`, or `target_generator` (leakage boundary). `live_extractor_v2_1` — cannot import
  `simulator`, `target_generator`, and must never construct a `LiveOccupantManager` itself.
  `crowd_intelligence`/`evacuation_progress` — must never import `predictive_dataset`/`predictive_model`
  (ML layer depends on the deterministic layer, never the reverse — verified this session).
- Designer never reaches into Simulation/Perception/Live Runtime internals — it authors `models.*` only;
  every downstream package treats `Building` as read-only.
- Live Runtime never reaches backward into Designer beyond initial project load
  (`live_runtime_launcher`), and never bypasses `BuildingStateEstimator` to hand-assemble a
  `BuildingState`.

**No improper cross-layer reach was found** beyond the two Phase-1 findings (which are both
LATERAL/misplacement issues, not a layer reaching in a forbidden DIRECTION — `event_bus` is still only
ever consumed as data/pub-sub, never as a backdoor for e.g. `live_occupants` to call into orchestration
logic).

## Phase 4 — Data flow audit

```
Camera → YOLO (human_detection) → Tracking (tracking/) → Cross-Camera Identity (cross_camera_identity/)
  → Calibration (camera_calibration/) → World Projection (WorldProjector.project())
  → Multi-Camera Fusion (multi_camera_fusion/) → LiveOccupant (live_occupants.manager.update())
  → canonical_occupancy() [ONE producer, memoized per (timestamp, mutation-version)]
  → Crowd Intelligence (CrowdIntelligenceEngine.compute()) [ONE producer, calls stair_flow/observable_assets, never recomputes occupancy]
  → Feature Extraction (predictive_dataset.live_extractor*) [reads CrowdIntelligenceSnapshot + stair_flow snapshot, never recomputes]
  → Prediction (ai_registry.LiveAIInferenceService, gateway exists, NOT production-wired)
  → Recommendation (EvacuationRecommendationEngine, reads BuildingState + PathfindingEngine, unaware of AI predictions today)
  → Guidance (EvacuationGuidanceEngine, reads Recommendation's own output, never re-plans)
```

Every transition has exactly one producer and well-defined, read-only consumers. **No duplicated
calculations, no hidden shortcuts** were found along this specific chain (this is the chain the six
prior milestones in this session repeatedly traced and tested end-to-end: multi-camera dedup proofs,
sim/live parity proofs, canonical occupancy memoization). The one genuine gap in this chain (not a
duplication, an absence): **Recommendation does not yet consume any AI prediction** — confirmed
deliberate (`live_ai_gateway`'s own docstring: "this milestone stops before any code path could let it
reach Decision Policy or Advisory System") and explicitly out of scope for this review to change.

## Phase 5 — Predictive AI readiness

- **Feature extraction**: schema-driven (`ai_features.feature_schema.AIFeatureField`), versioned
  (`SCHEMA_VERSION`/`SCHEMA_VERSION_V4`), with every field's SIM and LIVE source individually documented
  and, where proven, tested for numeric equivalence (this session's own Stair Predictive-Feature Live
  Parity milestone). Ready.
- **Observation semantics**: `ObservationStatus.OBSERVED`/`UNKNOWN` (observable_assets),
  `CoverageState` (camera_coverage), `StairFlowMetrics` gating — all follow one consistent "None means
  genuinely unavailable, never a fabricated zero" discipline, verified across every layer touched this
  session. Ready.
- **UNKNOWN handling**: consistent and tested at every layer (occupancy, coverage, flow, predictive
  feature extraction) — a real 0 and an honest UNKNOWN are never conflated anywhere in the traced chain.
  Ready.
- **Simulation/live parity**: proven with controlled equivalence tests for the one feature audited in
  depth this session (`candidate_recent_flow_rate`, Stair); the graph-context V4 fields are parity-by-
  construction (literal shared code); several base-schema fields remain honestly APPROXIMATE_PARITY
  (queue_length, approaching_count, adjacent_zone_occupancy, congestion_trend, congestion_level,
  Door's flow_rate, total_active_occupant_count) — disclosed, not fixed, not blocking (a model trained
  against these can still be evaluated honestly since the approximation is documented per-field).
- **Timestamp consistency**: every snapshot type in the traced chain carries its own `timestamp`, and
  `LiveOccupantManager.canonical_occupancy(timestamp)` memoizes per (timestamp, mutation version) so
  every same-cycle consumer sees identical data — verified.
- **Identity consistency**: `occupant_id` is the one canonical identity threaded through the entire
  chain from cross-camera resolution through to predictive feature rows; multi-camera dedup is proven
  structurally (not just tested) at the `LiveOccupantManager.update()` idempotency layer, and every
  downstream consumer inherits that guarantee rather than re-deriving it.
- **Extensibility**: the `ObservableAssetKind` registration pattern (Stair is the first of an anticipated
  Door/Exit/Assembly-Point/... family) and the `AIFeatureField`/schema-versioning pattern
  (`SCHEMA_VERSION` → `SCHEMA_VERSION_V4`, additive-only) are both proven extension points, not
  aspirational ones — a second asset kind was already registered zero-framework-change in a prior
  milestone's own test suite.

**Verdict: the architecture is ready for Shadow-Mode Prediction without structural redesign.** The
gateway abstraction (`live_system.live_ai_gateway.LiveAIInferenceGateway`) already exists, is already
tested, and already enforces exactly the safety posture Shadow-Mode requires (typed
AVAILABLE/PARTIAL/UNAVAILABLE/INCOMPATIBLE/ERROR status, exceptions never propagate into the live
cycle, explicit scoping that keeps predictions away from Decision Policy/Advisory System). **What remains
is wiring** (`live_runtime.factory.build_live_runtime()` does not yet construct/attach this gateway in
production), not architecture.

## Phase 6 — Threading / update order

SynEvac is **single-threaded and deterministic** throughout the traced chain — no subsystem in
Simulation, Perception, Crowd Intelligence, or Feature Extraction spawns its own thread or async task.
`MultiAgentSimulation` uses a single deterministic event heap (`heapq`, tie-broken by an
`itertools.count()` counter — verified this session while auditing Stair capacity/congestion). Live
Runtime's own cycle (`live_system.orchestrator`/`update_loop`) runs subsystems in a fixed, documented
order per cycle (perception → BuildingState → CrowdIntelligence → Recommendation → Guidance →
dispatch), with `LiveOccupantManager.canonical_occupancy()`'s memoization specifically existing to
prevent a stale-read/double-compute hazard within one cycle (traced and documented in
`canonical_live_occupancy.md`).

**Future multithreading hazards, not yet present but worth flagging**: `LiveOccupantManager`,
`CrowdIntelligenceEngine`, and `stair_flow`'s own functions are plain Python objects/functions with no
locking of any kind — completely correct for the current single-threaded cycle, but if a future
architecture introduces concurrent camera ingestion threads writing into the SAME `LiveOccupantManager`
instance without funneling through one serialized cycle boundary, `_store()`/`_version` bumping is not
thread-safe today. Not a current bug (nothing does this yet) — a documented risk for Phase 7's own
"future ML models"/scaling discussion.

## Phase 7 — Extensibility review

| New sensor/capability | Estimated effort without redesign | Why |
|---|---|---|
| Additional detector types (e.g. new sensor kind) | **Low** | `sensor_manager`/`SensorStatus`/`DetectorAssetState` already generic across Smoke/Heat/MCP; a new kind is a new reading type + registration, proven pattern |
| Future ML models | **Low** | `ai_registry.ModelRegistry` + `Deployability` classification (`PRODUCTION_CANDIDATE`/`EXPERIMENTAL`) + `ai_features.feature_schema` versioning already designed for exactly this — proven by `BottleneckOccurrenceModel_LiveCompatible` vs `EvacuationTimeModel_LiveCompatible` coexisting today under different deployability tiers |
| A second Observable Asset kind (Door/Exit/Assembly Point) | **Low** | proven zero-framework-change in `tests/test_observable_asset_extensibility.py` (a prior milestone's own fake asset kind registers cleanly) |
| Drone cameras | **Medium** | `camera_calibration`'s pinhole model assumes a fixed, calibrated mount (`CameraExtrinsics.position`/`mount_height`); a moving platform would need per-frame extrinsics, a genuinely new (if analogous) calibration concept — not a redesign of `WorldProjector`'s math, but a new extrinsics-source abstraction |
| Thermal cameras | **Medium** | `human_detection` is YOLO-backend-shaped (`HumanDetector` interface already abstract); a thermal-specific detector implementing the same interface fits the existing seam, but thermal-specific pre/post-processing (no RGB assumption) is new code, not a redesign |
| LiDAR | **Medium-High** | no existing 3D-point-cloud consumer anywhere in the perception chain; would need a new localization path parallel to (not replacing) `WorldProjector`, likely feeding the SAME `LiveOccupant.world_position` contract — the CONTRACT is reusable, the producer is genuinely new |
| RFID / BLE positioning | **Medium** | these are non-visual identity+position sources; `LiveOccupantManager.update()` already accepts `camera_id=None`/`world_position` from any source in principle (verified: `Optional[str]` camera_id, `Optional[Tuple[float,float]]` position) — the seam exists, but occupant IDENTITY would need a new resolver parallel to `cross_camera_identity` (an RFID tag isn't a camera track) |

**No subsystem examined requires architectural redesign to add any of these** — every one is additive
against an existing seam (a new detector/model/asset-kind registration, or a new producer feeding an
already-generic `LiveOccupant`/`BuildingState` contract). The higher-effort items (LiDAR, drone cameras)
are higher effort because they are genuinely new PHYSICS/GEOMETRY, not because the architecture resists
them.

## Phase 8 — Performance review (identification only, no optimization)

- **Largest computational surface**: `predictive_dataset/` campaigns (millions of rows across
  thousands of scenarios) — already an established, accepted cost center for OFFLINE dataset generation,
  not a live-path concern.
- **Live-path hotspot candidates**: `CrowdIntelligenceEngine.compute()` recomputes zone/door/exit/stair
  metrics every cycle over the full Building index (`self._zones`/`_doors`/`_exits`/`_stairs`) — O(assets
  × occupants) per cycle via `compute_queue_metrics`/`compute_zone_metrics`; `stair_flow.compute_stair_
  flow_snapshot()` is O(occupants × bounded_history_length) — empirically benchmarked this session at
  ~2ms for 100 occupants/20 stairs/20 cameras, and ~1.2ms for the full predictive-feature extraction path
  at the same scale. Neither shows superlinear scaling risk at building-realistic occupant counts
  (hundreds, not tens of thousands).
- **Memory hotspots**: `OccupantHistory`'s bounded ring buffers (`max_length=30` default, independently
  per history type) already cap per-occupant memory; `LiveOccupantManager`'s secondary indices (`_by_zone`/
  `_by_floor`/`_by_stair`/etc.) are O(active occupants), not a concern at realistic scale.
- **Future scaling risk worth flagging**: `crowd_intelligence.trends.TrendTracker` and similar per-
  cycle-keyed trackers (`f"asset_congestion:{asset_id}"`, `f"stair_flow:{stair_id}"`-shaped keys) grow
  one entry per distinct asset id ever observed, for the lifetime of the process — no eviction policy
  was found. Not a problem for a single building's fixed asset count, but worth a documented note for a
  future long-running, multi-building, or dynamically-reconfigured deployment.
- **Not examined in depth** (out of this review's realistic scope at this effort level): `advisory_system/
  advisory_engine.py` (1532 lines, the single largest file in the codebase) and `command_center/
  recommendation_center.py` (1076 lines) were not traced for internal hotspots — flagged as a specific
  follow-up in Phase 9/11 rather than asserted clean.

## Phase 9 — API review

- **Confusing/overlapping names**: `live_system.integration.AIInferenceGateway` (legacy, unused in
  production) vs. `live_system.live_ai_gateway.LiveAIInferenceGateway` (current, proven) — both exported
  from the same `live_system/__init__.py`, both plausible-sounding to a new reader, only one actually
  wired anywhere. This is the single clearest "confusing duplicate interface" finding in the review.
- **Leaky abstraction, minor**: `camera_coverage.compute.compute_stair_flow_snapshot()`'s optional
  `camera_coverage` parameter exists ONLY to enrich provenance TEXT (a debug string), never gating logic
  — documented clearly in-code, but a reader skimming the signature could reasonably expect it to affect
  the OBSERVED/UNKNOWN determination itself. Low severity, already disclosed in the docstring.
- **Oversized files** (candidates for eventual splitting, not reviewed line-by-line this session):
  `advisory_system/advisory_engine.py` (1532 lines), `command_center/recommendation_center.py` (1076),
  `predictive_dataset/topologies_v3.py` (1041 — largely repetitive topology-builder functions, likely
  fine as-is given its declarative nature), `live_system/orchestrator.py` (1031).
- **Unused abstractions**: `live_system.integration.*` (see above) is the clearest case — kept alive only
  because `LiveOrchestrator`'s own constructor still type-annotates against it, per its own documented
  disclosure. A candidate for removal in a future minor-cleanup pass, not blocking anything.
- **Otherwise**: the packages this session worked in directly (`observable_assets`, `camera_coverage`,
  `stair_flow`, `crowd_intelligence`, `predictive_dataset.live_extractor*`) all follow one consistent,
  well-documented naming/shape convention (`compute_*_snapshot()` pure functions, frozen dataclasses,
  `ObservationStatus`-style enums, `.for_stair()`/`.observation_for()`-style total accessors) — no
  confusing names or leaky abstractions found there.

## Phase 10 — Test coverage review

- **Well-covered** (verified via this session's own audits and existing suite size): Stair perception/
  flow/predictive-parity chain (151 dedicated tests), Crowd Intelligence (7 files), navigation/pathfinding
  (proven via 394 passing tests in a broad sweep this session), predictive_dataset (38 test files — the
  single most heavily tested package in the repo).
- **Meaningful gap 1**: `live_system` has only 1 file matching a direct `test_live_system*` naming
  pattern despite being 4765 lines and the busiest orchestration package in the codebase (many of its
  individual gateways ARE tested under other names — `test_live_command_center.py`,
  `test_application_live_runtime_launcher.py`, etc. — but there is no single test suite exercising
  `LiveOrchestrator`'s own full-cycle ordering end-to-end with ALL gateways wired, matching Phase 6's own
  "deterministic update order" concern). Worth a dedicated orchestrator-level integration test in a
  future milestone, not urgent.
- **Meaningful gap 2**: `advisory_system/` (the single largest file in the repo) was not examined this
  session and its test coverage was not verified — flagged for a future targeted review rather than
  asserted adequate or inadequate.
- **Meaningful gap 3**: the newly-identified `live_system.live_ai_gateway`/`ai_registry` Shadow-Mode
  gateway path has real tests (per `tests/test_live_ai_runtime_integration.py`, `tests/test_ai_registry.py`
  seen in this session's own full-suite runs), but **no test exercises it wired into
  `live_runtime.factory.build_live_runtime()`** — because it isn't wired there today. This is expected
  (not a bug), but is precisely the gap a future Shadow-Mode wiring milestone would need to close with
  new tests, not an existing coverage hole.
- Trivial getter/setter coverage was not inventoried, per this phase's own instruction to ignore it.

## Phase 11 — Technical debt review

| Severity | Item | Why it matters |
|---|---|---|
| **High** | `live_system.event_bus` physically nested inside the eagerly-`__init__`-importing `live_system` package, creating a package-level circular dependency and forcing every consumer of a trivial pub-sub type to transitively load `ai_registry`/orchestrator/sensor-registry | Real hidden coupling (Phase 1); fragile (works today only because nothing accesses a not-yet-initialized attribute mid-cycle); could break under reordering of `live_system/__init__.py`'s own import list |
| **Medium** | `behavior_library` ↔ `human_decision_engine` bidirectional package coupling | Real, pre-existing, simulation-only (Human Behavior Layer); does not affect the perception/predictive-AI chain this review gates |
| **Medium** | `live_system.integration.AIInferenceGateway` (legacy, unused-in-production seam) coexists with `live_system.live_ai_gateway.LiveAIInferenceGateway` (current) in the same public API surface | Genuine "which one do I use" risk for a future contributor wiring Shadow-Mode; disclosed in-code, but not removed |
| **Low** | No eviction policy on `TrendTracker`-style per-asset-id-keyed caches | Only matters for a long-running, dynamically-reconfigured, or multi-building deployment — not today's single-building live session shape |
| **Low** | `advisory_system/advisory_engine.py` (1532 lines) and `command_center/recommendation_center.py` (1076 lines) not reviewed for internal structure this session | Flagged for future targeted review, not asserted as a problem |

**No Critical-severity debt was found** — nothing in this list affects current correctness; every item is
either dormant/fragile-but-working, or explicitly out of the perception/predictive-AI chain this review
was gating.

## Phase 12 — Freeze recommendation

| Subsystem | Recommendation | Basis |
|---|---|---|
| Designer | **FROZEN** | stable across all six recent Stair-focused milestones with zero changes required |
| Navigation | **FROZEN** | root-caused and fixed its one known defect this session (zero-duration Stair bug); mechanically re-verified across 24 real topologies + explicit failure-case suite |
| Simulation | **FROZEN** | same audit; capacity/congestion/multi-floor routing all proven correct and tested |
| Perception | **FROZEN** | Stair-camera stack (calibration → projection → tracking → coverage) proven stable across 151 tests spanning 5 dedicated milestones |
| Crowd Intelligence | **FROZEN** | canonical, single-producer, proven extensible (stair_flow integrated with zero engine duplication) |
| Feature Extraction | **MINOR CLEANUP** | architecturally sound and schema-versioned, but several base-schema fields remain honestly APPROXIMATE_PARITY (disclosed, not urgent) — ready to use as-is, not ready to claim EXACT parity everywhere |
| Recommendation | **FROZEN** | verified Stair-aware by direct trace and controlled test this session; zero AI coupling today (correct, by design) |
| Guidance | **FROZEN** | same verification; `ordered_stair_ids` already a first-class field |
| Live Runtime | **MINOR CLEANUP** | functionally sound (222/222 architecture guards, full suite green) but carries the two real findings from Phase 1/11 (`event_bus` placement, legacy `integration.AIInferenceGateway`) — worth a cleanup pass BEFORE, not necessarily blocking, Shadow-Mode wiring |

No subsystem examined this session **NEEDS REDESIGN**.

## Phase 13 — Architecture scorecard (1–10)

| Subsystem | Architecture | Correctness | Maintainability | Extensibility | Testing | Documentation | AI Readiness |
|---|---|---|---|---|---|---|---|
| Designer | 9 | 9 | 8 | 8 | 8 | 8 | n/a |
| Navigation | 9 | 9 (post-fix) | 9 | 8 | 9 | 9 | 8 |
| Simulation | 9 | 9 (post-fix) | 8 | 8 | 8 | 9 | 7 |
| Perception | 9 | 9 | 8 | 8 | 9 | 9 | 9 |
| Crowd Intelligence | 9 | 9 | 9 | 9 | 8 | 9 | 9 |
| Feature Extraction | 8 | 8 | 8 | 9 | 9 | 9 | 8 |
| Recommendation | 8 | 9 | 8 | 7 | 7 | 8 | 6 (by design, no AI coupling yet) |
| Guidance | 8 | 9 | 8 | 7 | 7 | 8 | 6 (same reason) |
| Live Runtime | 7 | 8 | 6 | 7 | 6 | 8 | 8 (gateway already built) |

**Overall (unweighted mean across the nine subsystems above): ≈8.3/10.**

## Phase 14 — Shadow-Mode readiness

**If Shadow-Mode Predictive AI started tomorrow: YES, approve — with one prerequisite (wiring, not
architecture) still open.**

Prerequisites already satisfied:
- A typed, safe gateway abstraction exists and is tested (`LiveAIInferenceGateway`/
  `RegistryLiveAIInferenceGateway`/`ThrottledLiveAIInferenceGateway`, `AISystemStatus`,
  `LiveAIPredictionSnapshot`).
- Explicit deployability tiering exists (`PRODUCTION_CANDIDATE` vs. `EXPERIMENTAL`), with the current
  gateway already scoped to treat only the production-candidate model as operational.
- Feature extraction has schema-versioned, per-field-documented SIM/LIVE parity, including one field
  (`candidate_recent_flow_rate` for Stair) with this session's own controlled-equivalence proof.
- UNKNOWN/observed-zero distinction is consistent and tested at every layer feeding a feature row.
- Exception safety is already enforced ("never allowed to propagate out of this gateway and crash the
  live cycle").
- Explicit, already-enforced scoping keeps predictions away from Decision Policy/Advisory
  System/Recommendation until a deliberate future decision reopens that boundary.

Remaining, explicitly disclosed assumptions (not blockers):
- `live_runtime.factory.build_live_runtime()` does not yet construct/attach the gateway in production —
  pure wiring work, no new architecture.
- Several base-schema features remain APPROXIMATE_PARITY, not EXACT — acceptable for a model already
  trained against that same approximate live signal, but should be re-stated explicitly in whatever
  Shadow-Mode milestone follows, not assumed away.
- The two Phase-1/11 technical-debt items (`event_bus` placement, legacy `integration` gateway) are
  fragile-but-working — a Shadow-Mode wiring milestone touching `live_runtime.factory`/`live_system`
  should be aware of them, though neither blocks the work.
