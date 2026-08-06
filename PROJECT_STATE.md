# SynEvac Project State

> **Purpose of this document**: a self-contained architectural summary of the SynEvac codebase, written so another AI (or a human) can understand the current state of the project without inspecting the repository. Update this file after every completed milestone. Do not rewrite from scratch — append/edit incrementally.

> **Full architectural reference**: `docs/architecture/system_capability_inventory.md` — a comprehensive, verified, class-by-class inventory of all ~20 major subsystems (purpose, main classes, data model, runtime ownership, entry points, capabilities, limitations, tests, production-vs-dormant status), plus dependency graphs and a full capability matrix. This file (`PROJECT_STATE.md`) remains the quick-glance summary; consult the inventory doc for exact file/class references before extending any subsystem, to avoid duplicating something that already exists.

---

## Quick Snapshot

- **Current Phase**: System Capability Inventory Complete (documentation-only milestone, no code changed)
- **Current Version**: v0.9.1
- **Last Completed Milestone**: SynEvac Capability & Architecture Inventory — a full, verified, class-by-class inventory of ~20 major subsystems, dependency graphs, and a capability matrix, written so future milestones extend existing architecture instead of duplicating it. See `docs/architecture/system_capability_inventory.md`. Previous milestone: Camera -> Zone Assignment V1 (Camera.zone_ids now genuinely multi-assignable and authorable in Studio; storage/serialization/query only, no coverage/topology/tracking reasoning yet).
- **Next Planned Milestone**: TBD. The inventory's own §23 ("What Future Milestones Naturally Build Upon") lists the highest-leverage next steps: closing the perception dormancy chain (YOLO/Tracking/Cross-Camera-Identity/Live-Occupants are one config parameter away from activation), Camera Topology activation (now unblocked by Camera -> Zone Assignment V1), and camera-to-camera overlap (the one genuinely missing camera-reasoning concept, already designed in `docs/architecture/building_camera_topology_design.md`).
- **Frozen Subsystems**: Builder, Navigation, Simulation, Perception, Crowd Intelligence, Designer, Evacuation Recommendation, Evacuation Guidance, Recommendation Layer, Execution Layer *(full list with rationale in §5)*
- **Total Tests**: ~5967 passing, 2 known flaky/deselected, 25 skipped (pre-existing/unrelated) *(full suite run this session, all 430 test files covered across batched runs — see Known Flaky Tests below)*
- **Known Flaky Tests**:
  - `tests/test_builder_project_management.py::BuilderNewProjectTests::test_new_project_replaces_current_project_and_resets_dirty_flag` — pre-existing, unrelated to any recent milestone, a real unmocked `QMessageBox.question()` in `new_project()`'s discard-changes confirmation. Can hang indefinitely under `QT_QPA_PLATFORM=offscreen` (offscreen suppresses rendering, not the blocking modal event loop) rather than always resolving quickly.
  - `tests/test_builder_validation.py::BuilderMainWindowValidationGatingTests::test_save_succeeds_once_error_is_resolved` — newly discovered this session, same root cause: `BuilderMainWindow._save_to()` (builder/windows/builder_main_window.py:734) calls a real unmocked `QMessageBox.critical()` when validation fails, and this test's first `_save_to()` call (expected to fail validation) never patches it, unlike its sibling test directly above which does. Both are pre-existing Builder test bugs, confirmed unrelated to Live Camera Viewer (neither `builder/` nor these test files were touched this milestone) and unrelated to each other's discovery.
- **Known worktree-only test artifact (not a real bug, disclosed this session)**: `tests/test_detector_identity_unification.py::ArchitectureGuardTests::test_only_one_detector_migration_module_exists` fails ONLY when the entire test suite is run from inside a nested `.claude/worktrees/...` checkout — the guard's own `rglob("detector_migration.py")` filter deliberately excludes any match whose path contains `"worktrees"` (to avoid double-counting a stray sibling checkout), but when the *whole repo* being tested is itself under a `worktrees` directory, that filter incidentally excludes the one legitimate `models/detector_migration.py` too, leaving zero matches. Confirmed unrelated to this milestone (neither the test nor `models/detector_migration.py` was touched) — passes normally when run from the main checkout.

---

## 1. Project Overview

**Purpose**: SynEvac is a fire-evacuation digital-twin platform. It models a building (geometry, zones, doors, exits, stairs, sensors, fire-safety assets), simulates occupant behavior and hazard evolution, ingests real or simulated perception data (CCTV/YOLO human detection), fuses it into a live building-state estimate, runs predictive AI, produces evacuation recommendations, and — as of the two most recent milestones — coordinates (but does not itself perform) real-world execution actions (voice announcements, building control, dynamic signage, warden notification), all gated behind explicit human operator approval.

**Overall architecture**: a strictly layered pipeline where each layer consumes the previous layer's already-computed output and never reaches backward or sideways into another layer's internals. Two parallel authoring/runtime surfaces exist:
- **SynEvac Studio** (`designer/`) — the full-featured digital-twin authoring tool (the Building Designer *is* the digital twin; there is deliberately no separate "digital twin" package). Launched via `main.py` → `core/app.py::SynEvacApp` → `designer/windows/main_window.py::MainWindow`.
- **SynEvac Builder** (`builder/`, `builder_main.py`) — a separate, lightweight, dependency-clean authoring-only executable (geometry + scale calibration, zero Simulation/AI/Perception coupling), for users who only need to author a building model.

**High-level pipeline** (see §3 for the exact live-runtime stage order):
```
Perception (CCTV/YOLO or Simulation) → Building State Estimator → Crowd/Progress/Trajectory/Emergency-Response Intelligence
    → Predictive AI (Shadow-Mode) → Evacuation Recommendation → Evacuation Guidance → Recommendation Layer
    → Execution Layer (coordinates) → Voice/Building-Control/Dynamic-Signage/Warden-Notification controllers (execute, human-approved)
    → Command Center (operator UI) / Studio (author + observe)
```

**Current development phase**: the live execution/coordination layers are now complete (Recommendation Layer + Execution Layer, both shipped this development arc). The platform's live pipeline is fully wired end-to-end from perception through to human-approved execution, all running under `OFFLINE_DEMO` mode with Simulation-only providers (no real hardware/PLC/BMS/SMS integration exists anywhere yet — this is deliberate and disclosed, not a gap to silently fill). The next major frontier is either (a) a future Guidance v2 / mobile / API layer consuming the Execution Layer's public API, or (b) closing disclosed gaps (real hardware providers, `advisory_system` live-wiring, recommendation-id traceability for Voice/BuildingControl/Signage).

---

## 2. Completed Subsystems

### 2.1 Core Digital Twin & Authoring

#### Building Designer (Studio)

**Status**: Complete (continuously extended)

**Purpose**: The building authoring tool *is* the digital twin — there is no separate "digital twin" data model. Authoring a building in Designer produces the exact `Building`/`Floor`/`Zone`/`Door`/`Exit`/`Stair`/asset objects every other subsystem (Simulation, Perception, Recommendation, Execution) reads.

**Responsibilities**: floor/zone/door/exit/stair/camera/sensor/fire-safety-asset placement and editing; property panel editing; scene rendering; toolbar-driven tool switching; project save/load (`.syn` files via `serialization.Serializer` + `models.Project`); dock-based panel system for every downstream subsystem's live/debug views.

**Inputs**: user interaction (mouse/toolbar/property panel).

**Outputs**: `models.project.Project` (containing `models.building.Building`), serialized to `.syn`.

**Major public classes**: `designer.windows.main_window.MainWindow`, `designer.scene.graphics_scene.GraphicsScene`, `designer.widgets.property_panel.PropertyPanel` (7800+ lines — the single largest widget file), one `*Item` class per asset type (`designer/items/`), one `*Panel` class per live-data dock (`designer/widgets/`).

**Important files/folders**: `designer/windows/`, `designer/scene/`, `designer/items/`, `designer/widgets/`, `designer/campaign/` (Scenario Campaign Studio — batch simulation runner UI).

**Connects to**: everything. Owns the `Building` instance every downstream engine reads; hosts the Manual Simulation Sandbox (`sandbox/`) AND the real Live Runtime session (`designer/live_runtime_controller.py`) as two parallel, independent tick sources — this distinction matters a great deal (see §9).

**Important design decisions**: toolbar reorganized into Building / Perception & Alarm / Guidance & Output / Simulation groups; 9 fire-safety assets moved to an "Advanced Fire-Safety Tools" submenu (not deleted); legacy generic `Detector` hidden from new authoring (superseded by Smoke/Heat Detector). **Camera -> Zone Assignment milestone**: `Camera.zone_ids` (already a `Tuple[str, ...]` on `EngineeringAsset` since the Digital Twin Asset -> Zone Assignment milestone, but capped at one zone by the old single-`QComboBox` Property Panel UI) is now genuinely multi-assignable — widened to the same `_populate_zone_checklist()` multi-select `QListWidget` pattern `Speaker.zone_ids` already established, zero model change. `Floor.get_zone(zone_id)` (new, mirrors `Building.get_floor()`) is the honest way a runtime caller resolves `zone_ids` into real `Zone` objects. This is Milestone 1 of the Building Topology Foundation investigation (`docs/architecture/building_camera_topology_design.md`) — storage/query only; coverage visualization, camera-to-camera overlap/transition modeling, tripwires, and cross-camera reasoning are explicitly deferred to later milestones, not built here. SynEvac Builder's own separate `BuilderPropertyPanel.camera_zone` (still a single `QComboBox`) was deliberately left untouched — out of scope for this milestone.

**Known limitations**: `PropertyPanel` is a single 7800+ line file (never refactored into per-asset panels); the Manual Simulation Sandbox has no perception/AI/recommendation pipeline behind it at all (see §9's "two tick sources" discovery).

**Future extension points**: any new Designer-authorable asset type follows the established `models/<asset>.py` + `designer/items/<asset>_item.py` + toolbar action + property panel section pattern.

---

#### SynEvac Builder

**Status**: Complete (V1)

**Purpose**: a separate, standalone executable for authoring building geometry + scale calibration only, with zero Simulation/AI/Perception dependency — for a lighter-weight authoring-only use case.

**Responsibilities**: geometry authoring, floor scale calibration (a real gap identified and filled — scale calibration did not exist anywhere else in the repo).

**Inputs**: user interaction. **Outputs**: `.syn` project files, fully compatible with and openable in full Studio.

**Major public classes**: `builder.windows.builder_main_window.BuilderMainWindow`.

**Important files/folders**: `builder/`, `builder_main.py`.

**Connects to**: `models/`, `navigation/`, `serialization/`, `designer.items`/`designer.scene`/`designer.validation` (confirmed 100% free of Simulation/AI/Perception coupling before Builder was built on top of them).

**Important design decisions**: does NOT reuse Studio's 7800-line `PropertyPanel` (which imports camera_calibration) — has its own minimal Builder-only property panel.

**Known limitations**: fire-safety asset palette deferred (not yet in Builder).

---

#### Navigation & Pathfinding

**Status**: Complete / Frozen

**Purpose**: derives a navigation graph from building geometry and provides shortest/safe-path computation.

**Major files**: `navigation/graph_builder.py::NavigationGraphGenerator`, `pathfinding/`.

**Connects to**: `evacuation_recommendation/` (safe-exit routing), `sandbox/` (manual simulation occupant movement), `simulation_interactive/` (action executor for interactive door/exit state).

**Important design decisions**: a Stair's `vertical_height=0`-when-`from_floor_id`-unresolved bug (edge cost=0, not unreachable) was root-caused and fixed with a minimal `graph_builder.py` None-fallback — historical "cannot approach Stair" reports were never actually "unreachable," they were "instantaneous."

---

### 2.2 Simulation & Hazard

#### Manual Simulation Sandbox

**Status**: Complete

**Purpose**: a lightweight, Designer-embedded, manually-driven occupant simulation (Start/Pause/Stop/Reset) for quick authoring-time visualization — entirely separate from the "real" Live Runtime pipeline.

**Major classes**: `sandbox.manager.SandboxManager`, `sandbox.occupant.SandboxOccupant`.

**Connects to**: Designer's own debug panels only (`perception_debug_panel.py`, `building_state_debug_panel.py`) — has **no** connection to Perception/Crowd Intelligence/Predictive AI/Recommendation/Execution. This is a deliberate, disclosed architectural boundary, not an oversight (see §9).

---

#### Hazard / Fire Growth Simulation

**Status**: Complete

**Purpose**: models fire/smoke/hazard evolution over time for a scenario.

**Major files/folders**: `hazard/`, `hazard_evolution/`, `fire_growth/`, `smoke_propagation/`, `tenability/`.

**Connects to**: `building_state.models.BuildingState.hazard_summary`; `evacuation_recommendation` (excludes unsafe exits); `emergency_response` (hazard-present evidence).

---

#### Scenario Generation & Dataset Pipelines

**Status**: Complete (research/offline pipeline, separate from live production)

**Purpose**: generates synthetic training/evaluation scenarios at scale, for predictive-model training and stress-testing.

**Major folders**: `scenario/`, `scenario_definition/`, `scenario_generator/`, `scenario_pipeline/`, `scenario_validator/`, `scenario_storage/`, `scenario_runner/`, `predictive_dataset/`, `dataset_builder/`, `ground_truth/`.

**Important discovery**: two independent dataset pipelines exist (`predictive_dataset/` research pipeline vs. `ai_registry`-facing production pipeline) sharing zero code — confirmed deliberate, not accidental duplication, during a dedicated audit.

**Connects to**: `ai_training/`, `model_benchmark/`, `predictive_model/`, Designer's Campaign Studio (`designer/campaign/`) for batch-running scenarios through the real simulation and recording `ground_truth.json`.

---

### 2.3 Perception

#### Camera Calibration, CCTV Ingestion & Human Detection

**Status**: Complete (production-validated against real hardware)

**Purpose**: derives camera world-position calibration, ingests real RTSP camera streams (or replay/simulation sources), runs YOLO-based human detection, tracks and fuses identities across cameras into world positions.

**Major files/folders**: `camera_calibration/`, `camera_manager/`, `camera_validation/`, `camera_coverage/` (derives Camera↔Observable-Asset coverage purely from calibration+geometry), `live_camera_pipeline/`, `human_detection/`, `tracking/`, `multi_camera_fusion/`, `cross_camera_identity/`.

**Important milestones folded in**: real YOLO model validated against real photos/video; real correspondence-solver camera calibration with RMSE validation; `OpenCVFrameDecoderBackend` replacing a fake RTSP backend; a full physical-CCTV field-validation runner (6 progressive modes, JSON reports); a real CP PLUS NVR (CP-UNR-32) integration issue diagnosed and resolved (stale RTSP Digest auth fixed by an NVR restart).

**Known limitations**: no real measured/calibrated physical scene exists yet in production use — the pipeline is proven-ready, not yet deployed against a live building.

---

#### Building State Estimator & Live Occupants

**Status**: Complete / Frozen

**Purpose**: fuses perception evidence (camera detections, sensor readings, FACP alarms, manual call points) into one canonical `BuildingState` per live cycle; separately, `live_occupants/` tracks persistent occupant identity/lifecycle across cycles (distinct from the transient per-frame detections).

**Major classes**: `building_state.models.BuildingState`, `live_occupants.manager.LiveOccupantManager`.

**Important design decision**: legacy-vs-canonical detector identity unification handled via `models/detector_migration.py`.

---

#### Live Crowd/Progress/Trajectory/Emergency-Response Intelligence

**Status**: Complete / Frozen (all four)

**Purpose**: four sibling "live intelligence" engines, each deriving a different lens on the same live occupant/building-state evidence:
- **Crowd Intelligence** (`crowd_intelligence/`) — density, congestion, queueing, per-asset approach metrics, stair flow (`stair_flow/` — entry/exit/rate/direction derived purely from existing evidence, multi-camera dedup inherited not reimplemented).
- **Evacuation Progress** (`evacuation_progress/`) — zone clearance status, exit flow trends.
- **Trajectory Intelligence** (`trajectory_intelligence/`) — movement anomaly, route-deviation detection.
- **Emergency Response** (`emergency_response/`) — zone response-priority ranking (LOW/MODERATE/HIGH/CRITICAL), assistance-required detection (possible/confirmed/being-assisted, careful never to over-claim from a geometric heuristic).

**Important design decisions**: every engine's own `*Weights`/`*Thresholds` config is explicitly disclosed as "a documented project assumption, never a validated life-safety standard." None of these four ever recomputes another's logic — they all read the same upstream `BuildingState`/each other's snapshot as plain, duck-typed compute() parameters, never an import-level dependency on each other's `.engine` module.

---

#### FACP, Manual Call Point & Emergency Lighting

**Status**: Complete / Frozen

**Purpose**: `facp/` is a Fire Alarm Control Panel state machine (`SimulatedFACP`) + append-only event log — pure state machine, **no actuator concept at all**, ticked automatically every live cycle via `live_system.facp_gateway.EngineFACPGateway.evaluate()`. Manual Call Points feed FACP alarm evidence (and, since a later milestone, reach Emergency Response/Recommendation/Advisory/Command Center too, as a structurally distinct alarm source). Emergency Lights are isolated from route-safety logic (never affect evacuation routing).

**Known limitation (disclosed, deliberately unfixed)**: `facp/provider.py::FACPEventProvider` is an unimplemented placeholder for a future real hardware adapter — zero concrete subclasses exist.

---

#### Fire Suppression & Fire Water Infrastructure (Designer Assets)

**Status**: Complete as Designer/Command-Center digital-twin assets; **execution-inert by design**

**Purpose**: Sprinkler/FireExtinguisher/FireHydrant/HoseReel (`fire_safety_manager/`) and Tank/Pump/JockeyPump/Inlet (`fire_water_manager/`) are modeled, authored in Designer, and displayed in Command Center — but deliberately kept **out of** FACP alarm evidence and route-safety physics (no backing hazard-effect model exists for them). `enable_asset()`/`disable_asset()` exist but are unused outside tests (Designer-authoring toggles, not runtime commands).

**Important discovery**: a dedicated connectivity audit found these (plus several others) are genuine "dead ends" — fully modeled, fully displayed, zero influence on any decision — and this was found to still be true as of the most recent Execution Layer investigation.

---

### 2.4 Predictive AI

#### Predictive Dataset & Model Training (research arc)

**Status**: Complete through V4 (research), one model (V2.2/V3.1) selected for production registration

**Purpose**: an extensive, multi-generation research investigation into localized (per-Door/Exit/Stair-candidate) congestion prediction, iterating through dataset campaigns V1→V4 and model versions V1→V3.1, root-causing and fixing real bugs at each stage (zero-duration timestamp artifacts, cross-topology generalization gaps, dataset diversity vs. representation).

**Verdict progression**: V1 "promising but needs more data" → V2 fixed Stair prediction but broke Exit → V2.1/V2.2 fixed Exit via new features (validated at scale) but found the Door/Stair *target itself* was 100% a zero-duration artifact → Target V2 (onset-based) fixed that → V3 first full-rigor eval (PR-AUC 0.615) revealed topology-holdout generalization gap + shuffle-test memorization → V3.1 proved structural-diversity (not just row-count) is the real lever → Dataset V3/V4 campaigns validated diversity at scale (variant-level generalization dramatically improved; family-level, i.e. unseen building *type*, did not — still an open research question).

**Major folders**: `predictive_dataset/`, `predictive_model/`, `ai_training/`, `model_benchmark/`, `research_framework/`.

**Important discovery**: the cross-topology generalization gap is dominated by genuine concept shift (35x conditional-rate spread across building families) plus insufficient family diversity — not a fixable feature-engineering problem alone; graph-context features (`networkx`-based) gave a modest, non-negative but inconsistent benefit.

---

#### Shadow-Mode Live AI Inference

**Status**: Complete / Frozen

**Purpose**: runs a registered predictive model (`bottleneck_occurrence` classification, `evacuation_time_experimental` regression — the only two real model outputs that exist) against live `BuildingState` every cycle, in "shadow mode" (predictions recorded, never influencing any decision with execution authority).

**Major classes**: `live_system.live_ai_gateway.{LiveAIInferenceGateway, RegistryLiveAIInferenceGateway, ThrottledLiveAIInferenceGateway, LiveAIPredictionSnapshot}`, backed by `ai_registry.ModelRegistry`/`LiveAIInferenceService`.

**Important discovery (critical, still true)**: `evacuation_recommendation.scoring.score_candidate()` already accepts and uses `ai_prediction_snapshot` — but applies it as an *identical* additive term to every candidate exit in a zone, which is **mathematically provably rank-invariant** (proven symbolically and empirically). AI can shift a displayed number, never a decision. This predates the Shadow-Mode milestone itself (an even earlier "Live Dynamic Evacuation Recommendation Engine" milestone already built this coupling).

**Performance**: ~11.4ms overhead at 20 cameras/100 occupants/20 stairs.

**Not yet done**: `live_ai_gateway`/`live_advisory_gateway` have **zero default-construction path** anywhere in the codebase — a caller must explicitly build and wire one; `build_offline_demo_runtime()`/`build_live_runtime()` never do this automatically (unlike every "engine-shaped" stage, which the factory always constructs).

---

#### Prediction Evaluation

**Status**: Complete / import-isolated

**Purpose**: `prediction_evaluation/` compares Shadow-Mode predictions against actual future building state (windowed ground truth), for both Simulation and Live Runtime, entirely out-of-band (never on the runtime's critical path, mechanically proven bidirectionally import-isolated from Recommendation/Guidance/Execution).

**Important discovery**: the windowed ground-truth definition used here is a genuinely different question from the model's own whole-scenario training target — evaluation metrics from this framework are not directly comparable to training-time metrics.

---

### 2.5 Decision & Recommendation

#### Evacuation Recommendation Engine

**Status**: Complete / **FROZEN**

**Purpose**: the canonical, deterministic, per-zone "which exit should this zone's occupants use" routing engine. Sits immediately after deterministic safety evaluation — never scores or recommends an exit that hazard/structural evidence considers unsafe.

**Responsibilities**: safe-exit candidate filtering, distance/congestion/queue/throughput/trajectory/emergency-response/AI-support additive scoring, confidence computation, human-readable explanation generation.

**Inputs**: `BuildingState`, `CrowdIntelligenceSnapshot`, `EvacuationProgressSnapshot`, `TrajectoryIntelligenceSnapshot`, `EmergencyResponseSnapshot`, `LiveAIPredictionSnapshot` (all optional).

**Outputs**: `EvacuationRecommendationSnapshot` (one `ZoneEvacuationRecommendation` per *occupied* zone, never a fabricated empty one).

**Major public classes**: `evacuation_recommendation.engine.EvacuationRecommendationEngine`.

**Major files**: `evacuation_recommendation/` (`models.py`, `engine.py`, `scoring.py`, `ranking.py`, `evidence.py`, `explanation.py`).

**Connects to**: `evacuation_guidance/` (downstream), `recommendation_layer/` (downstream, read-only adapter), `live_system/orchestrator.py` (wired every live cycle via `evacuation_recommendation_gateway`).

**Important design decisions**: AI support is deliberately the smallest weight (5%) and applied uniformly per zone, so it can never change relative candidate order — mechanically proven by a dedicated test. Confidence is evidence-quality-based, never AI certainty.

**Known limitations**: has no per-zone lifecycle/dedup manager of its own — every `compute()` call is a fresh, stateless snapshot (this is exactly the gap Recommendation Layer's own manager fills, one layer downstream).

**Approx. completion**: mid-development-arc milestone, well before Recommendation Layer.

---

#### Evacuation Guidance Engine

**Status**: Complete / **FROZEN**

**Purpose**: converts "which exit" (Recommendation's output) into "how to reach it" — an ordered route/instruction plan per zone, plus a planned-only voice message (never itself broadcast — zero execution authority).

**Major classes**: `evacuation_guidance.engine.EvacuationGuidanceEngine`, `EvacuationGuidancePlan`.

**Inputs**: `EvacuationRecommendationSnapshot`, `BuildingState`, `speaker_manager` (for coverage-only checks). **Outputs**: `EvacuationGuidanceSnapshot`.

**Connects to**: `recommendation_layer/` (as `SYSTEM_WARNING` input — guidance inconsistencies), `voice_evacuation/adapter.py::guidance_plan_to_voice_message()` (downstream content generation).

---

#### Advisory System

**Status**: Complete, but **not FROZEN** and **not a live subsystem** — Replay-only in practice

**Purpose**: a broader, older, multi-audience reporting layer producing `AdvisoryReport` (civilian announcements, firefighter intelligence, building-systems recommendations, incident-commander dashboard) from a completed simulation's `Scenario`+`GroundTruth`.

**Major classes**: `advisory_system.orchestrator.AdvisoryOrchestrator`, `advisory_system.recommendation_models.{AdvisoryReport, BuildingRecommendation, CivilianAnnouncement, FirefighterIntelligenceReport, IncidentCommanderDashboard}`.

**Critical discovery (load-bearing, confirmed by direct code audit)**: `AdvisoryOrchestrator.generate_report()` has **exactly one real, non-test invocation path** anywhere in the codebase — `command_center/incident_data.py`'s Replay/"Load Incident" flow, fed by Designer's Campaign Studio batch-simulation output. The live-compatible adapter (`live_system.live_advisory_gateway.ReplayCompatibleAdvisoryGateway`) is **never constructed outside tests** — confirmed via exhaustive grep, zero non-test construction sites. Consequently `AdvisoryReport` is `None` in the overwhelming majority of live/offline-demo cycles. This is explicitly disclosed (not silently assumed) in both the Recommendation Layer's and Execution Layer's own documentation.

**Not FROZEN**: absent from `core_architecture_freeze_review.md`'s own FROZEN table, and explicitly flagged there as "the single largest file in the repo, not examined this session."

---

#### Recommendation Layer

**Status**: **Frozen** (shipped this development arc, considered complete)

**Purpose**: the single, canonical, unified public interface for evacuation recommendations — an orchestration/coordinating layer over `evacuation_recommendation`, `evacuation_guidance`, `emergency_response`, `crowd_intelligence`, and (optionally) `advisory_system`. **Not** a competing recommendation engine — none of those five packages were modified; this layer only reads their already-computed output.

**Responsibilities**: normalize 6 recommendation categories into one vocabulary; deduplicate same-cycle candidates from multiple providers (fixed provider-priority order); track provenance (`primary_source`, `supporting_sources`, `evidence_origin`); manage lifecycle (stable IDs across cycles, grace-period expiration); rank by priority.

**Inputs**: `EvacuationRecommendationSnapshot`, `EvacuationGuidanceSnapshot`, `EmergencyResponseSnapshot`, `CrowdIntelligenceSnapshot`, `LiveAIPredictionSnapshot`, `AdvisoryReport` (all optional — designed to produce meaningful output even when Advisory/AI are `None`, the common live case).

**Outputs**: `RecommendationSet` (a `Tuple[Recommendation, ...]`, priority-sorted).

**Major public classes**: `recommendation_layer.layer.RecommendationLayer` (facade, `compute(time, **snapshots) -> RecommendationSet`), `recommendation_layer.manager.RecommendationManager` (dedup/lifecycle), 6 adapter modules in `recommendation_layer/adapters/`.

**The 6 categories**: `OCCUPANT_ROUTING`, `HAZARD_AVOIDANCE`, `CONGESTION_MITIGATION`, `EXIT_UTILIZATION` (the one self-derived category — no upstream provider computes cross-zone exit-load redistribution), `WARDEN_DISPATCH`, `SYSTEM_WARNING`.

**Major files**: `recommendation_layer/` (`models.py`, `manager.py`, `layer.py`, `adapters/*.py`).

**Connects to**: `live_system/orchestrator.py` (new `recommendation_layer_gateway` stage, runs LAST among intelligence/advisory stages), `live_system/state_manager.py` (`recommendation_set` field), Studio (`designer/widgets/recommendation_panel.py`), **and now** Command Center (`command_center/data_source.py::CommandCenterSnapshot.recommendation_set`, added by the Execution Layer milestone) and Execution Layer (downstream consumer).

**Important design decisions**: dedup key = `type|trigger_condition|sorted(affected_zones)|sorted(affected_exits)` — independent of source, so two providers corroborating the same real-world claim collapse into one `Recommendation`. Recommendation IDs are minted once and stay stable for the entire active lifetime, including through a grace-period (default 5s) before expiring. Adapters are deliberately thin (near-verbatim passthrough for 5 of 6 categories) — almost no new decision logic anywhere in this package.

**Known limitations (disclosed)**: `advisory_report` is `None` in the overwhelming majority of live cycles (see Advisory System entry above) — every category is designed to still work without it.

**Example execution flow**: `evacuation_recommendation` computes zone→exit ranking → `RecommendationLayer.compute()` runs 6 adapters → `RecommendationManager` dedups/tracks lifecycle → `RecommendationSet` returned → stored on `LiveBuildingSnapshot.recommendation_set` → read by Studio panel and (new) Command Center + Execution Layer.

**Tests**: 53 new tests at ship time; full suite 5218 passing, zero regressions.

---

### 2.6 Execution

#### Voice Evacuation, Building Control & Dynamic Signage (pre-existing execution controllers)

**Status**: Complete, **not FROZEN**, production-reachable — discovered (not built) during the Execution Layer investigation

**Purpose**: three independent, structurally identical human-in-the-loop execution controllers, each following: `submit()` → `PENDING_APPROVAL` → explicit operator click → `approve()`/`reject()` → `_dispatch()` → `provider.execute()/apply()/send()` → `CONFIRMED`/`FAILED`, with an append-only history.

**Major classes**: `voice_evacuation.controller.VoiceEvacuationController` (+ `SimulationVoiceOutputProvider`), `building_control.controller.BuildingControlController` (+ `SimulationControlProvider`), `dynamic_signage.controller.DynamicSignageController` (+ `SimulationDynamicSignageProvider`).

**Critical discovery**: `command_center/live_operator_action_gateway.py::LiveOperatorActionGateway` already existed as **the** execution seam — real, tested, wired into production (`main.py` → Designer → Live Runtime panel → Command Center), with a real approve/reject UI and status tracking. This was discovered via a dedicated pre-implementation architectural review before the Execution Layer milestone began, specifically to avoid building a duplicate mechanism.

**Known limitations (disclosed)**: under `ApplicationMode.LIVE`, all three controllers are `None` (`NO_PROVIDER`) — only `OFFLINE_DEMO` mode gets real (Simulation-only) providers. No real hardware/PLC/BMS/protocol integration exists anywhere, and is mechanically forbidden by dedicated architecture-guard tests (`modbus`/`bacnet`/`mqtt`/`opcua`/`socket`/`serial` imports forbidden). Door/Exit control requests always resolve `FAILED` in every real production path found, because no caller ever wires a real `action_executor` into `SimulationControlProvider`.

---

#### Warden Notification (new controller)

**Status**: Complete / new this milestone

**Purpose**: the one genuinely new execution controller — notifies a human warden/responder of a zone needing attention. Mirrors `BuildingControlController`'s shape verbatim (submit→approve→dispatch→confirm/fail, append-only history).

**Major classes**: `warden_notification.controller.WardenNotificationController`, `warden_notification.provider.{WardenNotificationProvider, SimulationWardenNotificationProvider}`.

**Major files**: `warden_notification/` (`types.py`, `requests.py`, `history.py`, `controller.py`, `provider.py`).

**Important design decision**: `SimulationWardenNotificationProvider` is explicitly disclosed as pure bookkeeping — no real SMS/push/email/webhook transport exists anywhere in the codebase; "confirmed" never means a real person was reached.

**Connects to**: `command_center.live_operator_action_gateway.LiveOperatorActionGateway` (new `warden_controller` param + `ingest_warden_recommendations()`/`approve_warden_notification()`/`reject_warden_notification()`), `execution_layer.adapters.warden_adapter` (both submit-side translation and read-side status reporting).

---

#### Execution Layer

**Status**: **Frozen** (shipped this development arc, considered complete)

**Purpose**: an orchestration/coordinating layer over all four execution controllers (Voice, Building Control, Dynamic Signage, Warden Notification). **It never calls a provider itself and never bypasses the operator-approval gate** — those four controllers remain the sole execution authority. Answers "how is a recommendation carried out," never "what should happen" (that's Recommendation Layer's job) and never "did it actually happen" without an honest, disclosed provenance trail.

**Responsibilities**: read-only normalization of all four controllers' already-recorded state into one unified `ExecutionRequest` vocabulary; four-timestamp audit trail (created/approved/dispatched/completed); honest `recommendation_id` traceability disclosure; submit-side translation of `Recommendation(type=WARDEN_DISPATCH)` into `WardenNotificationRequest` (the one category with genuine, real, end-to-end traceability, built correctly from day one).

**Inputs**: the four controllers (any subset may be `None`).

**Outputs**: `ExecutionSet` (a `Tuple[ExecutionRequest, ...]`).

**Major public classes**: `execution_layer.layer.ExecutionLayer` (`compute(time) -> ExecutionSet`), 4 adapter modules in `execution_layer/adapters/`.

**Major files**: `execution_layer/` (`models.py`, `layer.py`, `adapters/{voice,building_control,signage,warden}_adapter.py`).

**Connects to**: `live_runtime/factory.py`/`runtime.py` (new `tick_execution_layer(time)` method — **deliberately separate from `run_cycle()`**, see §9's critical architectural constraint), `command_center/live_operator_action_gateway.py` (Warden bridge), Studio (`designer/widgets/execution_panel.py`, read-only, no Approve/Reject affordance at all), Command Center (`command_center/warden_notifications_panel.py`, the one panel with real Approve/Reject buttons).

**Important design decisions**: `recommendation_id_provenance` field (`"recommendation_layer"`/`"advisory_system"`/`"unavailable"`) — an explicit, honest disclosure of which id-space `originating_recommendation_id` came from, since Voice/BuildingControl/Signage's underlying requests only ever carry an `advisory_adapter`-synthesized hash id (never the real `recommendation_layer` id) or nothing at all (Signage has no traceability field whatsoever today).

**Known limitations (disclosed)**: closing the traceability gap for the three pre-existing categories (making `evacuation_guidance`/`advisory_adapter` carry the real `recommendation_layer` id through) is future work, explicitly out of scope for V1.

**Example execution flow** (the one fully-traceable path): `Recommendation(type=WARDEN_DISPATCH, recommendation_id="rec-X")` → `gateway.ingest_warden_recommendations()` → `WardenNotificationController.submit()` (`PENDING_APPROVAL`) → operator clicks Approve in Command Center → `approve()` → `_dispatch()` → `SimulationWardenNotificationProvider.notify()` → `CONFIRMED` → `runtime.tick_execution_layer(time)` → `ExecutionRequest` with all 4 timestamps and `recommendation_id_provenance="recommendation_layer"`.

**Tests**: 49 new tests. Full suite: 5315 passing, zero regressions from this work (one unrelated pre-existing flaky test in `builder/`'s own suite, see §9).

**Approx. completion**: immediately following Recommendation Layer, same development arc.

---

### 2.7 Live Runtime & Command Center

#### Live Runtime Composition Root

**Status**: Complete / Frozen (core), continuously extended (additive params)

**Purpose**: the one production composition root assembling a coherent `LiveRuntime` out of already-existing, already-tested components — camera/sensor/fusion/FACP/AI/advisory/voice/building-control/recommendation/execution, all wired together.

**Major classes**: `live_runtime.factory.{build_live_runtime, build_offline_demo_runtime}`, `live_runtime.runtime.LiveRuntime`.

**Important design decision (mechanically enforced)**: `LiveRuntime` imports **zero concrete collaborator classes** — every attribute is untyped, injected by the factory. `LiveOrchestrator` is mechanically forbidden from importing `voice_evacuation`/`building_control.controller`/`dynamic_signage.controller`/`live_runtime` at all (`LiveOrchestratorCannotDirectlyCallControllersTests`) — this is *the* reason `ExecutionLayer` had to be ticked from a new `LiveRuntime.tick_execution_layer()` method, never from inside `run_cycle()`.

**Two composition modes**: `ApplicationMode.LIVE` (real hardware — camera/voice/control/signage/warden providers all `NO_PROVIDER` until someone implements real ones) vs. `ApplicationMode.OFFLINE_DEMO` (all Simulation-only providers auto-defaulted — zero network I/O, zero real hardware, the mode used for every demo/test in this codebase today).

---

#### `live_system` Orchestrator & State Manager

**Status**: Complete / Frozen core, additive extension pattern established

**Purpose**: `LiveOrchestrator.run_cycle()` is the single per-cycle tick driving every optional intelligence/advisory/recommendation stage in a fixed order (see §3); `StateManager` holds the immutable `LiveBuildingSnapshot`, one field + `component_timestamps` entry per stage.

**Major classes**: `live_system.orchestrator.LiveOrchestrator`, `live_system.state_manager.{StateManager, LiveBuildingSnapshot}`, `event_bus.bus.EventType`.

**Important discovery**: a legacy "Phase 7, generation 1" seam (`decision_policy_gateway`/`recommendation_builder`, via `live_system/integration.py`) is genuinely dead in production — `live_runtime/factory.py` never passes either parameter; only `tests/test_live_system.py` ever constructs them. Left in place, not removed (out of scope for cleanup).

**The established "add a new stage" pattern** (used identically by every intelligence/recommendation milestone): add an `Optional[XGateway] = None` constructor param + 1:1 assignment; add a `latest_x` forwarding `@property`; insert a guarded `if self.x_gateway is not None:` block in `run_cycle()`; add a `LiveBuildingSnapshot` field + entry in `replace()`'s field dict; add `latest_x()`/`update_x()` on `StateManager`; add an `EventType.X_UPDATED` member. Recommendation Layer followed this exactly. Execution Layer deliberately did **not** (see above) — it's the one exception, driven by the orchestrator-cannot-reach-controllers guard.

---

#### Command Center

**Status**: Complete, continuously extended

**Purpose**: the operator-facing live/replay monitoring and action-approval application — a separate `QMainWindow` opened from Designer's Live Runtime panel once a session is running.

**Major classes**: `command_center.main_window.MainWindow`, `command_center.dashboard.Dashboard`, `command_center.data_source.{CommandCenterSnapshot, LiveCommandCenterDataSource}`, `command_center.live_operator_action_gateway.LiveOperatorActionGateway`, `command_center.recommendation_center.RecommendationCenter` (a `QTabWidget` hosting 7 panels as of the Execution Layer milestone: Civilian Announcements, Voice Evacuation, Firefighter Intelligence, Building Recommendations, Building Controls, **Warden Notifications** (new), Commander Summary).

**Two modes**: Live (real-time, driven by `LiveRuntimeController`'s 1Hz tick) and Replay (post-hoc, reading a completed `IncidentData`/`GroundTruth`-backed simulation via "Load Incident").

**Important design decision**: `command_center/live_operator_action_gateway.py` and `command_center/incident_data.py` are the *only* two files in `command_center/` allowed to import execution-controller internals directly — every panel goes through the gateway, never the controller.

---

#### Live Runtime Launcher (Designer-side session management)

**Status**: Complete

**Purpose**: mediates between Designer's Live Runtime panel and a `LiveRuntimeSession` (the one place `build_live_runtime()`/`build_offline_demo_runtime()` actually get called), and drives the real per-cycle tick.

**Major classes**: `live_runtime_launcher.session.LiveRuntimeSession`, `designer.live_runtime_controller.LiveRuntimeController` (owns a `QTimer` at 1000ms, calls `session.runtime.run_cycle(time.time())` then an `on_cycle_callback` — now used to refresh both the Recommendation panel and the Execution panel in Studio).

**Critical discovery**: this tick timer is the **only** place in the shipped application that actually calls `run_cycle()` — without it, every live panel would stay frozen forever regardless of how well-wired the backend is (a gap discovered and fixed in an earlier milestone; still true and load-bearing today).

---

## 3. Current Live Runtime Pipeline

Exact `LiveOrchestrator.run_cycle()` stage order (each stage individually optional/guarded, `None` gateway = stage skipped honestly):

```
1.  Sensor read                          (sensor_registry.read_all)
2.  Perception                           (perception_gateway)
3.  FACP evaluation                      (facp_gateway.evaluate — state machine only, no snapshot)
4.  Building State                       (building_state_gateway)
5.  Live Occupants                       (live_occupants_gateway)
6.  Crowd Intelligence                   (crowd_intelligence_gateway)
7.  Evacuation Progress                  (evacuation_progress_gateway)
8.  Trajectory Intelligence              (trajectory_intelligence_gateway)
9.  Emergency Response                   (emergency_response_gateway)
10. Predictive AI (Shadow-Mode)          (live_ai_gateway — usually None in practice)
11. Evacuation Recommendation            (evacuation_recommendation_gateway)
12. Evacuation Guidance                  (evacuation_guidance_gateway)
13. Dynamic Signage Planning              (evacuation_signage_gateway — PLANNING only, zero dispatch authority)
14. Advisory                              (live_advisory_gateway — usually None in practice, Replay-only in reality)
15. Recommendation Layer                  (recommendation_layer_gateway — unifies 9-14's outputs)
16. Decision Policy (legacy, dead)        (decision_policy_gateway — never configured in production)
17. Recommendation Builder (legacy, dead) (recommendation_builder — never configured in production)
18. Command Center notify                 (command_center_gateway)
```

**Separately, NOT part of `run_cycle()`** (mechanically forbidden from being inside it):
```
LiveRuntime.tick_execution_layer(time)  -- reads Voice/BuildingControl/Signage/Warden controllers
                                             (which live on LiveRuntime, one layer above the orchestrator)
                                          -> ExecutionSet
```

**Execution** (human-approval-gated, driven by explicit operator clicks in Command Center, never automatic):
```
Recommendation → {Voice: broadcast()} / {BuildingControl: submit→approve→dispatch} /
                  {DynamicSignage: submit→approve→dispatch} / {Warden: submit→approve→dispatch}
               → provider.{send/execute/apply/notify}() → CONFIRMED/FAILED → ExecutionRequest (via tick_execution_layer)
```

**Driving tick**: `designer/live_runtime_controller.py::LiveRuntimeController` — a 1000ms `QTimer` calling `session.runtime.run_cycle(time.time())` then `on_cycle_callback()` (which refreshes both the Recommendation and Execution Studio panels). This is the only place `run_cycle()` is ever called in the shipped application.

---

## 4. Repository Architecture

One paragraph per major package (grouped by pipeline stage; see §2 for full detail on the architecturally central ones):

- **`designer/`** — SynEvac Studio, the full authoring tool and digital twin (see §2.1).
- **`builder/`** — SynEvac Builder, a lightweight standalone authoring-only executable (see §2.1).
- **`models/`** — every core data model (`Building`, `Floor`, `Zone`, `Door`, `Exit`, `Stair`, every asset type, `Project`) — the shared vocabulary every other package reads.
- **`navigation/`, `pathfinding/`** — navigation graph construction and shortest/safe-path routing.
- **`sandbox/`, `simulation_interactive/`, `simulation_runtime/`, `simulator/`** — the Manual Simulation Sandbox and its interactive action executor (Designer-only, no perception/AI behind it).
- **`hazard/`, `hazard_evolution/`, `fire_growth/`, `smoke_propagation/`, `tenability/`** — hazard/fire/smoke evolution modeling.
- **`scenario/`, `scenario_definition/`, `scenario_generator/`, `scenario_pipeline/`, `scenario_validator/`, `scenario_storage/`, `scenario_runner/`, `scenario_event_executor/`, `scenarios/`** — scenario authoring, generation, validation, storage, and execution for batch/research simulation runs.
- **`behavior/`, `behavior_library/`, `behaviour_profile_resolver/`, `human_decision_engine/`** — occupant behavior modeling for simulation.
- **`camera_calibration/`, `camera_manager/`, `camera_validation/`, `camera_coverage/`** — camera geometry calibration (pinhole model, world-coordinate projection), asset management, validation, and coverage-derivation.
- **`automatic_calibration/`, `calibration_studio/`, `calibration_benchmark/`** — a *different* sense of "calibration": grid-search tuning of the *simulator's own physics parameters* (walking speed, herding, congestion factors) against published evacuation-research benchmarks, with a full session/dashboard/report "studio." Substantial, tested (17+ dedicated test files), real research-arc work — previously undocumented in this file; surfaced by the System Capability Inventory milestone. See `docs/architecture/system_capability_inventory.md` §17.
- **`live_camera_pipeline/`, `human_detection/`, `tracking/`, `multi_camera_fusion/`, `cross_camera_identity/`** — real-time camera ingestion, YOLO human detection, tracking, and multi-camera identity fusion.
- **`live_perception/`, `perception/`, `sensor_manager/`, `sensor_fusion/`, `sensors/`** — perception evidence gateways and sensor fusion.
- **`building_state/`, `live_occupants/`** — canonical live building-state and persistent occupant identity/lifecycle.
- **`crowd_intelligence/`, `evacuation_progress/`, `trajectory_intelligence/`, `emergency_response/`, `stair_flow/`** — the four sibling live-intelligence engines plus stair-flow derivation.
- **`facp/`, `human_evidence/`** — Fire Alarm Control Panel state machine and human-state evidence reconciliation.
- **`fire_safety_manager/`, `fire_water_manager/`, `emergency_light_manager/`** — Designer/Command-Center fire-safety asset digital twins (execution-inert by design).
- **`ai_registry/`, `ai_inference/`, `ai_training/`, `ai_features/`, `ai_explainability/`, `ai_decision/`, `model_benchmark/`, `predictive_dataset/`, `predictive_model/`, `dataset_builder/`, `ground_truth/`, `research_framework/`** — the full predictive-AI research and production-registry pipeline.
- **`prediction_evaluation/`** — out-of-band evaluation of Shadow-Mode predictions against ground truth.
- **`evacuation_recommendation/`** — canonical, FROZEN, per-zone exit-routing engine.
- **`evacuation_guidance/`** — canonical, FROZEN, "how to reach it" route/message planning engine.
- **`advisory_system/`** — multi-audience Replay-only reporting layer (civilian/firefighter/building/commander).
- **`recommendation_layer/`** — the unified, canonical recommendation interface (see §2.5).
- **`voice_evacuation/`, `speaker_manager/`, `building_control/`, `dynamic_signage/`, `sign_manager/`, `warden_notification/`** — the four execution-controller pairs plus their two pure asset-registry siblings.
- **`execution_layer/`** — the unified, canonical execution-status coordination interface (see §2.6).
- **`live_runtime/`, `live_runtime_launcher/`, `live_system/`, `event_bus/`** — the live-runtime composition root, session launcher, orchestrator/state-manager, and pub-sub event bus.
- **`command_center/`** — the operator-facing live/replay monitoring and action-approval application.
- **`decision_policy/`** — an older, simulation-only, modular rule-based decision layer (unrelated to the live Recommendation/Execution chain; never imported by it).
- **`credential_store/`** — camera/NVR credential persistence.
- **`scenario_event_executor/`, `scenario_runner/`** — scripted scenario replay/execution.
- **`campaign_analytics/`** — post-campaign batch-simulation analysis (unrelated same-named `generate_report()` to Advisory's, confirmed no code relationship).
- **`serialization/`, `core/`** — project file (de)serialization and the application composition root (`core/app.py::SynEvacApp`).
- **`config/`, `utils/`, `analysis/`, `validation/`, `validation_framework/`, `validation_media/`** — supporting/cross-cutting utility and validation packages.
- **`rl/`, `rl_training/`** — reinforcement-learning research scaffolding (peripheral to the main pipeline).
- **`occupancy/`, `observable_assets/`, `visibility/`** — supporting occupancy/visibility geometry utilities.
- **`tests/`** — the full test suite (5300+ tests as of this writing).
- **`docs/architecture/`** — the authoritative, dated architecture-review/milestone-documentation trail (one doc per major milestone).
- **`scripts/`** — standalone diagnostic/benchmark/validation CLI scripts (not part of the shipped app).

---

## 5. Frozen Architecture

The following are **FROZEN** — do not modify unless a minimal, absolutely-unavoidable integration seam is required (and even then, prefer additive changes over rewrites):

- **`evacuation_recommendation/`** — canonical exit-routing engine. Frozen per `docs/architecture/core_architecture_freeze_review.md`'s own Phase-12 table.
- **`evacuation_guidance/`** — canonical route/message-planning engine. Same table.
- **`recommendation_layer/`** — the unified recommendation interface. Frozen as of its own completion; existing providers it adapts (above) were never modified to build it.
- **`execution_layer/`** — the unified execution-coordination interface. Frozen as of its own completion.
- **Designer, Navigation, Simulation, Perception, Crowd Intelligence** — all frozen per the same core freeze review.
- **`voice_evacuation.controller.VoiceEvacuationController`, `building_control.controller.BuildingControlController`, `dynamic_signage.controller.DynamicSignageController`** — the pre-existing execution controllers. Not formally in the FROZEN table (predate/postdate the review), but treated as frozen-in-practice by the Execution Layer milestone's own explicit design decision: extend via new sibling controllers/gateway methods, never modify their internals.

**Why**: each of these represents a mature, extensively tested, mechanically-guarded architectural boundary. Every recent milestone (Recommendation Layer, Execution Layer) has independently rediscovered the same lesson — reflexively check whether a "new" subsystem the brief describes already exists before building a duplicate. Future work should default to **coordinating/adapting** existing frozen layers, never reimplementing or bypassing them.

---

## 6. Active Development

**No subsystem is currently mid-implementation.** Recommendation Layer and Execution Layer are both complete and frozen as of this update. The project is at a natural milestone boundary — the next piece of work has not yet been chosen/scoped.

---

## 7. Future Roadmap

**Already completed** (see §2 for full detail): Building Designer/Studio, Builder, Navigation/Pathfinding, Manual Simulation Sandbox, Hazard/Fire Growth Simulation, Scenario Generation & Dataset Pipelines, Camera Calibration/CCTV/Human Detection, Building State Estimator, Live Occupants, Crowd/Progress/Trajectory/Emergency-Response Intelligence, FACP/MCP/Emergency Lighting, Fire Suppression & Water Infrastructure assets, Predictive Dataset/Model research arc (V1-V4), Shadow-Mode Live AI Inference, Prediction Evaluation, Evacuation Recommendation Engine, Evacuation Guidance Engine, Advisory System, **Recommendation Layer**, Voice Evacuation/Building Control/Dynamic Signage controllers, **Warden Notification**, **Execution Layer**, Live Runtime Composition Root, `live_system` Orchestrator/State Manager, Command Center, Live Runtime Launcher.

**In progress**: none currently.

**Not yet started / disclosed open gaps** (candidates for future milestones, none committed to yet):
1. Real hardware provider implementations for Voice/BuildingControl/Signage/Warden (currently Simulation-only everywhere; LIVE mode has `NO_PROVIDER` for all four).
2. Giving `advisory_system` a genuine live-reachable path (today it is Replay-only — `live_advisory_gateway` is never constructed outside tests).
3. Closing the `recommendation_id` traceability gap for Voice/BuildingControl/Signage (only Warden Notification has real, honest, end-to-end traceability today).
4. A future Guidance v2 / mobile app / external REST API layer consuming `RecommendationLayer`/`ExecutionLayer`'s public APIs (explicitly named as a design goal for `ExecutionLayer` to be *capable of supporting*, not yet built).
5. Wiring an `action_executor` into `SimulationControlProvider` in a real production path (Door/Exit control currently always resolves `FAILED` — a pre-existing, disclosed gap, not yet fixed).
6. Minor cleanup: removing the dead legacy `decision_policy_gateway`/`recommendation_builder` orchestrator seam (never configured in production).

---

## 8. Architectural Rules

- **Layers only ever read the previous layer's already-computed output** — never recompute, never reach sideways into a sibling engine's `.engine`/`.controller` internals, only its plain output models (mechanically enforced by dozens of dedicated `test_*_architecture_guards.py` files, one per subsystem).
- **A "coordinating/orchestration layer" never becomes a second execution/decision authority.** Recommendation Layer never generates new recommendation *intelligence* beyond thin adapters; Execution Layer never calls a provider itself. Both were designed this way specifically to avoid duplicating pre-existing frozen subsystems.
- **AI/Advisory can never reach execution authority directly.** Mechanically proven by dedicated guard tests scanning `advisory_system/`, `live_ai_gateway.py`, `live_advisory_gateway.py`, `decision_policy/`, and `live_system/orchestrator.py` for any import of an execution controller or gateway.
- **`LiveOrchestrator` never imports execution controllers or `live_runtime` itself** — the controllers live one layer up, on `LiveRuntime`. Any new subsystem needing to read them must be ticked from `LiveRuntime` directly (a new sibling method to `run_cycle()`), never from inside the orchestrator's own cycle.
- **Every optional pipeline stage follows the same "add a gateway" pattern**: `Optional[XGateway] = None` constructor param, a `latest_x` forwarding property, a guarded block in `run_cycle()`, a `LiveBuildingSnapshot` field + `replace()` entry, `StateManager.latest_x()`/`update_x()`, an `EventType.X_UPDATED` member. Deviate from this only when a specific architectural constraint (like the orchestrator-guard above) forces a different seam — and disclose why when it happens.
- **Human approval is a real two-step gate, never a rubber stamp.** Every execution controller requires an explicit `submit()` (never dispatches) followed by a *separate*, explicit `approve()`/`reject()` call; approving/rejecting an already-resolved request raises rather than silently succeeding.
- **Never fabricate a value that wasn't actually observed/computed.** `None`/"UNAVAILABLE"/"NO_PROVIDER" is always the honest default when evidence is missing — disclosed prominently in code comments and docs, never silently guessed or defaulted to something plausible-looking.
- **Naming convention**: `Engine<X>Gateway` = the real, factory-constructed adapter wrapping an `<X>Engine`/`<X>Controller`; a bare `<X>Gateway` (no `Engine` prefix) = the `Protocol`/ABC defining the seam's shape.
- **New Designer-authorable asset type** = `models/<asset>.py` + `designer/items/<asset>_item.py` + toolbar action + property panel section, following the existing ~30-asset precedent.
- **Public API is always one facade class per subsystem** (`RecommendationLayer`, `ExecutionLayer`, `EvacuationRecommendationEngine`, etc.) — future consumers import that facade, never reach into the subsystem's own internal modules.
- **Ownership of decisions**: Recommendation = "what should happen" (informational only, zero execution authority — mechanically enforced, no `.execute(`/`.dispatch(` call anywhere in that package). Execution = "how it's carried out" (coordination only, same zero-new-authority discipline — the four controllers remain the sole place a command is actually dispatched).

---

## 9. Important Discoveries

- **Existing execution gateway already existed.** Before the Execution Layer milestone began, a dedicated architectural review discovered `command_center.live_operator_action_gateway.LiveOperatorActionGateway` was already a complete, tested, production-wired execution mechanism for Voice/BuildingControl/Signage — the milestone extended it (adding Warden Notification as a fourth sibling) rather than building a competing seam.
- **Advisory system is Replay-only, not a live subsystem.** `AdvisoryOrchestrator.generate_report()` has exactly one real, non-test invocation path (`command_center/incident_data.py`'s Load-Incident flow); the live-compatible gateway is never constructed outside tests. `AdvisoryReport` is `None` in the overwhelming majority of live cycles — this is now explicitly disclosed in both Recommendation Layer's and Execution Layer's own documentation, rather than silently assumed.
- **Recommendation Layer introduced the first unified, cross-provider recommendation lifecycle** (stable IDs, dedup, provenance, expiration) — nothing before it tracked "is this the same recommendation reappearing, or a new one" across cycles at all; `evacuation_recommendation` itself is fully stateless per call.
- **`LiveOrchestrator` cannot reach execution controllers — by mechanically enforced design**, not oversight. This is *the* reason `ExecutionLayer` needed a new `LiveRuntime.tick_execution_layer()` sibling method to `run_cycle()`, rather than being wired in as just another orchestrator stage the way Recommendation Layer was.
- **Two entirely independent "tick" sources exist in Designer**: the Manual Simulation Sandbox (no perception/AI/recommendation pipeline behind it at all) and the real Live Runtime session (the only path that actually produces Recommendation/Guidance/Advisory/AI/Execution data). Any new Studio feature needing real pipeline data must hook into the Live Runtime tick (`LiveRuntimeController.on_cycle_callback`), never the Sandbox loop.
- **Voice Evacuation has no internal PENDING_APPROVAL phase of its own** (unlike BuildingControl/Signage) — `broadcast()` dispatches synchronously and atomically the moment it's called; the pre-dispatch "has an operator reviewed this" bookkeeping lives entirely one layer up, in the gateway. This is why Voice's `ExecutionRequest.created_at`/`approved_at` are always `None` in the Execution Layer's read-side adapter — an honest, disclosed limitation, not a bug.
- **Only Warden Notification has genuine, real `recommendation_id` traceability today.** Building Control/Voice only ever carry an `advisory_adapter`-*synthesized* content-hash id (never the real `recommendation_layer` id); Dynamic Signage's `SignageInstruction` has no traceability field at all. This is disclosed via the `recommendation_id_provenance` field rather than glossed over.
- **A pre-existing, unrelated flaky test exists in `builder/`'s own suite** (`tests/test_builder_project_management.py::BuilderNewProjectTests::test_new_project_replaces_current_project_and_resets_dirty_flag`) — it calls a real, unmocked `QMessageBox.question()` during `new_project()`, and its outcome depends on how the Qt platform resolves a modal dialog with no real display attached (observed run times: 3s / 12s / 106s / 145s across different invocations, and — newly observed this session — an indefinite hang lasting 48+ minutes before being force-killed). Confirmed via `git status` that zero files it depends on were touched by any recent milestone. **Caution for future sessions**: running this test file (or anything that constructs a dirty `BuilderMainWindow`) without `QT_QPA_PLATFORM=offscreen` forced can pop a real, visible dialog box on the developer's actual screen, and `QT_QPA_PLATFORM=offscreen` itself does NOT guarantee a bounded run time — it suppresses rendering, not the blocking modal event loop, so an unmocked `QMessageBox.exec()` can still hang indefinitely.
- **A second, previously-undiscovered instance of the same root cause exists**: `tests/test_builder_validation.py::BuilderMainWindowValidationGatingTests::test_save_succeeds_once_error_is_resolved` calls `BuilderMainWindow._save_to()` expecting a validation failure (which triggers a real, unmocked `QMessageBox.critical()` at `builder/windows/builder_main_window.py:734`) without patching it, unlike its sibling test directly above (`test_save_blocked_when_project_has_critical_errors`), which correctly does. Found only by tracing an anomalously long full-suite hang back to its exact stall point in test-collection order — a full suite run must currently `--deselect` both of these tests to complete in bounded time. Neither is related to Live Camera Viewer or to each other's discovery; both are disclosed, not fixed (out of scope for this milestone).
- **AI's coupling into Recommendation is mathematically rank-invariant, proven not assumed.** `ai_prediction_snapshot` is applied as an identical additive term to every candidate in a zone — provably incapable of changing relative order, verified both symbolically and empirically. This predates the Shadow-Mode/Recommendation Layer milestones (an even earlier milestone already built it in).

---

## 10. Session Changelog

### Session: Recommendation Layer V1

**Date**: (this development arc, prior to Execution Layer)

**Subsystem**: Recommendation Layer (new)

**Files created**: `recommendation_layer/` (`__init__.py`, `models.py`, `manager.py`, `layer.py`, `adapters/{__init__,occupant_routing,hazard_avoidance,congestion_mitigation,exit_utilization,warden_dispatch,system_warning}_adapter.py`), `live_system/recommendation_layer_gateway.py`, `designer/widgets/recommendation_panel.py`, `docs/architecture/recommendation_layer.md`, 12 new test files under `tests/`.

**Files modified**: `live_system/orchestrator.py` (new gateway param + `run_cycle()` stage + `latest_recommendation_set` property), `live_system/state_manager.py` (`recommendation_set` field), `event_bus/bus.py` (`RECOMMENDATION_SET_UPDATED`), `live_runtime/factory.py` + `runtime.py` (construction/threading), `designer/live_runtime_controller.py` (`on_cycle_callback` hook), `designer/windows/main_window.py` (panel/dock wiring), `designer/scene/graphics_scene.py` (`highlight_recommendation`/`clear_recommendation_highlight`), `designer/items/exit_item.py` (added `set_highlighted()`, previously missing).

**Architectural decisions**: dedup key = `type|trigger_condition|zones|exits`; fixed provider-priority collision order; 5-adapter-thin-passthrough discipline; `EXIT_UTILIZATION` as the one self-derived category.

**Tests added**: 53. **Tests passed**: full suite 5218, zero regressions.

**Known issues**: none new; disclosed `advisory_report`-usually-`None` limitation carried forward from upstream.

---

### Session: Execution Layer V1

**Date**: (this development arc, immediately following Recommendation Layer)

**Subsystem**: Execution Layer (new) + Warden Notification (new controller)

**Files created**: `warden_notification/` (`__init__.py`, `types.py`, `requests.py`, `history.py`, `controller.py`, `provider.py`), `execution_layer/` (`__init__.py`, `models.py`, `layer.py`, `adapters/{__init__,voice,building_control,signage,warden}_adapter.py`), `command_center/warden_notifications_panel.py`, `designer/widgets/execution_panel.py`, `docs/architecture/execution_layer.md`, ~13 new test files under `tests/`.

**Files modified**: `live_runtime/factory.py` (Warden provider default + `ExecutionLayer` construction + gateway wiring), `live_runtime/runtime.py` (new `warden_notification_controller`/`execution_layer` attributes + new `tick_execution_layer(time)` method, deliberately separate from `run_cycle()`), `command_center/live_operator_action_gateway.py` (additive `warden_controller` param + `ingest_warden_recommendations()`/`approve_warden_notification()`/`reject_warden_notification()`/read accessors), `command_center/recommendation_center.py` (7th tab), `command_center/dashboard.py` (`recommendation_set` threaded into `show_live()`), `command_center/data_source.py` + `live_system/live_command_center_gateway.py` (`recommendation_set` field forwarded into `CommandCenterSnapshot`), `designer/windows/main_window.py` (Execution panel/dock/toggle-action wiring, `_on_live_runtime_tick()` wrapper), `tests/test_command_center.py` (panel-count assertion updated 6→7).

**Architectural decisions**: pre-implementation architectural review discovered `LiveOperatorActionGateway` already existed — Execution Layer extends it rather than duplicating it; `ExecutionLayer` ticked via a new `LiveRuntime` method (not inside `run_cycle()`) due to the orchestrator-cannot-reach-controllers guard; `recommendation_id_provenance` field added for honest traceability disclosure across the three pre-existing categories vs. the one new (fully traceable) Warden category.

**Tests added**: 49. **Tests passed**: full suite 5315, zero regressions from this work (one pre-existing, unrelated flaky test in `builder/`'s own suite — see §9).

**Known issues**: none new. Disclosed: no real hardware providers anywhere; Voice/BuildingControl/Signage traceability incomplete; `builder/` flaky test caution noted for future sessions running Qt tests without forcing `QT_QPA_PLATFORM=offscreen`.

---

### Session: Live Camera Viewer V1 (video presentation only)

**Date**: (this development arc, immediately following Execution Layer V1)

**Subsystem**: `live_camera_pipeline` (additive), Designer (new panel)

**Context**: Camera 1's real RTSP connectivity had been independently verified (bounded, one-shot preflight, same session) before this milestone began. This milestone lets an operator see that stream inside Studio — video only, explicitly no YOLO overlays, no detection/tracking changes, no runtime redesign, and no second RTSP connection/decoder of any kind.

**Files created**: `designer/widgets/live_camera_view_panel.py` (`LiveCameraViewPanel` — dumb widget, `refresh(camera_name, connection_state, frame)` entry point, same convention as `RecommendationPanel`/`ExecutionPanel`), `tests/test_designer_live_camera_view_panel.py`.

**Files modified**: `live_camera_pipeline/pipeline.py` (`LiveCameraPipeline` gained a single-slot `_latest_frames` cache + public `latest_frame(camera_id)` getter, populated inside the existing `run_cycle()` loop right after its own pre-existing `read_frame()` call — no second read, no change to detection/tracking/fusion logic), `designer/windows/main_window.py` (panel/dock/toggle-action wiring mirroring Recommendation/Execution panels exactly; `_on_live_runtime_tick()` extended with a third call, `_refresh_live_camera_view_panel()`, reading `session.runtime.camera_pipeline`/`.camera_manager`/`.frame_sources` — all pre-existing `LiveRuntime` attributes, so no factory/runtime/orchestrator/gateway changes were needed), `tests/test_live_camera_pipeline.py` (new `LiveCameraPipelineLatestFrameCacheTests`), `tests/test_application_live_runtime_launcher.py` (one added assertion confirming `runtime.camera_pipeline.latest_frame()` is reachable through the real composition path).

**Architectural decisions**: the smallest clean integration point was a cache on the one class that already legitimately owns the single `read_frame()` call (`LiveCameraPipeline`), read by a new Designer panel via already-public `LiveRuntime` attributes — zero changes needed to `factory.py`, `runtime.py`, `orchestrator.py`, `state_manager.py`, or any gateway. No new thread was introduced: the entire live pipeline already runs on the Qt main thread inside the existing 1 Hz `QTimer` tick. `QImage` construction from the BGR ndarray explicitly `.copy()`s the buffer (OpenCV can reuse/overwrite it on the next decode) to avoid a use-after-free-style corruption hazard. The panel clears its displayed pixmap whenever connection status is not `Online` (never leaves a stale frame after a disconnect), and shows exactly one camera (the first configured `frame_source`) — a channel selector is explicitly out of scope for V1.

**Tests added**: 9 new + 1 amended assertion. **Tests passed**: full suite 5324 passed, 0 failed, 2 deselected (both pre-existing, unrelated — see below), zero regressions from this work.

**Known issues**: none new in this milestone's own code. **Two pre-existing, unrelated Builder test bugs were discovered/re-confirmed while chasing an anomalous full-suite hang** (both real, unmocked Qt modal dialogs — `QMessageBox.question()`/`QMessageBox.critical()` — that can block indefinitely even under `QT_QPA_PLATFORM=offscreen`; see §9 for full detail and exact deselect arguments). **Real Camera 1 verification was environmentally blocked**: the physical RTSP stream was unreachable at verification time (TCP port 554 open and responsive, but RTSP stream negotiation failed) — confirmed via the same pre-existing, previously-successful `scripts/test_camera_connection.py` diagnostic failing identically, proving this is not a defect in the new code. A full real, production-path run (`LiveRuntimeSession` with the real Camera 1 config + real credential store + real YOLO weights) was executed instead, proving the entire pipeline — connection attempt, `CameraConnectionState` propagation, `latest_frame()` cache, and `LiveCameraViewPanel.refresh()` — handles this real disconnected state correctly (correct "Stream Unavailable" display, no stale frame, no crash). The moment the physical stream is reachable again, the same scripted proof is ready to re-run for a full live-frame confirmation.

---

### Session: Camera -> Zone Assignment V1 (Building Topology Foundation, Milestone 1)

**Date**: 2026-08-06 (immediately following the Building Topology & Camera Topology Foundation investigation — see `docs/architecture/building_camera_topology_design.md`, produced the same session on branch `worktree-building-camera-topology-design`)

**Context**: the investigation found that a camera-to-zone assignment mechanism already existed end-to-end (`Camera.zone_ids` on `EngineeringAsset`, a Property Panel combo box, `_check_zone_assignment` Designer validation, `cross_camera_identity.topology.build_topology_from_navigation_graph`'s own `camera_zone_ids` parameter) but was silently capped at exactly one zone by the Property Panel's UI, even though the model field was already a tuple with no cardinality limit. The user explicitly scoped this session to closing *only* that one gap — no coverage polygons, camera overlap, transition probabilities, tripwires, cross-camera tracking, or occupancy prediction.

**Subsystem**: Designer Property Panel (Camera section), Designer validation, `models/floor.py` (additive).

**Files modified**: `designer/widgets/property_panel.py` (`self.camera_zone`: `QComboBox` → `QListWidget` + new `camera_zone_warning` label, reusing the exact `_populate_zone_checklist`/`_checked_zone_ids`/`_update_zone_warning` helpers `Speaker.zone_ids` already established; `update_camera_zone()` rewritten from an index-based combo handler to a checklist handler), `designer/validation.py` (`validate_building_authoring()` gained one `for camera in floor.cameras: _check_zone_assignment(...)` line, new `camera_missing_zone` WARNING code, same pattern as Speaker/SmokeDetector/etc.), `models/floor.py` (new `get_zone(zone_id)` lookup, mirroring `Building.get_floor(floor_id)` exactly — the honest way runtime code resolves `zone_ids` into real `Zone` objects), `tests/test_camera.py` (two pre-existing Property Panel tests updated from `QComboBox` APIs — `currentData()`/`findData()`/`setCurrentIndex()` — to the new checklist APIs; behavior they assert is unchanged), `tests/test_property_panel_zone_assignment.py` (new `CameraMultiZoneUITests`, 7 tests, mirroring `SpeakerMultiZoneUITests`), `tests/test_designer_validation_zone_assignment.py` (new `camera_missing_zone` coverage), `PROJECT_STATE.md`.

**Files created**: `tests/test_camera_zone_assignment.py` (`Floor.get_zone()` tests, the "which zones does this Camera observe" query demonstration, and a full `Building`-level `to_dict()`/`from_dict()` round trip + a from-scratch legacy-dict backward-compatibility test — the level actually written by `serialization.Serializer`, not just `Camera`'s or `Floor`'s own in isolation, which `tests/test_camera.py` already covered).

**Explicitly NOT changed**: `models/camera.py`/`models/engineering_asset.py` (`Camera.zone_ids` was already a correctly-shaped `Tuple[str, ...]` — zero model change needed), `navigation/*`, `cross_camera_identity/*`, `camera_coverage/*`, `builder/widgets/builder_property_panel.py` (SynEvac Builder's own, separate, simpler Camera zone `QComboBox` — deliberately left single-zone, out of scope for this milestone).

**Verification against a real project**: `C:\Users\riddh\Desktop\p3.syn` (a real, `.gitignore`d, personally-authored project outside the repo — not `p3.syn` in the repo root, a different, older file). Loaded via `serialization.Serializer.load()`, found Camera 7 (`ffbdac49-7982-4b61-86cc-98cbbc73b899`, real `Live`-mode RTSP config, previously `zone_ids=()`), assigned it to Zone 4 + Zone 9 (its real containing/neighboring zones, derived from its actual authored position — zero hard-coded ids), saved to a temp copy (`Serializer.save()`, original file never touched — confirmed by construction, only `Serializer.load()` was ever called on the real path), reloaded the temp copy, and confirmed both `reloaded_camera7.zone_ids` round-tripped exactly and `floor.get_zone(zone_id)` resolves each id to the correct real `Zone` (name + `zone_type`) — all without touching RTSP/live CCTV code (`connection`/`mode` fields read back unchanged, never exercised).

**Architectural decisions**: reused every existing seam verbatim — no new model class, no new package, no parallel representation of zone membership. The only genuinely new code is `Floor.get_zone()` (4 lines, mirrors an existing pattern) and the UI widening (which is a mechanical copy of Speaker's own already-proven pattern). This directly followed the investigation's own "if it already exists, reuse it" instruction.

**Tests added**: 7 new UI tests + 2 amended + 5 new validation tests + 11 new model/query/round-trip tests = 25 new/amended, all passing. **Full suite**: every one of the 430 files in `tests/` was run this session (in batches — a single unbroken `pytest tests/` invocation was not achievable in this environment/session, see below) — **zero failures caused by this milestone**. Exactly one genuine (pre-existing, environmental) failure surfaced and was root-caused: see "Known worktree-only test artifact" in the Quick Snapshot above. A second pre-existing gap was found and fixed as a byproduct (not a code change): this session's fresh git worktree was missing the gitignored `weights/yolov8n.pt` binary present in the main checkout, causing `tests/test_application_live_runtime_launcher.py`'s real-YOLO-pipeline test to fail purely from the missing file; copying the weights file from the main checkout resolved it (no repository change — `weights/*.pt` stays gitignored).

**Known issues**: none new in this milestone's own code. **Environment note for future sessions**: full-suite `pytest tests/` runs in this session's background-bash environment were repeatedly killed by the harness partway through (at inconsistent points, 7%–52%) regardless of the explicit timeout passed — root cause not fully identified, but batching the suite into ~15 alphabetical-prefix groups (`test_a*.py test_b*.py`, `test_ca*.py`, etc.) reliably completed every batch. A single `pytest tests/ -x` (stop-on-first-failure) run did also complete normally in ~130s. Future sessions needing a full-suite confirmation in this environment should default to batching rather than one long invocation.

---

### Session: SynEvac Capability & Architecture Inventory

**Date**: 2026-08-06 (immediately following Camera -> Zone Assignment V1, same worktree/branch `milestone-camera-zone-assignment`)

**Subsystem**: documentation only. **No code was changed** — this milestone was explicitly scoped by the user as investigation-and-documentation-only, to give future milestones a verified architectural reference before extending anything further.

**Context**: the user asked for a complete architectural inventory of the whole system — purpose, main classes, data model, runtime ownership, entry points, capabilities, limitations, tests, and production-vs-dormant status for ~20 major subsystems (Building Model, Navigation, Cameras, Live Runtime, CCTV, Human Detection, Live Occupants, Building State, Command Center, Designer, Simulation, Recommendation Layer, Execution Layer, Decision Engine, AI/reasoning modules, Cross-Camera Identity, Calibration, Geometry utilities, Visualization utilities, dormant/partial systems) — plus dependency graphs and a capability matrix, so future work extends the architecture instead of accidentally duplicating it.

**Method**: own verified knowledge from the two immediately-preceding milestones (Building Model, Navigation, Camera model/coverage/topology, Cross-Camera Identity — all freshly investigated against real code this same session) combined with four parallel `Explore` subagent investigations, each independently verifying its own cluster against real source (exact file paths and class names throughout, entry points confirmed by grepping for actual non-test call sites, not just definitions) rather than trusting prior documentation blindly: (1) CCTV/Human Detection/Tracking/Identity/Live Occupants, (2) Live Runtime/Building State/Command Center, (3) Designer/Simulation/Decision-Engine/AI/Recommendation/Execution, (4) Calibration/Geometry/Visualization utilities + a dedicated dormant-and-partially-implemented-systems scan.

**Files created**: `docs/architecture/system_capability_inventory.md` — the full inventory (23 numbered sections: 20 subsystem write-ups + 4 dependency graphs + 1 capability matrix + "what future milestones build upon this" summary).

**Files modified**: `PROJECT_STATE.md` (this file) — added a pointer to the new inventory doc right under the purpose blockquote, updated the Quick Snapshot for this milestone, and folded in a real, previously-undocumented gap the inventory's dormant-systems scan found: `automatic_calibration/`, `calibration_studio/`, `calibration_benchmark/` (simulator-physics calibration against published evacuation research, 17+ dedicated test files) had zero mention anywhere in this file before now.

**Key findings** (full detail in the inventory doc itself):
- **A single, recurring dormancy pattern** across four independent subsystems (Human Detection, Tracking, Behavior Recognition, Cross-Camera Identity): all four are real, tested, correctly wired all the way through `live_runtime/factory.py::build_live_runtime()` — but `live_runtime_launcher/session.py::LiveRuntimeSession.construct()` only actually constructs them when a `human_detector_weights_path` is explicitly supplied, and `designer/live_runtime_controller.py::LiveRuntimeController.on_start()` never supplies one. One missing parameter gates four otherwise-complete subsystems from the shipped Designer app; each is proven only via `scripts/demo_*.py`/`scripts/benchmark_*.py` and the test suite.
- **Camera Topology** (`cross_camera_identity/topology.py::build_topology_from_navigation_graph`) independently reconfirmed dormant (zero production callers) — consistent with the Building Topology investigation's own finding two milestones ago, now cross-verified from a different angle.
- **Shadow-Mode AI** (`live_ai_gateway`) confirmed dormant in the shipped Designer app for the same reason as the perception chain — a real, tested, one-parameter-away-from-activation gap, not a missing feature.
- **Camera-to-camera geometric overlap** reconfirmed genuinely missing (not merely dormant) — matches the Building Topology investigation's own finding.
- **A real, previously-undocumented body of work found**: `automatic_calibration/`, `calibration_studio/`, `calibration_benchmark/` (simulator-physics calibration, not camera-geometry calibration) — substantial and tested, absent from `PROJECT_STATE.md`'s "Completed Subsystems" and "Future Roadmap" sections entirely until this session's fix.
- **Evidence of uncommitted, in-progress work** found in the user's live working tree (outside this worktree's committed history): a Command Center multi-camera grid view (`camera_tile_widget.py`, `live_camera_grid_panel.py`, `live_camera_view_gateway.py` + modifications to `building_view.py`/`dashboard.py`/`main_window.py`) — flagged in the inventory (§9, §20) as evidence, not verified in depth (out of scope for a documentation-only milestone against a specific worktree).

**Architectural decisions**: none — this was investigation and documentation only, per explicit user instruction. No files outside `docs/architecture/system_capability_inventory.md` and `PROJECT_STATE.md` were touched.

**Tests added**: none (no code changed). **Tests passed**: not applicable — no code was run beyond the read-only investigation itself (agents used Read/Glob/Grep only, no test execution needed for a pure inventory task).

**Known issues**: none new. The two not-yet-merged branches from this development arc (`worktree-building-camera-topology-design` for the Building/Camera Topology investigation doc, and this branch `milestone-camera-zone-assignment` for the Camera -> Zone Assignment milestone + this inventory) still need a merge decision — out of scope for this milestone, noted for whoever manages branch integration next.
