# SynEvac System Capability Inventory

**Status**: investigation and documentation only — no code was changed to produce this document. It is the architectural reference the project's own `PROJECT_STATE.md` points to for full detail; `PROJECT_STATE.md` remains the quick-glance summary.

**Scope**: as of this writing, the codebase at commit `67159fc` (local `main`) plus two not-yet-merged branches produced earlier this development arc — `worktree-building-camera-topology-design` (Building/Camera Topology investigation + proposal, no code) and `milestone-camera-zone-assignment` (Camera → Zone Assignment V1, shipped). This document describes the codebase as it exists in the `milestone-camera-zone-assignment` worktree, i.e. `main` + the Camera → Zone Assignment milestone. Every claim below was verified against real source (file paths and class names are exact), not inferred from prior documentation alone — where an existing doc's claim was spot-checked and found accurate, that is noted; where a gap in `PROJECT_STATE.md` itself was found, that is flagged in §20 and §21.

**How to read each section**: Purpose → Main classes → Data model → Runtime ownership → Entry points → Capabilities → Limitations → Tests → Status (production-ready / experimental / dormant / missing).

---

## 1. Building Model

**Purpose**: the single source of truth for a building's physical and engineering-asset structure. Per this project's own standing architectural rule, **the Building Designer IS the Digital Twin** — no parallel "digital twin" data model exists or should be built.

**Main classes** (`models/`):
- `building.py::Building` — `floors: list[Floor]`, `fire_water_systems`. `floor_elevation()` is derived (never stored) from cumulative `Floor.height` in `display_order`; `next_zone_name()` is derived from existing Zone names, never a stored counter.
- `floor.py::Floor` — ~25 asset-type lists (`zones, exits, stairs, elevators, cameras, detectors, smoke_detectors, heat_detectors, speakers, signs, manual_call_points, emergency_lights, sprinklers, fire_extinguishers, fire_hydrants, hose_reels, fire_water_tanks, fire_pumps, jockey_pumps, fire_service_inlets, assembly_points, obstacles, doors`), each with `add_x()`/`remove_x()`/`x_count`. `floor_plan_scale`/calibration-point fields support floor-plan-image-to-meters scale calibration (SynEvac Builder milestone). `get_zone(zone_id)` (added by the Camera → Zone Assignment milestone) mirrors `Building.get_floor()`.
- `zone.py::Zone(BaseObject)` — `x, y, width, height, polygon` (rectangle-only in every real project seen so far; `polygon` exists but is unused), `zone_type` (`Generic, Room, Corridor, Lobby, Stair Lobby, Mechanical, Electrical, Storage, Office, Outdoor`), `max_occupancy`.
- `door.py::Door(BaseObject)` — `start_point, end_point, zone_a_id, zone_b_id` (connects any `connectable_space` — Zone or AssemblyPoint), `door_type, normally_open, locked, active`.
- `exit.py::Exit(BaseObject)` — `start_point, end_point, zone_id, width, capacity, is_blocked`.
- `staircase.py::Staircase(BaseObject)` — one object spans exactly two floors: `from_position/to_position` (each in its own floor's local coordinates), `from_floor_id/to_floor_id, from_zone_id/to_zone_id, width`. `vertical_height()`/`travel_distance()` are derived via `Building.floor_elevation()`, never stored. `StairObservableRegion` (optional, per-side) supports camera-based stair perception.
- `obstacle.py::Obstacle(BaseObject)` — `traversability, traversal_cost`; checked live against Door/Exit edge geometry on every `Edge.traversable` access, never baked into the graph at build time.
- `engineering_asset.py::EngineeringAsset(BaseObject)` — shared base for every physical device (Camera, Detector-family, Speaker, Sign, MCP, Emergency Light, fire-safety/fire-water assets): `floor_id, zone_ids: Tuple[str,...], position, mount_height, active, mode: DeviceMode (SIMULATION/REPLAY/LIVE), connection: ConnectionInfo` (`rtsp_address, ip_address, username, credential_ref` — plaintext password never persisted, captured into `credential_store` on save).
- `connectable_space.py` — the registry of what a Door may connect (`Zone`, `AssemblyPoint` today); explicitly designed as the seam for a future third type (e.g. Virtual Zone) without touching Door, the Property Panel, or the Navigation Graph builder.

**Data model**: plain dataclasses, JSON-serializable via `to_dict()`/`from_dict()` on every class, backward-compatible field defaults throughout (an old `.syn` missing a newer key loads with an honest default, never a crash).

**Runtime ownership**: owned exclusively by whichever `Project`/`Building` instance Designer (or Builder) has loaded; every other subsystem (Navigation, Simulation, Perception, Command Center) reads it, none duplicates it.

**Entry points**: `serialization.Serializer.save/load` ↔ `.syn` JSON files; authored interactively via Designer's `GraphicsScene`/`PropertyPanel` (§10) or SynEvac Builder.

**Capabilities**: full engineering-asset authoring across ~25 asset types; multi-floor; explicit (never geometry-inferred) connectivity via Door/Exit/Staircase zone references; Camera → Zone assignment is now genuinely multi-zone (Milestone 1, this session).

**Limitations**: `Zone.polygon` (non-rectangular zones) is modeled but unused by every real project inspected; no "Virtual Zone" type yet exists (the seam for one does); Camera zone assignment being non-empty is still operator-manual, never auto-derived from FOV geometry.

**Tests**: `test_camera.py`, `test_engineering_asset.py`, `test_zone_assignment_full_e2e.py`, `test_zone_assignment_failure_modes.py`, `test_zone_assignment_architecture_guards.py`, `test_camera_zone_assignment.py`, plus one `test_<asset>_model.py`/`test_<asset>_designer.py` pair per asset type.

**Status**: production-ready, actively used everywhere. No missing pieces relative to what every other subsystem currently needs.

---

## 2. Navigation

**Purpose**: derives the building's zone-connectivity graph and provides shortest/safe-path computation. **This already IS the "zone topology"/"building topology" concept** any future virtual-zone or camera-topology work should build on, not duplicate (confirmed explicitly by the Building Topology investigation, §16/§17 below).

**Main classes** (`navigation/`, `pathfinding/`):
- `graph_builder.py::NavigationGraphGenerator.build(building) -> NavigationGraph` — a **pure, stateless function**. Never persisted independently of the Building that produced it; rebuilding after any edit reproduces the same graph.
- `node.py::Node` — one per Zone/AssemblyPoint, plus a single shared `"Outside"` node; holds a live `reference` back to the real object, never a copy. `space_type` exposes `Zone.zone_type`.
- `edge.py::Edge` — one per Door/Exit/Staircase; genuine `walking_distance` (meters, geometrically derived), `traversal_cost`/`traversal_time`, live `traversable` (checks Door `locked`/`active`, Exit `is_blocked`, and obstacle blocking on every access).
- `graph.py::NavigationGraph` — `nodes`, `edges`, `floor_ids`, `find_neighbors()`, `validate()` (reports `zone_without_connections`, `isolated_zone`, `disconnected_floor`).
- `flow_region.py` / `flow_region_inference.py` — Hybrid Flow Regions: groups edges representing one continuous crowding phenomenon (e.g. a multi-flight stairwell); purely additive, never changes routing, feeds future capacity-formula work only.
- `obstacle_geometry.py::segment_blocked_by_obstacles()` — reuses `visibility.geometry` primitives (§18), never a second implementation.
- `cost.py` — `CostModel` wrapping a `DefaultCostModel`; the documented seam for a future Simulation/Dynamic-Hazard layer to apply smoke/fire/congestion penalties without changing `Edge` itself.
- `pathfinding/` — `PathfindingEngine` (shortest/safe-path over the graph); consumed by the Manual Simulation Sandbox (`nearest_exit()`/`nearest_assembly_point()`) and `evacuation_recommendation`.

**Data model**: `Node`/`Edge` are thin, read-only views — never own or duplicate engineering-object data.

**Runtime ownership**: rebuilt fresh wherever needed (Designer validation, Sandbox routing, Evacuation Recommendation); never cached as a mutable singleton.

**Entry points**: `NavigationGraphGenerator().build(building)`, called from Designer validation, Sandbox, `evacuation_recommendation`, and multiple predictive-dataset/research pipelines.

**Capabilities**: multi-floor graph via Staircase edges; live obstacle-aware traversability; Hybrid Flow Region grouping for future capacity modeling; structural validation (disconnected floors/isolated zones).

**Limitations**: no built-in hazard-aware costing yet (the `CostModel` seam exists, unused by default); no notion of Virtual Zones/tripwires yet (deferred, per the Building Topology investigation).

**Tests**: `test_navigation.py`-family (graph/node/edge), `test_obstacle_navigation_integration.py`, `test_stair_simulation_reliability_audit*.py`, `test_flow_region*.py`.

**Status**: production-ready, FROZEN in spirit (core graph-derivation logic has been stable across many milestones), actively used by nearly every downstream subsystem.

---

## 3. Cameras

**Purpose**: model a physical (or planned) camera as a Digital Twin asset, and — separately — everything that reasons about what it sees.

**Camera model** (`models/camera.py::Camera(EngineeringAsset)`): `rotation, horizontal_fov (default 90°), max_range (default 25m), resolution/fps` (logical-only display strings, never parsed), `coverage_polygon()` (derived sector-fan geometry from position/rotation/FOV/range, never stored).

**Camera configuration / Live vs Simulation modes**: `EngineeringAsset.mode` (`DeviceMode.SIMULATION/REPLAY/LIVE`) + `ConnectionInfo` (`rtsp_address, credential_ref`). Only `LIVE` mode with a configured `rtsp_address` produces a real `RTSPFrameSource` (§5); `Simulation`/no-mode cameras never touch the network.

**Zone assignment**: `Camera.zone_ids: Tuple[str,...]` (inherited from `EngineeringAsset`) — as of the **Camera → Zone Assignment V1 milestone (this development arc)**, genuinely multi-zone via a checklist UI in `designer/widgets/property_panel.py` (`_populate_zone_checklist`, mirroring `Speaker`'s pre-existing pattern), with a matching `camera_missing_zone` Designer-validation warning and `Floor.get_zone()` for runtime resolution. Previously capped at one zone by the UI alone (the model field was always a tuple).

**Coverage** (`camera_coverage/` package — Camera ↔ Observable Asset, e.g. a Stair's `StairObservableRegion`, **not** Camera ↔ Camera): `models.py::CameraCoverage, AssetCoverage, CoverageState (FULLY_VISIBLE/PARTIALLY_VISIBLE/NOT_VISIBLE/UNKNOWN), CameraCoverageSnapshot`; `discovery.py::compute_camera_coverage()` derives a camera's calibrated sector-fan (reusing `Camera.coverage_polygon()`'s exact math, but from `CalibrationProfile.extrinsics` when available) and tests it against each asset's authored observable region — geometry only, **zero occupancy counts** (that stays in `observable_assets.models.ObservableAssetSnapshot`). Wired into `BuildingState.camera_coverage` (§8).

**Camera topology** (`cross_camera_identity/topology.py`): `CameraTopology` (hand-buildable camera-adjacency graph: `add_camera`, `add_transition(min/max_transition_time)`, `possible_destinations()`, `is_plausible_transition()` — a **boolean** plausibility check today, not a graded confidence score) and `build_topology_from_navigation_graph(navigation_graph, camera_zone_ids, ...)` — the automatic derivation function (two cameras become adjacent if their assigned zones are the same or one real `NavigationGraph` edge apart, using genuine `Edge.walking_distance`). **Fully engineered and unit-tested, but confirmed (again, this session) to be called only from `tests/test_cross_camera_identity.py` and standalone `scripts/demo_*`/`scripts/benchmark_*` files — never from `live_runtime_launcher/session.py` or any reachable production path.**

**Cross-camera support**: see §16.

**Camera-to-camera geometric overlap** ("do two cameras see the same physical patch of floor"): confirmed **missing entirely** — neither `camera_coverage` (Camera↔Asset) nor `cross_camera_identity` (Camera↔Camera adjacency/timing, not simultaneous visibility) answers this. Proposed (not implemented) in `docs/architecture/building_camera_topology_design.md` as a new `building_topology/overlap.py::CameraOverlapSnapshot`, reusing the same `visibility.geometry` polygon-intersection primitives `camera_coverage/discovery.py` already uses.

**Tests**: `test_camera.py`, `test_camera_zone_assignment.py`, `test_property_panel_zone_assignment.py`, `test_camera_coverage_discovery.py`, `test_camera_coverage_designer.py`, `test_cross_camera_identity.py`, `test_cross_camera_identity_architecture_guards.py`.

**Status**: model + zone assignment + coverage = production-ready and actively used. Camera topology (adjacency/transition-time) = engineered and tested but **dormant** (zero production callers). Camera-to-camera overlap = **missing** (designed, not built).

---

## 4. Live Runtime

**Purpose**: the single production composition root assembling every already-existing, already-tested engine (camera/sensor/fusion/FACP/perception/intelligence/recommendation/execution) into one coherent `LiveRuntime`.

**Main classes** (`live_runtime/`): `factory.py::build_live_runtime()` (production `ApplicationMode.LIVE` — no output provider defaulted; Voice/BuildingControl/Signage/Warden stay `None`/`NO_PROVIDER` unless a caller explicitly supplies one) and `build_offline_demo_runtime()` (`ApplicationMode.OFFLINE_DEMO` — all four default to their `Simulation*Provider`, fully offline). `runtime.py::LiveRuntime` — imports **zero concrete collaborator classes** (every attribute injected by the factory, mechanically enforced).

**Runtime lifecycle**: `live_runtime_launcher/session.py::LiveRuntimeSession` + `modes.py::RuntimeLifecycleState` (`STOPPED → STARTING → RUNNING`/`DEGRADED → STOPPED`, or `FAILED` on any construction exception, never a crash). `LiveRuntimeSession.construct(building)` picks the factory function per `ApplicationMode`, and under `LIVE` additionally builds real `RTSPFrameSource`s (`rtsp_camera_sources.py::build_rtsp_frame_sources()`) for every `Camera` with `mode=LIVE` + a real `rtsp_address` — and, **only if a YOLO weights path was explicitly supplied** (which the shipped Designer UI never does — see §6), a real `YOLOHumanDetector`/`SimpleSingleCameraTracker`.

**Update cadence**: `designer/live_runtime_controller.py::LiveRuntimeController` — a `QTimer` at `LIVE_CYCLE_INTERVAL_MS = 1000`, calling `session.runtime.run_cycle(time.time())` once per second, then `on_cycle_callback()` (refreshes Designer's Recommendation/Execution/Live-Camera-View panels). **This is the only place `run_cycle()` is ever called in the shipped application.**

**Snapshot flow**: `run_cycle()` (owned by `live_system.orchestrator.LiveOrchestrator`, §9) computes and stores an immutable `LiveBuildingSnapshot` on `StateManager`; Command Center's own `live_refresh_timer` (1000ms) only re-renders the already-computed snapshot, never advances the cycle.

**Capabilities**: fully offline `OFFLINE_DEMO` mode needs only a `Building`; `LIVE` mode is real-hardware-capable (RTSP) end to end when configured.

**Limitations**: no real hardware provider exists anywhere for Voice/BuildingControl/Signage/Warden/FACP-event/most sensors (Simulation-only, by design and mechanically enforced — `modbus`/`bacnet`/`mqtt`/`opcua`/`socket`/`serial` imports are forbidden by architecture-guard tests).

**Tests**: `test_live_runtime.py`, `test_live_runtime_architecture_guards.py`, `test_live_runtime_architecture_cleanup.py`, `test_live_runtime_e2e.py`, `test_live_runtime_failure_modes.py`, ~13 `test_live_runtime_<stage>_e2e.py` files, `test_application_live_runtime_launcher.py`.

**Status**: production-ready, actively used, the real spine of the live application.

---

## 5. CCTV

**Purpose**: turn a configured Camera asset into decoded frames, decode-library-independent.

**Main classes** (`live_camera_pipeline/`, `camera_manager/`): `frame_source.py::CameraFrame` (dataclass: `camera_id, timestamp, frame_sequence, payload_ref, width, height, codec`), `CameraFrameSource` (ABC); `rtsp_frame_source.py::RTSPFrameSource` — bounded retry/backoff, credential redaction, five-state status vocabulary; `rtsp_backend.py::FrameDecoderBackend` (ABC), `human_detection/opencv_decoder_backend.py::OpenCVFrameDecoderBackend` (the one real decoder, OpenCV-based); `replay_frame_source.py::ReplayFrameSource` (offline/demo, no hardware); `camera_manager/manager.py::CameraManager` (registry + per-`DeviceMode` provider routing, never does CV itself), `status.py::CameraStatus`, `connection_status.py::CameraConnectionState`.

**Frame sources / Live CCTV dashboard**: `live_runtime_launcher/rtsp_camera_sources.py::build_rtsp_frame_sources()` builds one `RTSPFrameSource` per LIVE camera with a real RTSP address, threaded into `build_live_runtime()`. The one video-showing widget is `designer/widgets/live_camera_view_panel.py::LiveCameraViewPanel` — lives in **Designer, not Command Center** (Command Center has data/state panels only, no raw video widget). Shows exactly one camera (first configured frame source, no channel selector), video-only by design (no YOLO overlay), reading `LiveCameraPipeline.latest_frame()` — a single-slot cache populated inside the real `run_cycle()` loop, never a second decode. Genuinely wired to the real 1 Hz tick via `MainWindow._on_live_runtime_tick()`.

**Capabilities**: real RTSP connectivity, bounded retry, credential-safe (never logs/persists plaintext passwords), decode via OpenCV, live video display in Designer.

**Limitations**: single-channel view only (no multi-camera grid in the shipped app — though `command_center/camera_tile_widget.py`/`live_camera_grid_panel.py`/`live_camera_view_gateway.py` exist as **uncommitted, in-progress work** in the user's live working tree, outside the scope of this worktree/commit — see §20); no detection overlay in the video panel.

**Tests**: `test_rtsp_frame_source.py`, `test_rtsp_failure_modes.py`, `test_rtsp_camera_manager_status_integration.py`, `test_rtsp_camera_sources.py`, `test_rtsp_offline_e2e.py`, `test_opencv_decoder_backend.py`, `test_camera_manager*.py`, `test_real_decoder_full_chain_e2e.py`, `test_cctv_offline_pipeline_validation.py`, `test_physical_camera_validation_field_runner.py`, `test_designer_live_camera_view_panel.py`.

**Status**: production-ready and actively reachable end-to-end from the shipped Designer app, given a real RTSP camera configured in the project — genuinely requires real network/hardware to show live video (Simulation/OFFLINE_DEMO never builds an `RTSPFrameSource`).

---

## 6. Human Detection

**Purpose**: person-only bounding-box detection on a decoded frame.

**Main classes** (`human_detection/`): `yolo_backend.py::UltralyticsYOLOBackend` (real, lazily loads `ultralytics.YOLO`; requires an existing local `.pt` weights path, never auto-downloads — raises `ModelWeightsNotFoundError` otherwise), `YOLOInferenceBackend` (ABC), `BoundingBoxDetection`; `yolo_human_detector.py::YOLOHumanDetector`. Repo ships real weights at `weights/yolov8n.pt` (gitignored, not tracked — must exist locally).

**Data model**: `RawHumanDetection(camera_id, local_track_id, timestamp, bounding_box, confidence, classification_evidence=UNKNOWN, state_evidence=None, floor_id, zone_id, world_position, world_velocity, ...)` — detection-only, no pose/classification.

**Runtime ownership / entry points**: `live_runtime_launcher/human_detector_wiring.py::build_yolo_human_detector()`, called by `LiveRuntimeSession.construct()` **only when `human_detector_weights_path` is explicitly supplied**.

**Confirmed this session**: `designer/live_runtime_controller.py::LiveRuntimeController.on_start()` constructs `LiveRuntimeSession(mode, credential_store=...)` **without** `human_detector_weights_path` — so the real YOLO detector is never constructed from any reachable Designer UI path today. It is exercised only by `scripts/dry_run_physical_cctv.py`, `scripts/run_physical_camera_validation.py`, `scripts/benchmark_yolo_human_detector.py`, `scripts/demo_real_yolo_tracking.py`, and tests. This matches and extends `PROJECT_STATE.md`'s existing "Camera 1 milestone was single-camera, deliberately minimal" framing — the gap is one config parameter, not missing engineering.

**Capabilities**: real, validated YOLO inference (per the earlier "Real YOLO Model Validation" milestone, run against real photos/video).

**Limitations**: not reachable from the shipped GUI without a code/config change; no in-app control to point Designer at a weights file.

**Tests**: `test_yolo_human_detector.py`, `test_human_detector_wiring.py`, `test_human_detection_architecture_guards.py`, `test_real_yolo_model_validation.py`, `test_yolo_tracking_integration.py`, `test_yolo_rtsp_live_runtime_compatibility.py`.

**Status**: production-quality code, **dormant in the shipped application** — proven via scripts and tests only.

---

## 7. Live Occupants

**Purpose**: the single canonical, cross-cycle occupant registry — the one place tracking/behavior/identity/state/classification evidence persists across cycles with an explicit lifecycle.

**Main classes** (`live_occupants/`): `manager.py::LiveOccupantManager` (O(1) dict + secondary indices by zone/floor/behavior/camera/stair; memoized `canonical_occupancy()`); `occupant.py::LiveOccupant` (frozen dataclass); `state.py::OccupantStatus`; `lifecycle.py` (exit-proximity/expiry rules); `history.py::OccupantHistory`; `occupancy.py::OccupancyFacts`.

**Data model**: `LiveOccupant(occupant_id, current_camera_id, current_track_id, current_zone_id, current_floor_id, world_position, world_velocity, behavior, confidence, first_seen, last_seen, status, history, human_classification[_confidence/source/last_observed_at], human_state[_confidence/source/last_observed_at], world_position_provenance, current_stair_id)`.

**Runtime ownership**: always constructed unconditionally by `live_runtime/factory.py::build_live_runtime()` — one shared instance exists regardless of whether any camera pipeline is configured. Consumed by `CrowdIntelligenceEngine`, `EvacuationProgressEngine`, `TrajectoryIntelligenceEngine`, `EmergencyResponseIntelligenceEngine`, `EvacuationRecommendationEngine`, `command_center/live_occupant_panel.py`.

**Occupant lifecycle / snapshots**: status transitions (present → near-exit → departed/expired) driven by `lifecycle.py`; `LiveOccupantManager.update()` is the per-cycle ingestion point from `LiveCameraPipeline`.

**Confirmed this session**: because `LiveCameraPipeline`'s `tracker` is never actually supplied in the shipped GUI (§6's gap cascades here), `LiveOccupantManager.update()` is never invoked from a real camera in production today — the manager itself is live and correctly wired, but has no real input source until §6's gap closes.

**Tests**: `test_live_occupants.py`, `test_live_occupants_architecture_guards.py`, `test_live_occupants_gateway.py`, `test_live_occupants_human_evidence.py`, `test_live_occupant_panel.py`, `test_live_camera_pipeline_occupant_integration.py`, `test_live_perception_double_counting.py`, `test_canonical_live_occupancy.py`.

**Status**: production-ready plumbing, correctly wired into every downstream intelligence engine — but effectively idle in the shipped GUI today because its only real-world input source (§6) is dormant. Fully exercised via Simulation/test paths.

---

## 8. Building State

**Purpose**: the single authoritative snapshot of "what is true right now" — fuses hazard/occupancy/camera/detector/FACP/control/fire-safety/fire-water/observable-asset/camera-coverage evidence into one immutable object per cycle. Deliberately carries no AI/decision-policy field.

**Main classes** (`building_state/`): `models.py::BuildingState` — fields: `state_id, timestamp, occupant_tracks, zone_occupancy (OccupancySnapshot), camera_observations, smoke_detector_states, heat_detector_states, manual_call_point_states, hazard_summary (HazardSummary), building_alarm_status, active_assets (ActiveAssetsSummary), facp_status, control_status, fire_safety_status, fire_water_status, observable_assets, camera_coverage`. `estimator.py::BuildingStateEstimator.estimate(...)` — a pure fusion function, agnostic to whether inputs came from Simulation/Replay/Live. `consistency.py::check_consistency()` — diagnostic-only `ConsistencyWarning`s (camera-occupancy mismatch, alarm-without-hazard, etc.), never fed back into state.

**Runtime ownership**: `live_system/building_state_gateway.py::EstimatorBuildingStateGateway` wraps the estimator behind per-cycle provider callables, invoked once per `run_cycle()` (§9, stage 4). A `None` provider yields an honest empty default, never a fabricated reading.

**Capabilities**: fuses everything currently perceivable; runs fully offline.

**Limitations**: entirely dependent on upstream perception actually running (§5-§7's dormancy means, in the shipped GUI today, `camera_observations`/`occupant_tracks` stay effectively empty for real cameras).

**Tests**: `test_building_state.py`, `test_building_state_consistency.py`, `test_sensor_fusion_building_state_integration.py`, `test_camera_detection_tracking_building_state_integration.py`.

**Status**: production-ready, actively used every cycle.

---

## 9. Command Center

**Purpose**: the operator-facing live/replay monitoring and action-approval application, a separate `QMainWindow` opened from Designer's Live Runtime panel.

**Main classes**: `main_window.py::MainWindow` (owns `Dashboard` + two mutually-exclusive `QTimer`s: `playback_timer` 200ms for Replay frame-advance, `live_refresh_timer` 1000ms for Live re-render only, never recomputation); `dashboard.py::Dashboard` (central widget, `apply_snapshot(CommandCenterSnapshot)` for Live, `set_incident()`/`show_frame()` for Replay, toggles Live-only vs Replay-only tabs); `data_source.py::CommandCenterSnapshot` (presentation dataclass — `mode, timestamp, building, frame, advisory_report, decision_policy` + live-only intelligence-snapshot fields + `consistency: SnapshotConsistency (CURRENT/STALE/PARTIAL/UNAVAILABLE)` + `recent_events`); `live_system/live_command_center_gateway.py::LiveCommandCenterDataSource` (reads only `StateManager.current()`, avoiding a circular import); `live_operator_action_gateway.py::LiveOperatorActionGateway` (the one seam allowed to route operator intent into real execution).

**Every distinct panel class found**: `hazard_panel.py::HazardPanel`, `event_timeline_panel.py::EventTimelinePanel`, `building_controls_panel.py::BuildingControlsPanel`, `building_view.py::BuildingView` (2D floor-plan render, both modes), `incident_panel.py::IncidentPanel`, `live_ai_panel.py::LiveAIPanel`, `live_dynamic_signage_panel.py::LiveDynamicSignagePanel`, `live_emergency_response_panel.py::LiveEmergencyResponsePanel`, `live_evacuation_guidance_panel.py::LiveEvacuationGuidancePanel`, `live_evacuation_progress_panel.py::LiveEvacuationProgressPanel`, `human_panel.py::HumanPanel`, `live_evacuation_recommendation_panel.py::LiveEvacuationRecommendationPanel`, `incident_status_bar.py::IncidentStatusBar`, `live_occupant_panel.py::LiveOccupantPanel`, `live_events_panel.py::LiveEventsPanel`, `live_status_panel.py::LiveStatusPanel`, `live_trajectory_intelligence_panel.py::LiveMovementIntelligencePanel`, `occupancy_panel.py::OccupancyPanel`, `occupant_inspector_panel.py::OccupantInspectorPanel`, `recommendation_panel.py::RecommendationPanel`, `recommendation_timeline_panel.py::RecommendationTimelinePanel`, `statistics_panel.py::StatisticsPanel`, `timeline_panel.py::TimelinePanel`, `warden_notifications_panel.py::WardenNotificationsPanel`; `recommendation_center.py::RecommendationCenter` hosts exactly 7 tabs (Civilian Announcements, Voice Evacuation, Firefighter Intelligence, Building Recommendations, Building Controls, Warden Notifications, Commander Summary).

**Live vs Replay**: Replay reads a static, offline `IncidentData`/`IncidentFrame` (via `command_center/incident_data.py`, needs no runtime); Live reads a fresh `CommandCenterSnapshot` every second from `LiveCommandCenterDataSource`. Same window, same widget tree, mutually exclusive.

**Note found this session, uncommitted**: `command_center/building_view.py, dashboard.py, main_window.py` show as modified in the user's live working tree (outside this worktree's committed history), alongside three new files `camera_tile_widget.py`, `live_camera_grid_panel.py`, `live_camera_view_gateway.py` — evidence of in-progress, uncommitted work toward a multi-camera Command Center grid view. Not evaluated here (uncommitted, not part of this worktree) — flagged for whoever picks it up next.

**Tests**: `test_command_center.py`, `test_command_center_facp_sources.py`, `test_dynamic_signage_command_center.py`, `test_fire_safety_command_center.py`, `test_fire_water_infrastructure_command_center.py`, `test_live_command_center.py`, `test_live_command_center_operator_actions.py`.

**Status**: production-ready, actively used, reachable via `LiveRuntimeSession.open_command_center()` (Live) and `MainWindow.load_incident_dialog()` (Replay).

---

## 10. Designer

**Purpose**: the primary building-authoring surface — SynEvac Studio.

**Main classes**: `designer/windows/main_window.py::MainWindow` — the composition root; wires `GraphicsScene`/`GraphicsView`, `PropertyPanel`, `SimulationPanel` (Sandbox controls, own 50ms `QTimer`), `PerceptionDebugPanel`, `BuildingStateDebugPanel`, `CameraManagerPanel`, `SpeakerManagerPanel`, `CameraValidationPanel`, a lazily-built Campaign Studio, `LiveRuntimePanel`/`LiveRuntimeController`, `RecommendationPanel`/`ExecutionPanel` (read-only, refreshed only by the Live Runtime tick — **not** the Sandbox loop), `LiveCameraViewPanel`, `ProjectTree`, `FloorList`, `FireWaterSystemList`.

**`designer/scene/graphics_scene.py::GraphicsScene`** (3052 lines) — the authoring canvas; `rebuild_scene()` redraws every asset type from the current `Floor`, plus zone-coverage and camera-visibility overlays; also owns `self.sandbox_manager = SandboxManager()` for the Manual Simulation Sandbox display.

**`designer/widgets/property_panel.py::PropertyPanel`** (7872 lines, one class, never split — a known, disclosed limitation) — one `show_<asset>()` method per asset type (23 found).

**`designer/validation.py::validate_building_authoring(building)`** — Designer-specific completeness pass (stricter than `navigation/validation.py`): **ERROR** for Door/Exit/Stair missing a zone reference (would silently produce no graph edge) or duplicate/overlapping Stairs; **WARNING** for any of ~14 asset types (now including Camera, per this session's milestone) missing `zone_ids`, and dangling `FireWaterSystem` references.

**Serialization**: `serialization/serializer.py::Serializer.save/load` + `models/project.py::Project` — `.syn` files are plain indented JSON; `Serializer` optionally threads a `CredentialStore` so camera passwords are captured out of the file, never persisted in plaintext.

**Tests**: `test_designer_main_window.py`, `test_builder_*.py` (dock management, navigation preview, project management, property panel, scale calibration, validation, widgets), `test_designer_validation*.py`, `test_designer_zone_autoassignment.py`, `test_property_panel_*.py`, plus one `test_*_designer.py` per asset type.

**Status**: production-ready, the primary authoring surface, continuously extended.

---

## 11. Simulation

**Purpose**: three genuinely separate simulation surfaces exist, each scoped differently — this is a real architectural fact to keep straight, not a naming inconsistency.

1. **Manual Simulation Sandbox** (`sandbox/manager.py::SandboxManager`, `occupant.py::SandboxOccupant`) — a lightweight, Designer-embedded, manually-driven demo: places occupants in a Zone, routes them via the real `NavigationGraph`/`PathfindingEngine`, steps them each tick. **Confirmed (again) NOT connected to Perception/AI/Recommendation** — imports only `navigation`/`pathfinding`; occupants are never persisted. Driven by `MainWindow`'s own simulation `QTimer`, entirely independent of the Live Runtime tick.
2. **Batch/offline simulation** (`simulation_interactive/` — `RouteManager`, `action_executor.py`; `simulation_runtime/runtime.py::SimulationRuntime`; `simulator/` — `MultiAgentSimulation`, `OccupantState`, `congestion.py`, `capacity.py`, `discharge.py`) — the heavier engine behind Campaign Studio and research scripts; integrates `behavior.orchestrator.HumanBehaviorLayer`, hazard-aware/recommendation-aware routing, and a genuine `ai_decision.engine.DecisionEngine` (§14).
3. **Fire/smoke/tenability**: `hazard/` (`HazardProvider`, `HazardAwareCostModel`), `hazard_evolution/engine.py::HazardEvolutionEngine` (merges multiple `HazardSource`s over time), `fire_growth/model.py::FireGrowthModel`, `smoke_propagation/model.py::SmokePropagationModel`, `tenability/model.py::TenabilityModel` (survivability/visibility scoring).

**Replay/playback**: `scenario_runner/runner.py` builds a `SimulationContext` from a `ScenarioDefinition`; `scenario_event_executor/` applies scripted events. Command Center's "Load Incident" Replay (`command_center/incident_data.py`) independently imports `scenario_generator`, `scenario_pipeline`, `scenario_runner`, `ai_decision.engine.AIDecisionEngine`, `simulation_runtime.SimulationRuntime`, `decision_policy`, `advisory_system` — entirely separate from `live_runtime/factory.py`.

**Designer's Campaign Studio** (`designer/campaign/campaign_window.py`, `campaign_worker.py`) — batch-runs many scenarios end to end, producing datasets/ground-truth for research (`dataset_builder`, `ground_truth`).

**Tests**: `test_sandbox.py`, `test_simulation_interactive.py`, `test_fire_growth.py`, `test_hazard_evolution.py`, `test_hazard_layer.py`, `test_smoke_propagation.py`, `test_tenability.py`, `test_scenario_runner.py`, `test_scenario_event_executor.py`, `test_simulation_replay_studio_e2e.py`, `test_campaign_studio.py`, `test_campaign_pipeline_integration.py`.

**Status**: batch/offline simulation + fire/smoke/replay/Campaign Studio = production-ready, actively used for research/offline workflows. Manual Simulation Sandbox = production-ready but **deliberately disconnected** from the live AI/Perception/Recommendation chain (a real architectural boundary, not a bug).

---

## 12. Recommendation Layer

**Purpose**: the single, canonical, unified interface for evacuation recommendations — orchestrates (never recomputes) `evacuation_recommendation`, `evacuation_guidance`, `emergency_response`, `crowd_intelligence`, optionally `advisory_system`.

**Main classes**: `recommendation_layer/layer.py::RecommendationLayer` (`compute(time, **snapshots) -> RecommendationSet`), `manager.py::RecommendationManager` (dedup + lifecycle: stable ids, 5s grace-period expiry). Six categories, each with a thin adapter in `recommendation_layer/adapters/`: `OCCUPANT_ROUTING, HAZARD_AVOIDANCE, CONGESTION_MITIGATION, EXIT_UTILIZATION` (the one self-derived category), `WARDEN_DISPATCH, SYSTEM_WARNING`.

**Underlying engines** (both **FROZEN**, confirmed against `docs/architecture/core_architecture_freeze_review.md`): `evacuation_recommendation/engine.py::EvacuationRecommendationEngine` (deterministic, explainable per-zone exit ranking; provably zero AI-driven rank changes — AI support is the smallest weight, applied identically to every candidate) and `evacuation_guidance/engine.py::EvacuationGuidanceEngine` ("how to reach it" route/message planning).

**Runtime ownership**: `live_system/orchestrator.py::LiveOrchestrator.run_cycle()` stage 11-12 (recommendation), stage 15 (recommendation layer); ticked every second via §4.

**Tests**: `test_recommendation_layer.py`, `test_recommendation_layer_architecture_guards.py`, six `test_recommendation_layer_<category>_adapter.py` files, `test_evacuation_recommendation.py`, `test_evacuation_recommendation_architecture_guards.py`.

**Status**: FROZEN, production-ready, actively used every cycle.

---

## 13. Execution Layer

**Purpose**: coordinates (never dispatches) the four execution controllers (Voice, Building Control, Dynamic Signage, Warden Notification) into one unified, auditable `ExecutionSet`.

**Main classes**: `execution_layer/layer.py::ExecutionLayer` (`compute(time) -> ExecutionSet`), `models.py` (`ExecutionCategory, ExecutionStatus, RecommendationIdProvenance, ExecutionRequest`), four adapters in `execution_layer/adapters/` (`voice_adapter.py, signage_adapter.py, building_control_adapter.py, warden_adapter.py`). Underlying controllers: `voice_evacuation.controller.VoiceEvacuationController`, `building_control.controller.BuildingControlController`, `dynamic_signage.controller.DynamicSignageController`, `warden_notification.controller.WardenNotificationController` — each `submit() → PENDING_APPROVAL → approve()/reject() → dispatch() → CONFIRMED/FAILED`, real human-approval gate, never a rubber stamp.

**Runtime ownership**: `LiveRuntime.tick_execution_layer(time)` — deliberately a **separate method from `run_cycle()`**, because `LiveOrchestrator` is mechanically forbidden from importing execution controllers at all.

**Capabilities**: full audit trail (created/approved/dispatched/completed timestamps); `recommendation_id_provenance` honestly discloses which id-space (`recommendation_layer`/`advisory_system`/`unavailable`) an execution request's traceability came from — only Warden Notification has real, complete traceability today.

**Limitations**: no real hardware providers exist for any of the four controllers (Simulation-only everywhere, `LIVE` mode = `NO_PROVIDER`); Voice/BuildingControl/Signage traceability is incomplete by disclosed design.

**Tests**: `test_execution_layer.py`, `test_execution_layer_architecture_guards.py`, three `test_execution_layer_<controller>_adapter.py` files, `test_warden_notification_architecture_guards.py`, `test_warden_notification_controller.py`, `test_voice_evacuation.py`, `test_building_control.py`, ten `test_dynamic_signage_*.py` files.

**Status**: FROZEN, production-ready, actively used every time an operator approves an action in Command Center.

---

## 14. Decision Engine

**Purpose**: two genuinely distinct "decision engines" exist under this name — keeping them straight matters for future work.

1. **`decision_policy/`** — an older, fully deterministic, rule-based layer (`policy.py::DecisionPolicy`, `announcement_policy.py, exit_policy.py, human_priority_policy.py, rescue_policy.py, stair_policy.py, zone_policy.py`). **Actively used**, but only by Replay/Campaign/offline-analysis paths (`advisory_system/advisory_engine.py`, `command_center/incident_data.py`, `designer/campaign/campaign_worker.py`, `voice_evacuation/adapter.py`). **Confirmed not used by the live chain**: `live_runtime/factory.py` never imports it. The narrower `decision_policy_gateway`/`recommendation_builder` **orchestrator seam** in `live_system/integration.py` is genuinely dead — both parameters default to `None` and are never populated in production; listed in `PROJECT_STATE.md` for future removal.
2. **`ai_decision/engine.py::DecisionEngine`** (ABC) / `AIDecisionEngine` (concrete) — despite the name, this is **deterministic**, hazard-aware pathfinding/priority logic (`HazardAwareCostModel`, `BuildingAnalysisEngine.critical_connectors()`, `SeverityOccupancyPriorityRule`), not a trained ML model. It is `SimulationRuntime`'s and Campaign/Replay's decision engine — unrelated to `ai_registry`'s trained models (§15).

**Status**: `decision_policy/` = production-ready, actively used (Replay/offline only). `ai_decision/` = production-ready, actively used (offline simulation only). Neither is part of the live production recommendation chain — that's `evacuation_recommendation`/`recommendation_layer` (§12), by deliberate design.

---

## 15. AI / Reasoning Modules

**Purpose**: the predictive-AI research and production-registry pipeline (distinct from the deterministic engines in §12/§14).

**Main packages**: `ai_registry/` (`registry.py::ModelRegistry`, `inference_service.py::LiveAIInferenceService`, `baselines.py, campaign.py, training.py, uncertainty.py` — the real model-lifecycle registry); `ai_inference/` (`predictor.py, ensemble.py, confidence.py, cache.py, loader.py, recommendation.py` — inference-time wrappers); `ai_training/` (`dataset.py, preprocessing.py, split.py, experiment.py, evaluation.py, metrics.py`); `ai_features/` (`building_state_extractor.py, simulation_extractor.py, feature_schema.py` — converts `BuildingState`/simulation output into model-ready features); `ai_explainability/` (`feature_importance.py, permutation_importance.py, benchmark.py, comparison.py, prediction_report.py, visualization.py`); `model_benchmark/` (`algorithms.py` — factory for `random_forest, gradient_boosting, xgboost, decision_tree, linear_regression, mlp, dummy`; `search.py, robustness.py`).

**Production model selection** (spot-checked against `docs/architecture/model_benchmark.md`, still accurate): **Gradient Boosting** for `bottleneck_occurrence` (classification), **Linear Regression** for `evacuation_time` (regression) — selected dynamically by `scripts/run_model_benchmark_registration.py`, not hardcoded; no persisted registry artifact ships in the repo (only `weights/yolov8n.pt` for human detection is checked in).

**Shadow-Mode construction (confirmed this session)**: `live_runtime/factory.py::build_live_runtime()` accepts `live_ai_gateway` as an **opaque, already-constructed** object and never imports `ai_registry`/`ai_inference`/`ai_training`/`decision_policy` itself (an enforced architecture-guard boundary) — but `live_runtime_launcher/session.py` (the real caller in Designer) **never passes `live_ai_gateway`**. `live_system/live_ai_gateway.py::RegistryLiveAIInferenceGateway` is constructed only in tests and `scripts/run_model_benchmark_registration.py`.

**Prediction Evaluation** (`prediction_evaluation/`) — compares Shadow-Mode predictions against actual future state, entirely out-of-band, mechanically proven import-isolated from Recommendation/Guidance/Execution.

**Tests**: extensive — one `test_ai_*.py` per package, `test_model_benchmark*.py`, `test_shadow_mode_prediction.py`, `test_live_ai_runtime_integration.py`, `test_prediction_evaluation*.py`.

**Status**: production-quality research/registry pipeline, **dormant in the shipped Designer app** — real, tested, one parameter away from activation, same shape of gap as §6/§7's Human Detection chain. Even if activated, its only coupling into a live decision is `evacuation_recommendation`'s provably rank-invariant additive AI term (§12) — AI can shift a displayed number, never change which exit is recommended.

---

## 16. Cross-Camera Identity

**Purpose**: maintain one persistent global occupant identity as a person moves between cameras, using camera topology + time continuity + track age + (optional) behavior evidence — deliberately no deep-learning ReID/facial recognition/appearance embeddings (mechanically enforced by architecture-guard tests).

**Main classes** (`cross_camera_identity/`): `identity_registry.py::IdentityRegistry` (pure storage: `create/touch/release/unbound_records/delete`, sequential `"OCC-N"` ids, never reused); `topology.py::CameraTopology`/`CameraTransition`/`build_topology_from_navigation_graph` (§3 above); `matching.py::CrossCameraMatcher`(ABC)/`RuleBasedCrossCameraMatcher`; `resolver.py::CrossCameraIdentityResolver`(ABC)/`RuleBasedCrossCameraIdentityResolver`; `transition_model.py::TransitionModel` (identity timeout policy + `pending_transition_for()`).

**Pipeline position**: `CameraFrame → YOLOHumanDetector → SimpleSingleCameraTracker → TrackedHuman → RuleBasedBehaviorRecognizer → BehaviorObservation → RuleBasedCrossCameraIdentityResolver → ResolvedIdentity → [identity-stabilized] RawHumanDetection`.

**Confirmed this session (again, independently)**: `RuleBasedCrossCameraIdentityResolver` is constructed only in `scripts/benchmark_cross_camera_identity.py`, `scripts/demo_cross_camera_identity.py`, `scripts/demo_live_occupants.py`, and tests — **zero production call sites**. `live_runtime/factory.py::build_live_runtime()` accepts `cross_camera_identity_resolver` as an optional parameter and threads it through, but `live_runtime_launcher/session.py::construct()` never supplies one — the real single-camera Live path uses `SimulationIdentityResolver` (local_track_id IS the global id) with zero cross-camera reconciliation.

**Tests**: `test_cross_camera_identity.py`, `test_cross_camera_identity_architecture_guards.py`, `test_live_camera_pipeline_cross_camera_integration.py`, `test_live_camera_pipeline_occupant_integration.py`.

**Status**: fully engineered, extensively tested, documented (`docs/architecture/cross_camera_identity.md`) — **dormant**, waiting on (a) more than one real Live camera being configured at once and (b) someone wiring `cross_camera_identity_resolver`/a real `CameraTopology` into `LiveRuntimeSession`. The Camera → Zone Assignment milestone (this development arc) specifically closed one prerequisite gap: `Camera.zone_ids` is now genuinely multi-assignable, which `build_topology_from_navigation_graph()` needs to produce a non-trivial topology.

---

## 17. Calibration

**Purpose**: converts one detection's pixel bounding box into a floor-plan world coordinate — a pinhole-camera model kept deliberately separate from the Digital Twin's `Camera` asset itself.

**Main classes** (`camera_calibration/`): `camera_model.py::CameraIntrinsics/CameraExtrinsics/CalibrationProfile/CalibrationQuality` (`quality=None` means "never validated" — an honest three/four-way status: `NOT CONFIGURED / CONFIGURED -- UNVALIDATED / VALIDATED -- RMSE: …`); `calibration.py::CalibrationRegistry` (plain `camera_id → CalibrationProfile` map, owned by the Designer session); `calibration_loader.py` (manual + JSON entry paths — OpenCV auto-calibration explicitly out of scope); `calibration_solver.py::solve_calibration_from_correspondences()` (fits yaw/pitch via `scipy.optimize.minimize` against ≥3 measured correspondences; position/height always given, never fitted); `validation.py::validate_calibration()`; `geometry.py` (pure 3D vector math, deliberately never duplicates 2D polygon logic — see §18); `projection.py::WorldProjector` (the production pipeline class: bbox → ground-contact pixel → ray → floor intersection → Zone lookup → observable-asset lookup → confidence); `asset_lookup.py`/`stair_lookup.py` (generic + Stair-specific spatial-asset lookup).

**Workflow end-to-end**: Designer property panel (or `scripts/calibrate_camera_scene.py`) → `CalibrationProfile` → `CalibrationRegistry` → `live_camera_pipeline.pipeline.Pipeline`'s `WorldProjector` → per-detection Zone/asset/provenance resolution → `RawHumanDetection`/`LiveOccupant`/Command Center.

**Confirmed this session**: no committed calibration JSON with real, physically-measured data exists anywhere in the repo (searched for real `focal_length_x`/`yaw_degrees` values) — every calibration is either synthetic/test-fixture or explicitly "configured but unvalidated" by construction. The full architecture for a genuine site-measured calibration exists (`calibration_solver.py`, `validate_calibration()`) but no such dataset ships — matches `docs/architecture/cctv_connection_and_calibration_readiness.md`'s own "readiness, not done" framing.

**Newly flagged this session (not in `PROJECT_STATE.md` at all)**: `automatic_calibration/`, `calibration_studio/`, `calibration_benchmark/` — three substantial packages (grid-search engine, a benchmark-publishing "studio" with dashboard/session/report, and a harness comparing the *simulator's* physics parameters — walking speed, herding, congestion — against published evacuation-research values) with 17+ dedicated test files. This is a **different sense of "calibration"** (simulator physics, not camera geometry) and appears to be a real, substantial, tested body of work with zero mention in `PROJECT_STATE.md`'s "Completed Subsystems" or "Future Roadmap" sections. Flagged in §20 as a documentation gap, not a code defect.

**Tests**: `test_camera_calibration.py`, `test_camera_calibration_architecture_guards.py`, `test_camera_calibration_failure_modes.py`, `test_camera_calibration_validation.py`, `test_world_position_provenance.py`, `test_live_camera_pipeline_calibration_integration.py`, plus (newly flagged) `test_automatic_calibration_*.py` (7 files), `test_calibration_benchmark_*.py` (8 files), `test_calibration_studio_*.py` (17 files).

**Status**: Camera-geometry calibration = production-ready architecture, actively used, but running on synthetic/unvalidated data only (no real site measurement yet). Simulator-physics calibration (`automatic_calibration`/`calibration_studio`/`calibration_benchmark`) = production-ready and actively used for research, but architecturally undocumented until now.

---

## 18. Geometry Utilities

**Purpose**: the shared 2D geometric-primitive layer every spatial-reasoning subsystem reuses, deliberately kept as one canonical source.

**Primitives**: `visibility/geometry.py::point_in_polygon(), segment_intersection(), closest_point_on_segment(), perpendicular_distance_to_line()`. Confirmed reused (not reimplemented) by: `camera_calibration/projection.py` (Zone lookup), `navigation/obstacle_geometry.py::segment_blocked_by_obstacles()` (explicit in-code comment about avoiding a second implementation), `camera_coverage/discovery.py`, `camera_validation/recommendations.py`, `visibility/coverage.py`, `visibility/engine.py`, `visibility/segments.py`.

**One deliberate, documented exception**: `perception/providers/ground_truth_camera_provider.py` reimplements a private `_point_in_polygon()` rather than importing `visibility.geometry`, explicitly to keep the dependency direction one-way (Perception must never depend on Visibility) — noted in `visibility/geometry.py`'s own comment.

**Higher-level geometry**: `visibility/coverage.py::FloorCoverage` (multi-camera zone coverage/overlap fractions), `visibility/engine.py::VisibilityEngine`/`CameraVisibility` (occlusion-aware FOV-wedge raycasting against `visibility/segments.py::floor_opaque_segments()`), `camera_calibration/geometry.py` (pure 3D vector math — camera basis vectors, pixel-ray direction, ray-floor intersection — deliberately never duplicates 2D polygon logic).

**Tests**: `test_visibility_coverage.py`, `test_visibility_engine.py`.

**Status**: production-ready, actively used, correctly kept as a single canonical layer across the codebase (the geometry-duplication risk this document's own investigation was watching for was checked and found clean).

---

## 19. Visualization Utilities

**Purpose**: everything that renders the building/data visually — two entirely separate rendering paths exist (interactive Qt UI vs. offline research charting).

**Designer Qt scene**: `designer/scene/graphics_scene.py::GraphicsScene` (3052 lines, the `QGraphicsScene` hosting every asset item); `designer/items/` — one `QGraphicsItem` subclass per asset type (`AssemblyPointItem, CameraItem, DetectorItem, DoorItem, EmergencyLightItem, ExitItem, FireExtinguisherItem, FireHydrantItem, FirePumpItem, FireServiceInletItem, FireWaterTankItem, HeatDetectorItem, HoseReelItem, JockeyPumpItem, ManualCallPointItem, ObstacleItem, OccupantItem, SensorItemBase, SignItem, SmokeDetectorItem, SpeakerItem, SprinklerItem, StairItem, VertexHandle, VertexLayer, ZoneItem, ZoneRectangle`).

**Offline charting** (all headless matplotlib, `Agg` backend forced before `pyplot` import — a repeated, explicitly-documented convention): `research_framework/figures.py` (confusion matrix, PR/ROC curves, congestion heatmaps, evacuation-time distributions, RL reward curves); `validation_framework/figures.py` (imports and reuses `research_framework/figures.py`'s save-functions rather than reimplementing — confirmed in-code); `campaign_analytics/visualizations.py` (single-campaign charts, reads only `CampaignAnalysis` output, never raw artifacts) + `report.py` (assembles the narrative report); `ai_explainability/visualization.py` (same `Agg` convention).

**Status**: both paths production-ready, actively used — the Qt path for live authoring/monitoring, the matplotlib path for research/validation/campaign reporting. Entirely separate, no shared rendering code between them (correctly so, given their different runtime contexts).

---

## 20. Dormant or Partially-Implemented Systems (consolidated)

This section merges the dedicated dormant-systems scan with every dormancy noted in §1-§19, plus a cross-check against `PROJECT_STATE.md`'s own disclosed gaps (§7 there).

**Confirmed dormant (real, tested, zero/near-zero production reachability)**:
- Human Detection / Tracking / Behavior Recognition / Cross-Camera Identity (§6, §7, §16) — one missing config parameter (`human_detector_weights_path`) in `LiveRuntimeController.on_start()` gates all four.
- Camera Topology (`cross_camera_identity/topology.py`, §3/§16) — engineered, unit-tested, zero production callers.
- Shadow-Mode AI Inference (`live_ai_gateway`, §15) — `live_runtime_launcher/session.py` never wires it in.
- Advisory System (`advisory_system/`) — confirmed Replay-only; `AdvisoryOrchestrator.generate_report()`'s only non-test call site is `command_center/incident_data.py`'s Load-Incident flow; `live_advisory_gateway` is never constructed outside tests.
- Real hardware providers for Voice/BuildingControl/Signage/Warden-Notification/FACP events/most sensors — `NO_PROVIDER` in `LIVE` mode everywhere; `modbus`/`bacnet`/`mqtt`/`opcua`/`socket`/`serial` imports mechanically forbidden by architecture-guard tests. `facp/provider.py::FACPEventProvider` is an explicit unimplemented placeholder (zero concrete subclasses).
- Fire-safety/fire-water/emergency-light Designer assets (`fire_safety_manager/`, `fire_water_manager/`, `emergency_light_manager/`) — modeled, authored, displayed, deliberately execution-inert (kept out of FACP alarm evidence and route-safety physics by design).
- `decision_policy_gateway`/`recommendation_builder` legacy orchestrator seam in `live_system/integration.py` — both parameters default to `None`, never populated in production; PROJECT_STATE.md already lists this for removal.
- Camera-to-camera geometric overlap — designed (`docs/architecture/building_camera_topology_design.md`) but not implemented; genuinely missing, not merely dormant.
- Virtual Zones / tripwires — the seam exists (`models/connectable_space.py`'s extensibility design) but no concrete implementation anywhere.
- `predictive_dataset/graph_context_v4.py` — explicitly "never wired into LiveRuntime this milestone" per its own comment.
- `perception/providers/ground_truth_camera_provider.py` — has an explicit `TODO(future version)` for per-occupant Ground Truth positions.
- Undo/Redo (`designer/widgets/toolbar.py`) and Elevator authoring (`builder/widgets/builder_toolbar.py`) — tooltips literally read "is not implemented yet."
- `rl_training/` — a complete Gymnasium + stable-baselines3 (PPO/A2C/DQN) RL scaffold, real and tested, but confined entirely to the research/validation arc (`research_framework/`, `validation_framework/`, `validation/phase3_rl_validation.py`) — zero reachability from `live_runtime/`, `live_system/`, or `command_center/`. Not broken, just scoped away from live execution.

**Newly discovered this session, not previously documented anywhere in `PROJECT_STATE.md`**:
- `automatic_calibration/`, `calibration_studio/`, `calibration_benchmark/` — three substantial, tested packages calibrating the *simulator's* physics parameters against published evacuation research (see §17). Real gap in `PROJECT_STATE.md`'s coverage, not a code defect — recommended for a future `PROJECT_STATE.md` pass to fold in properly (partially addressed by this session's own `PROJECT_STATE.md` update, see below).
- Uncommitted, in-progress Command Center multi-camera grid work (`camera_tile_widget.py`, `live_camera_grid_panel.py`, `live_camera_view_gateway.py`, plus modifications to `building_view.py`/`dashboard.py`/`main_window.py`) present in the user's live working tree but outside this worktree's committed history — not part of this inventory's verified claims, flagged for whoever resumes that work.

**Confirmed NOT dormant (verified still active despite superficially similar naming)**:
- `decision_policy/` the package itself — actively used by Replay/Campaign/offline paths (only the narrow orchestrator *seam* is dead, not the package).
- `ai_decision/engine.py::AIDecisionEngine` — actively used by `SimulationRuntime`/Campaign/Replay (it is a deterministic engine despite the "AI" name, unrelated to `ai_registry`'s trained models).

---

## 21. Dependency Graphs

### Building → Live Runtime → Command Center (structural/authoring chain)

```
Building (models/)
    ↓  (derived, never duplicated)
Navigation  (NavigationGraphGenerator.build())
    ↓  (consumed by routing/validation)
Live Runtime  (live_runtime/factory.py::build_live_runtime())
    ↓  (fused every cycle)
Building State  (building_state/estimator.py::BuildingStateEstimator)
    ↓  (read every second)
Live System Orchestrator  (live_system/orchestrator.py::LiveOrchestrator.run_cycle())
    ↓  (snapshot stored)
Command Center  (command_center/dashboard.py::Dashboard, via LiveCommandCenterDataSource)
```

### Camera → RTSP → YOLO → Tracking → Live Occupants → Digital Twin (perception chain — mostly DORMANT past step 2)

```
Camera (models/camera.py, Digital Twin asset -- Zone-assigned per Milestone 1)
    ↓  (mode=LIVE + rtsp_address)
RTSP  (live_camera_pipeline/rtsp_frame_source.py::RTSPFrameSource)                    [ACTIVE — real hardware capable]
    ↓  (decoded via OpenCVFrameDecoderBackend)
YOLO  (human_detection/yolo_human_detector.py::YOLOHumanDetector)                      [DORMANT — no weights path wired in Designer UI]
    ↓
Tracking  (tracking/simple_tracker.py::SimpleSingleCameraTracker)                       [DORMANT — same gate]
    ↓
Cross-Camera Identity  (cross_camera_identity/resolver.py)                              [DORMANT — never constructed in production]
    ↓
Live Occupants  (live_occupants/manager.py::LiveOccupantManager)                        [ACTIVE plumbing, idle input in production]
    ↓
Building State → Command Center / "Digital Twin" (= the Building model itself, live-annotated)
```

### Recommendation & Execution chain (fully active)

```
Building State + Crowd/Progress/Trajectory/Emergency-Response Intelligence
    ↓
Evacuation Recommendation (FROZEN)  →  Evacuation Guidance (FROZEN)
    ↓
Recommendation Layer  (unifies 6 categories, dedup + lifecycle)
    ↓  (operator approves in Command Center)
Execution Layer  (coordinates, never dispatches)
    ↓
Voice / Building Control / Dynamic Signage / Warden Notification controllers
    ↓
Simulation-only providers  (NO real hardware anywhere — by design)
```

### Calibration chain (real math, synthetic data)

```
Camera (Designer-authored position/rotation/FOV)
    ↓
CalibrationProfile  (camera_calibration/camera_model.py — manual or correspondence-solved)
    ↓
CalibrationRegistry  (owned by Designer session)
    ↓
WorldProjector  (camera_calibration/projection.py — per-detection ray-cast to floor)
    ↓
Zone / Observable-Asset resolution  →  RawHumanDetection.world_position (+ provenance flag)
```

---

## 22. Capability Matrix

Legend: ✓ = exists & used in production · ⚠ = exists but dormant / partially implemented · ✗ = missing entirely.

| Capability | Status | Notes |
|---|---|---|
| **Building authoring (multi-floor, ~25 asset types)** | ✓ | Designer + Builder, production-ready |
| **Explicit (never geometry-inferred) connectivity** | ✓ | Door/Exit/Stair zone references |
| **Virtual Zones** | ✗ | Extensibility seam exists (`connectable_space.py`), no implementation |
| **Navigation graph derivation** | ✓ | `NavigationGraphGenerator`, pure/stateless, FROZEN in spirit |
| **Hazard-aware pathfinding cost** | ⚠ | `navigation/cost.py` seam exists, unused by default; `ai_decision`'s `HazardAwareCostModel` used only offline |
| **Camera asset modeling** | ✓ | Position/rotation/FOV/range/coverage_polygon |
| **Camera → Zone assignment (multi-zone)** | ✓ | Shipped this development arc (Milestone 1) |
| **Camera ↔ Observable-Asset coverage** | ✓ | `camera_coverage/`, wired into `BuildingState` |
| **Camera ↔ Camera adjacency/topology** | ⚠ | `cross_camera_identity/topology.py`, fully built, zero production callers |
| **Camera ↔ Camera geometric overlap** | ✗ | Designed, not implemented |
| **RTSP live video ingestion** | ✓ | Real hardware-capable, shown in Designer's `LiveCameraViewPanel` |
| **Multi-camera grid / Command Center live video** | ⚠ | In-progress, uncommitted (outside this worktree) |
| **YOLO human detection** | ⚠ | Real, validated, but no weights path wired into shipped Designer UI |
| **Single-camera tracking** | ⚠ | Same gate as YOLO |
| **Cross-camera identity resolution (ReID)** | ⚠ | Fully engineered, zero production callers |
| **Behavior recognition (rule-based)** | ⚠ | Same gate as YOLO/Tracking |
| **Live Occupants registry** | ✓ (plumbing) / ⚠ (real input) | Always constructed; idle without §6 |
| **Building State fusion** | ✓ | Every cycle, offline-capable |
| **Live Runtime composition (OFFLINE_DEMO)** | ✓ | Fully self-contained, no hardware needed |
| **Live Runtime composition (LIVE)** | ✓ | Real-hardware-capable when configured |
| **Command Center (Live + Replay)** | ✓ | Full dual-mode operator UI |
| **Crowd / Progress / Trajectory / Emergency-Response Intelligence** | ✓ | Four FROZEN sibling engines |
| **Manual Simulation Sandbox** | ✓ | Production-ready, deliberately disconnected from AI/Recommendation |
| **Batch/offline simulation (Campaign Studio, research)** | ✓ | Production-ready, active research/dataset use |
| **Fire/smoke/tenability modeling** | ✓ | Production-ready, offline |
| **Evacuation Recommendation** | ✓ | FROZEN, deterministic, zero AI-driven rank changes |
| **Evacuation Guidance** | ✓ | FROZEN |
| **Recommendation Layer (unified)** | ✓ | 6 categories, dedup + lifecycle |
| **Execution Layer (unified)** | ✓ | 4 controllers, full audit trail |
| **Real hardware execution (Voice/Control/Signage/Warden)** | ✗ | Simulation-only everywhere, mechanically enforced |
| **Advisory System (multi-audience reports)** | ⚠ | Real, Replay-only reachability |
| **Decision Policy (rule-based, legacy)** | ✓ (Replay/offline only) | Not part of the live chain by design |
| **AI Decision Engine (deterministic)** | ✓ (offline only) | Distinct from `ai_registry` ML models |
| **Predictive AI model registry** | ✓ (registry itself) / ⚠ (live wiring) | Gradient Boosting + Linear Regression selected; Shadow-Mode gateway never constructed in Designer |
| **Prediction Evaluation (offline)** | ✓ | Import-isolated from the live decision chain |
| **Camera Calibration (math/architecture)** | ✓ | Full pinhole model, correspondence solver, validation framework |
| **Camera Calibration (real measured data)** | ✗ | No physically-measured calibration ships in the repo |
| **Simulator-physics calibration (`automatic_calibration`/`calibration_studio`/`calibration_benchmark`)** | ✓ | Real, tested, previously undocumented in PROJECT_STATE.md |
| **Geometry utilities (single canonical layer)** | ✓ | One deliberate, documented exception (Perception/Visibility one-way dependency) |
| **Visualization (Designer Qt scene)** | ✓ | One `QGraphicsItem` per asset type |
| **Visualization (offline research charting)** | ✓ | Headless matplotlib, shared save-functions |
| **Fire-safety/fire-water Designer assets** | ✓ (modeled) / ✗ (execution) | Deliberately execution-inert |
| **Reinforcement Learning scaffold** | ✓ (research only) | Not reachable from live/production code |

---

## 23. What Future Milestones Naturally Build Upon

- **Closing the perception dormancy chain** (§6/§7/§16): a single config change (`human_detector_weights_path` wired into `LiveRuntimeController`) activates YOLO → Tracking → Behavior → (optionally) Cross-Camera Identity → Live Occupants all at once — the plumbing is already correct end to end.
- **Camera Topology activation** (§3/§16): now that Camera → Zone Assignment is genuinely multi-zone, `build_topology_from_navigation_graph()` can produce a real `CameraTopology` for any project with cameras assigned — the remaining work is wiring `cross_camera_identity_resolver` into `LiveRuntimeSession`, not building anything new.
- **Camera-to-camera overlap** (§3): the one genuinely missing camera-reasoning concept; `docs/architecture/building_camera_topology_design.md` already proposes the smallest seam (`building_topology/overlap.py`, reusing existing `visibility.geometry` primitives).
- **Shadow-Mode AI activation** (§15): same shape of gap as perception — one wiring change in `live_runtime_launcher/session.py` would surface Gradient-Boosting/Linear-Regression predictions in Command Center's `LiveAIPanel`, which already exists and already expects `ai_prediction_snapshot`.
- **Virtual Zones / tripwires**: `models/connectable_space.py` was explicitly designed as the seam — a third `CONNECTABLE_SPACE_TYPES` entry, without touching Door, the Property Panel, or the Navigation Graph builder.
- **Real hardware execution providers**: every one of Voice/BuildingControl/Signage/Warden already has a clean `Provider` ABC seam (`SimulationXProvider` is just today's only concrete implementation) — a real provider is an additive class, not a redesign.
- **Multi-camera Command Center grid**: already in progress (uncommitted, this session found evidence of it) — a natural extension once merged.
- **PROJECT_STATE.md documentation debt**: `automatic_calibration/`, `calibration_studio/`, `calibration_benchmark/` need folding into the main architectural summary properly (this document's own update is a starting point, not a full fix).

---

*This document was produced by investigation only — no source files were modified to create it, other than this document itself and the `PROJECT_STATE.md` cross-reference described in that file's own changelog.*
