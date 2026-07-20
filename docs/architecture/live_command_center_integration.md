# Live Command Center Integration

Status: **Command Center is now a dual-mode operational interface. The same `Dashboard`/`MainWindow` instance renders either a completed-run Replay session (unchanged from before this milestone) or a real-time view of whatever `LiveOrchestrator`/`StateManager` currently hold, through one clean data-source abstraction. Command Center remains a presentation layer throughout: it never runs perception, fusion, AI inference, or advisory generation itself, and never executes a voice broadcast or a building control action.**

## 1. Replay vs. Live modes

`command_center.data_source.CommandCenterMode` — `REPLAY` (the default, this widget's own pre-existing behavior, entirely unchanged) and `LIVE`. `Dashboard.mode` tracks the current mode; `Dashboard.set_mode(mode)` toggles which side-tabs are visible (the four Replay-only tabs — Incident, Occupancy, Hazard, Decision Policy (Raw) — hide in Live mode rather than being fed fabricated `IncidentData`; the three new Live-only tabs — Live Status, Live AI, Live Events — hide in Replay mode), disables the timeline-scrubbing controls in Live mode (§5), and relabels `IncidentStatusBar`'s Mode tile.

`MainWindow` exposes the toggle as a `View → Mode → Replay/Live` checkable action pair (mirroring the pre-existing Overlay submenu's own convention). Replay is available immediately (an incident can be loaded via `File → Load Incident...` exactly as before); Live only becomes selectable once a caller has handed `MainWindow` a `LiveCommandCenterDataSource` via `enable_live_mode(data_source)` — attempting to select Live before that shows an honest message rather than switching to a mode with nothing to show.

Switching modes is never destructive: `switch_to_replay_mode()` only pauses the live-refresh timer and calls `data_source.stop()`; it never discards the loaded `IncidentData` or the live data source. Re-entering Live mode resumes the same `LiveCommandCenterDataSource`, and re-entering Replay mode resumes exactly where the operator left the timeline.

## 2. Data-source abstraction

`command_center.data_source.CommandCenterDataSource` — a `Protocol` with `mode`, `current_snapshot() -> CommandCenterSnapshot`, `start()`, `stop()`. Two implementations:

- `ReplayCommandCenterDataSource` (`command_center/data_source.py`) — a thin, additive wrapper over the existing, completely untouched `IncidentData`/`IncidentFrame` API (`frame_at_index`, `advisory_report_at_index`, `frame_count`, `building`, `decision_policy`). `start()`/`stop()` are no-ops (an `IncidentData` is always already fully resolved). Not load-bearing for `Dashboard`'s existing Replay rendering path — `Dashboard.set_incident()`/`show_frame()`/`set_frame_index()` are unmodified by this milestone and remain how `MainWindow`'s own playback `QTimer` drives Replay — this class exists as the symmetric, testable `CommandCenterDataSource` implementation Phase 3 asks for.
- `LiveCommandCenterDataSource` (`live_system/live_command_center_gateway.py`) — the real Live adapter. Lives in `live_system`, not `command_center`, because `live_system.integration` already imports `command_center` (the pre-existing `DashboardCommandCenterGateway`); `command_center` must therefore never import `live_system`, or that would be circular. This mirrors exactly how `advisory_system.ai_evidence.AIDecisionEvidence` is defined in `advisory_system` while its live *source* adapter lives in `live_system.live_advisory_gateway`.

Command Center's own live-facing modules (`command_center/live_status_panel.py`) defer their one `live_system` import to inside a method body for the same reason, so `command_center`'s own module-level import graph never has to resolve `live_system` at package-load time.

## 3. `CommandCenterSnapshot`

`command_center.data_source.CommandCenterSnapshot` — the one presentation type both modes produce. Deliberately not a new state model: `building_state`, `advisory_report`, and `ai_prediction_snapshot` are carried as direct references to the already-computed canonical objects (never copied, never re-derived), and `frame` reuses `IncidentFrame` — the one per-tick shape every existing Command Center panel already renders — rather than inventing a second one. For Replay, `frame` is exactly `IncidentData.frame_at_index()`'s own return value; for Live, it is adapted from `BuildingState` by `frame_from_building_state()` (§6). `replay_incident`/`replay_frame_index`/`replay_frame_count` are Replay-only navigation context; `building_state`/`ai_prediction_snapshot`/`recent_events` are Live-only. `consistency` and the three per-component timestamps are Phase 14's own honesty mechanism (§5).

## 4. `StateManager` integration

`LiveCommandCenterDataSource` reads exactly three `StateManager` accessors already established by prior milestones — `current().building_state`, `.ai_prediction_snapshot`, `.advisory_report`, plus `.component_timestamps` — and nothing else. It never instantiates `BuildingStateEstimator`, `LiveAIInferenceService`, or `AdvisoryOrchestrator`; it never calls `CameraManager`/`SensorManager`/`MultiCameraFusionEngine`/`SimulatedFACP` directly. An optional `EventBus` reference (the same public `orchestrator.event_bus` attribute prior milestones already exposed) powers the bounded recent-events list (§6) — without one, `recent_events` stays honestly empty. `start()`/`stop()` are this class's own lifecycle: while stopped, `current_snapshot()` unconditionally reports `SnapshotConsistency.UNAVAILABLE`, regardless of what `StateManager` holds.

## 5. Snapshot consistency

`SnapshotConsistency` — `CURRENT`, `STALE`, `PARTIAL`, `UNAVAILABLE`. `LiveCommandCenterDataSource._resolve_consistency()` compares `component_timestamps["building_state"]`/`["ai_prediction_snapshot"]`/`["advisory_report"]`: no `building_state` timestamp at all is `UNAVAILABLE`; a missing AI/Advisory timestamp (never populated this run — e.g. no `live_ai_gateway` configured) is `PARTIAL`; three timestamps that disagree (the brief's own "BuildingState from cycle 12, AI from cycle 11, Advisory from cycle 10" example) is `STALE`; three matching timestamps is `CURRENT`.

`Dashboard.apply_snapshot()` acts on this: a `STALE` cycle never renders its carried-over `AdvisoryReport` as though it were current — every `AdvisoryReport`-driven panel is handed `None` instead (degrading to its own pre-existing empty-state rendering), and `Dashboard.live_consistency_banner` (visible only in Live mode) shows the exact reason — for `STALE`, literally "Advisory unavailable for current state -- BuildingState/AI prediction/Advisory Report timestamps do not match this cycle; the previous Advisory Report is withheld rather than shown as current." This is the mechanism, not merely the wording, the brief's Phase 14 asks for.

## 6. Live `BuildingState` rendering

`live_system.live_command_center_gateway.frame_from_building_state(building_state) -> Optional[IncidentFrame]` is the one adapter. Every field it can honestly source from `BuildingState` is populated: `zone_occupancy` from `BuildingState.zone_occupancy.observations` (aggregate per-zone counts — Phase 7's own explicit "Estimated Occupants: N, never invented coordinates" instruction, satisfied structurally since `BuildingState.occupant_tracks` carries no per-person classification this adapter could honestly turn into anything richer); `current_fire_zone_ids` from zones whose `hazard_summary.zone_severities` entry is `HIGH`/`CRITICAL`; `door_states`/`exit_states` from `BuildingState.control_status.entries` (CONFIRMED control state only); `detector_states`/`camera_states` from each asset's own `status.active` flag, in the same `DeviceAvailability` vocabulary `_baseline_frame()`/`_frame_from_row()` already use for Replay. Every field with no live source — `zone_smoke`/`zone_temperature`/`zone_visibility` (no live per-zone sensor path exists), `stair_states` (no live door/exit-shaped state for stairs), `ignition_zone_id`/`current_hazard_score` (no live single-scalar or "ignition" concept), `people_evacuated`/`people_trapped`, `human_observations` — stays at its own honest empty/`None` default, never guessed. Because `frame_from_building_state()` produces the same `IncidentFrame` type Replay already uses, `BuildingView.show_frame(frame)` needed **zero changes** for Live mode.

Richer per-asset condition (ALARM/FAULT/NORMAL, not just AVAILABLE/FAILED) surfaces separately, in the new Live Status panel (§8), via `detector_condition(asset)` — which reuses `facp.models.DetectorConditionReport.from_status_and_reading()`'s own already-established "health fault outranks alarm, alarm outranks clear" priority rule rather than re-deriving one.

## 7. AI rendering

`command_center.live_ai_panel.LiveAIPanel` — two visually and structurally separate group boxes. "Operational AI -- Bottleneck Occurrence" shows the `PRODUCTION_CANDIDATE` probability/model id/version/timestamp, with an honest "AI prediction unavailable" (never a fabricated 0%) when no snapshot or no `bottleneck` exists, and a `STALE` label supplied explicitly by the caller (`Dashboard.apply_snapshot()`, from the same `SnapshotConsistency.STALE` check as §5 — this widget never infers staleness itself). "Experimental AI -- Evacuation-Time Estimate" is always labeled `EXPERIMENTAL`, never presented with equal authority. `IncidentStatusBar`'s own new "AI Bottleneck Probability" tile is a separate, additive field, kept structurally distinct from "Occupancy Confidence"/"Recommendation Confidence" (Phase 10's own confidence-separation requirement, restated at the UI layer). The status bar's pre-existing "Predicted RSET" tile is deliberately blanked (`"-"`) whenever `live=True` — that value derives from `GroundTruth.total_evacuation_time`, a caller-supplied, non-live artifact (§ Advisory rendering below), and showing it unlabeled in Live mode would overclaim exactly what Phase 6 forbids.

## 8. Advisory rendering

`RecommendationCenter.show_live(report)` fans the real, unmodified `advisory_system.AdvisoryReport` out to its six sub-panels. Four are reused with **zero changes**: `CivilianAnnouncementsPanel`/`FirefighterIntelligencePanel`/`BuildingRecommendationsPanel`/`CommanderSummaryPanel` already read only `(frame, report)`, never `self._incident`, so their existing `show_frame(None, report)` call is correct for Live as-is. The remaining two — `VoiceEvacuationPanel`/`BuildingControlsPanel` — reach into `self._incident` in their Replay path (§9/§10), so each gained an additive `show_live(report)` method instead. Explainability is unchanged from the prior milestone: `confidence_source` still distinguishes `RULE_BASED` (empty tuple) from `AI_SUPPORTED` (`("ai",)`), and no panel ever renders a localized claim like "AI predicts North Stair" — `AIDecisionEvidence` itself carries no such field (see `docs/architecture/ai_augmented_advisory_integration.md` §2).

Also: `command_center.live_status_panel.LiveStatusPanel` (Cameras/Sensors/FACP tables, display-only, straight off `BuildingState.camera_observations`/`smoke_detector_states`/`heat_detector_states`/`facp_status`) and `command_center.live_events_panel.LiveEventsPanel` (Phase 15's bounded, most-recent-first list of already-formatted event strings from `LiveCommandCenterDataSource._recent_events()`, honestly empty when no `EventBus` was supplied) round out the three new Live-only tabs.

## 9. Voice Evacuation non-execution boundary

`VoiceEvacuationPanel.show_live(report)` renders `report.civilian_announcements` directly into the existing active-messages table, relabeled (`set_live_mode()`, a one-time header switch) "Recommended Message"/"Message Status"/"Broadcast Status" — every row's Broadcast Status column reads exactly `"Broadcast Status: NOT SENT"`. The history table stays empty. Nothing in this path — or anywhere in `command_center/`'s live-facing files — imports `voice_evacuation.controller.VoiceEvacuationController`, `voice_evacuation.provider`, or `speaker_manager` (mechanically enforced, §11). This is not a placeholder for a future confirmation the code could produce; it is the only honest value until a real output provider is wired in a later, separately-scoped milestone.

## 10. Building Control non-execution boundary

`BuildingControlsPanel.show_live(report)` renders `report.building_recommendations` directly into the existing pending-requests table as read-only rows — no Approve/Reject buttons (a disabled button would still imply an execution path exists; omitting the affordance entirely is the honest choice), Decision column reading `"Execution Provider: Not Connected"`. The active/history tables stay empty — nothing is submitted to, or read from, a `BuildingControlController`. Nothing in this path imports `building_control.controller`/`building_control.providers` (mechanically enforced, §11).

## 11. Failure behavior

`LiveCommandCenterDataSource.current_snapshot()` never raises. Every gap degrades to an honest, disclosed value, never a fabricated one:

| Scenario | `consistency` | UI behavior |
|---|---|---|
| Data source never started | `UNAVAILABLE` | banner: "No live BuildingState available yet" |
| `LiveOrchestrator` started but no cycle run yet | `UNAVAILABLE` | same |
| No `building_state_gateway` configured | `UNAVAILABLE` every cycle | camera/sensor/detector tables stay empty, never a fabricated "unknown" row |
| No `live_ai_gateway` configured | `PARTIAL` | Live AI panel: "AI prediction unavailable"; status bar AI tile: `-` |
| AI/Advisory timestamp mismatched vs. `building_state` | `STALE` | previous `AdvisoryReport` withheld (§5); AI panel labeled `STALE` |
| No `live_advisory_gateway` configured, or its `decision_policy_provider` returns `None`/raises | `PARTIAL`/previous report kept under its own honest timestamp | Recommendation Center panels degrade to empty/`-` |
| Camera/detector inactive | reflected via `status.active=False` → `"FAILED"` | never rendered as `"AVAILABLE"` |
| `LiveOrchestrator`/data source stopped mid-session | `UNAVAILABLE` | `Dashboard.apply_snapshot()` never raises on a stopped/never-started source |
| Mode switched Live→Replay→Live | unaffected | `switch_to_replay_mode()`/`enable_live_mode()` never discard state; each resumes cleanly |

`tests/test_live_command_center.py::LiveCommandCenterFailureTests` proves all fifteen of Phase 17's named scenarios directly.

## 12. Offline testing

`tests/test_live_command_center.py::EndToEndOfflineLiveCommandCenterTests` drives the complete chain — `ReplayFrameSource → MockHumanDetector → MappingIdentityResolver → CameraManager → MultiCameraFusionEngine → SimulatedFACP → BuildingState (via EstimatorBuildingStateGateway) → a real trained BottleneckOccurrenceModel_LiveCompatible (via ai_registry, one small campaign trained once per test module) → AIDecisionEvidence → a real ReplayCompatibleAdvisoryGateway → AdvisoryReport → StateManager → LiveCommandCenterDataSource → a real Dashboard/MainWindow` — and asserts all fifteen UI facts Phase 16 names: Live mode, building name, occupant count, camera status, detector status, FACP state, bottleneck probability, civilian recommendation, firefighter intelligence, commander summary, AI explainability with no localized claim, recommendation confidence, AI probability shown as a distinct value, zero voice messages sent, and zero building controls executed. Zero network access, zero real CCTV, zero physical hardware anywhere in the test.

## 13. Physical CCTV readiness

Unchanged from every prior milestone's own scope: `RTSPFrameSource`, a real `HumanDetector`, and Live ReID remain correctly out of scope and unbuilt. Every collaborator this milestone's own E2E test exercises (`ReplayFrameSource`, `MockHumanDetector`, `MappingIdentityResolver`) is already the offline-proven substitute the CCTV Pipeline milestone established; `LiveCommandCenterDataSource` itself has no dependency on any of them at all — it reads only `StateManager`, which is equally correct whether the `BuildingState` beneath it came from `ReplayFrameSource` or a future `RTSPFrameSource`. No part of this milestone required physical camera access, and no part of it is blocked by its continued absence.
