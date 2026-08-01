# Simulation Replay Studio V1

A read-only "video player" for completed SynEvac simulations. Opens a generated scenario and
replays occupant movement, hazard evolution, engineering state, and recommendation/execution
history exactly as recorded — never by re-running the simulator.

---

## 1. Existing replay infrastructure discovered

Before writing any code, the repository was searched for replay/playback/timeline/history/
visualization/scenario-viewer functionality (keywords: replay, playback, timeline, history,
simulation recording, visualization, scenario viewer, campaign viewer, command center replay,
incident replay). Finding: **a working, file-based replay engine already existed**, built for
Command Center's own operator-review use case:

- `command_center/incident_data.py` — `IncidentData`/`IncidentFrame`: loads a completed
  scenario's on-disk artifacts (`building.syn`, a stored `Scenario`, `ground_truth.json`,
  `decision_policy.json`, `timeline_rows.json`) via `load_incident()` and resolves them into one
  `IncidentFrame` per recorded tick, with `frame_at(time)`/`frame_at_index(index)` already
  supporting "scrub to any time/tick."
- `command_center/timeline_panel.py` + `command_center/main_window.py` — a real video-player
  control set: Play/Pause/Stop/speed-combo/scrub-slider/"Jump to time," driven by one `QTimer`.
- `command_center/occupancy_panel.py`, `hazard_panel.py`, `human_panel.py`,
  `recommendation_center.py`, `recommendation_timeline_panel.py` — existing Building/Occupant/
  Recommendation inspector panels, already wired to `IncidentFrame`.
- `designer/campaign/campaign_worker.py` — the producer: every scenario a Designer campaign
  generates already writes every one of the artifacts above, and its own comment confirms the
  JSON timeline shape exists specifically for Command Center playback to consume.

**What was missing** (verified, not assumed): no occupant spatial/route history was ever
persisted (`MultiAgentSimulationResult.occupants` was read once for aggregates and discarded, so
occupants never visually moved and no route could be drawn); no Occupant Inspector strategy/
profile detail; no unified Event Timeline; no per-scenario statistics charts; no "open scenario by
id" workflow; and the two newest layers (`recommendation_layer`/`execution_layer`) were live-only
adapters never wired into the campaign/export pipeline.

**Decision**: reuse Command Center's Dashboard/IncidentData/BuildingView/panel machinery as-is
(per the explicit instruction to reuse existing infrastructure) rather than building a parallel
application, and add only the one new, read-only artifact pair needed to close the occupant-motion
gap.

---

## 2. New architecture

```
Builder -> Simulation -> Scenario Generator -> Dataset Generator (campaign_worker.py /
research_framework/runner.py)
        |                                              |
        |  (unchanged: timeline.csv, ground_truth.json, decision_policy.json, ...)
        |                                              |
        +----------------------> simulation_recording/ (NEW, read-only) --> occupant_routes.json
                                                                          --> decision_events.json
                                                                                    |
                                                                                    v
                                            command_center.incident_data.IncidentData
                                            (occupant_positions: additive per-frame field,
                                             interpolated from occupant_routes.json)
                                                                                    |
                                                                                    v
                                    replay_studio/ (new launcher, reuses MainWindow/Dashboard
                                    as a library) -> Occupant Inspector / Event Timeline /
                                    Statistics panels (new) + BuildingView occupant/route
                                    rendering (new, additive)
```

- **`simulation_recording/`** (new package) — the only seam near the frozen Simulation package,
  and it is a pure reader: `occupant_routes.py` converts `MultiAgentSimulationResult.occupants`
  (already-public `OccupantTimeline`/`OccupantTimelineStep` objects) into plain-id/scalar
  dataclasses (`OccupantRouteRecord`/`OccupantRouteHop`) and saves/loads them via
  `serialization/json_writer.py`/`json_reader.py` — the same convention `scenario_storage`
  already uses, not pickle. `decision_events.py` does the same for
  `human_decision_engine.events.DecisionEvent.to_dict()`'s already-existing shape, which was
  computed but discarded before this milestone. `occupant_position.py` interpolates an
  occupant's world position at any time from their own recorded hops, mirroring (not importing)
  `sandbox/manager.py`'s own position-resolution formula, including the "hold position across a
  Stair hop" rule for cross-floor coordinate spaces. No line of `simulator/`, `behavior/`,
  `navigation/`, or any other frozen package was modified — only read.
- **`command_center/incident_data.py`** — `IncidentData` gained an additive `occupant_routes`
  field; `IncidentFrame` gained an additive `occupant_positions` mapping, computed once per frame
  from `occupant_routes` + the Scenario's own authored starting zones. Every pre-existing
  `IncidentData` construction/`load_incident()` call is unaffected (both new fields default to
  empty).
- **`command_center/building_view.py`** — two new, purely additive rendering methods
  (`render_occupants`/`highlight_route`, reusing the existing generic `_add_marker`/`_add_line`),
  plus click-to-select occupant markers (a new `occupant_clicked` signal via an event filter on
  the view's viewport).
- **Three new Command Center panels**, following the existing panel convention exactly
  (`set_incident()` once, `show_frame()` per tick, no owned timer/logic):
  `occupant_inspector_panel.py`, `event_timeline_panel.py`, `statistics_panel.py`.
- **`replay_studio/`** (new package) — `session.py` resolves a scenario id against a campaign
  output directory (mirroring the shared directory layout both producers already write);
  `open_scenario_dialog.py` is the "Scenario #4832" picker UI; `app.py`/`replay_studio_main.py`
  is a thin launcher that reuses `command_center.main_window.MainWindow` wholesale as a library —
  the same "separate executable, reuse existing packages" pattern this codebase's own Builder V1
  milestone already established relative to the Designer app.
- **`command_center/timeline_panel.py`** — speed options aligned to the requested
  0.25x/0.5x/1x/2x/5x set, and Step Back/Step Forward buttons added (both simply move the
  existing slider by one frame).

---

## 3. Public API

- `simulation_recording.occupant_routes`: `OccupantRouteHop`, `OccupantRouteRecord`,
  `build_occupant_route_records(movement_result)`, `save_occupant_routes(records, path)`,
  `load_occupant_routes(path)`.
- `simulation_recording.decision_events`: `save_decision_events(events, path)`,
  `load_decision_events(path)`.
- `simulation_recording.occupant_position`: `OccupantPosition`, `BuildingPositionIndex`,
  `interpolate_occupant_position(record, start_zone_id, time, position_index)`.
- `command_center.incident_data.load_incident(...)`: two new optional keyword arguments,
  `occupant_routes_path`, `decision_events_path`.
- `command_center.building_view.BuildingView`: `set_occupant_routes(records)`,
  `select_occupant(occupant_id)`, `occupant_clicked` signal, `selected_occupant_id` property.
- `command_center.occupant_inspector_panel.OccupantInspectorPanel`: `set_incident`, `show_frame`,
  `select_occupant`.
- `command_center.event_timeline_panel.EventTimelinePanel`: `set_incident`, `show_frame`,
  `jump_to_time` signal.
- `command_center.statistics_panel.StatisticsPanel`: `set_incident`, `show_frame`.
- `command_center.dashboard.Dashboard`: new `jump_to_time(time)` method.
- `command_center.main_window.MainWindow`: new `open_scenario_dialog()` method (additive, sits
  alongside the pre-existing `load_incident_dialog()`).
- `replay_studio.session`: `discover_scenario_ids(output_dir)`, `resolve_scenario_artifacts(
  output_dir, scenario_id)`.
- `replay_studio.app.ReplayStudioApp`, launched via `replay_studio_main.py`.

---

## 4. Replay storage format

Two new plain-JSON artifacts per scenario, written to the same campaign output directory every
existing artifact already uses:

- `<output_dir>/occupant_routes/<scenario_id>/occupant_routes.json` — a JSON list, one object per
  occupant/firefighter: `occupant_id`, `state`, `depart_time`, `arrival_time`, and `hops` (a list
  of `{from_node_id, to_node_id, edge_id, edge_type, start_time, end_time, distance,
  queue_wait_time}`). Every value is a plain string/float/null — no live `Node`/`Edge` reference.
- `<output_dir>/decision_events/<scenario_id>/decision_events.json` — a JSON list of
  `human_decision_engine.events.DecisionEvent.to_dict()`'s own existing shape.

Both follow the exact convention `scenario_storage` already established for persisting a
`Scenario` (`JsonWriter.write`/`JsonReader.read`, not pickle, not `Serializer`) and are written by
both known scenario producers (`designer/campaign/campaign_worker.py` and
`research_framework/runner.py`), which call the identical `simulation_recording` functions so no
logic is duplicated between them.

---

## 5. UI

- **Playback**: Play/Pause/Stop (pre-existing, reused), Step Back/Step Forward (new), speed
  0.25x/0.5x/1x/2x/5x (aligned to spec), scrub slider + "Jump to time" (pre-existing, reused).
- **Building/occupant view**: `BuildingView`'s existing floor-plan rendering now also draws a
  colored marker per occupant (color by live state — PENDING/AT_NODE/TRAVERSING/ARRIVED/
  UNREACHABLE/STATIONARY), filtered to the currently selected floor; clicking a marker selects
  that occupant everywhere (Occupant Inspector updates, their actual travelled route is drawn as
  a highlighted polyline).
- **Occupant Inspector** (new tab): id, current zone/stair, behaviour profile, base Decision/
  Route-Choice/Pre-movement-delay strategy names, current state/destination/speed, assisting/
  assisted/leader-follower status, derived attributes (compliance, panic susceptibility, etc.),
  and a decision-event timeline for the selected occupant.
- **Event Timeline** (new tab): one merged, chronological, double-click-to-jump table combining
  occupant depart/arrive, engineering state changes (a diff between consecutive frames),
  decision events, recommendation changes, and voice broadcasts.
- **Statistics** (new tab): occupants evacuated/remaining vs. time, congestion/smoke vs. time,
  door/exit/stair utilization vs. time, average speed vs. time, and a behavior-profile
  distribution table — all rendered with a small custom-painted chart widget matching the
  existing `_TrendChart` style (no new charting library dependency introduced).
- **Open Scenario** (new menu action in `File`, alongside the pre-existing `Load Incident...`):
  pick a campaign output directory, pick a discovered scenario id, replay opens.

---

## 6. Performance

- Occupant position interpolation is O(hops) per occupant per frame lookup — negligible at
  per-occupant scale (a handful to a few dozen hops per occupant).
- `StatisticsPanel`'s door/exit/stair utilization series is O(frames × occupants × hops) in the
  current implementation — fine at the scale exercised by this milestone's own tests and typical
  single-scenario review, but would be the first thing to optimize (e.g. precomputing a sorted
  hop-interval index) if used against a scenario with thousands of occupants and hundreds of
  ticks.
- No new heavy dependency was introduced (no charting library, no NavigationGraph rebuild) —
  `BuildingPositionIndex` is a small dict built once per `Building` load.

---

## 7. Integration

- `designer/campaign/campaign_worker.py::_export_scenario_artifacts()` and
  `research_framework/runner.py::run_scenario_artifacts()` both now additionally write
  `occupant_routes.json`/`decision_events.json`, keeping their shared "byte-for-byte the same
  layout" claim true.
- Zero modifications to any frozen subsystem (Builder, Navigation, Simulation, Hazard Evolution,
  Perception, Crowd Intelligence, Recommendation, Guidance, Recommendation Layer, Execution
  Layer) — `simulation_recording` only reads already-public output types.
- `recommendation_layer`/`execution_layer` were confirmed to be live-only adapters with no
  campaign/export wiring; replay's own recommendation/voice/control history continues to come
  from `IncidentData`'s existing deterministic reconstruction (`advisory_system`, unaffected).

---

## 8. Test results

New tests (all passing):

| File | Tests |
|---|---|
| `tests/test_simulation_recording.py` | 11 |
| `tests/test_incident_data_occupant_positions.py` | 3 |
| `tests/test_replay_studio_session.py` | 4 |
| `tests/test_simulation_replay_studio_panels.py` | 12 |
| `tests/test_simulation_replay_studio_e2e.py` | 1 (real, unmocked campaign run) |
| `tests/test_research_framework_simulation_recording.py` | 1 |

Regression verification: every test file in the repository that imports `command_center`,
`simulation_recording`, or `replay_studio` (32 files, found by grep, not assumed) — **580 tests,
all passing**, run together in one session. A literal full-repository pytest run (~382 test
files) was attempted twice but exceeded this session's background-command time budget (killed at
its cap both times); `PROJECT_STATE.md` in this repository records the last known full-suite
result (5324 passing) from a prior session, and nothing in this milestone's diff touches any
frozen package, so a full-suite regression beyond the 580 directly-verified tests is considered
low-risk but not independently re-confirmed here — disclosed rather than assumed.

---

## 9. Remaining limitations

- **Occupant Inspector strategy names are template-level, not fully wrapped-chain-accurate.**
  "Decision Strategy"/"Route Choice Strategy" show the profile template's own base strategy class
  (e.g. `ComplianceDecisionStrategy`), not the full runtime wrapper chain (assistance/social-group/
  crowd-following/attribute-aware) — disclosed in-panel via the "(from profile template)" label.
- **"Exit/Stair Guidance" is building-wide, not personalized.** No per-occupant routing
  recommendation is ever generated by the batch scenario-generation pipeline (that mechanism
  exists only in the separate interactive/live simulation path) — the panel shows the exit/stair's
  own decision-policy status instead of fabricating a personalized "recommendation you received."
- **No cross-floor route interpolation.** A Stair hop's two endpoints live in different, non-
  comparable coordinate spaces (each floor's own local system) — position correctly holds rather
  than drawing a misleading straight line, but there is no single combined view showing a
  multi-floor route continuously.
- **Decision-event timestamps are mostly registration-order, not simulation-clock time**, since
  `human_decision_engine` decisions are made before movement begins; shown as "Pre-departure"
  rather than a fabricated numeric time, except the two post-hoc event types that do carry a real
  time.
- **Event Timeline row highlighting for "current frame" is not implemented** — the timeline is
  static and double-click-to-jump works, but there is no visual indicator of which row corresponds
  to the currently displayed frame.
- **Statistics utilization chart is O(frames × occupants × hops)** — see Performance above.
- A full, literal 382-file/5000+ test repository run was not completed within this session's time
  budget (see Test Results) — targeted, complete-file-level coverage was substituted and is
  disclosed as such rather than presented as an exhaustive run.

---

## 10. Commit hash

`7da0de586e81c8ba48ccbb3df63fd843e9ddd01f`

(This commit also bundles several previously-completed, uncommitted milestones — Recommendation
Layer, Execution Layer, Warden Notification framework, Live Camera Viewer/Occupants integration —
included at the user's explicit request after being flagged, since they shared several of the
same files this milestone needed to edit.)
