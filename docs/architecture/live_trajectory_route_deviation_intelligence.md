# Live Occupant Trajectory, Movement Anomaly & Route-Deviation Intelligence

Status as of this milestone: SynEvac can now maintain a persistent, graph-aware trajectory for every live occupant, determine whether their movement constitutes genuine progress toward a currently-safe exit, and detect (with configurable persistence, never from a single noisy frame) route reversal, stalled route progress, hazardous-zone movement, against-flow movement, and shared/group route deviation — via a new `trajectory_intelligence/` package. This is movement geometry only: it never infers panic, confusion, or non-compliance, and it never overrides a deterministic safety decision.

## 1. Investigation findings (Phase 1)

Verified directly against the current source before writing anything:

1. `live_occupants.occupant.LiveOccupant.history` (`live_occupants/history.py`) already retains a **bounded** (`max_length`, default 30, configurable via `LiveOccupantManager(history_length=...)`), **immutable** history of `position_samples` (timestamp + world position) and `velocity_samples`, updated every cycle `world_position`/`world_velocity` is genuinely known. This is already sufficient raw material for trajectory analysis — no separate position-history store was created (Phase 4's own "reuse existing history if suitable" requirement).
2. `OccupantHistory.zone_transitions` already retains one record per *genuine* zone change (`from_zone_id`/`to_zone_id`/`timestamp`) — reused directly for both graph-aware route continuity and zone-based reversal detection (`trajectory_intelligence/anomaly.py::repeated_route_reversal`). There is **no** equivalent `floor_transitions` list; floor changes are validated on demand against the Navigation Graph's own Stair edges instead (`trajectory_intelligence/route_progress.py::_floor_transition_uncertain`).
3. `navigation.graph.NavigationGraph` + `pathfinding.engine.PathfindingEngine` already provide everything Phase 5/6/7 needed — a real, generic, decision_policy-free routing engine (`shortest_path`/`nearest_exit`/`distances_from`, Dijkstra/A*/Yen's-algorithm) operating purely on Node/Edge (`Edge.traversable` already refuses to cross a locked Door/blocked Exit). It was **never previously wired into any live package** — this milestone is the first live consumer.
4. `decision_policy/` is confirmed **entirely simulation-only**: every rule module reads exclusively from `GroundTruth`'s own post-hoc, completed-simulation analytics (mirrors `live_system/live_advisory_gateway.py`'s own, earlier, identical finding) — never imported here.
5. Live `BuildingState` exposes **zone-level hazard only** (`BuildingState.hazard_summary.zone_severities: Mapping[zone_id, HazardSeverity]`). The richer `HazardSnapshot` (with per-edge `HazardEdgeState.traversable`) is consumed *inside* `BuildingStateEstimator._summarize_hazard()` and never retained on `BuildingState` itself — confirmed by direct inspection, not assumption. Consequently: safe-route exclusion uses **zone-level hard node exclusion** (never a soft cost penalty), and hazardous-zone movement detection is honestly zone-level only — `MOVES_TOWARD_HAZARD` is **not implemented** (no gradient exists to honestly support it).
6. `crowd_intelligence.flow.evaluate_approach()` computes a transient, per-occupant-per-asset `ApproachEvidence` — never persisted, never exposed on `CrowdIntelligenceSnapshot`. Not duplicated; Trajectory Intelligence computes its own independent, graph-aware (not asset-relative) movement direction/route-progress instead.
7. `evacuation_progress.models.ZoneClearanceStatus.STALLED` is exclusively **per-zone** (a clearance-fraction trend). `trajectory_intelligence.models.AnomalyFlag.MOVEMENT_STALLED` is a **different, per-occupant** concept (insufficient graph-route-distance improvement for a configurable duration) — never duplicated, never conflated (Phase 27 test 13).
8. No trajectory/anomaly/reversal/against-flow logic existed anywhere in the repository prior to this milestone (confirmed by broad grep) — genuinely greenfield.

## 2. Terminology — these are different concepts

| Concept | Owner | Meaning |
|---|---|---|
| `RecognizedBehavior` | `behavior_recognition` | A hedged, single-cycle geometric heuristic (WALKING/RUNNING/STATIONARY/POSSIBLY_FALLEN) |
| `HumanState` | `human_evidence`/perception | Reconciled, cross-cycle physical state evidence (FALLEN/CRAWLING/BEING_ASSISTED/…) |
| `OccupantStatus` | `live_occupants.lifecycle` | Presence lifecycle (NEW/ACTIVE/TEMPORARILY_LOST/EXITED/EXPIRED) — orthogonal to physical state |
| `MovementStatus` (new) | `trajectory_intelligence` | Purely physical: is this occupant currently MOVING or STATIONARY, from position-sample speed |
| `RouteProgressStatus` (new) | `trajectory_intelligence` | Graph-aware: is this occupant's zone getting closer to a *currently safe* exit |
| `ZoneClearanceStatus.STALLED` | `evacuation_progress` | Zone-level: has this zone's own clearance fraction stopped improving |
| `AnomalyFlag.MOVEMENT_STALLED` (new) | `trajectory_intelligence` | Occupant-level: has THIS occupant's own route distance stopped improving, regardless of physical motion |

**ROUTE DEVIATION ≠ PANIC. AGAINST FLOW ≠ NON-COMPLIANCE. MOVEMENT STALLED ≠ FALLEN. MOVING AWAY ≠ CONFUSION.** Every flag `trajectory_intelligence.models.AnomalyFlag` defines is a deterministic, disclosed geometric observation — never a psychological or compliance judgment. Command Center never renders "PANICKING"/"CONFUSED"/"DISOBEDIENT" (`command_center/live_trajectory_intelligence_panel.py` only ever displays the neutral vocabulary this package itself defines).

## 3. Architecture

```
Camera → Tracking → World Projection → Cross-Camera Identity → LiveOccupant
                                                                     |
                                     (LiveOccupantManager.active_occupants(),
                                      OccupantHistory.position_samples/
                                      zone_transitions -- read-only)
                                                                     v
                                              trajectory_intelligence/
                                                     |
                       trajectory.py   -- movement facts (distance, speed,
                                           direction, MovementStatus) from
                                           position_samples alone
                       route_progress.py -- SafeRouteCalculator (cached,
                                           hazard-excluded PathfindingEngine.
                                           distances_from(Outside)) ->
                                           RouteProgressStatus, route
                                           distance trend, nearest safe exit
                       anomaly.py       -- moving-away/reversal/stall/
                                           hazardous-zone detection
                                           (persistence-gated)
                       flow_alignment.py -- per-floor dominant flow,
                                           AGAINST_DOMINANT_FLOW
                       history.py       -- the ONE small piece of state
                                           this package owns beyond
                                           OccupantHistory: bounded
                                           per-occupant route-distance
                                           samples
                       engine.py        -- TrajectoryIntelligenceEngine,
                                           the single per-cycle entry point
                                                                     |
                                                                     v
                                       TrajectoryIntelligenceSnapshot
                                          /            |            \
                                         v             v             v
                          EmergencyResponseIntelligenceEngine   Advisory (via
                          (optional trajectory_snapshot param,  TrajectoryDecisionEvidence,
                           severe-anomaly evidence only)        advisory_system/trajectory_evidence.py)
                                         |                             |
                                         v                             v
                                Command Center (LiveMovementIntelligencePanel, decision support only)
```

`LiveOccupantManager` remains the sole canonical occupant registry — `trajectory_intelligence/` reads it but owns no occupant identity/lifecycle state of its own.

## 4. Safe-route candidate handling (Phase 6)

`SafeRouteCalculator` (`trajectory_intelligence/route_progress.py`) runs exactly one `PathfindingEngine.distances_from(Node.OUTSIDE_NODE_ID, excluded_node_ids=...)` Dijkstra pass per cycle, cached by a `(excluded_zone_ids, non_traversable_edge_ids)` fingerprint (mirrors `ai_decision.engine.AIDecisionEngine`'s own established per-cycle route-cache pattern — recomputing per-occupant was that class's own documented "~72% of tick time" cost before caching). Zones whose `BuildingState.hazard_summary.zone_severities` reading is at/above a configurable floor (default `HIGH`) are **hard-excluded** as graph nodes — never merely cost-penalized — so an unsafe zone can structurally never appear on any `Route` this calculator returns. Structural blockage (a locked Door, a blocked Exit) is separately and unconditionally enforced by `Edge.traversable` inside `PathfindingEngine` itself, with no configuration needed here. An occupant standing inside a now-excluded zone honestly receives `NO_SAFE_ROUTE`, never a fabricated escape route through it.

## 5. Uncertainty handling (Phase 15)

| Missing evidence | Result |
|---|---|
| No `world_position` | Route/zone-transition analysis may still work; `position_available=False`, `position_sample_count=0`, never a fabricated position |
| No `current_zone_id` | `route_progress_status = ROUTE_UNCERTAIN` (graph route cannot be honestly claimed); world-space reversal fallback may still apply |
| No hazard information at all | Every zone treated as unexcluded (no fabricated hazard) — never `NO_SAFE_ROUTE` from absent data alone |
| Camera offline / long cross-camera blind interval | `stale=True` once `now - LiveOccupant.last_seen` exceeds a configurable threshold (independent of, and typically shorter than, `LiveOccupantManager`'s own `expire_after_seconds`) |
| Fewer than `against_flow_min_occupants` moving occupants, or coverage below `against_flow_min_coverage_fraction`, on a floor | That floor is simply absent from `dominant_flow_direction_by_floor` — never a fabricated 0.0 |

## 6. Safety precedence (Phase 23) — mechanically proven

`tests/test_live_runtime_trajectory_intelligence_e2e.py::SafetyPrecedenceRecomputeTests` drives the full offline production pipeline, makes zone `z1` (containing `EXIT-1`) hazardous mid-run, and proves `nearest_safe_exit_id` for an occupant elsewhere in the building switches from `EXIT-1` to `EXIT-2` — and never switches back — for as long as the hazard reading says so. No automatic voice/building-control/dispatch/FACP action is ever taken as a result (`test_no_automatic_action_taken_when_hazard_appears`).

## 7. No execution authority (Phase 26)

Mechanically enforced by `tests/test_trajectory_intelligence_architecture_guards.py`: `trajectory_intelligence/` may not import `advisory_system`, `command_center`, `voice_evacuation`, `speaker_manager`, `building_control`, `decision_policy`, `ai_*`, `rl_training`, RTSP/YOLO/torch modules, or `ground_truth`/`simulator`; it may never call `.broadcast(`, `.announce(`, `.execute_control(`, `.dispatch(`, `.confirm(`. It has no reverse dependency either — `decision_policy` never imports it. Evidence reaches Emergency Response/Advisory only as an already-computed, optional, plain-value parameter (`trajectory_snapshot`/`TrajectoryDecisionEvidence`), the same one-way, no-cycle shape every prior live-intelligence milestone already established.

## 8. Event vocabulary (Phase 25)

`live_system.event_bus.EventType` gains: `TRAJECTORY_INTELLIGENCE_UPDATED` (general "fresh snapshot" signal, every cycle), plus **transition-only** events — fired the cycle a per-occupant condition newly starts or stops, never repeated every cycle it merely continues to hold: `OCCUPANT_ROUTE_DEVIATION_DETECTED`/`OCCUPANT_ROUTE_RECOVERED`, `OCCUPANT_MOVEMENT_STALLED`/`OCCUPANT_MOVEMENT_RESUMED`, `OCCUPANT_ENTERED_HAZARDOUS_ZONE`/`OCCUPANT_EXITED_HAZARDOUS_ZONE`, `SHARED_ROUTE_DEVIATION_DETECTED`.

## 9. What this milestone deliberately does not do

No psychological panic/confusion inference, no compliance classification, no deep-learning trajectory prediction, no pose estimation, no new YOLO models, no face recognition, no Decision Policy redesign, no automatic voice broadcast, no automatic building control, no firefighter auto-dispatch. `MOVES_TOWARD_HAZARD` is not implemented (no live edge-level hazard gradient exists to honestly support it — see §1 item 5); only `ENTERED_HAZARDOUS_ZONE`/`REMAINS_IN_HAZARDOUS_ZONE` (zone-level) are.
