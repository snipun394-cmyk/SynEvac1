# Obstacle → Navigation & Evacuation Connectivity

Status: Obstacle now genuinely participates in Navigation Graph traversability, Pathfinding, Evacuation Recommendation, Evacuation Guidance, Simulation, Scenario Generation, and Live Runtime — closing the specific gap `docs/architecture/designer_asset_connectivity_audit.md` found ("Obstacle... currently lacking meaningful runtime decision connectivity"). No Designer asset was added, no Navigation Graph redesign occurred, no CFD/fire physics was added, and Obstacle was never given execution authority.

## 1. What Obstacle stores (Phase 1)

`models/obstacle.py::Obstacle` — an axis-aligned **rectangle** in meters: `x`, `y` (top-left corner), `length` (x-extent), `width` (y-extent). Not a polygon, not a point. It already carried, before this milestone:

- `active: bool` — whether the obstacle currently exists physically (already the exact "enabled/blocked/active concept" Phase 2 asked to reuse, never reinvented).
- `traversability: str` — one of `"Passable"`, `"Reduced Width"`, `"Blocked"`.
- `traversal_cost: float` — present but explicitly documented as "Never interpreted here... Used by future simulation/navigation movement-cost calculations." **Still not interpreted by this milestone** — see §2.

## 2. Honest semantics adopted (Phase 2)

**Obstacle means exactly one thing:** a physical region that cannot be traversed by occupants while `active=True` **and** `traversability == "Blocked"`. `"Passable"` and `"Reduced Width"` obstacles never affect traversability at all — `traversal_cost` remains uninterpreted, deliberately. This milestone does not invent a congestion/cost meaning for `"Reduced Width"` or `traversal_cost`; doing so was explicitly out of scope. Obstacle is not hazard, not fire, not smoke, not a door/zone closure, and not dynamic crowd blockage — those remain entirely separate concepts, untouched.

## 3. The integration point (Phase 3) — Edge.traversable, live-referenced

`navigation/node.py` and `navigation/edge.py` both document a deliberate design: no dynamic state is ever baked into a Node/Edge, because the graph is rebuilt fresh from the Building on every `NavigationGraphGenerator().build()` call. Yet `Door.locked`/`Exit.is_blocked` already reach `Edge.traversable` **live** — the property does `getattr(self.reference, "locked", False)` against the *same* Door object Floor.doors owns, so a state change is reflected the instant it happens, with **no graph rebuild required**.

Obstacle follows the identical discipline. `Edge` gained one new field:

```python
blocking_obstacles: Tuple[Any, ...] = field(default_factory=tuple)
```

populated once, at build time, by `NavigationGraphGenerator._add_door_edges()`/`_add_exit_edges()`, with `tuple(floor.obstacles)` — the **same live Obstacle object references** the Floor already owns, not a snapshot of their state. `Edge.traversable` (for Door and Exit only — see §4 for why Stair is out of scope) additionally calls `navigation.obstacle_geometry.segment_blocked_by_obstacles(start_point, end_point, self.blocking_obstacles)`, which live-checks each referenced obstacle's *current* `active`/`traversability`/`x`/`y`/`length`/`width` and performs real segment-vs-rectangle geometry intersection (reusing the existing `visibility.geometry.segment_intersection` primitive rather than a second implementation — Phase 1's own "what existing geometry utilities can be reused" question).

**What this buys**: toggling `obstacle.active`, changing `obstacle.traversability`, or moving an obstacle (`obstacle.x`/`.y`) is reflected on the *very next* `edge.traversable` read — no rebuild. **What still needs a rebuild**: a brand-new obstacle appearing, or an existing one being deleted, since `blocking_obstacles` is a tuple snapshot of *which* objects exist on that floor at build time — exactly the same limitation a brand-new Door already has (it needs a rebuild to produce its own Edge at all). This is proven directly in `tests/test_obstacle_navigation_integration.py::EvacuationGuidanceStalenessTests::test_stale_route_invalidated_then_fresh_valid_guidance_replaces_it`.

`PathfindingEngine` itself was **not modified at all** — `_relax()` already gated traversal purely on `edge.traversable` before this milestone; extending what that property considers required zero changes to Dijkstra/A*/Yen's algorithm.

## 4. Geometric resolution — what can honestly be represented (Phase 3)

The Navigation Graph represents Zones, Assembly Points, and the Door/Exit/Stair connections between them — it has **no finer geometric resolution than that**. This milestone does not fabricate arbitrary within-zone obstacle avoidance (a route within a Zone's own interior is not represented at all, at any resolution, by this graph). What genuinely has enough geometric resolution to check honestly:

- **Door and Exit** both carry a real `start_point`/`end_point` line segment — an Obstacle rectangle can be checked for genuine intersection against that segment. **Implemented.**
- **Stair** connects two zones via two bare `(x, y)` points (`from_position`/`to_position`), not a segment, and — more importantly — its own approach geometry spans *two different floors*, so a single `blocking_obstacles` tuple (populated from one floor) cannot honestly represent both ends. **Explicitly out of scope, documented here rather than faked.** A future milestone could check each endpoint point-in-rectangle against its own floor's obstacles if this gap is ever prioritized.
- **Zone interiors**: no representation exists for "avoid this specific corner of a Zone" — an obstacle placed inside a Zone, touching no Door/Exit segment, correctly has **zero** effect (see §11 test below). This is the honest limit of a graph that only models Zones as single nodes, not sub-Zone geometry.

## 5. Pathfinding & 6. Evacuation Recommendation & Guidance (Phases 4/5/6) — zero code changes required

Because obstacle-blocking reaches `Edge.traversable` and nothing else, every consumer already built on top of it inherited the behavior automatically:

- `pathfinding/engine.py::PathfindingEngine._relax()` already skips non-traversable edges — an obstacle blocking a transition makes that route genuinely unavailable; removing/deactivating it restores the route, still on the same graph object. **Proven directly** (`EdgeTraversableObstacleTests`, `WorkedBuildingMigrationTests`).
- `evacuation_recommendation/ranking.py::SafeExitDistanceCalculator.compute()` already filters `edge.traversable` for exit candidates **and** its own per-cycle cache fingerprint already includes `frozenset(edge.id for edge in graph.edges if not edge.traversable)` — an obstacle toggling changes that fingerprint, forcing a genuine recomputation with the new blocked state honored. A worked building (Zone A → Door D1 → Zone B → Exit E1 vs. Zone A → Door D2 → Zone C → Exit E2) proves the recommendation migrates from E1 to E2 the moment an obstacle blocks D1, and reverts the moment it's deactivated — with **zero changes to `evacuation_recommendation/`**.
- `evacuation_guidance/route_planner.py::resolve_route()` already runs a fresh `PathfindingEngine.distances_from()` every call — an obstacle-blocked Door is structurally invisible to that search, so Guidance can never route through it. **Zero changes to `evacuation_guidance/`.** No obstacle-specific ranking bonus or penalty was added anywhere — graph connectivity alone already solves Phase 5/6's entire requirement.

## 7. Simulation (Phase 7)

`simulator/engine.py::OccupantSimulator` is a **pure graph-level executor** — it walks whatever `Route` `PathfindingEngine` already computed, using `edge.traversal_time`/`edge.walking_distance`. It performs **no continuous geometric collision avoidance of any kind**, confirmed by direct inspection (no spatial/geometry logic exists in this class at all). This was true before this milestone and remains true after it.

**GRAPH-LEVEL OBSTACLE EFFECT: YES.** A route that would have crossed a blocked Door/Exit is never returned by `PathfindingEngine` in the first place, so simulated occupants automatically take an alternate route or fail to reach the goal, exactly matching real routing behavior.

**CONTINUOUS LOCAL OBSTACLE AVOIDANCE: NO.** SynEvac does not, and this milestone does not add, any per-step steering/collision geometry for an occupant walking near (but not through) an obstacle inside a Zone's interior. Claiming otherwise would be dishonest; it is not implemented.

## 8. Scenario Engine (Phase 8) — found already fully wired

Investigation found the Scenario Definition/Generator/Runner/Event-Executor layers already supported obstacle presence end-to-end, **before this milestone touched anything**:

- `scenario_generator/generator.py::_generate_obstacle_states()` samples a per-obstacle `PresenceState.ACTIVE`/`INACTIVE` from `engineering.obstacle_state_distribution`.
- `scenario_runner/building_initializer.py::apply_obstacle_state(obstacle, presence)` mutates `obstacle.active` directly on the **same live Obstacle object** the building copy owns, applied at scenario setup (`_apply_obstacle_states()`).
- `scenario_event_executor/handlers.py::_handle_obstacle_event()` applies the identical mutation **mid-scenario**, from a scripted event.

Because all three already mutate the real, live `Obstacle.active` field, and because `Edge.blocking_obstacles` holds live references to those same objects, this milestone's navigation-side change is the **only** piece that was missing — connecting it required zero changes to any scenario package. Proven directly in `tests/test_obstacle_navigation_integration.py::ScenarioObstacleActivationTests`, calling the real `apply_obstacle_state()`/`_handle_obstacle_event()` functions, not reimplementations.

## 9. Live Runtime (Phase 9)

`live_runtime/factory.py::build_live_runtime()` builds one `NavigationGraph` from the caller's own `Building` — the canonical Digital Twin, never a second building model. Because `Edge.traversable` reads live Obstacle references, a change to `Obstacle.active` on that same `Building` object is honored on the **very next** `orchestrator.run_cycle()` — no rebuild, no restart. Proven directly in `LiveRuntimeObstacleTests::test_obstacle_activation_changes_the_live_recommendation_on_the_next_cycle`: an inactive obstacle produces `recommended_exit_id == "E1"` at cycle 0; flipping `obstacle.active = True` on the canonical Building and running cycle 1 produces `recommended_exit_id == "E2"`, with no other code touched.

No AI involvement was required for this, matching the milestone's own instruction.

## 10. Worked building test (Phase 10)

`tests/test_obstacle_navigation_integration.py::WorkedBuildingMigrationTests` and `EvacuationGuidanceStalenessTests` build exactly the milestone's own named structure: Zone A (occupied) → Door D1 → Zone B → Exit E1 (short path) and Zone A → Door D2 → Zone C → Exit E2 (long path), with Obstacle O1 positioned across D1's own line segment.

- Without O1: `EvacuationRecommendationEngine` recommends **E1**; `EvacuationGuidanceEngine` builds a route through **D1**.
- O1 active (`Blocked`): recommendation migrates to **E2**; guidance builds a route through **D2**, never through D1.
- O1 deactivated: recommendation and guidance both revert to **E1**/**D1**, on the same graph object, no rebuild.

## 11. Non-blocking obstacle test (Phase 11)

`NonBlockingObstacleTests` places a `"Blocked"`, active obstacle well inside Zone A's own footprint, nowhere near any Door/Exit segment. Every edge (`D1`, `D2`, `E1`, `E2`) remains traversable, and `PathfindingEngine.nearest_exit("ZONE-A")` still finds a route — the entire Zone is never incorrectly disabled by an obstacle that happens to exist somewhere inside it. This is the direct proof against an overly coarse ("obstacle anywhere in a Zone blocks the whole Zone") implementation.

## 12. Failure / edge cases tested (Phase 12)

`FailureAndEdgeCaseTests` and `EdgeTraversableObstacleTests` cover: an obstacle never placed on any floor (`floor_id=""`, absent from `Floor.obstacles`) has no effect by construction; an obstacle touching a Door segment's boundary exactly still blocks it; an obstacle overlapping a Door, and separately an Exit, blocks each independently; multiple obstacles on one floor are each evaluated independently (one blocking, one harmless); an inactive obstacle directly overlapping a Door never blocks it; a deleted obstacle stops blocking after the next rebuild; a legacy project saved with no `"obstacles"` key at all (pre-dating this milestone) loads and builds cleanly; an obstacle on a different floor never affects another floor's edges; degenerate geometry (zero or negative length/width) never crashes graph build or the geometry check. No test produced a crash or a fabricated block.

## 13. Architecture guards (Phase 13)

Mechanically proven (`ArchitectureGuardTests`): `navigation/obstacle_geometry.py`, `navigation/edge.py`, and `models/obstacle.py` import none of `hazard`, `hazard_evolution`, `fire_growth`, `smoke_propagation`, `ai_decision`, `ai_registry`, `ai_inference`, `ai_training`, `ai_features`, `advisory_system`, `building_control`, `voice_evacuation`, `dynamic_signage`, or `decision_policy`. `BuildingState.hazard_summary` remains entirely hazard-snapshot-derived, untouched by Obstacle. Recommendation/Guidance compute() calls in the presence of an active, blocking obstacle produce plain data snapshots only — both engines' own pre-existing architecture guards (`test_evacuation_recommendation_architecture_guards.py`, `test_evacuation_guidance_architecture_guards.py`) already mechanically forbid importing Voice/Signage/BuildingControl at all. Obstacle's influence flows exclusively through `Edge.traversable` — no other path exists for it to reach anything.

## 14. Performance (Phase 14)

`scripts/benchmark_obstacle_navigation.py`, 50 zones / 100 doors / 100 obstacles:

| Measurement | Mean | p95 |
|---|---|---|
| Graph build, no obstacles | 0.662 ms | 0.819 ms |
| Graph build, 100 obstacles added | 0.666 ms | 0.859 ms |
| Incremental build overhead from 100 obstacles | 0.003 ms | — |
| Dijkstra `shortest_path()`, obstacle-aware graph | 0.223 ms | 0.292 ms |

Build-time overhead is negligible — `blocking_obstacles` is a plain tuple assignment per edge. The real (still small, at this scale) cost is paid per `Edge.traversable` access during pathfinding, where the O(obstacles-on-that-floor) geometry check actually runs. No premature optimization was performed.

## 15. Files created / modified

**New:** `navigation/obstacle_geometry.py` (`segment_blocked_by_obstacles()`), `tests/test_obstacle_navigation_integration.py` (44 tests), `scripts/benchmark_obstacle_navigation.py`, this document.

**Modified (additively):** `navigation/edge.py` (`blocking_obstacles` field, `_blocked_by_obstacle()`, extended `traversable`), `navigation/graph_builder.py` (`_add_door_edges`/`_add_exit_edges` populate `blocking_obstacles=tuple(floor.obstacles)`), `tests/test_engineering_navigation_graph.py` (updated the exact-field-set assertion to include the one new, deliberate field).

**Unchanged:** `pathfinding/engine.py`, `evacuation_recommendation/*`, `evacuation_guidance/*`, `simulator/*`, `scenario_generator/*`, `scenario_runner/*`, `scenario_event_executor/*`, `live_runtime/factory.py`, every hazard/AI/advisory/voice/signage/building-control package, and `models/obstacle.py` itself (its own `active`/`traversability` fields were already exactly what this milestone needed).
