# Stair Simulation Reliability & Multi-Floor Reachability Audit

Status: audit complete, two root-caused fixes applied (both minimal, both preserve existing
architecture, neither redesigns Stair as a Zone/node, neither touches Stair-camera perception). Builds
on `docs/architecture/live_stair_perception.md`, `docs/architecture/stair_flow_intelligence.md`, and
`docs/architecture/stair_predictive_feature_live_parity.md`, none of which are modified by this
milestone.

---

## Phase 1: traced architecture (production code, not documentation)

```
Designer Stair (models.staircase.Staircase)
  -> Floor.stairs (physical placement -- the Floor a Staircase is added to)
  -> NavigationGraphGenerator._add_stair_edges(graph, building, floor)   [navigation/graph_builder.py]
       -> resolves from_zone (against `floor`, the CONTAINING floor) and
          to_zone (against building.get_floor(stair.to_floor_id))
       -> Edge(id=stair.id, edge_type=STAIR, from_node, to_node,
               walking_distance=stair.travel_distance(building))         [navigation/edge.py]
  -> pathfinding.engine.PathfindingEngine (Dijkstra/A*, floor-agnostic --
     treats a Stair edge exactly like Door/Exit: cost_model.cost(edge),
     edge.traversable)
  -> pathfinding.route.Route (a fixed, pre-planned sequence of nodes/edges)
  -> simulator.coordinator.MultiAgentSimulation (walks the FIXED Route
     one edge at a time via Occupant.current_edge_index; a Stair edge is
     structurally indistinguishable from any other edge except for
     StairCapacityModel/StairAwareCongestionModel's own opposing-
     occupant-count adjustment)
  -> continued routing after traversal: NOT a separate mechanism -- the
     entire origin-to-goal Route (every Door/Stair/Exit hop) was already
     computed as ONE PathfindingEngine call before simulation starts;
     the coordinator only ever advances an index through it
  -> Exit edge -> Node.OUTSIDE_NODE_ID ("outside")
```

- **Stair endpoints**: `Staircase.from_zone_id`/`to_zone_id` (Zone references) -- `from_floor_id`/
  `to_floor_id` are used ONLY by `Staircase.vertical_height()`/`travel_distance()` for the DISTANCE
  computation, never for graph connectivity itself (see Phase 2's root cause below).
- **`from_floor_id`/`to_floor_id` meaning**: whichever two floors this Staircase connects -- NEVER
  "lower"/"upper" by field position (verified: `Building.floor_elevation()` is the only authoritative
  source of vertical ordering, and a Staircase authored with `from_floor_id` on the HIGHER floor still
  produces the geometrically correct positive `vertical_height()`).
- **Approach nodes**: a Stair edge's `from_node`/`to_node` are ordinary Zone nodes -- reaching a Stair is
  exactly "can `PathfindingEngine` find a path from the occupant's zone to the Stair's own approach
  zone," no different from reaching a Door.
- **`walking_distance`**: `stair.travel_distance(building)`, computed ONCE at graph-build time
  (`NavigationGraphGenerator`), never cached on `Staircase` itself.
- **`vertical_height`**: `abs(Building.floor_elevation(to_floor) - Building.floor_elevation(from_floor))`,
  `0.0` if either floor fails to resolve via `Building.get_floor()`.
- **Traversability**: `Edge.traversable` always returns `True` for a Stair edge that was built at all --
  Staircase has no `active`/`locked` flag in V1 (a genuine, confirmed finding, not a bug -- see Phase 16).
- **Capacity**: `StairCapacityModel` (narrower than `DefaultCapacityModel`, floored at 1 -- see Phase 10).
- **Congestion**: `StairAwareCongestionModel` counts opposing-direction occupants on the same Stair edge
  and reduces effective speed accordingly (bidirectional counter-flow).
- **Edge directionality**: every V1 edge (Door/Exit/Stair) is inherently bidirectional -- no separate
  direction flag anywhere.
- **Floor transition**: not a special mechanism -- `OccupantTimelineStep.is_floor_transition` is a
  read-only derived property (`from_node.floor_id != to_node.floor_id`), never consulted by the
  coordinator's own control flow.
- **Continued routing**: guaranteed by construction -- the Route is one fixed, complete origin-to-goal
  path; there is no "re-routing after the Stair" step to get wrong.

## Phase 2: historical failure, reproduced and root-caused

**Reproduced**: yes, at both the graph-build level and the simulation level.

**Root cause, precisely**: `Staircase.travel_distance(building)` calls `vertical_height(building)`,
which resolves `building.get_floor(self.from_floor_id)`/`building.get_floor(self.to_floor_id)`. If
`from_floor_id` is empty/wrong (does NOT match the floor the Staircase is actually placed on, i.e. the
Floor whose `.stairs` list contains it), `get_floor()` returns `None`, and `vertical_height()` honestly
returns `0.0` -- **this by itself does NOT make the Stair unreachable**: `_add_stair_edges()` resolves
`from_zone`/`to_zone` against the real containing `floor` (not against `stair.from_floor_id`), so
connectivity is completely unaffected. Only the DISTANCE is wrong. This 0.0 then flowed through two
places with no safeguard:

1. `Edge.walking_distance = 0.0` (never `None` -- `Staircase.travel_distance()` always returns a float).
2. `Edge.traversal_cost` returned `0.0` (its own `if walking_distance is not None: return it` branch),
   making the broken Stair the CHEAPEST possible edge for Dijkstra -- never unreachable, actually
   over-preferred.
3. `MultiAgentSimulation._admit_onto_edge()`'s own `distance = edge.walking_distance or 0.0` computed
   `duration = 0.0`, producing `start_time == end_time` -- an instantaneous, physically nonsensical
   traversal.

**So the historical symptom was never "cannot approach or use a Stair"** in the sense of routing
failure -- pathfinding always succeeded, and in fact preferred the broken Stair (cost 0). The real,
demonstrated symptom is **an instantaneous, zero-time traversal**, which can *look* like "the occupant
never really used the stair" when inspecting a timeline (no time elapses on it) -- plausibly the origin
of the "unable to approach or use" framing this milestone's motivating report used.

**Not previously fixed everywhere**: the predictive-dataset topology fix (referenced in this milestone's
own brief) corrected `from_floor_id` on the AFFECTED topology *data* -- it did not add any mechanical
guard against the SAME mistake recurring anywhere else (Designer-authored buildings, Scenario Generator
inputs, future topologies). That gap is what Phase 17 closes.

## Fixes applied (both minimal, both preserve existing architecture)

**1. `navigation/graph_builder.py::_add_stair_edges()`** -- when a Stair's computed `travel_distance()`
is `<= 0` (regardless of cause), `Edge.walking_distance` is now set to `None` instead of `0.0` -- reusing
`Edge.walking_distance`'s own PRE-EXISTING "not derivable" contract (no new concept). A new validation
issue (`stair_zero_traversal_distance`, WARNING) is recorded via the same `graph.record_issue()`
mechanism every other Stair validation problem already uses. A SEPARATE new issue
(`stair_same_floor_both_ends`, WARNING) is recorded when a Stair's two ends resolve to the identical
floor. **Connectivity, edge construction, and graph structure are completely unchanged** -- the edge is
still built, still traversable; only its cost/distance degrades to the SAME safe default
(`Edge.DEFAULT_TRAVERSAL_COST = 1.0`) every other edge with undeterminable geometry already falls back
to.

**2. `simulator/coordinator.py::MultiAgentSimulation._admit_onto_edge()`** -- `distance = edge.
walking_distance or 0.0` silently coerced BOTH "genuinely unknown" (`None`) and "genuinely zero" to the
identical `0.0`, which would have completely NEUTRALIZED fix #1 for the multi-agent simulation path
specifically (a `None` walking_distance still produced `duration = 0.0`). Changed to
`distance = edge.traversal_cost` -- numerically IDENTICAL to the old expression for every already-working
edge (including a genuinely-computed `0.0`), differing ONLY for the previously-broken `None` case, where
it now correctly falls back to `1.0`. `simulator/engine.py::OccupantSimulator` (the frozen, single-
occupant V1 sibling) already treated `None` correctly (`distance_known = False`/`time_known = False`,
never coerced to 0) -- this fix brings the multi-agent coordinator's own behavior back in line with its
own sibling's existing discipline, not a new one.

Neither fix touches `models/staircase.py`, `NavigationGraph`'s public shape, `Edge`'s public shape (only
values passed to already-existing fields), `PathfindingEngine`, or Stair-camera perception.

## Phase 3: multi-floor reachability matrix

`tests/test_stair_simulation_reliability_audit.py::MultiFloorReachabilityMatrixTests` -- 2, 3, and 4-floor
linear buildings (one Stair per adjacent floor pair, one Exit on the ground floor), one occupant per
floor:

| Start Floor | Stair(s) Required | Exit Floor | Route Exists | Traversal Completes | Evacuates |
|---|---|---|---|---|---|
| Ground | 0 | Ground | ✓ | n/a | ✓ |
| Floor 1 | 1 | Ground | ✓ | ✓ | ✓ |
| Floor 2 | 2 (chained) | Ground | ✓ | ✓ | ✓ |
| Floor 3 | 3 (chained) | Ground | ✓ | ✓ | ✓ |

Every structurally reachable case succeeded, at every floor count tested.

## Phase 4: chained stairs

`ChainedStairsTests` -- Floor 3 → STAIR-3 → Floor 2 → STAIR-2 → Floor 1 → STAIR-1 → Ground → EXIT-G.
Proven: 4 steps total (3 Stair hops + 1 Exit hop), each stair step's `from_node.floor_id` matches the
expected floor at that exact point in the sequence (no teleportation), every step has `end_time >
start_time` (no zero-distance stair), every Stair `distance > 0`, and the occupant reaches `ARRIVED`.

## Phase 5: shared multi-floor staircase architecture

**Investigated, not guessed**: `models.staircase.Staircase` has exactly `from_floor_id`/`to_floor_id` --
one physical connector spans EXACTLY two floors, structurally (`Staircase.__dataclass_fields__` proven to
have no `floor_ids`/`intermediate_floor_id` field). **A 3+ floor stairwell requires one separate
`Staircase` object per adjacent floor pair**, each ending at the SAME physical landing Zone on the shared
middle floor (`SharedMultiFloorStaircaseArchitectureTests` proves two such Staircases correctly chain
through one shared landing Zone, appearing exactly once in the resulting Route, never duplicated).

**Correct Designer-authoring procedure for a 3+ floor stairwell**: for a stairwell spanning Ground/Floor
1/Floor 2, author TWO Staircase objects: one `from_floor_id=Ground, to_floor_id=Floor 1` ending at a
Zone `LANDING-1` on Floor 1, and a second `from_floor_id=Floor 1, to_floor_id=Floor 2` STARTING from that
SAME `LANDING-1` Zone. `models/staircase.py`'s own docstring already anticipates a future "Stairwell"
grouping object (multiple flights belonging to one physical stairwell) layered on top of this without
changing what a Staircase is -- not needed, and not built, by this milestone.

## Phase 6: stair directionality

`StairDirectionalityTests` -- both directions traversable (`lobby -> upstairs` and `upstairs -> lobby`,
both resolve to the same Stair edge); downward evacuation from an upper floor to a ground Exit succeeds;
a Staircase authored with `from_floor_id` on the physically HIGHER floor (reversed vs. the usual
convention) still produces the correct positive `vertical_height()` and a usable, positive-distance
Edge -- direction is derived from `Building.floor_elevation()`, never from `from_floor_id`/`to_floor_id`
field position.

## Phase 7: approachability

`ApproachabilityTests` -- occupants starting near the Stair, in a far zone (crossing one Door), and
behind a second chained Door all successfully route to the Stair's approach zone and beyond. **One real
failure mode confirmed and correctly attributed**: a Stair whose own approach Zone has no Door connecting
it to the rest of the floor IS genuinely unreachable from those other zones -- this is a floor-wide Zone-
connectivity completeness property (the same as any Door/Exit would be), not something Stair-specific
code could or should silently invent connectivity for. The Stair edge itself, once its own approach zone
IS reached, was always correctly built and traversable.

## Phase 8: multiple stairs

`MultipleStairsTests` -- both of two Stairs (STAIR-A, STAIR-B) independently reachable; blocking Door
A's own segment with an Obstacle correctly makes STAIR-A's approach non-traversable, and a fresh route
migrates entirely to STAIR-B; two independent `MultiAgentSimulation` runs against the identical Building
(only the Obstacle differing) prove an occupant is never hard-wired to a specific Stair id -- the SAME
occupant that used STAIR-A before the block uses STAIR-B after, purely from re-running pathfinding
against the current graph state.

## Phase 9: blocked stair approach

`BlockedStairApproachTests` -- confirms the EXISTING Obstacle mechanism (`Edge.blocking_obstacles`,
Obstacle → Navigation & Evacuation Connectivity milestone) is sufficient: an Obstacle activated across
Door D-A's own segment makes that Door edge non-traversable (`edge.traversable == False`), forcing a
fresh route through Door D-B and STAIR-B instead, while Door D-B/STAIR-B remain completely unaffected
(never globally disabling the floor). **Stair edges themselves have no `blocking_obstacles` integration**
(confirmed pre-existing, unchanged, and correctly out of scope -- `docs/architecture/
obstacle_navigation_integration.md`'s own "Stair's cross-floor point-to-point connection is deliberately
out of scope" finding) -- blocking a Stair's *approach* (a Door/Zone leading to it) is the supported,
tested mechanism; no continuous collision physics was added.

## Phase 10: stair capacity & congestion

`StairCapacityCongestionTests` -- `capacity=1` for a narrow (0.5m) Stair is confirmed **intentional**:
both `DefaultCapacityModel`/`StairCapacityModel` floor at `MINIMUM_CAPACITY = 1` specifically so the
coordinator's own event queue can never deadlock (a queued occupant must always eventually be admitted).
It means "one occupant may physically occupy this edge at once," never "only one occupant may ever use
this Stair." Verified with 1, 2, 10, and 50 occupants through the same capacity-1 Stair: nobody
disappears (`len(result.occupants) == n` every time), every occupant reaches `ARRIVED`, every individual
step has `end_time > start_time`, capacity-1 queueing is strictly serialized (no two occupants ever
overlap on the edge), and `unreachable_occupant_ids` stays empty even at 50 occupants. **NAVIGATION
UNREACHABLE and WAITING FOR CAPACITY are confirmed structurally distinct**: capacity constrains
THROUGHPUT (how fast people cross), never REACHABILITY (whether they eventually do).

## Phase 11 / 13: traversal-time & predictive-topology audit

`tests/test_stair_predictive_topology_audit.py` mechanically audits **every currently-active
predictive-dataset structural topology** -- `predictive_dataset.topologies_v4.all_structural_variants_v4()`,
24 templates across 6 families (`single_exit_lowrise`, `twin_stair_highrise`, `multi_exit_wide`,
`v1_topology_fixed`, `multi_wing`, `ring_corridor`; V3's 16 reused + V4's 8 new) -- using the SAME new
`stair_zero_traversal_distance`/`stair_same_floor_both_ends` validation codes as the detector:

```
family                   variant                            floors  stairs  distances
single_exit_lowrise      single_exit_lowrise                1       0       []
single_exit_lowrise      single_exit_deep_corridor          1       0       []
single_exit_lowrise      single_exit_branching_deadends     1       0       []
single_exit_lowrise      single_exit_vertical               2       1       [5.23]
twin_stair_highrise      twin_stair_highrise                3       2       [5.23, 10.46]
twin_stair_highrise      twin_stair_highrise_3stair         4       3       [5.23, 10.46, 15.69]
twin_stair_highrise      twin_stair_low                     2       1       [5.23]
twin_stair_highrise      twin_stair_chained_core            3       2       [5.23, 5.23]
multi_exit_wide          (4 variants)                       1       0       []
v1_topology_fixed        v1_topology_fixed                  2       1       [5.23]
v1_topology_fixed        v1_fixed_dual_stair                2       2       [5.23, 5.23]
v1_topology_fixed        v1_fixed_three_floor               3       2       [5.23, 5.23]
v1_topology_fixed        v1_fixed_long_corridor             2       1       [5.23]
multi_wing               (3 variants)                       1       0       []
multi_wing               multi_wing_vertical                2       1       [5.23]
ring_corridor            (4 variants)                       1       0       []
```

**Result: zero degenerate Stairs across all 24 templates** -- every Stair edge has a positive
`walking_distance`, no `stair_zero_traversal_distance`/`stair_same_floor_both_ends` issue anywhere. Every
Zone on every topology (76 zone-level subtests) has a structural path to some Exit. `twin_stair_highrise_
3stair` (4 floors, 3 chained Stairs) is the active suite's own chained-Stair proof -- all three distances
positive, top-floor-to-Exit routing succeeds. **This confirms, rather than merely trusts, this
milestone's own premise**: the previously-fixed predictive-topology bug is genuinely fixed in every
currently-active template, AND (via the two new fixes above) the SAME class of bug is now mechanically
guarded against for any future or Designer-authored Stair, not just these specific 24 topologies.

## Phase 12: Scenario Generator

**Investigated, no Stair-topology risk found**: `scenario_generator/` never constructs a `Staircase`
object, never references `from_floor_id`/`to_floor_id` anywhere in its own source (confirmed by
exhaustive grep). `scenario_generator.generator.generate_scenario()` takes an ALREADY-BUILT `Building`
as input (`request.building`) and only randomizes OCCUPANT PLACEMENT and engineering-asset STATE (door/
exit/stair/obstacle/camera/detector states) on top of it -- it cannot structurally introduce a malformed
Stair, since it never creates Stair geometry. Any occupant-stranding a generated scenario produces is
therefore either (a) a genuine topology defect in whatever Building it was given (now covered by Phase
11/13's audit and Phase 17's guard), or (b) an intentional state randomization (e.g. a scenario where a
Stair-adjacent Door is randomly locked) -- a deliberate stress-test case, never a topology bug.

## Phase 14: Recommendation

`tests/test_stair_recommendation_guidance_audit.py::RecommendationUnderstandsStairRoutesTests` -- traced
`evacuation_recommendation.ranking.SafeExitDistanceCalculator` to confirm it constructs and uses a real
`pathfinding.engine.PathfindingEngine` directly (never a floor-local Euclidean shortcut). Proven with an
upper-floor occupant and two Exits (one genuinely shorter total route via the Stair + a short ground-
floor leg, one deliberately far): `EvacuationRecommendationEngine` correctly recommends the graph-cost-
cheaper Exit, and correctly migrates its recommendation when the only route to that Exit is blocked --
demonstrating Stair-aware, not proximity-blind, ranking. **No code change was needed or made.**

## Phase 15: Guidance

`GuidancePreservesStairTraversalTests` -- `EvacuationGuidancePlan.ordered_stair_ids` already exists and
is correctly populated for an upper-floor route (`("STAIR-1",)`); `ordered_zone_ids` correctly begins at
the occupant's own upper-floor zone and passes through the Stair's landing zone before reaching the
recommended Exit; a second, deliberately UNREACHABLE "ghost" Stair (no Door connects its own approach
zone to anything) never appears in the generated route. **No code change was needed or made.**

## Phase 16: explicit failure cases

`StairFailureCasesTests` -- all tested explicitly:

| Case | Result |
|---|---|
| Missing `from_floor_id` | Edge still built (connectivity unaffected), `walking_distance=None`, `stair_zero_traversal_distance` WARNING recorded |
| Missing `to_floor_id` | No edge built at all (pre-existing, unchanged behavior — `stair_missing_destination_floor`) |
| Same `from`/`to` floor | Edge still built, `walking_distance=None`, BOTH `stair_same_floor_both_ends` AND `stair_zero_traversal_distance` recorded |
| Deleted floor | No edge built (`invalid_reference`, pre-existing, unchanged) |
| Inactive Stair | **Not a real V1 concept** — `Staircase` has no `active`/`locked` field at all (confirmed via its own `__dataclass_fields__`), unlike Door/Exit; a built Stair edge is always `traversable` |
| Blocked Stair (approach) | Handled via the existing Door/Obstacle mechanism (Phase 9) — Stair edges themselves have no `blocking_obstacles` integration, by design |
| Disconnected Stair approach | Genuinely, honestly unreachable (`PathfindingEngine.dijkstra()` returns `None`) — correct, not a bug |
| Zero vertical height | Now flagged (`stair_zero_traversal_distance`) and degraded safely, never silently instantaneous |
| Malformed legacy Stair | Covered by the same missing-`from_floor_id` case above |
| Only Stair unavailable | Structural unreachability if genuinely the only path — correct, conservative |
| One of multiple Stairs unavailable | The remaining healthy Stair(s) stay fully usable — proven |

## Phase 17: historical zero-duration bug — permanent regression guard

Implemented as the two fixes described above. `HistoricalZeroDurationBugReproductionTests` is the
permanent guard: reproduces the exact historical configuration (`from_floor_id=""`), asserts
`Edge.walking_distance is None` (never `0.0`), asserts the validation issue is recorded, and asserts the
resulting `MultiAgentSimulation` step has `end_time > start_time` — any future regression of either fix
fails this suite immediately.

## Phase 18: Stair-camera architecture regression check

Both fixes are confined to `navigation/graph_builder.py` (Stair EDGE construction) and `simulator/
coordinator.py` (multi-agent traversal duration) — neither file is imported by, nor interacts with, any
part of the Stair-camera perception stack (`models.staircase.StairObservableRegion`,
`camera_calibration.projection.WorldProjection.stair_id`, `live_occupants.occupant.LiveOccupant.
current_stair_id`, `observable_assets.models.ObservableAssetSnapshot`, `camera_coverage`, `stair_flow`).
Confirmed empirically: the complete perception/flow/predictive-parity test suites (151 tests across
`test_*stair_perception*`, `test_*stair_flow*`, `test_*camera_coverage*`, `test_*observable_stair*`,
`test_*observable_asset*`, `test_*stair_observation*`, `test_*stair_predictive_feature*`) all pass
unchanged. Navigation Stair (this audit's own subject — a graph edge with a walking distance) and
Observable Stair geometry (a camera-observable physical region) remain the two separate concepts they
already were.

## Phase 19: end-to-end evacuation

`EndToEndEvacuationTests` — 3 floors, 8 zones, 4 Doors, 2 Exits, 2 chained Stairs, 10 occupants spread
across every floor, one blocked route (an Obstacle across a ground-floor Door), `StairCapacityModel` +
`StairAwareCongestionModel` both active:

```
initial_occupants=10  evacuated=10  stranded=0  max_stair_queue=1  total_evacuation_time≈152.0s
```

**STRANDED = 0**, as expected for a structurally reachable scenario.

## Remaining limitations (disclosed, not fixed by this milestone)

- Stair edges have no obstacle-blocking integration of their own (pre-existing, intentional scope
  boundary — Obstacle → Navigation & Evacuation Connectivity milestone) — only a Stair's *approach*
  (Door/Zone) can be blocked, never the flight itself.
- Staircase has no active/locked/inactive concept in V1 — a Designer wanting to represent a temporarily
  closed stairwell has no dedicated field for it today (out of scope; would be a genuine new-capability
  milestone, not a bug fix).
- A 3+ floor stairwell requires deliberate multi-Staircase authoring (documented in Phase 5) — there is
  no single-object shortcut, and no validation currently warns an operator who forgets the middle flight
  (a possible, narrowly-scoped future Designer-usability improvement, not attempted here).
- The two fixes degrade a broken Stair to `Edge.DEFAULT_TRAVERSAL_COST` (1.0m-equivalent) rather than
  refusing to build the edge at all — a deliberate, conservative choice (never remove connectivity a
  project may already depend on) that trades a small, flagged inaccuracy for guaranteed backward
  compatibility; an operator seeing the new WARNING can correct the underlying data at the source.
