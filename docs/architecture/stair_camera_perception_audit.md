# Stair / Traversal-Asset Perception Architecture Audit

Status: **investigation only**. No production code was modified to produce this document. All claims
below are traced to real, currently-running code (file:line citations throughout), not inferred from
class names or design intent.

## 0. Motivating observation

A real college laboratory building has CCTV cameras aimed directly at stair *flights* (not at the
landings/lobbies next to them, at the flight itself), so people using the stairs can be observed while
physically between floors. This raises the question this audit answers: **can SynEvac honestly
represent a person who is physically on a stair for several seconds, as opposed to instantaneously
crossing a portal between two zones?**

The short answer, established below with citations: **no, not today, at any layer** — not in live
perception, not in `BuildingState`, not in Crowd/Trajectory Intelligence, not in Evacuation Progress —
and, more subtly, **not even fully in the offline simulator's own runtime bridge**, despite the
simulator's raw scheduling data containing real per-edge dwell timestamps. The gap is real, it is
consistent across every layer (nobody is quietly cheating and no layer is quietly worse than another),
and one specific piece of it — a stairwell having no physical footprint in the Building Model — was
already flagged as an open, deliberately out-of-scope item in `docs/architecture/perception_roadmap.md`
before this audit began. This document is the first place that traces the full consequence chain of that
gap end-to-end and proposes what to do about it.

---

## 1. Current architecture: Stair as a navigation Edge (Phase 1)

**`Staircase`** (`models/staircase.py:9-43`) is a plain dataclass:

```python
from_position: tuple = (0.0, 0.0)      # in from_floor_id's own local coordinate space
to_position: tuple = (0.0, 0.0)        # in to_floor_id's own local coordinate space
from_floor_id: str = ""
to_floor_id: str = ""
from_zone_id: str = ""                 # connectivity only, never geometrically inferred
to_zone_id: str = ""
width: float = 1.50
```

Its own docstring is explicit that it is **two points, not a region**: "`from_position`/`to_position` are
each in their OWN floor's coordinate space; there is no meaningful single 'length' or 'center' between
them" (lines 13-18) — this was a deliberate simplification when Staircase was redesigned away from a
single-floor line. There is no polygon, no rectangle, no interior-of-the-shaft geometry, no `capacity`,
no `active`/`blocked` flag, and no occupancy field of any kind. The only derived quantities —
`vertical_height(building)` and `travel_distance(building) = vertical_height / sin(35°)`
(lines 64-87) — are scalars for pathfinding cost, not a walkable path.

A Staircase is stored **once**, in `from_floor.stairs` only, and rendered on the destination floor by a
scan for a matching `to_floor_id` (confirmed at `designer/scene/graphics_scene.py:769-771` and
`2287-2305`) — unlike `Zone`, which owns exactly one `floor_id` and belongs to exactly one floor's
`zones` list.

**NavigationGraphGenerator._add_stair_edges()** (`navigation/graph_builder.py:225-309`) turns each
Staircase into exactly one `Edge`:

```python
Edge(id=stair.id, edge_type=Edge.STAIR, from_node=from_zone.id, to_node=to_zone.id,
     reference=stair, walking_distance=stair.travel_distance(building))
```

**No physical geometry survives into the graph at all** — not even Staircase's own two points. The
resulting `Edge` (`navigation/edge.py`) has: `walking_distance` (yes, a scalar), `width` (derived via
`getattr(reference, "width")`), **no `capacity`** (`edge.py:101-107` — "Only Exit models a capacity
today"), **`traversable` is always `True`** for Stair ("Stair has no blocking flag in V1",
`edge.py:114-116`), **no direction** (every V1 edge is bidirectional, `edge.py:16-18`), and no
independent from/to-floor field (floor-ness is derived from each endpoint `Node.floor_id`).
`blocking_obstacles` is explicitly never populated for Stair (`edge.py:44`).

**Can a simulated occupant be "on Stair S1" for a nonzero period? Yes — and richly so, but only inside
the offline scheduler.** `Occupant` (`simulator/occupant.py:19-39`) has `current_edge_index: int` and a
`TRAVERSING` state. `OccupantTimelineStep` (`simulator/multi_agent_result.py:12-46`) is the actual
per-hop dwell record: `edge`, `start_time`, `end_time`, `queue_wait_time`, populated in
`MultiAgentSimulation._admit_onto_edge()` (`simulator/coordinator.py:321-370`) with a real nonzero
`duration = distance / effective_speed`. Stair edges additionally get **stair-specific physics no other
edge type has**: bidirectional counterflow via `_count_opposing_occupants()`
(`simulator/coordinator.py:374-401`) feeding `StairAwareCongestionModel.speed_factor()`
(`simulator/congestion.py:56-93`), and `StairCapacityModel.derive_stair_capacity()`
(`simulator/capacity.py:52-112`).

**But this richer state is thrown away before it reaches the tick loop any consumer actually reads.**
`simulation_runtime/occupancy_bridge.py`'s `MovementTimelineOccupancyProvider._node_at()` (lines 82-106)
deliberately returns `None` for any time strictly between a step's `start_time` and `end_time`, with an
explicit comment: *"a live occupant contributes to no zone while mid-traversal"* — faithfully replaying
what live tracking would see, not a bug. So the hazard/decision/perception tick loop
(`simulation_runtime/runtime.py::tick()`) sees **nobody, anywhere** for the whole interval a simulated
occupant is on a stair, even though `OccupantTimelineStep` privately knows exactly when they entered and
left. **This means the "simulation has richer stair state than live" framing this audit started with is
half right**: the *raw scheduler data* is richer than anything live perception has, but *neither of
SynEvac's two runtime consumers* (the tick-based `simulation_runtime` bridge, or the real live CCTV
pipeline) currently surfaces it. See §9 for why this matters for predictive-AI parity.

There is **no `EDGE_ENTERED`/`EDGE_EXITED`/`STAIR_ENTERED` event type anywhere in the codebase** (grepped
for all spellings, zero matches) — every occupant lifecycle event on the event bus
(`evacuation_progress/ledger.py:66-69`) is zone-granular (`OCCUPANT_ZONE_CHANGED` carries only
`to_zone_id`).

### Would modeling Stair as a Zone fix this? No — concrete, structural conflicts

1. **`Zone.floor_id` is a single scalar** (`models/zone.py:25`); Staircase intentionally has none,
   only `from_floor_id`/`to_floor_id`, because it spans two floors by design.
2. **`Floor` keeps `zones`/`exits`/`stairs` as separate parallel collections**
   (`models/floor.py:60-62`) with parallel `add_*`/`*_count`/`to_dict`/`from_dict` code; folding Stair
   into Zone means either collapsing this (touching every `floor.zones` iterator, e.g.
   `_add_zone_nodes`, `navigation/graph_builder.py:85-97`) or keeping the duplication *and* the subclass
   relationship, which is redundant.
3. **NavigationGraph's Node=Zone / Edge=Stair split is load-bearing**, not incidental — every pathfinding
   consumer (`PathfindingEngine._relax()`, `pathfinding/engine.py:320-397`) walks `graph.find_neighbors()`
   assuming routing hops through Edges and dwelling happens at Nodes. A Stair-as-Zone would need **two
   Edges** to connect through it (in on floor A, out on floor B) instead of one, changing
   `Route.total_distance` semantics and doubling queueing/capacity checkpoints in
   `MultiAgentSimulation._handle_try_enter_edge()`.
4. **`NavigationGraph._validate_floor_connectivity()`** (`navigation/graph.py:318-357`) keys its
   disconnected-floor check specifically off `edge.edge_type == Edge.STAIR` — a Node has no `edge_type`
   to check.
5. **Capacity/congestion machinery is Edge-shaped**: `StairCapacityModel`/`StairAwareCongestionModel`
   both special-case `edge.edge_type == Edge.STAIR`; `Node` deliberately carries no dynamic-state field
   at all (`navigation/node.py:15-23`). There is no Node-side equivalent to build on.
6. **Designer authoring is two-point-placement, not rectangle-drawing**
   (`designer/scene/graphics_scene.py:694-780`, `StairItem` — a point marker with a width tick, not a
   drag-a-polygon tool like `ZoneRectangle`). Rebuilding this as zone-style authoring is a real UX
   rewrite, not a data-model change.
7. **`scenario_generator/`** reads `floor.stairs` specifically, and separately notes
   `Staircase` has no `active`/`blocked` field at all (`scenario_generator/generator.py:589`) — any
   Zone-style occupancy semantics (`max_occupancy`) would need reconciling from scratch.

**Verdict: Stair must remain a navigation Edge.** A non-zero-dwell-time, camera-observable stair model
that keeps Stair as (or alongside) an Edge — giving it a genuinely new, additive "observable region"
concept instead of converting it into a Node — is a materially smaller, lower-risk change than
Stair-as-Zone. See §10-11.

**Serialization**: `Staircase.to_dict()`/`from_dict()` (`models/staircase.py:91-192`) already has one
precedent backward-compat shim (`from_dict` falls back from `from_position` to the pre-redesign
`start_point` key, and deliberately does *not* trust the old `end_point` as a cross-floor position,
defaulting `to_position = from_position` instead — lines 114-139). There is **no schema-version field**
anywhere in `Staircase`/`Floor`/`Project` serialization; backward compatibility is handled per-field via
`data.get(key, default)`. Any new Stair field must follow this exact convention.

---

## 2. Current architecture: camera perception and world projection (Phase 2)

**`WorldProjector._lookup_zone()`** (`camera_calibration/projection.py:143-160`) is the entire zone
resolution mechanism:

```python
def _lookup_zone(self, floor_id, world_position):
    zones = self._zones_by_floor.get(floor_id, ())
    for zone in zones:
        if zone.polygon:
            if point_in_polygon((x, y), zone.polygon):
                return zone.id
        elif zone.contains(x, y):
            return zone.id
    return None
```

`self._zones_by_floor` is populated, at the real deployment entrypoint
(`scripts/run_physical_camera_validation.py:571-572`), as `zones_by_floor[floor.id] = list(floor.zones)`
— **`Floor.doors`, `Floor.exits`, and Staircase are never passed in.** There is no code path anywhere
that adds a Door/Exit/Stair object into this lookup. A separate helper,
`nearest_navigation_node()` (`projection.py:187-230`), restricts itself even further
(`if node.node_type != "Zone": continue`, line 215) — and the Navigation Graph itself only has three
node types (`Node.ZONE`, `Node.OUTSIDE`, `Node.ASSEMBLY_POINT`); there is no `Node.STAIR`.

This is real, wired runtime code: `live_camera_pipeline/pipeline.py:297-302` calls
`self.world_projector.project(...)` every cycle for every tracked human, and `pipeline.py:361-366` reads
`projection.zone_id`.

**What happens when a camera sees someone physically on a stair flight?** The world position lands
outside every zone polygon on that floor (Staircase geometry is two points, never a zone polygon) →
`_lookup_zone` exhausts the loop and returns `None` → `WorldProjection.zone_id = None` →
`pipeline.py:363` passes it straight through, no substitution → `LiveOccupantManager.update()`
(`live_occupants/manager.py:244-256`) **unconditionally overwrites** `current_zone_id=zone_id` even when
`None` — **there is no "retain last known zone" branch**. The transition is still recorded in history
(`history.with_zone_transition(...)`, line 196), but the occupant's *current-cycle* location is honestly
lost. `compute_occupancy_facts()` (`live_occupants/occupancy.py:110-146`) buckets this occupant into
`unlocalized_occupant_ids` — a deliberate, documented, tested state ("never assign a fabricated zone,
never silently drop from the total headcount," lines 48-54) — not a bug, and not the same as "no
information" (the total observed-occupant count is preserved even though zone attribution is lost).

**`Camera.zone_ids`** (inherited from `EngineeringAsset.zone_ids`, `models/engineering_asset.py:124-135`)
is a real, persisted field — but it is **never read by `WorldProjector`, `LiveCameraPipeline`, or
`LiveOccupantManager`**. Its only live consumers:

| Site | Role |
|---|---|
| `designer/widgets/property_panel.py` (~30 sites) | Cosmetic — zone-assignment checklist UI |
| `designer/widgets/camera_manager_panel.py` | Cosmetic — filters camera list display by zone |
| `camera_manager/manager.py` → `CameraStatus.zone_ids` → `building_state/estimator.py:233-234` | Bookkeeping only (`active_camera_ids`/`offline_camera_ids`), never branched on for localization |
| `perception/providers/ground_truth_camera_provider.py` | **Simulation-mode only** — wired only through `designer/perception_debug_runner.py`, never through `live_runtime/factory.py` or the real pipeline |
| `cross_camera_identity/topology.py::build_topology_from_navigation_graph` | Real, operational code — but **orphaned**: called only from tests, never from any launcher |

**No mechanism exists for a Camera to declare "I observe Stair S1" or "I observe Door D1."**
`models/connectable_space.py`'s `CONNECTABLE_SPACE_TYPES = (Zone, AssemblyPoint)` structurally excludes
Door/Exit/Stair from ever appearing in the zone-assignment checklist that produces `Camera.zone_ids` in
the first place.

**Cross-camera identity** (`cross_camera_identity/`) is a real, tested, **topology + time-window rule
matcher** (`RuleBasedCrossCameraMatcher`, `cross_camera_identity/matching.py:16-135`) — explicitly and
mechanically **not** appearance ReID (its own docstring disclaims image embeddings/facial
recognition/deep models, and this exclusion is enforced by
`tests/test_cross_camera_identity_architecture_guards.py`). `CameraTopology` construction from the
Navigation Graph exists (`topology.py:166-221`) and could, in principle, derive plausible walking-time
windows between two cameras' declared coverage — **but it is never constructed by any real launcher**
(`live_runtime/factory.py:106` exposes `cross_camera_identity_resolver` as an unset `Optional` seam; the
real single-camera deployment wires nothing there — `live_runtime_launcher/human_detector_wiring.py:29-33`
says so explicitly). **No appearance-based ReID exists at all** — not stubbed, not faked, architecturally
excluded by design and by guard test.

---

## 3. Live occupant semantics (Phase 3)

`LiveOccupant` (`live_occupants/occupant.py:71-124`, frozen dataclass) carries `current_zone_id`,
`current_floor_id`, `world_position`, `world_velocity`, `behavior`, `status`, `history`,
`world_position_provenance`, plus human-classification fields. **There is no `current_edge_id`,
`current_stair_id`, `current_traversal_asset_id`, or any equivalent** — confirmed absent by full read of
the class, and by a repo-wide grep for `asset_occupancy|edge_occupancy|stair_occupancy|traversal_state|
current_edge|current_asset|transition_asset|portal_occupancy`, whose only hits are in the **offline**
`predictive_dataset/`, `simulator/`, `ground_truth/` packages — none in the live perception stack.

When an occupant leaves one zone's polygon before entering another, `current_zone_id` becomes `None`
that same cycle (§2) and the occupant is bucketed into `unlocalized_occupant_ids` — this is the
*closest* existing concept to "on a traversal asset," but it is a dead end: nothing downstream re-derives
"which stair are they probably on" from an unlocalized occupant's last-known zone/velocity.

---

## 4. BuildingState / perception boundary (Phase 4)

`compute_occupancy_facts()` (`live_occupants/occupancy.py:110-146`) groups occupants strictly by
`current_zone_id`/`current_floor_id`. `BuildingState.zone_occupancy: OccupancySnapshot`
(`building_state/models.py:130`, `occupancy/snapshot.py`) is keyed by `node_id` — and the Navigation
Graph's node types are `ZONE`/`OUTSIDE`/`ASSEMBLY_POINT` only. **`BuildingState` cannot express
"STAIR-1: occupant_count = 6" today, structurally** — there is no stair `node_id` to key it by, and no
occupant ever carries a stair identity to aggregate in the first place (§3).

**This is not a dead end architecturally, though.** `BuildingState` already has an established,
repeated pattern for exactly this shape of problem — an **additive, `Optional`, sibling snapshot**,
populated as a pure passthrough in `BuildingStateEstimator.estimate()` (`estimator.py:123-139`), never
computed inline:

```python
facp_status: Optional[FACPSnapshot] = None            # models.py:164
control_status: Optional[ControlStateSnapshot] = None  # models.py:176
fire_safety_status: Optional[FireSafetyStatusSnapshot] = None  # models.py:189
fire_water_status: Optional[FireWaterInfrastructureSnapshot] = None  # models.py:206
```

`ControlStateSnapshot.entries[].target_id` (`building_control/snapshot.py:17`) is direct precedent for
an **asset-keyed** (not node-keyed) sub-snapshot — it already accepts any asset id, including a stair
(stair pressurization is one of its `ControlSystemType` members). A `stair_occupancy:
Optional[StairOccupancySnapshot]` field would fit this exact, already-proven pattern.

---

## 5. Crowd Intelligence (Phase 5)

`AssetApproachMetrics` (`crowd_intelligence/models.py:143-178`) is the shared Door/Exit/Stair shape.
`stair_sides()` (`crowd_intelligence/flow.py:73-78`) returns **two degenerate, zero-length point
segments** — one at `from_position`, one at `to_position` — i.e. the two landings only, never the
stairwell interior:

```python
AssetSide(floor_id=stair.from_floor_id, segment_start=stair.from_position, segment_end=stair.from_position)
AssetSide(floor_id=stair.to_floor_id,   segment_start=stair.to_position,   segment_end=stair.to_position)
```

"Approaching Stair S1" (`occupants_near_asset()`/`evaluate_approach()`, `flow.py:125-180`) means
**proximity to a landing point** (within `DEFAULT_APPROACH_REGION_DEPTH = 3.0` m,
`crowd_intelligence/queue.py:11`) with decreasing distance across the last two position samples — never
"physically between the landings."

| Metric | Status |
|---|---|
| Stair occupancy (headcount physically on the stair) | **NOT SUPPORTED** |
| Stair flow rate | **NOT SUPPORTED** (no `ExitFlow`-equivalent exists for Stair) |
| Direction of movement through stair | **NOT SUPPORTED** |
| Stair traversal time | **NOT SUPPORTED** |
| Stalled-while-on-stair | **NOT SUPPORTED directly** — `trajectory_intelligence.anomaly.movement_stalled()` is route-level and stair-agnostic; it can flag "stalled somewhere near this route segment," never "stalled specifically on the stair" |

**"10 approaching Stair S1" and "10 physically ON Stair S1" cannot currently be distinguished** — both
would show up (if at all) as landing-proximity counts, because there is no signal for "between the
landings" at all.

---

## 6. Trajectory Intelligence (Phase 6)

`LiveOccupant.world_position` and `PositionSample.world_position` (`live_occupants/history.py:59-61`) are
flat 2D — **no `floor_id`/z field on a position sample**. `compute_movement_facts()`
(`trajectory_intelligence/trajectory.py:24-92`) computes distance/speed/direction from consecutive raw
samples **with no floor-awareness at all**. If a stair traversal happens between two samples, this
computes a Euclidean distance between two points from **two unrelated floor-local coordinate systems**
(per Staircase's own documented convention, §1) as if they were coplanar — a physically meaningless
"jump" can be summed into `distance_travelled`/`current_speed`/`movement_direction` for that one cycle,
unguarded. `crowd_intelligence/flow.py`'s own `ApproachEvidence` docstring explicitly names and mitigates
this exact risk for its own two-sample distance check (`flow.py:106-118`) — `trajectory.py` has no
equivalent guard.

**Route-level logic is, however, already correct.** `_floor_transition_uncertain()`
(`trajectory_intelligence/route_progress.py:151-177`) checks `_has_direct_stair_edge()` — if a valid
Stair edge connects the two zones, the transition is trusted and `route_progress_status` is computed via
graph distance (never raw Euclidean), never flagged as a teleport/stall. **The asymmetry is specifically
in `compute_movement_facts()`'s raw geometry facts, not in route-progress semantics.**

---

## 7. Evacuation Progress (Phase 7)

`EvacuationProgressEngine` builds `self._zones` and `self._exits` only
(`evacuation_progress/engine.py:120-121`) — **no `self._stairs`, ever.** `EvacuationProgressSnapshot`
(`evacuation_progress/models.py:159-196`) has `zones: Mapping[str, ZoneClearance]` and
`exits: Mapping[str, ExitFlow]` — no third stair mapping. A grep for `stair`/`Stair` across
`evacuation_progress/` returns **zero matches**. `EvacuationLedger` subscribes only to
`OCCUPANT_CREATED/UPDATED/ZONE_CHANGED/EXITED` and attributes crossings to the nearest **Exit** geometry
only (`_nearest_exit_id`, built from `exit_sides()`, never `floor.staircases`).

**Stair usage is not counted at all** — not from graph inference, not from live presence. No current
stair occupancy, no observed traversal time, no entry/exit rate split — none of these are supported,
because no stair-entry/stair-exit event pairing exists anywhere in the ledger. The only indirect artifact
is `TrajectoryResult.zone_transition_history` (a bare zone-id sequence) from which a consumer *could*
infer "this occupant's route crossed a floor boundary via a Stair edge" — but `evacuation_progress` never
reads or aggregates this into any stair-specific count today.

---

## 8. Simulation vs. live semantics (Phase 8)

| Fact | Simulation (`simulator/`) | Live (`live_occupants/`) |
|---|---|---|
| "Currently on this edge" | `Occupant.current_edge_index` (`simulator/occupant.py:38`) | absent |
| Edge dwell interval | `OccupantTimelineStep.start_time/end_time` (`multi_agent_result.py:19-26`) | absent |
| Queue admission bookkeeping | `queue_wait_time`, `join_time` | proxy only: `behavior == STATIONARY` within 3m of a landing (`crowd_intelligence/queue.py`) |
| Stair counterflow | `opposing_occupants` (`coordinator.py:374-401`) | absent, and not proxied by anything |

This asymmetry is real, but it is **not filled by reading simulation edge-state live** — the predictive
feature layer instead substitutes the geometric world-position + STATIONARY-behavior proxy above for
Door/Exit/Stair uniformly, and this substitution is honestly disclosed per-field (see §9). Separately,
and importantly: even `simulation_runtime`'s own tick-based bridge (§1,
`MovementTimelineOccupancyProvider`) discards the rich `OccupantTimelineStep` data before any hazard/
decision/perception consumer ever sees it — so the real gap is between **raw simulator scheduling data**
and **both runtime consumers** (sim-runtime and live), not strictly "simulation is better than live."

---

## 9. Connection to predictive AI work (Phase 14)

Every candidate feature is tagged with a `FeatureAvailability` in `predictive_dataset/schema.py` (v1.0)
and `schema_v4.py` (v4.0), reusing the same `AIFeatureField` vocabulary as the whole-building
`ai_features/feature_schema.py` framework "rather than inventing a second... independently-defined...
feature set" (`schema.py:14`).

| Feature | Sim computation | Live computation | Classification |
|---|---|---|---|
| `candidate_queue_length` | Exact `OccupantTimelineStep` interval test | `AssetApproachMetrics.queue_candidate_count` (STATIONARY-near-landing proxy) | **LIVE-ESTIMABLE-TODAY**, proxy not ground truth — weakest for Stair specifically (see below) |
| `candidate_recent_flow_rate` | Count of `end_time`s in trailing 60s | Door/Stair: `zone_transitions` count (zone-based, sidesteps edge-state gap entirely) | **LIVE-COMPUTABLE-TODAY** |
| `candidate_congestion_trend` | Demand-proxy delta | `TrendTracker` over `AssetApproachMetrics.trend` | **LIVE-COMPUTABLE-TODAY**, inherits queue proxy's limitations |
| `candidate_adjacent_zone_occupancy` | `OccupancySnapshot` at edge's from-node | `occupancy_facts.occupant_ids_by_zone` | **LIVE-COMPUTABLE-TODAY**, pure zone occupancy |
| `candidate_walking_distance` | `edge.walking_distance` | identical, structural | **LIVE-COMPUTABLE-TODAY**, trivially |
| Graph-context (betweenness, is_bridge, catchment) | `graph_context_v4.py`, structural | identical function, same `Building` object | **LIVE-COMPUTABLE-TODAY**, zero occupant dependence |

The one clearly `SIMULATION-ONLY` stair signal — `opposing_occupants` counterflow — is already labeled
as such in `docs/architecture/live_crowd_intelligence.md` §15's own data-source classification table, and
**it is not present in any `predictive_dataset` schema at all**, so it is not currently a sim/live
mismatch inside the trained model's actual inputs. It is a separate, already-documented gap in
simulation-realism reuse.

**The real Stair-specific gap lives in the Building Model, not the feature code.** Because
`stair_sides()` only instruments the two landing points, `queue_candidate_count`/`approaching_count` for
a Stair candidate are blind to an occupant physically mid-flight, several meters from either landing —
whereas simulation's `OccupantTimelineStep` counts that occupant as traversing regardless of exact
position. **This exact gap was already named, independently, before this audit**, in
`docs/architecture/perception_roadmap.md:126-128`:

> "Stairwell-shaft representation — corridors already work today (they're Zones), but a stairwell shaft
> between landings has no Zone of its own in the Building Model at all. A Building-Model-level gap, not
> something any Perception-side option can fix on its own."

A direct `observed_stair_occupancy` signal from a real stair camera would replace/augment the
landing-proximity proxy specifically inside `crowd_intelligence/queue.py`/`flow.py`'s Stair path — **not**
the graph-context features (already structural) and **not** `candidate_recent_flow_rate` (already
zone-transition-based, unaffected either way). This gap predates and is silently inherited by every
Predictive Dataset V1-V4 / Localized Predictive Model V1-V3.1 milestone — it was not introduced by that
work, and none of that work's verdicts (PR-AUC numbers, generalization findings) are invalidated by it.
But it does mean **Stair candidate queue/congestion features have always been measuring "near a landing,"
never "on the stair,"** in both the simulated and (proxy) live paths — an honest limitation, not a bug,
but one worth naming explicitly, which no prior document did until now.

---

## 10. Epistemic honesty / provenance mechanisms already in place (Phase 15)

SynEvac already has a strong, consistent, repo-wide convention for exactly the OBSERVED / INFERRED /
UNKNOWN distinction this audit's brief asks for — it does not need to be invented:

- `LiveOccupant.world_position_provenance` (`NONE`/`UNVALIDATED`/`VALIDATED`)
- `unlocalized_occupant_ids` — never a fabricated zone, never a silent drop
- `OccupancyEstimator`'s `UNOBSERVED` handling + max-not-sum duplicate resolution
- `position_available`/`position_coverage_fraction` flags throughout `crowd_intelligence`
- The published data-source classification table itself
  (`docs/architecture/live_crowd_intelligence.md` §15): `LIVE-OBSERVABLE`, `LIVE-OBSERVABLE, requires
  calibration`, `SIMULATION-STYLE ESTIMATE`, `ESTIMATED`, `SIMULATION-ONLY`,
  `UNAVAILABLE WITHOUT CALIBRATION → genuine zero`

Any future stair-camera signal should be slotted into this **same** taxonomy (e.g.
`observed_stair_occupancy` = `LIVE-OBSERVABLE` once a camera + stair region genuinely exist; "stair is
congested" = a separate `ESTIMATED` field built on top, never conflated with the raw count) rather than
inventing a parallel one.

---

## 11. What could a stair camera honestly measure? (Phase 11)

| Signal | Classification | Why |
|---|---|---|
| Current stair occupancy | **REQUIRES SMALL REPRESENTATION CHANGE** | Needs an occupant-on-asset location field + a stair observable region; the surrounding machinery (`WorldProjector`, `LiveOccupant`, `BuildingState`'s sibling-snapshot pattern) already exists and is proven for other asset types |
| Entry / exit count | **DERIVABLE WITH EXISTING DATA** once occupancy representation exists | Same mechanism as `ExitFlow.unique_exited_count`, applied to a stair boundary crossing |
| Entry / exit rate | **DERIVABLE WITH EXISTING DATA** | Same reasoning as `candidate_recent_flow_rate`, already live-computable for Door/Stair today |
| Dominant direction | **REQUIRES SMALL REPRESENTATION CHANGE** | Needs an ordered entry-side/exit-side per occupant |
| Bidirectional/opposing counts | **REQUIRES SMALL REPRESENTATION CHANGE** (crude) / **REQUIRES NEW PERCEPTION CAPABILITY** (precise per-lane, matching sim's `opposing_occupants`) | Crude counts follow directly from occupancy+direction; true lane-level counterflow needs new tracking precision |
| Average movement speed | **DERIVABLE WITH EXISTING DATA** once stair-local position/timestamps exist | Reuse trajectory speed logic, with the same floor-aware guard proposed in §6 |
| Traversal time | **REQUIRES SMALL REPRESENTATION CHANGE** (single camera covering the whole flight) / **REQUIRES NEW PERCEPTION CAPABILITY** (split across two cameras — needs cross-camera identity continuity, §12) | |
| Stalled occupants | **DERIVABLE WITH EXISTING DATA** | `trajectory_intelligence.anomaly.movement_stalled()` logic already exists; would apply directly once stair-local dwell exists |
| Occupancy / congestion trend | **DERIVABLE WITH EXISTING DATA** | `crowd_intelligence.trends.TrendTracker` is already generic over any bounded history |

---

## 12. Camera arrangement cases (Phase 12)

- **Case A (one camera, whole flight)**: everything in §11 marked "derivable"/"small change" becomes
  honestly measurable with single-camera tracking alone — no cross-camera identity needed.
- **Case B (camera sees only the entrance/landing)**: this is, in effect, **what SynEvac already has
  today**, silently, via the landing-proximity `AssetApproachMetrics` (§5) — it can see "entered," not
  true interior occupancy or exit-side flow.
- **Case C (camera at top + bottom, no full-flight visibility)**: entry count at one end, exit count at
  the other, but pairing them into one person's traversal time requires cross-camera identity continuity
  (§13/Phase 13) — a real, tested topology/time-window matcher exists in principle but is not wired into
  any real launcher today.
- **Case D (camera sees corridor + partial stairwell)**: needs the region-disambiguation from §9's
  Option 2/C (zone-region vs. stair-region lookup on the same frame); without hysteresis (none exists
  today — `LiveOccupantManager` overwrites `current_zone_id` unconditionally every cycle), a detection
  near the boundary risks flip-flopping cycle to cycle.
- **Case E (incomplete coverage)**: must honestly report `UNKNOWN`/`UNAVAILABLE WITHOUT CALIBRATION` for
  uncovered stair segments — direct precedent already exists (`OccupancyEstimator`'s `UNOBSERVED`
  handling, `position_available` flags) — never fabricate a zero.

---

## 13. Cross-camera identity boundary (Phase 13)

- **Works today**: nothing, in the real deployment — `CameraTopology` is never constructed by any
  launcher (§2).
- **Exists as a library, ready to wire**: `RuleBasedCrossCameraMatcher` + `build_topology_from_navigation_
  graph` — topology + time-window plausibility only.
- **Requires appearance ReID**: disambiguating near-simultaneous transits at the same boundary, or any
  case where topology/timing alone is ambiguous. **This does not exist anywhere in the codebase — not
  stubbed, not faked** — and is architecturally excluded by a guard test, so adding it would be a
  genuinely new component, not an upgrade.
- **Cannot currently be guaranteed**: "Camera A sees TRACK-17 enter Stair S1, Camera B sees the same
  person leave" as a *named-asset* event at all — because neither camera, nor `WorldProjector`, nor
  `LiveOccupant` can currently name a stair (§2/§3). This is a bigger gap than ReID; the ReID gap only
  matters once stair-naming exists.

---

## 14. Architectural options (Phase 16)

**Option 1 — Landing-only, minimal (do nothing structural).** Zero new code. **Rejected**: does not
honestly represent a whole-flight stair camera; an occupant mid-flight remains permanently unlocalized.

**Option 2 — Stair observable region + `BuildingState` sibling snapshot (recommended, see §15).**
Staircase gains an optional physical footprint, distinct from its existing `from_position`/`to_position`
connectivity points (Edge/NavigationGraph/pathfinding/simulation untouched). `WorldProjector` gains a
second lookup stage (zone, then stair-region). `LiveOccupant` gains an optional `current_stair_id`
(mirroring `current_zone_id`'s own `Optional` convention). `BuildingState` gains
`stair_occupancy: Optional[StairOccupancySnapshot]`, following the exact `facp_status`/`control_status`
precedent. Crowd Intelligence gets a genuine occupancy/flow/direction signal *alongside* (not replacing)
the existing landing-proximity "approaching" signal. Trajectory gets a floor-aware guard mirroring
`route_progress.py`'s existing one. Fully additive; old projects simply have no stair-region data
(`None`, same as an unconfigured FACP).

**Option 3 — Full traversal-asset observation layer (generalize to Door/Exit too).** Same idea as Option
2, generalized into a new perception sibling covering every edge-hosted asset uniformly. **Rejected for
now**: real-world trigger was specifically stair cameras; Door/Exit share the same landing-proximity
limitation but that is not what was reported from the field, and generalizing now is scope creep beyond
what this audit's evidence supports. Option 2 is this option's necessary first instance regardless.

**Option 4 — Camera declares "I observe this stair," no new geometry (cheapest hack).** Treat every
detection from a stair-flagged camera as automatically on that stair. **Rejected**: dishonest for Case D
(camera framing corridor + stair) and any mixed-coverage case; violates the "never fabricate" convention
this codebase already enforces everywhere else (`world_position_provenance`, `unlocalized_occupant_ids`).

Option 2 is the only one of the four that satisfies the audit's own explicit requirement: **Stair =
navigation Edge/traversal asset** and **Stair = observable physical asset** as two separate,
non-conflated roles on the same object.

---

## 15. Recommended architecture (Phase 17)

**Option 2.** It is the smallest change that is still geometrically honest: it does not touch
`NavigationGraph`, pathfinding, routing, or the simulator's scheduling logic at all (§1's conflicts are
entirely avoided because Stair never becomes a Zone or a Node); it reuses `BuildingState`'s already-proven
additive-sibling-snapshot pattern (§4) instead of inventing a new one; it reuses the existing
`Optional`/never-fabricate convention already on `LiveOccupant`/`unlocalized_occupant_ids` (§2-3) instead
of inventing a new failure mode; and it slots directly into the existing OBSERVED/ESTIMATED/
SIMULATION-ONLY taxonomy (§10) instead of inventing a new one. It keeps the existing landing-proximity
`AssetApproachMetrics` signal intact as "approaching" — genuinely distinct from, and not replaced by, the
new "on the stair" signal, honoring the audit's own instruction not to conflate them.

---

## 16. Staged implementation plan (Phase 18 — NOT executed in this milestone)

1. **Observable stair-region representation** — additive geometry field(s) on `Staircase`
   (`models/staircase.py`), per-floor-local-coordinate-space (consistent with the existing
   `from_position`/`to_position` convention), serialized via the existing `.get(key, default)` pattern.
   Zero `Edge`/`NavigationGraph`/pathfinding/simulator changes.
2. **Calibration / world-projection asset localization** — extend `WorldProjector`
   (`camera_calibration/projection.py`) with a stair-region lookup stage alongside `_lookup_zone`,
   returning an optional stair id on `WorldProjection`; extend the physical-camera validation wiring
   (`scripts/run_physical_camera_validation.py`-style) to supply stair regions.
3. **LiveOccupant integration** — add `Optional[str] current_stair_id` (or a generalized
   `current_traversal_asset_id`) to `LiveOccupant`; wire `live_camera_pipeline/pipeline.py` +
   `LiveOccupantManager.update()` to set/clear it each cycle with the same unconditional-overwrite,
   never-fabricate convention `current_zone_id` already has.
4. **Stair occupancy/flow snapshot** — new `StairOccupancySnapshot` sibling type +
   `BuildingState.stair_occupancy: Optional[...]`, populated as a pure passthrough in
   `building_state/estimator.py`, mirroring `facp_status`/`control_status` exactly.
5. **Crowd/Trajectory integration** — real stair occupancy/flow/direction/trend in `crowd_intelligence`
   from `current_stair_id` (existing landing-proximity `AssetApproachMetrics` kept, not replaced); a
   floor-aware guard in `trajectory_intelligence.trajectory.compute_movement_facts()` mirroring
   `route_progress.py`'s existing `_floor_transition_uncertain()`.
6. **Designer support** — authoring UI for the stair region (or a computed default from
   `from_position`/`to_position`/`width`), calibration UI recognizing the new region type, Camera property
   panel showing genuine observed-stair coverage.
7. **Simulation/offline parity** — a Ground Truth adapter (mirroring `GroundTruthCameraProvider`) that can
   populate the same `StairOccupancySnapshot` from `simulator.OccupantTimelineStep` data for offline
   validation, plus explicit sim/live parity tests mirroring the existing
   `tests/test_predictive_dataset_parity.py` pattern.
8. **Real-camera validation** — extend the existing physical CCTV field-validation runner with a
   dedicated stair-camera calibration/validation mode, following that runner's existing
   no-fabrication, progressive-mode precedent.

---

## 17. Test plan required before implementation is trustworthy (Phase 19)

- Occupant enters stair (zone→stair-region transition; `current_stair_id` set; decide and test explicitly
  whether `current_zone_id` is cleared or retained during the crossing)
- Occupant remains on stair for N cycles (dwell accumulates; no flicker/false zone reassignment)
- Occupant exits stair onto destination zone (`current_stair_id` clears; landing zone set; no double-count
  in either zone during the transition)
- Two occupants moving the same direction on the stair (both counted correctly)
- Opposing-direction (bidirectional) movement on the stair (both counted; no cancellation/undercounting)
- Camera sees only part of the stair (Case B/D — partial coverage honestly reflected, no fabricated
  full-flight occupancy)
- Detection temporarily lost mid-stair (occupant not silently dropped from total headcount, consistent
  with existing `TEMPORARILY_LOST`/`unlocalized` conventions)
- Uncalibrated camera aimed at a stair (stair occupancy `UNAVAILABLE WITHOUT CALIBRATION`, never a
  fabricated zero)
- Stair with no observing camera at all (`stair_occupancy` stays `None` entirely, same as an unconfigured
  FACP — not a missing-key error)
- Obstacle/hazard reroutes an occupant away from the stair (existing Obstacle-blocks-Edge logic
  unaffected; a routed-away occupant never legitimately gets a `current_stair_id` for that stair)
- Multiple cameras observe the same stair (Case C — no double counting of the same `occupant_id`;
  traversal-time pairing marked `ESTIMATED`/`UNKNOWN` where cross-camera identity continuity isn't
  guaranteed, never fabricated)
- Floor transition via stair correctly recognized by Trajectory Intelligence (not misread as a
  teleport/route-uncertainty artifact, per the guard proposed in step 5 above)
- Save/reload of a project containing the new stair-region data (round-trips through `to_dict`/`from_dict`)
- Legacy project compatibility (a pre-existing `.syn` file with a `Staircase` lacking the new region field
  loads without error; that stair's occupancy capability is simply absent/`None`, same convention as the
  existing `start_point`/`end_point` backward-compat shim)

---

## 18. Final report

1. **How is Stair represented today?** A `Staircase` model object: two points
   (`from_position`/`to_position`, each in its own floor's local coordinate space), floor/zone
   connectivity references, and a width — no polygon, no capacity, no occupancy field. It is compiled
   into exactly one bidirectional `Edge` (`edge_type=STAIR`) between two Zone nodes in the Navigation
   Graph, carrying only a scalar `walking_distance` — no physical geometry survives into the graph at all.
2. **Can a live occupant currently be explicitly "on Stair S1"?** No. `LiveOccupant` has no
   `current_stair_id`/`current_edge_id`/traversal-asset field of any kind — confirmed absent by direct
   read of the class.
3. **What happens today when CCTV detects someone physically on a stair?** `WorldProjector` resolves only
   against `Floor.zones`; a position on stair geometry matches no zone polygon, so zone resolution
   honestly returns `None`. The occupant is not dropped or fabricated into an adjacent zone — it is
   correctly counted in the total headcount and explicitly bucketed as `unlocalized`, every cycle it
   remains on the stair.
4. **Can `BuildingState` represent stair occupancy?** Not today — `OccupancySnapshot` is node-keyed and
   there is no Stair node type. But `BuildingState` already has a proven additive-sibling-snapshot pattern
   (`facp_status`, `control_status`, etc.) that a `stair_occupancy` field would fit cleanly.
5. **Can Crowd Intelligence represent actual stair occupancy?** No — `AssetApproachMetrics` for Stair
   measures proximity to the two landing points only (`stair_sides()`'s degenerate point geometry), never
   interior/on-stair presence. It cannot distinguish "approaching" from "physically on."
6. **Can Trajectory Intelligence understand stair traversal?** Partially. Route-level progress logic
   (`route_progress.py`) already correctly recognizes a stair crossing as legitimate via
   `_has_direct_stair_edge()`. Raw movement-facts computation (`compute_movement_facts()`) is floor-blind
   and can produce a spurious jump artifact across a stair transition — an existing, unguarded gap this
   audit surfaced.
7. **Can Evacuation Progress measure real stair usage?** No — `EvacuationProgressEngine` tracks zones and
   exits only; there is zero stair handling anywhere in that package.
8. **Does simulation have richer stair traversal state than Live?** The raw scheduler
   (`OccupantTimelineStep`) does, including stair-specific counterflow physics — but the tick-based
   `simulation_runtime` bridge deliberately discards it before any consumer sees it ("contributes to no
   zone while mid-traversal"), so the two actual runtime paths (sim-runtime and live) are closer to
   symmetric than the raw scheduler data suggests.
9. **Can Camera currently target/observe a Stair explicitly?** No — `Camera.zone_ids` exists but is
   inert for the real live pipeline (only read by cosmetic UI, bookkeeping, an orphaned simulation-mode
   provider, and an unwired cross-camera-topology builder), and there is no Door/Exit/Stair equivalent of
   `zone_ids` at all.
10. **What can one stair camera honestly provide?** With a single camera covering the whole flight
    (Case A): occupancy, entry/exit counts and rates, average speed, stalled-occupant detection, and
    occupancy/congestion trend — all classified `DERIVABLE WITH EXISTING DATA` or `REQUIRES SMALL
    REPRESENTATION CHANGE` in §11, once the stair-region/occupant-location gap is closed.
11. **What requires two cameras?** Splitting entry (top) and exit (bottom) observation for a flight
    longer than one camera's view (Case C).
12. **What requires cross-camera identity?** Pairing a Case-C entry sighting with its matching exit
    sighting into one person's traversal time. The topology/time-window matcher exists as a library but
    is unwired in every real launcher; genuine appearance ReID for disambiguating ambiguous cases does not
    exist anywhere in the codebase.
13. **Should Stair become a Zone?** No — concrete, multi-layer conflicts (single-floor `Zone.floor_id`,
    Floor's parallel collections, the Node=Zone/Edge=Stair pathfinding bipartite, Edge-shaped
    capacity/congestion machinery, two-point Designer authoring UX) make this a large, high-risk change
    for no benefit the smaller option doesn't already provide.
14. **Should Stair remain a navigation edge?** Yes, unconditionally — every routing/simulation/capacity
    system depends on this.
15. **Can Stair simultaneously remain a navigation edge while gaining an observable physical
    representation?** Yes — this is exactly what Option 2 (§14-15) proposes: an additive observable
    region alongside the existing Edge, never replacing or restructuring it.
16. **Smallest clean architecture?** Option 2 — stair observable region + `WorldProjector` fallback lookup
    + `current_stair_id` on `LiveOccupant` + `stair_occupancy` sibling snapshot on `BuildingState`, all
    additive, all reusing existing conventions (§15).
17. **Would this improve the scientific validity of Stair congestion features used by predictive AI?**
    Yes, materially — it would replace the landing-proximity-only queue/congestion proxy (§9) with a real
    on-stair occupancy signal for the one candidate type whose approach-metric geometry was always
    weakest, without touching any already-computable feature (graph-context, walking distance,
    zone-based flow rate).
18. **Does this need to be resolved before continuing the next predictive-AI feature milestone?** Not as
    a hard blocker — every currently-classified `LIVE-COMPUTABLE-TODAY` feature (§9) remains valid and
    honestly labeled regardless. But any *new* Stair-congestion feature work should be aware that
    `candidate_queue_length`/`candidate_congestion_trend` for Stair specifically are landing-proximity
    proxies, not on-stair measurements — continuing to build on top of them without closing this gap is
    fine short-term, but building a *new* Stair-specific feature that implicitly assumes on-stair accuracy
    would be premature.
19. **What exact implementation milestone should come next?** Stage 1 of §16 — the observable stair-region
    representation on `Staircase` — is the correct next step if/when this is picked up, since every later
    stage depends on it and it alone touches zero existing runtime behavior.
20. **Full-suite result.** See below.
21. **Commit hash.** This document (`docs/architecture/stair_camera_perception_audit.md`) is the only
    file this audit adds or modifies. No production code, tests, or other files were changed. Latest
    commit at the start of this audit: `c3d863522a3e293bee0c7776a2e3c50f5875923f`. Committing this
    document is left to the user, since the working tree currently has substantial unrelated in-progress
    work (uncommitted YOLO/RTSP/occupant-panel changes) that this audit did not touch and should not be
    bundled with.

### A-H

- **A. Is there a genuine stair-perception architecture gap?** Yes — confirmed at every layer (calibration
  → live occupant → BuildingState → Crowd Intelligence → Trajectory → Evacuation Progress), independently
  cross-checked by four separate research passes in this audit.
- **B. Is this a UI/modeling issue, or does it affect live intelligence?** It affects live intelligence
  directly — Crowd Intelligence's Stair congestion signal is landing-proximity-only, which is a real
  measurement limitation, not merely a missing UI affordance.
- **C. Are real stair CCTV cameras currently underutilized by SynEvac?** Yes — a camera aimed at a stair
  flight today can only contribute to the total headcount (via `unlocalized_occupant_ids`); it cannot
  contribute any zone-, stair-, or asset-level signal at all.
- **D. Does fixing this require redesigning NavigationGraph?** No — the recommended option (§15) makes
  zero changes to `NavigationGraph`, `Edge`, or pathfinding.
- **E. Does fixing this require turning Stair into a normal Zone?** No — and §1/§14 show concretely why
  that would be the wrong direction.
- **F. Can we obtain direct measured stair occupancy/flow without ML?** Yes, for the geometric/positional
  parts — this is the same calibration + world-projection + polygon-membership technique already used for
  Zones, generalized to a stair region. Human *detection* still depends on YOLO (already in the pipeline
  today); no *new* ML model is required for the stair-specific parts of this gap.
- **G. Could this eventually improve predictive congestion AI?** Yes — see final-report item 17.
- **H. Should we implement this before continuing deeper predictive-AI work?** Not as a hard prerequisite
  — but Stage 1 (§16, the Staircase observable-region representation) is low-risk, self-contained, and
  worth doing soon, since it unblocks everything else in this document and every later stage depends on
  it existing first.
