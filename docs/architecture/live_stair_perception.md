# Observable Stair Perception

Status: implemented (foundation only, per this milestone's own scope). Builds directly on
`docs/architecture/stair_camera_perception_audit.md` (the architecture audit this milestone
implements Stage 1 of). No predictive model, Recommendation scoring, Voice/Signage dispatch,
Building Control execution, or NavigationGraph redesign is touched by this milestone.

---

## 1. Problem discovered from the real laboratory CCTV layout

A real college laboratory building has CCTV cameras aimed directly at stair *flights* — not just the
landings next to them — so people using the stairs can be observed while physically between floors.
The prior audit (`stair_camera_perception_audit.md`) traced the full pipeline and found that SynEvac
had no way to represent this: a detection landing on stair geometry resolved to no zone at all, and
nothing downstream — `LiveOccupant`, `BuildingState`, Crowd Intelligence, Trajectory Intelligence,
Evacuation Progress — had any concept of "on a stair."

## 2. Previous architecture

Stair was represented purely as a `Staircase` model (two points, `from_position`/`to_position`, each
in its own floor's local coordinate space) compiled into exactly one bidirectional `Edge` in the
Navigation Graph. No physical geometry survived into the graph at all — not even Staircase's own two
points. `WorldProjector` only ever resolved a calibrated detection against `Floor.zones`; a point on
stair geometry produced `zone_id=None`, honestly bucketed as `unlocalized`, but with zero attribution
to the stair itself.

## 3. Why Stair remains a navigation traversal asset

Unchanged by this milestone, and deliberately so. The prior audit found concrete, multi-layer
conflicts with making Stair a Zone: `Zone.floor_id` is a single scalar (Staircase spans two floors by
design), `Floor` keeps `zones`/`stairs` as separate parallel collections, `NavigationGraph`'s
Node=Zone / Edge=Stair split is load-bearing for every pathfinding/simulation consumer, and Designer
authoring is a fundamentally different (two-point vs. rectangle) workflow. This milestone verified
those conflicts still hold and did not touch `navigation/graph.py`, `navigation/graph_builder.py`,
`navigation/edge.py`, or `navigation/node.py` at all — `Node.STAIR` still does not exist, `Edge.STAIR`
still does (mechanically proven by `tests/test_observable_stair_perception_architecture_guards.py`).

## 4. Observable physical Stair representation

`models/staircase.py` gains a new `StairObservableRegion` dataclass — an axis-aligned rectangle
(`center_x`, `center_y`, `width`, `depth`) in one floor's local coordinate space — and `Staircase`
gains two new, independently `Optional` fields: `from_observable_region` and `to_observable_region`,
one per floor side (mirroring `from_position`/`to_position`'s own per-side convention). This is
explicitly **not** a claim about the physical stair tread/riser footprint — it is the camera-observable,
top-down 2D region within which a calibrated detection is honestly considered "on this Staircase" for
perception purposes. Missing means `STAIR LOCALIZATION UNAVAILABLE` for that side, never a fabricated
default; nothing auto-derives a region from `Staircase.width` or any other existing field.
`Staircase.observable_region_for_floor(floor_id)` / `contains_world_point(floor_id, world_position)`
give the one place this geometry is tested, correctly picking the from- or to-side region (or `None`)
based on which floor is asking.

Serialization follows the existing `.get(key, default)` convention exactly (`Staircase.to_dict()` /
`from_dict()`), so pre-milestone `.syn` files load unchanged with both regions `None`.

## 5. Designer semantics

`designer/widgets/property_panel.py`'s existing Stair section gains four new fields: "From/To
Observable Region Width (m)" and "...Depth (m)". Both width and depth must be present and `> 0` for a
region to exist at all; leaving either blank means the region stays `None`. The region's center always
tracks the stair's *current* `from_position`/`to_position` (recomputed on every geometry edit), so
moving the stair marker later keeps an already-authored region correctly anchored. No new authoring
tool was added — this extends the existing Stair workflow exactly as the audit recommended, and no
on-canvas visual rendering of the region rectangle was added (see §18, Limitations).

## 6. Calibration / spatial lookup

`camera_calibration/stair_lookup.py` is a new, small, duck-typed module (no dependency on
`models.staircase` at all — mirrors `WorldProjector._lookup_zone`'s own convention):

- `locate_stair(stairs, floor_id, world_position) -> StairMatch` — exactly one of three honest
  outcomes: no match, exactly one match, or `ambiguous=True` with `stair_id=None` (never an arbitrary
  pick when regions overlap).
- `build_stairs_by_floor(building)` — the standard way to build a `floor_id -> stairs` mapping, mirroring
  the `zones_by_floor[floor.id] = list(floor.zones)` construction convention already established in
  `scripts/run_physical_camera_validation.py`.
- `covered_stair_ids(stairs_by_floor, calibrated_floor_ids)` — the honest, **geometrically derived**
  answer to "is Stair S1 observable this cycle": a stair id is covered iff it has an observable region
  on a floor that has at least one calibrated camera. This deliberately does **not** repeat the prior
  audit's finding that `Camera.zone_ids` is cosmetic/non-authoritative — no manual Camera-to-Stair
  assignment exists or is needed.

`camera_calibration/projection.py`'s `WorldProjector` gains an optional `stairs_by_floor` constructor
parameter and a `_lookup_stair()` method, called from `project()` alongside (never instead of, never
duplicating) the existing zone lookup — the same already-computed `world_position`/`floor_id` feeds
both. `WorldProjection` gains two new fields, `stair_id` and `stair_localization_ambiguous`, both
independent of `zone_id`. Putting Stair lookup inside `WorldProjector` (rather than a separate pipeline
stage) mirrors the existing precedent that Zone lookup — itself "semantic," not purely geometric — already
lives there; Phase 6's "preserve separation of concerns" instruction is satisfied because no projection
math is duplicated, only the already-computed `world_position` is tested against a second geometry set.
`WorldProjector.calibrated_floor_ids()` exposes exactly the floor set `covered_stair_ids()` needs.

## 7. LiveOccupant semantics

`LiveOccupant` gains `current_stair_id: Optional[str]`, deliberately named for Stair specifically (not
a generic `current_traversal_asset_id`) — Stair is the only traversal asset type this milestone gives
an observable region and lookup; a generic name would overclaim a generality the mechanism doesn't have.
It follows `current_zone_id`'s own convention exactly: unconditionally overwritten every `update()`
cycle (even to `None`), no per-field evidence timestamp or staleness-decay reconciliation (unlike
`human_classification`/`human_state`) — it is a per-cycle geometric fact from exactly one spatial
lookup, not evidence from possibly-disagreeing sources. `current_stair_id` and `current_zone_id` are
never conflated: both are independently computed and can genuinely both be set (overlapping authored
geometry) or one-without-the-other (the common case). `OccupantHistory` gains a parallel
`stair_transitions` tuple (mirroring `zone_transitions` exactly), giving Phase 12's flow derivation
honest data without a new subsystem. `PositionSample` gains an optional `floor_id` (additive, appended
after `world_position` so every existing positional call site is unaffected) — see §12.

**Temporary detection loss** required zero new code: `LiveOccupantManager.sweep_missing()` already
never calls `update()` for a camera that simply missed a frame, so `current_stair_id` (like
`current_zone_id`) stays frozen at its last value while `status` becomes `TEMPORARILY_LOST`, correctly
excluding the occupant from active-occupancy groupings for that cycle without fabricating an exit. If
they're seen again within `expire_after_seconds`, `update()` runs again and recovers cleanly — proven
in `tests/test_stair_perception_pipeline.py::test_11_temporary_detection_loss_preserves_current_stair_id`
and the equivalent E2E scenario.

## 8. Zone vs. Stair occupancy

Deliberately **not mutually exclusive**. A world point may satisfy a Zone polygon, a Stair observable
region, both, or neither — each is computed independently and both fields are set honestly based on
their own geometry test. The dominant real case (a stairwell shaft with no Zone polygon over it)
naturally produces `current_zone_id=None` / `current_stair_id=<id>`; an authored overlap (e.g. a "Stair
Lobby" Zone drawn over part of a landing region) can legitimately produce both set at once — this is
not double counting, it is two independent, non-conflicting facts about the same point (see §9 for why
the total headcount is unaffected either way).

## 9. Canonical occupancy relationship

`live_occupants/occupancy.py`'s `OccupancyFacts` gains `occupant_ids_by_stair`, computed in the exact
same single pass over occupants `compute_occupancy_facts()` already makes — no second scan, no
duplicate computation. An occupant with no `current_stair_id` contributes nothing there and is **not**
added to `unlocalized_occupant_ids` on that account ("not on a stair" is the ordinary case, not a
localization failure). `total_observed_occupant_ids` stays keyed by `occupant_id` set membership,
exactly as before — an occupant appearing in both `occupant_ids_by_zone` and `occupant_ids_by_stair`
never inflates the total. `LiveOccupantManager` gains a `_by_stair` secondary index and
`occupants_on_stair(stair_id)` query, mirroring `occupants_in_zone()` exactly.

## 10. Stair observation/facts model

A new, small, self-contained package, `stair_perception/`, mirrors the `facp/`/`building_control/`
convention (one package, one immutable snapshot type, `BuildingStateEstimator` only ever passes it
through). `StairObservationStatus` is `OBSERVED` or `UNKNOWN` — the mandatory distinction the audit's
Phase 18 required: a stair simply absent from occupant groupings must never be silently read as "zero
people" unless it is *known* to be genuinely observable this cycle. `StairObservation` carries
`stair_id`, `status`, and `occupant_ids` (never re-derived — identical to `OccupancyFacts.
occupant_ids_by_stair`'s own tuple for that id). `compute_stair_occupancy_snapshot()` is a pure function
combining `occupant_ids_by_stair` (from `OccupancyFacts`) with `covered_stair_ids()` (from
`camera_calibration.stair_lookup`) — deliberately narrow, no congestion score, no safety score, no
predicted bottleneck, measured occupancy truth only.

`BuildingState` gains `stair_occupancy: Optional[StairOccupancySnapshot] = None`, following the exact
`facp_status`/`control_status`/`fire_safety_status`/`fire_water_status` precedent: additive, `None`
when not configured, a pure passthrough parameter on `BuildingStateEstimator.estimate()`
(`stair_occupancy_snapshot`), never computed inline, never folded into `zone_occupancy` (which stays
node-keyed and untouched — Stair is not, and must not become, a Navigation Graph node).

## 11. Crowd Intelligence integration

`crowd_intelligence/models.py`'s `AssetApproachMetrics` gains `observed_on_stair_count: Optional[int]`
— a genuinely separate signal from `approaching_count`/`queue_candidate_count` (landing-proximity,
world-position-based), never a replacement for either. `None` means "not measured" (every Door/Exit
today, or a Stair with no calibrated coverage this cycle); an int (including honestly `0`) means a
calibrated camera covers it. `CrowdIntelligenceEngine.compute()` gains an optional
`observed_stair_occupancy: Dict[str, Optional[int]] = None` parameter — deliberately a **plain dict**,
never a `stair_perception` import, so `crowd_intelligence/`'s own existing, separately-guarded import
allow-list (`tests/test_crowd_intelligence_architecture_guards.py`) is completely untouched by this
milestone (mechanically proven in `tests/test_observable_stair_perception_architecture_guards.py`). A
caller (e.g. live orchestration glue) reduces a `StairOccupancySnapshot` to this simple shape before
calling `compute()`. Every existing caller/test that never supplies it keeps its exact pre-milestone
behavior.

## 12. Trajectory implications

Route-level logic (`trajectory_intelligence/route_progress.py`) already correctly recognized a stair
crossing via `_has_direct_stair_edge()` — untouched by this milestone. The actual gap the audit found
was in `trajectory_intelligence/trajectory.py`'s `compute_movement_facts()`, which computed raw
Euclidean distance/direction/speed from consecutive position samples with no floor-awareness — a
spurious "jump" artifact across any floor transition (stair-observed or not). The fix stays inside this
module's own documented boundary ("no Navigation Graph, no BuildingState here at all"): a new
`_confirmed_floor_change()` helper compares two samples' own (now-optional) `floor_id` — pure
positional metadata, not a graph query — and returns `True` only when both are known and different.
`distance_travelled` excludes any pair straddling a confirmed floor change; `net_displacement` and
`movement_direction`/the `current_speed` fallback are guarded the same way against their own relevant
sample pair. Unknown floor context (the common case for every pre-milestone caller that never threads
`floor_id`) is treated exactly as before — never suppressing legitimate same-floor data just because
floor context happens to be missing.

## 13. Evacuation Progress implications

Deliberately **not integrated** this milestone. `EvacuationProgressEngine` is fundamentally zone/exit
event-driven (`EvacuationLedger` subscribes to `OCCUPANT_ZONE_CHANGED`/`OCCUPANT_EXITED` only); adding
real stair flow would need a new `STAIR_ENTERED`/`STAIR_EXITED` event pathway, which is a larger,
separate change than this milestone's own "smallest trustworthy foundation" scope. The boundary is
documented here rather than forced — `occupant.history.stair_transitions` (§7) already carries
everything a future integration would need, without altering any durable evacuation ledger semantics
today.

## 14. Multi-camera semantics

Proven directly in `tests/test_observable_stair_perception_e2e.py::MultiCameraStairE2ETests` using the
existing, already-proven identity architecture — no deep ReID added or required:

- Two cameras observing the same stair, both detections resolved to the same global identity by an
  explicit `MappingIdentityResolver` entry (a real, authored topology mapping, not inference) — Stair
  occupancy correctly reports **1**, not 2.
- Two cameras with no such mapping — `MappingIdentityResolver`'s own documented fallback (a
  per-camera-namespaced synthetic id, never a silent merge) honestly reports **2** separate identities
  rather than guessing they're the same person.

This is exactly the boundary the prior audit's Phase 13 already established: topology/mapping-based
resolution works today; appearance ReID remains genuinely absent, and this milestone does not add it.

## 15. Provenance / UNKNOWN-vs-zero semantics

`StairObservationStatus.OBSERVED` vs. `UNKNOWN` slots directly into the same taxonomy this codebase
already uses everywhere else — `LiveOccupant.world_position_provenance`, `unlocalized_occupant_ids`,
`OccupancyObservation.occupant_count`'s own "`None` means no reading, never zero" convention, and the
published data-source classification table in `docs/architecture/live_crowd_intelligence.md` §15. No new
taxonomy was invented. Mechanically tested in
`tests/test_stair_perception_pipeline.py::StairOccupancySnapshotTests` and
`tests/test_observable_stair_perception_failure_modes.py::MissingOrInvalidCalibrationTests`: a covered
stair with genuinely nobody there reports `OBSERVED`/`occupant_count=0`; an uncovered stair (no region,
or a region on an uncalibrated floor) reports `UNKNOWN`/`occupant_count=0` — same numeric zero, honestly
distinguishable `status`.

## 16. Simulation/live parity

Not merged, by design. Simulation's own `OccupantTimelineStep`/`opposing_occupants` ground truth
(documented in the prior audit) remains a separate, richer, simulation-only concept; this milestone adds
no Ground Truth adapter for Stair occupancy and does not wire simulation data into `stair_perception` at
all. The two concepts — simulation's "occupants currently traversing the Stair edge" and live's
"occupants currently observed within the Stair region" — are comparable physical concepts with different
epistemic sources, exactly as the milestone brief required; no simulation ground truth leaks into the
live path anywhere in this milestone's code (mechanically checked by
`tests/test_observable_stair_perception_architecture_guards.py`, which confirms `stair_perception/` and
`camera_calibration/stair_lookup.py` import nothing from `simulator`/`ground_truth`/`predictive_dataset`).

## 17. Predictive-AI implications

No `predictive_dataset` schema was changed and no model was trained (out of scope, per this milestone's
own explicit instruction). What changed for future feature-parity work: `candidate_queue_length`/
`candidate_congestion_trend` for Stair candidates were previously sourced entirely from a
landing-proximity proxy (`crowd_intelligence.queue`/`flow`, blind to the stairwell interior). A genuine
`observed_on_stair_count` measurement now exists and could, in a *future* milestone, be wired into that
same feature computation to replace or augment the proxy specifically for Stair — the graph-context
features and `candidate_recent_flow_rate` (already zone-transition-based) are unaffected either way.
This milestone does not perform that wiring; it only establishes that the measurement now honestly
exists to wire in later.

## 18. Failure behavior / limitations

- No on-canvas visual rendering of the observable region rectangle was added to the Designer scene
  (`StairItem` still renders only the existing point-and-width-tick marker) — the region is editable
  and functional via the Property Panel, but not yet visualized on the floor plan. Deferred as a
  presentation-layer nicety, not a correctness gap.
- Coverage (`covered_stair_ids()`) is a **necessary-condition** check (region exists + floor has a
  calibrated camera), not real camera-FOV/frustum geometry — the same limitation the existing live Zone
  path already has (no true visibility computation exists for Zones either in the live path).
- Direction of movement (Phase 13 of the audit's own numbering) and precise per-lane bidirectional
  counterflow were investigated and deferred: a 2D point moving within an axis-aligned observable
  region does not, on its own, provide a defensible vertical-direction axis without additional geometry
  this milestone does not add. Not implemented, not faked.
- Evacuation Progress integration deferred (§13).
- Ambiguous stair-region overlaps are reported, never silently resolved — a real deployment should treat
  persistent ambiguity as a Designer authoring issue to fix (shrink/reposition the overlapping regions),
  not something the runtime resolves for the operator.

## 19. Performance

`scripts/benchmark_stair_perception.py`, at the milestone's required scale (50 zones, 20 stairs, 20
cameras, 100 occupants):

- World-position → Stair lookup (`locate_stair`, single occupant): ~0.005 ms/call (mean).
- Stair coverage derivation (`covered_stair_ids`, whole building): ~0.003 ms/call (mean).
- Full Stair occupancy derivation (`canonical_occupancy` + `compute_stair_occupancy_snapshot`): ~0.017
  ms/call (mean).

All three are negligible relative to a 1-second live cycle budget; no optimization was necessary.

## 20. Remaining limitations (summary)

See §18. In addition: this milestone proves the chain through real production classes
(`WorldProjector`, `LiveOccupantManager`, `CrowdIntelligenceEngine`, `BuildingStateEstimator`) driven by
a fake frame source and fake human detector (mirroring `tests/test_live_camera_pipeline.py`'s own
established convention) — it does not exercise a real YOLO detector, a real RTSP stream, or the full
production `live_runtime`/`live_system` orchestrator wiring, several of which had unrelated in-progress
changes in the working tree at the time of this milestone and were deliberately left untouched (see
Phase 0/30). Wiring this into the real orchestrator/launcher is explicitly left for a future milestone.
