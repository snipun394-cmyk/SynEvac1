# Observable Asset Perception Framework

Status: implemented (generalization/refactor only — no new asset type's perception was implemented).
Builds directly on `docs/architecture/live_stair_perception.md` (the Observable Stair Perception
milestone this one generalizes) and `docs/architecture/stair_camera_perception_audit.md` (the original
audit). No NavigationGraph redesign, simulation redesign, LiveOccupant redesign, calibration-math
change, new ML, ReID, or Recommendation-scoring change is made by this milestone.

---

## Motivation

The Observable Stair Perception milestone gave Stair an observable physical region, a spatial lookup,
and an OBSERVED/UNKNOWN occupancy snapshot. The real laboratory building that motivated it has more
than one kind of asset a camera could plausibly be aimed at — Door, Exit, Assembly Point, Refuge Area,
Lift Lobby, Escalator, Corridor, Fire Door, Hose Reel Area, Hydrant Area. Building each of those as its
own hand-copied "DoorMatch/DoorObservation/DoorOccupancySnapshot/locate_door/covered_door_ids" would
duplicate the exact same logic under a new name every time, and the duplication would only grow. This
milestone is a Phase-1 audit finding turned into an architecture: almost none of the Observable Stair
Perception milestone's own lookup/snapshot code was actually about stairs. It only had Stair's *name* on
it. This document records what stayed genuinely Stair-specific, what moved to a generic, reusable
framework, and how a future asset type plugs into it.

## Phase 1 audit: what was actually generic in disguise

| Component (pre-refactor) | Verdict | Disposition |
|---|---|---|
| `StairMatch` / `locate_stair()` | 100% generic — duck-typed only against `.id`/`.contains_world_point()`, no Stair-specific field ever read | Moved, unchanged in behavior, to `camera_calibration.asset_lookup.AssetMatch`/`locate_asset()` |
| `covered_stair_ids()` | 100% generic — duck-typed only against `.observable_region_for_floor()` | Moved to `camera_calibration.asset_lookup.covered_asset_ids()` |
| `StairObservationStatus` / `StairObservation` / `StairOccupancySnapshot` / `compute_stair_occupancy_snapshot()` | 100% generic — no Stair-specific field, computation, or assumption anywhere | Moved to `observable_assets.models.ObservationStatus`/`AssetObservation`/`ObservableAssetSnapshot` and `observable_assets.facts.compute_asset_occupancy_snapshot()`; the redundant `stair_perception/` package was deleted rather than kept as a parallel copy |
| `AssetApproachMetrics.observed_on_stair_count` | Generic in disguise (the surrounding `AssetApproachMetrics` type is already shared across Door/Exit/Stair) | Renamed `observed_occupant_count`, populated for whichever `asset_type` a caller supplies data for |
| `BuildingState.stair_occupancy` | Generic in disguise (its type had no Stair-specific logic) | Renamed `BuildingState.observable_assets: Optional[ObservableAssetSnapshot]` |
| `build_stairs_by_floor()` | **Genuinely Stair-specific** — the only code that actually knows a Staircase lives in `Floor.stairs` and spans `from_floor_id`/`to_floor_id` | Stays in `camera_calibration.stair_lookup`, untouched |
| `Staircase` / `StairObservableRegion` / `LiveOccupant.current_stair_id` | Genuinely Stair-specific by design (this milestone's own explicit "do not redesign LiveOccupant" instruction) | Untouched |

## Architecture

```
                     ┌─────────────────────────────┐
                     │  camera_calibration.asset_lookup   (GENERIC)   │
                     │  AssetMatch, locate_asset(),                    │
                     │  ObservableAssetKind, build_assets_by_floor(),  │
                     │  covered_asset_ids()                            │
                     │  -- zero project-package dependencies, zero    │
                     │  knowledge of any concrete asset type           │
                     └───────────────┬─────────────────────────────────┘
                                     │ registers
                     ┌───────────────┴─────────────────────────────────┐
                     │  camera_calibration.stair_lookup   (ADAPTER)     │
                     │  build_stairs_by_floor()  (genuinely Stair-only) │
                     │  STAIR_ASSET_KIND, DEFAULT_OBSERVABLE_ASSET_KINDS│
                     └───────────────────────────────────────────────────┘

                     ┌─────────────────────────────┐
                     │  observable_assets            (GENERIC)         │
                     │  ObservationStatus, AssetObservation,           │
                     │  ObservableAssetSnapshot,                       │
                     │  compute_asset_occupancy_snapshot()             │
                     │  -- zero project-package dependencies           │
                     └─────────────────────────────────────────────────┘
```

`camera_calibration.projection.WorldProjector` consults the generic registry internally
(`self._assets_by_floor`, built from either the new `assets_by_floor` constructor parameter or the
backward-compatible `stairs_by_floor` alias) and exposes both the generic result (`WorldProjection.
asset_id`/`asset_type`/`asset_localization_ambiguous`) and a thin, backward-compatible Stair-only view
(`stair_id`/`stair_localization_ambiguous`, derived, never a second computation). No existing caller of
`WorldProjector` needed to change.

## Relationship to Stair

Stair is the framework's first, and as of this milestone still only, registered
`ObservableAssetKind` (`STAIR_ASSET_KIND` in `camera_calibration/stair_lookup.py`,
`DEFAULT_OBSERVABLE_ASSET_KINDS = (STAIR_ASSET_KIND,)`). Every behavior this milestone's regression
suite proves is Stair behavior flowing through the now-generic pipes, not a new Stair feature.

## Future extensibility

Registering a new asset kind requires exactly three things, none of which touch the framework itself:

1. A per-kind builder function — `build_<kind>_by_floor(building) -> Mapping[floor_id, Sequence[asset]]`
   (mirrors `build_stairs_by_floor()`), reading wherever that asset type's geometry actually lives
   (`floor.doors` for Door, etc.).
2. One `ObservableAssetKind(asset_type="Door", build_by_floor=build_doors_by_floor)` registration record.
3. Adding that record to whatever tuple of kinds a caller passes to `build_assets_by_floor()` (and, for
   live projection, to `WorldProjector`'s `assets_by_floor` parameter).

`tests/test_observable_asset_extensibility.py` proves this concretely: a test-only fake object
satisfying the same `.id`/`.contains_world_point()`/`.observable_region_for_floor()` contract as
`Staircase` registers as a second kind, and `locate_asset()`, `covered_asset_ids()`, and
`compute_asset_occupancy_snapshot()` all handle it — including cross-type ambiguity (a point matching
both a Stair and the fake Door-like asset resolves `ambiguous=True`, never an arbitrary pick) — with
**zero changes to any production module**. This milestone does not implement real Door/Exit/Assembly
Point/etc. perception; it only proves the framework can carry it.

## Separation from navigation

Unchanged, verified mechanically (`tests/test_observable_stair_perception_architecture_guards.py`):
`Staircase` is not a `Zone` subclass, `navigation.node.Node` still has no `STAIR` node type, and
`navigation.edge.Edge` still has its `STAIR` edge type. No observable asset this framework anticipates
(Door, Exit, Assembly Point, Refuge Area, Lift Lobby, Escalator, Corridor, Fire Door, Hose Reel Area,
Hydrant Area) is, or should become, a Navigation Graph node — every one of them is a traversal/portal
asset or a room-like space already representable as a Zone; "observable" is a perception-layer concept
layered *alongside* the navigation role, never a replacement for it.

## Separation from simulation

Untouched. `camera_calibration.asset_lookup` and `observable_assets` import nothing from `simulator`,
`ground_truth`, `simulation_runtime`, or `predictive_dataset` (mechanically checked by the same guard
test file). Simulation's own ground truth remains a separate, richer, differently-sourced concept, exactly
as `docs/architecture/live_stair_perception.md` §16 already established for Stair specifically.

## BuildingState integration

`BuildingState.observable_assets: Optional[ObservableAssetSnapshot]` replaces the prior milestone's
Stair-named field, following the exact same `facp_status`/`control_status`/`fire_safety_status`/
`fire_water_status` precedent: additive, `None` when not configured, a pure passthrough parameter on
`BuildingStateEstimator.estimate()` (`observable_asset_snapshot`), never computed inline, never folded
into `zone_occupancy` (which stays exactly what it always was — node-keyed occupancy only). A future
Door/Exit integration would populate the *same* field with a snapshot whose `observations` mapping simply
has more entries (keyed by asset_id, tagged by `asset_type`) — no new `BuildingState` field, ever, per
asset type.

## LiveOccupant integration

**Deliberately unchanged**, per this milestone's own explicit instruction ("Do NOT redesign
LiveOccupant"). `LiveOccupant.current_stair_id` stays exactly what it was — a per-cycle geometric fact,
independent of `current_zone_id`, following that field's own unconditional-overwrite convention (see
`live_stair_perception.md` §7). A future Door/Exit integration would need its **own** analogous field
(e.g. `current_door_id`) if live occupant localization to that asset type were ever required — this
framework does not attempt to generalize `LiveOccupant` itself, and doing so is explicitly out of scope
here. The generic layer stops at `WorldProjection.asset_id`/`asset_type` and `ObservableAssetSnapshot`;
per-asset-type occupant fields, if ever added, remain a deliberate, reviewed decision each time, not an
automatic consequence of registering a new `ObservableAssetKind`.

## Future asset roadmap

Stair is implemented. Door, Exit, Assembly Point, Refuge Area, Lift Lobby, Escalator, Corridor, Fire
Door, Hose Reel Area, and Hydrant Area are all anticipated by this framework's shape (a plain
`asset_type: str`, never a closed enum) but **none are implemented by this milestone**. Each would need,
at minimum: (a) an observable-region concept on that asset's own model (mirroring
`StairObservableRegion` — most of these assets are already single-floor, simpler than Stair's two-floor
case), (b) a per-kind `build_<kind>_by_floor()` adapter, (c) an `ObservableAssetKind` registration, and
(d) a deliberate decision about whether `LiveOccupant` needs a new per-type field for that asset (see
above) — none of which requires touching `camera_calibration.asset_lookup`, `observable_assets`,
`CrowdIntelligenceEngine`, or `BuildingState` again.

## Regression safety

Every test written against the Observable Stair Perception milestone's original API was updated to the
generalized names (`locate_stair`→`locate_asset`, `covered_stair_ids`→`covered_asset_ids`,
`compute_stair_occupancy_snapshot`→`compute_asset_occupancy_snapshot`, `StairObservationStatus`→
`ObservationStatus`, `BuildingState.stair_occupancy`→`observable_assets`,
`AssetApproachMetrics.observed_on_stair_count`→`observed_occupant_count`) and proves IDENTICAL Stair
behavior through the new, generic call surface — never merely "still compiles." `WorldProjector`'s own
constructor kept its `stairs_by_floor` parameter as a genuine backward-compatible alias (not just a
renamed test fixture): an existing caller that only ever knew about `stairs_by_floor` needs zero code
changes. `NavigationGraph`, `models/zone.py`, simulation, calibration ray-projection math, and
`LiveOccupant`'s own field set are all untouched — verified both by the architecture-guard tests
(structural assertions against the real classes, not text search) and by the fact that this milestone's
diff never touches `navigation/*.py`, `simulator/*.py`, `simulation_runtime/*.py`, or
`camera_calibration/geometry.py`.
