# Camera Coverage Intelligence & Observable Asset Mapping

Status: implemented (metadata/intelligence layer only — no new perception, no new calibration math,
no UI). Builds directly on `docs/architecture/observable_asset_perception.md` (the generic Observable
Asset framework) and `docs/architecture/camera_calibration_and_world_projection.md` (calibration).
Does not redesign `WorldProjector`, `NavigationGraph`, `CameraIntrinsics`/`CameraExtrinsics`, or any
Observable Asset type. Implements no new ML and no camera hardware protocol.

---

## Motivation

Before this milestone, a Camera Asset (`models.camera.Camera`) knew its own position, FOV, and
detection range, and a calibrated camera could resolve an *individual detection* to an observable
asset (`camera_calibration.projection.WorldProjector.project()` → `WorldProjection.asset_id`). But
nothing answered the standing, detection-independent question: **which observable assets does this
camera cover at all?** The closest existing answer,
`camera_calibration.asset_lookup.covered_asset_ids()`, only ever asked a *floor-level* question ("does
this floor have *any* calibrated camera?") — it could not name *which* camera, and could not
distinguish a camera that fully frames a Stair from one that grazes its edge. This milestone makes
cameras first-class perception devices: a camera now has a queryable, geometrically-derived coverage
picture over every observable asset on its floor.

## Phase 1 audit: Camera model

| Field (`models.camera.Camera`) | Status | Notes |
|---|---|---|
| `position`, `rotation`, `horizontal_fov`, `max_range` | **Authoritative** for 2D coverage-sector geometry | Drives `Camera.coverage_polygon()` (the Visibility engine's own sector fan) — a Designer-authored, logical description, independent of any real lens. |
| `resolution`, `fps` | **Legacy/logical-only** | Display-only strings, "never parsed, validated, or used to configure an actual video pipeline" (see that field's own docstring). Ignored by this milestone entirely. |
| `EngineeringAsset.zone_ids` | **Legacy/cosmetic** | The Observable Asset Perception Framework milestone already found this non-authoritative for the real live pipeline (`camera_calibration/asset_lookup.py`'s own docstring). This milestone does not read it, and does not repeat the mistake for observable-asset coverage. |
| `to_dict()` / `from_dict()` | Standard `EngineeringAsset` round-trip | Unchanged by this milestone. |
| **Calibration** (`camera_calibration.camera_model.CalibrationProfile`) | **Authoritative** for camera coverage, deliberately **not** stored on `Camera` itself (a pre-existing, unrelated-to-this-milestone design decision) | `CalibrationProfile.extrinsics.position`/`.yaw_degrees` are the camera's real measured pose; `.intrinsics.image_width`/`.focal_length_x` yield a derived horizontal FOV. This milestone reads these fields, never the raw `Camera.rotation`/`Camera.horizontal_fov` directly, so a camera whose calibration was refined independently of its Designer-authored fields is honestly represented. `Camera.max_range` is still reused as-is for the sector's radius (calibration carries no analogous field) — the same "one distance an engineering camera asset needs" precedent `perception.providers.ground_truth_camera_provider.GroundTruthCameraProvider` already established. |

Conclusion: geometry, never manual assignment, is the only honest source of camera coverage — exactly
what Phase 2/3 below implement.

## Architecture

```
                    ┌───────────────────────────────────────────┐
                    │  camera_coverage.models        (GENERIC)   │
                    │  CoverageState, AssetCoverage,              │
                    │  CameraCoverage, CameraCoverageSnapshot     │
                    │  -- zero project-package dependencies       │
                    └───────────────────┬─────────────────────────┘
                                        │ produced by
                    ┌───────────────────┴─────────────────────────┐
                    │  camera_coverage.discovery                   │
                    │  compute_camera_coverage()                    │
                    │  compute_camera_coverage_snapshot()            │
                    │  -- consumes CalibrationProfile (unchanged)    │
                    │     + camera_calibration.asset_lookup.         │
                    │       build_assets_by_floor() output (unchanged)│
                    └─────────────────────────────────────────────┘
```

`camera_coverage` is a new, standalone package — it imports `camera_calibration.camera_model`
(read-only: `CalibrationProfile.extrinsics`/`.intrinsics`, existing public fields) and
`visibility.geometry` (existing `point_in_polygon`/`segment_intersection` primitives), and is imported
by `building_state`. It is never imported by `camera_calibration`, `observable_assets`, or
`camera_calibration.projection.WorldProjector` — the dependency direction is one-way, and
`WorldProjector` itself has zero changes in this milestone.

## Phase 2/3: the `CameraCoverage` model and automatic discovery

`camera_coverage.models.AssetCoverage` is the one honest per-(camera, asset) coverage fact:

- `asset_id`, `asset_type` — the same identity `observable_assets.models.AssetObservation` already uses.
- `state` — one of four `CoverageState` values (Phase 4, below).
- `region_polygon` — the asset's own observable-region corners, present **only** when `state` is
  `FULLY_VISIBLE` or `PARTIALLY_VISIBLE` ("coverage geometry if appropriate": nothing honest to show
  otherwise). Never new geometry — always the asset's own already-authored region, restated as corners.
- `provenance` — a short diagnostic string, never a source of truth (mirrors
  `AssetObservation.provenance`).

`camera_coverage.discovery.compute_camera_coverage(camera, calibration, candidates)` computes this for
one camera against a list of `(asset_type, asset)` candidates (exactly the shape
`camera_calibration.asset_lookup.build_assets_by_floor()` already produces — never re-derived here).
`compute_camera_coverage_snapshot(cameras, calibrations, assets_by_floor)` does this for every camera
in a building, producing a `CameraCoverageSnapshot`. **No manual Camera-to-asset assignment is read or
accepted anywhere in this module** — geometry (calibration + the asset's own observable region) is the
only input.

### How the coverage sector is built

The camera's coverage sector is the exact same fan/wedge shape `models.camera.Camera.coverage_polygon()`
already draws (apex at the camera's position, an arc from `-FOV/2` to `+FOV/2` around its facing
direction, out to its detection range, `0° = +x, increasing clockwise`) — not a new geometric model,
just built from the camera's **calibrated** pose instead of its raw Designer-authored fields:

- Center: `CalibrationProfile.extrinsics.position` (not `Camera.position`).
- Facing angle: `CalibrationProfile.extrinsics.yaw_degrees` (not `Camera.rotation` — same convention,
  reused directly, since `CameraExtrinsics.yaw_degrees`'s own docstring already establishes it as
  identical to `Camera.rotation`'s convention).
- Horizontal FOV: **derived** from `CalibrationProfile.intrinsics` — `fov = 2·atan((image_width/2) /
  focal_length_x)`, the exact algebraic inverse of `CameraIntrinsics.from_horizontal_fov()`. This
  derivation lives entirely inside `camera_coverage.discovery` as a private helper; **no field, method,
  or behavior in `camera_calibration.camera_model` is added, changed, or redesigned.**
- Radius: `Camera.max_range` (calibration carries no detection-range field of its own).

A camera with no `CalibrationProfile` at all, or one whose intrinsics/`max_range` cannot yield a usable
sector (non-positive focal length or detection range), produces `sector_polygon = None` and every
candidate asset reports `UNKNOWN` — never a guessed sector.

## Phase 4: coverage states, precisely

`camera_coverage.models.CoverageState` has exactly four values, determined by
`camera_coverage.discovery._classify_overlap()` in this priority order, once both the camera's sector
polygon and the asset's own observable-region rectangle (`StairObservableRegion` today — see
`_rectangle_corners()`) are known:

1. **`FULLY_VISIBLE`** — all four of the region's corners lie inside the sector polygon.
   `point_in_polygon()` on a range-bounded fan inherently enforces *both* the angular (FOV) and the
   distance (max range) limit in one test — there is no separate range check.
2. **`PARTIALLY_VISIBLE`** — some, but not all, of the region's corners lie inside the sector polygon
   (the common "half in frame" case); **or**, if zero corners are inside, the sector nonetheless reaches
   into the rectangle — checked two ways: any sector-polygon vertex lies inside the rectangle (a large
   asset region can swallow the camera's position or an arc point without containing it), or any sector
   edge crosses any rectangle edge (a narrow sector wedge can pass through a large region without ever
   containing one of its four corners). Either check passing is enough.
3. **`NOT_VISIBLE`** — both geometries are known, and none of the above hold: a provably empty
   intersection.
4. **`UNKNOWN`** — no honest geometric basis exists at all: the camera has no calibration, its
   calibration cannot yield a usable sector, or the asset has no observable region authored on the
   camera's floor (mirrors `observable_assets.models.ObservationStatus.UNKNOWN`'s own "no honest basis"
   discipline exactly). `UNKNOWN` is the default a caller gets for any (camera, asset) pair this module
   never even considered.

A consumer must always check `state`; `AssetCoverage.is_covered` is the one place "`FULLY_VISIBLE` or
`PARTIALLY_VISIBLE`" is spelled out, so no caller re-derives that boolean differently.

## Phase 5: multi-camera coverage, without duplicate occupancy

`CameraCoverageSnapshot` (building-wide) holds every camera's own `CameraCoverage`, keyed by
`camera_id`. Multiple cameras can — and routinely will — each report `FULLY_VISIBLE`/`PARTIALLY_VISIBLE`
for the same `asset_id`; `cameras_observing(asset_id)` returns all of them.

This cannot cause double-counted occupancy, **by construction**: `camera_coverage` carries **zero
occupancy counts anywhere in it** — it is pure visibility metadata (which cameras see which assets, and
how completely). The one and only occupant count for an asset remains exactly where the Observable
Asset Perception Framework milestone put it — `observable_assets.models.ObservableAssetSnapshot`,
computed once per `asset_id` regardless of how many cameras cover it (see
`observable_assets.facts.compute_asset_occupancy_snapshot()`, unchanged by this milestone). A future
consumer wanting "N cameras confirm M people currently on Stair S1" reads **both** snapshots off
`BuildingState` and combines them itself; neither snapshot needs to know the other exists.

## Phase 6: `BuildingState` integration

`BuildingState.camera_coverage: Optional[CameraCoverageSnapshot]` — additive, following the exact same
`facp_status`/`control_status`/`observable_assets` precedent: `None` when not configured, a pure
pass-through parameter on `BuildingStateEstimator.estimate()` (`camera_coverage_snapshot`), never
computed inline, never merged into `observable_assets` (a genuinely separate, purely-metadata field —
see Phase 5 above on why the two must stay separate).

## Phase 7: what a future Command Center / consumer can already answer

No UI was implemented by this milestone. What exists is enough for a future consumer to answer both
motivating questions directly, with zero new plumbing:

- **"Which cameras observe Stair S1?"** → `building_state.camera_coverage.cameras_observing("S1")`.
- **"Which observable assets does Camera C3 cover?"** → `building_state.camera_coverage.assets_observed_by("C3")`.

Both default to an empty tuple for an unknown id, and both accept an optional `states=` override (e.g.
`states=(CoverageState.FULLY_VISIBLE,)` to ask only about complete coverage).

## Regression safety / boundaries respected

- `camera_calibration/camera_model.py`, `camera_calibration/projection.py`, `camera_calibration/geometry.py`
  — **zero lines changed**. `camera_coverage.discovery` only reads existing public fields.
- `navigation/*.py` — zero lines changed; no observable asset is, or becomes, a Navigation Graph node
  (unchanged from `docs/architecture/observable_asset_perception.md`).
- `observable_assets/*.py` — zero lines changed; occupancy truth is untouched.
- No new ML, no `Recommendation` change, no camera hardware protocol implementation.
- `models/camera.py`, `models/staircase.py` — zero lines changed (Phase 1 was audit-only; this
  milestone reads both, modifies neither).
- `building_state/models.py`, `building_state/estimator.py` — one new additive field, one new additive
  pass-through parameter, mirroring five prior milestones' own identical pattern.

## Future extensions

- A future Door/Exit/etc. `ObservableAssetKind` registration (per `observable_asset_perception.md`'s own
  roadmap) needs zero changes here: `compute_camera_coverage()`/`compute_camera_coverage_snapshot()` are
  already type-agnostic over `asset_type`, and only require a per-kind corner/boundary adapter in
  `camera_coverage.discovery._rectangle_corners()`-equivalent if that asset's observable-region shape is
  not axis-aligned-rectangular.
- A real lens's non-uniform distortion, a true 3D view frustum (accounting for pitch/roll, not just
  yaw), or occlusion by intervening geometry (walls, other assets) are all deliberately out of scope —
  the same "2D coverage-sector wedge" simplification `models.camera.Camera.coverage_polygon()` and the
  Visibility engine already use elsewhere in this codebase, not a new limitation this milestone
  introduces.
- A UI panel surfacing `cameras_observing()`/`assets_observed_by()` (Command Center or Designer) is
  anticipated but explicitly not built by this milestone.
