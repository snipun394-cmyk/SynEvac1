# Building Topology & Camera Topology — Investigation + Architecture Proposal

Status: **design only, nothing implemented**. This document is Phases 1–3 of the Building
Topology & Camera Topology Foundation milestone. It does not touch the LAN, NVR, RTSP, or
any live CCTV path — everything here was verified against `p3.syn` and the committed source
tree only.

---

## Phase 1 — Investigation report (`p3.syn`)

`p3.syn` is one `Building` with two `Floor`s.

### Ground Floor (`a1937a63…`, `display_order=0`)

| Kind | Count | Detail |
|---|---|---|
| Zones | 5 | Zone 1–5, all `zone_type="Generic"`, all axis-aligned rectangles (`x/y/width/height`); `polygon` is empty on every one — no non-rectangular zone shape is actually in use anywhere in this project. |
| Exits | 1 | Exit 1, `zone_id` → Zone 1. |
| Stairs (owned here) | 0 | — Ground Floor is only ever a Stair *destination* (see Floor 1 below), never an origin. |
| Cameras | 4 | Camera 1–4. `position`, `rotation`, `horizontal_fov=90°`, `max_range` 5–15 m, `mount_height=3m`, all `active=true`. **None of the four cameras has a `zone_ids` value in the file at all** — the key is absent, not an empty list. |
| Obstacles | 1 | One `Barrier`, `traversability="Blocked"`. |
| Doors | 4 | Door 1: Zone 3 ↔ Zone 1. Door 3: Zone 2 ↔ Zone 1. Door 3 (duplicate name, distinct id): Zone 1 ↔ Zone 4. Door 4: Zone 5 ↔ Zone 1. **Zone 1 is a hub — every door, the only exit, and the stair landing all converge on it.** |

### Floor 1 (`901b1ce0…`, `display_order=1`)

| Kind | Count | Detail |
|---|---|---|
| Zones | 3 | Zone 6–8, all `Generic`, same rectangle-only shape as Ground Floor. |
| Exits | 0 | — |
| Stairs | 1 | Stair 1: `from_floor_id`=Floor 1, `from_zone_id`=Zone 6 → `to_floor_id`=Ground Floor, `to_zone_id`=Zone 1. |
| Cameras | 0 | **Floor 1 has zero cameras — no perception coverage upstairs at all.** |
| Obstacles | 1 | One `Barrier`, `traversability="Blocked"`. |
| Doors | 2 | Door 1: Zone 6 ↔ Zone 7. Door 2: Zone 7 ↔ Zone 8. |

### Existing adjacency already present in the file

There is **no separate "connections" list anywhere in `p3.syn`.** Adjacency is already
distributed across the objects that create it, exactly as the model layer defines it:

- `Door.zone_a_id` / `Door.zone_b_id` → same-floor Zone↔Zone adjacency.
- `Exit.zone_id` → Zone↔Outside adjacency.
- `Staircase.from_zone_id`/`to_zone_id` + `from_floor_id`/`to_floor_id` → cross-floor Zone↔Zone adjacency.

No navigation graph is ever serialized into the `.syn` file itself — it is derived at load
time (see Phase 2). Camera **position/rotation/FOV/range** are fully authored; Camera
**zone membership** (`zone_ids`) is not authored anywhere in this project.

---

## Phase 2 — Existing architecture report

Everything below is real, committed, tested production code — traced to exact classes, not
inferred.

### 2.1 Building topology already exists: `navigation.graph.NavigationGraph`

`Building` (`models/building.py`) → `Floor` (`models/floor.py`) → `Zone`/`Door`/`Exit`/
`Staircase`/`Camera` remains the single Digital Twin source of truth (per this project's own
standing rule: **the Building Designer IS the Digital Twin** — no parallel model is ever
built beside it).

`navigation/graph_builder.py::NavigationGraphGenerator.build(building) -> NavigationGraph` is
a **pure, stateless function** that walks the Building and derives:

- `Node` (`navigation/node.py`) — one per `Zone`/`AssemblyPoint`, plus a single shared
  `"Outside"` node. Holds a live `reference` back to the real Zone/AssemblyPoint — never a
  copy. `Node.space_type` already exposes `Zone.zone_type` (`Generic`, `Room`, `Corridor`,
  `Lobby`, **`Stair Lobby`**, `Mechanical`, `Electrical`, `Storage`, `Office`, `Outdoor`) —
  the exact "Zone → Corridor → Lobby → Staircase" typing the brief asks for **already
  exists on `Zone` today**, it's just unused in `p3.syn` (every zone there is `Generic`).
- `Edge` (`navigation/edge.py`) — one per `Door`/`Exit`/`Staircase`, with a genuine
  `walking_distance` (meters, geometrically derived) and `traversal_cost`/`traversal_time`.
- `FlowRegion` (`navigation/flow_region.py`) — groups edges that represent one continuous
  crowding phenomenon (e.g. a multi-flight stairwell). Purely additive, doesn't change
  routing.
- `NavigationGraph.validate()` already reports `zone_without_connections`, `isolated_zone`,
  and `disconnected_floor` — exactly the kind of topology sanity check a Building Topology
  layer would otherwise need to reinvent.

**This already is the "Zone topology" the brief asks for** (Zone → Zone → Corridor → Lobby →
Staircase). Building a second, parallel `ZoneGraph` class would directly violate this
project's own "don't duplicate the Building model" rule.

### 2.2 Camera topology already exists — but has never been called in production

`cross_camera_identity/topology.py` (fully documented in
`docs/architecture/cross_camera_identity.md`, §5–6) already contains exactly the camera
adjacency model the brief describes:

- `CameraTransition` — a directed edge between two camera ids: `min_transition_time`,
  `max_transition_time`, `expected_transition_time`.
- `CameraTopology` — a hand-buildable graph: `add_camera`, `add_transition(...,
  bidirectional=True)`, `possible_destinations(camera_id)`, and `is_plausible_transition(
  from, to, elapsed, default_min, default_max)`, which distinguishes three honest cases
  (registered pair / neither camera known / one camera known but no registered pair).
- `build_topology_from_navigation_graph(navigation_graph, camera_zone_ids, walking_speed,
  slack_factor) -> CameraTopology` — the **automatic derivation** function: two cameras
  become topologically connected if their assigned zones are the same zone, or exactly one
  real `NavigationGraph` edge apart, using that edge's genuine `walking_distance` to compute
  an honest `expected_transition_time` (never a fabricated number).

**Verified finding**: `build_topology_from_navigation_graph` is exercised only inside
`tests/test_cross_camera_identity.py`. A repo-wide search of every non-test `.py` file finds
**zero production call sites** for it, for `CameraTopology()`, or for
`RuleBasedCrossCameraIdentityResolver`. `live_runtime/factory.py::build_live_runtime()`
already accepts an optional `cross_camera_identity_resolver` parameter — but nothing that
constructs a live runtime (`designer/live_runtime_controller.py`,
`live_runtime_launcher/session.py`) ever supplies one. `live_runtime_launcher/
human_detector_wiring.py` even says so explicitly in its own comment: *"No tracker/
behavior_recognizer/cross_camera_identity_resolver/world_projector/live_occupant_manager is
wired here"* — current Live Mode wiring is deliberately single-camera only (the Camera 1
milestone).

So: **the camera topology model is fully engineered and tested, but it is a dead seam in
production today** — a plug with nothing plugged into it.

### 2.3 Camera-to-zone assignment already exists, but is single-zone only

`designer/widgets/property_panel.py` already has a **Camera → Zone** combo box
(`self.camera_zone`), backed by `Camera.zone_ids` (`models/engineering_asset.py`, a `Tuple[
str, ...]`). The UI currently only ever writes a single element:
`model.zone_ids[0] if model.zone_ids else ""` — even though the underlying model field is
already a tuple, and even though a camera's 90°/15 m FOV can obviously reach more than one
zone (Camera 1 in `p3.syn`, at the corner of Zone 1, plausibly sees Zone 1 and Zone 5 both).
`Speaker` (a peer `EngineeringAsset`) already has a **multi-zone checklist** widget
(`_populate_zone_checklist`) for exactly this reason. Camera does not yet reuse it.

This matters directly for §2.2: `build_topology_from_navigation_graph` needs real
`camera_zone_ids` to produce a non-empty `CameraTopology`. In `p3.syn` today, all four
cameras have empty `zone_ids`, so running that function against this exact project would
currently yield **zero transitions**.

### 2.4 Camera coverage already exists — but answers a different question

`camera_coverage/` (models.py + discovery.py) answers **"which cameras observe which
observable asset"** (a Stair's `StairObservableRegion`, today) — `CameraCoverage`,
`AssetCoverage`, `CoverageState` (`FULLY_VISIBLE`/`PARTIALLY_VISIBLE`/`NOT_VISIBLE`/
`UNKNOWN`). It reuses `Camera.coverage_polygon()`'s exact sector-fan geometry, built from a
camera's **calibrated** pose (`camera_calibration.camera_model.CalibrationProfile`) when
available. It is geometry-only, by design carries **zero occupancy counts** (that stays in
`observable_assets.models.ObservableAssetSnapshot`), and is wired into
`BuildingState.camera_coverage` already (Camera Coverage Intelligence milestone).

**What this package does not do, and nothing else does either: camera-to-camera overlap** —
"do Camera A's and Camera B's coverage sectors cover the same patch of floor." That is a
different question from both `camera_coverage` (camera vs. asset) and `cross_camera_identity`
(camera vs. camera, but *adjacency/transition-time*, not *simultaneous visibility*). A
repo-wide search (models, navigation, camera_coverage, cross_camera_identity, visibility)
found no existing concept of camera-to-camera geometric overlap anywhere in this codebase.

### 2.5 Summary table

| Concept the brief asks about | Exists today? | Where |
|---|---|---|
| Building topology (Floors → Zones → connections) | **Yes** | `models/building.py`, `models/floor.py`, `models/zone.py`, `models/door.py`, `models/exit.py`, `models/staircase.py` |
| Navigation graph | **Yes** | `navigation/graph.py`, `navigation/graph_builder.py` |
| Zone connectivity graph | **Yes — same object as the navigation graph** | `navigation/graph.py` (`Node`/`Edge`) |
| Camera adjacency / transition time | **Yes, engineered + tested, never called in production** | `cross_camera_identity/topology.py` |
| Camera overlap (simultaneous visibility of the same floor patch) | **No** | — genuinely missing |
| Transition confidence (graded, not boolean) | **No** — only a boolean in/out-of-window check | `CameraTopology.is_plausible_transition()` |
| Coverage model (camera → asset) | **Yes** | `camera_coverage/` |
| Camera → Zone assignment (data) | **Yes, but never populated in `p3.syn`; UI restricts to one zone** | `models/engineering_asset.py` (`Camera.zone_ids`), `designer/widgets/property_panel.py` |
| A single facade that exposes all of the above together | **No** | — this is the one real gap |

---

## Phase 3 — Proposed architecture

### 3.1 Governing principle

**Building topology and zone topology are not being (re)built — they already exist as
`NavigationGraph`, derived straight from `Building`.** The only genuinely new work is:

1. A small **read-only facade** that composes the navigation graph with camera-facing views,
   so a future consumer (coverage viz, tripwires, virtual zones, cross-camera tracking) has
   one stable object to depend on instead of three unrelated imports.
2. **Camera-to-camera overlap** — the one concept that is truly absent.
3. **Actually calling** `build_topology_from_navigation_graph` from somewhere real, instead of
   only from its own test file.

Cameras remain sensors that *observe* Building structure; they never own movement logic.
Movement is described exclusively by `Building → Floor → Zone → Door/Exit/Stair`
(`NavigationGraph`), precisely as the brief requires. Camera-facing objects (`CameraTopology`,
the new overlap model) are always **derived from** that graph, never the other way around.

### 3.2 New package: `building_topology/`

A new top-level package, sibling to `navigation/`, `camera_coverage/`, and
`cross_camera_identity/` — matching this codebase's own established convention of one
narrow-scope package per concern that *reads* Building/NavigationGraph/BuildingState and
never duplicates them.

```
building_topology/
    __init__.py
    models.py    # BuildingTopology, CameraOverlap, CameraOverlapSnapshot, OverlapState
    overlap.py   # compute_camera_overlap() / compute_camera_overlap_snapshot()
    builder.py   # BuildingTopologyBuilder.build(building, calibrations=None) -> BuildingTopology
```

`BuildingTopology` (frozen dataclass, same "immutable snapshot" convention every other
`*Snapshot`/read-model in this codebase already follows — `CameraCoverageSnapshot`,
`ObservableAssetSnapshot`, `BuildingState` itself):

```python
@dataclass(frozen=True)
class BuildingTopology:
    building: Building                        # reference only, never copied
    navigation_graph: NavigationGraph          # reused as-is, unchanged
    camera_topology: CameraTopology            # reused as-is, unchanged
    camera_overlap: CameraOverlapSnapshot       # NEW
```

`BuildingTopologyBuilder.build(building, calibrations=None)` is a **pure function**, the same
shape as `NavigationGraphGenerator.build(building)`:

1. Call `NavigationGraphGenerator().build(building)` — unchanged, existing code.
2. Read each `Camera.zone_ids` per floor → call the existing
   `build_topology_from_navigation_graph(navigation_graph, camera_zone_ids)` — unchanged,
   existing code, **first real caller**.
3. Call the new `compute_camera_overlap_snapshot(cameras, calibrations)` (§3.3).
4. Return `BuildingTopology(building, navigation_graph, camera_topology, camera_overlap)`.

Never persisted independently of the Building that produced it — rebuilding it after any
Designer edit always reproduces the same result, exactly like `NavigationGraph` today.

### 3.3 New concept: camera-to-camera overlap

`building_topology/overlap.py` reuses — never reinvents — the exact geometry primitives
`camera_coverage/discovery.py` already proved: `Camera.coverage_polygon()` for an
uncalibrated camera's raw Designer-authored sector, or the calibrated sector-fan derivation
`camera_coverage/discovery.py::_sector_polygon()` already performs when a
`CalibrationProfile` exists; polygon intersection via `visibility/geometry.py::
point_in_polygon`/`segment_intersection` (the same functions `camera_coverage/discovery.py`
already imports).

```python
class OverlapState(Enum):
    NOT_OVERLAPPING = auto()   # sectors provably don't intersect
    PARTIAL = auto()           # sectors intersect, but neither is wholly inside the other
    SUBSTANTIAL = auto()       # one sector's own polygon lies almost entirely inside the other
    UNKNOWN = auto()           # same "no calibration, no usable sector" honesty as CoverageState.UNKNOWN

@dataclass(frozen=True)
class CameraOverlap:
    camera_a_id: str
    camera_b_id: str
    state: OverlapState
    overlap_polygon: Optional[Tuple[Tuple[float, float], ...]] = None

@dataclass(frozen=True)
class CameraOverlapSnapshot:
    by_pair: Mapping[Tuple[str, str], CameraOverlap]
    def overlap_for(self, camera_a_id, camera_b_id) -> CameraOverlap: ...
    def cameras_overlapping(self, camera_id) -> Tuple[str, ...]: ...
```

Only cameras on the **same floor** are ever compared (mirrors `camera_coverage`'s own
floor-scoping). This is the one net-new geometric computation this milestone would introduce;
everything else is composition of existing, already-tested code.

### 3.4 Camera-to-zone assignment: widen the existing UI, not the model

`Camera.zone_ids` (`models/engineering_asset.py`) is already a tuple — **no model change**.
Only `designer/widgets/property_panel.py` changes: replace the single `camera_zone`
`QComboBox` with the same `_populate_zone_checklist` pattern `Speaker` already uses, so an
operator can honestly record every zone a camera's FOV actually reaches. This is what makes
`build_topology_from_navigation_graph` produce a real, non-trivial `CameraTopology` for a
project like `p3.syn` instead of an empty one.

### 3.5 Wiring into Live Mode

`live_runtime/factory.py::build_live_runtime()` gains **one new optional parameter**,
`building_topology: Optional[BuildingTopology] = None` — additive, mirrors the existing
`cross_camera_identity_resolver` optional-seam convention exactly. When supplied, and no
explicit `cross_camera_identity_resolver` was already passed, the factory constructs
`RuleBasedCrossCameraIdentityResolver` from `building_topology.camera_topology` —
`cross_camera_identity`'s own classes, first real production caller. Every existing call site
and test that doesn't pass `building_topology` keeps behaving exactly as it does today.

`designer/live_runtime_controller.py` (or wherever the Designer already resolves the current
`Building` to start Live Mode) builds one `BuildingTopology` via `BuildingTopologyBuilder.
build(building)` and threads it into `build_live_runtime(...)`.

**Note for implementation, not now**: the Shadow-Mode Prediction milestone already found and
reverted a similar factory convenience-parameter once, because it tripped an existing
architecture-guard test. Whoever implements §3.5 needs to re-check that guard test first —
flagged here so it isn't rediscovered the hard way.

---

## Zone topology — explicit answer

No new zone graph is proposed. `navigation.graph.NavigationGraph`, built by
`NavigationGraphGenerator.build(building)`, **is** the Zone → Zone → Corridor → Lobby →
Staircase graph the brief describes — `Node.space_type` already reads `Zone.zone_type`,
which already has `Corridor`, `Lobby`, and `Stair Lobby` as first-class values.
`BuildingTopology.navigation_graph` simply exposes this existing object under one stable name
alongside the camera-facing views. Camera reasoning (`CameraTopology`, `CameraOverlapSnapshot`)
is explicitly built **on top of** this graph (§3.2 step 2), never the reverse.

---

## Future compatibility — why no redesign is needed later

| Future feature | How it's already served without changing `BuildingTopology`'s shape |
|---|---|
| Camera coverage visualization | `BuildingTopology` already exposes `navigation_graph` (geometry) + `camera_topology`; `Camera.coverage_polygon()`/`camera_coverage` package supply sector polygons untouched. |
| Virtual zones | `models/connectable_space.py` was explicitly designed for this: *"adding a future type... means adding one entry here, not touching Door, the Property Panel, or the Navigation Graph builder again."* A Virtual Zone becomes one new `CONNECTABLE_SPACE_TYPES` entry; `BuildingTopology.navigation_graph` picks it up automatically since it only ever reflects whatever `NavigationGraphGenerator` produces. |
| Tripwires | A new, separate package that *reads* `BuildingTopology` (camera sector geometry + zone adjacency) the same way `camera_coverage`/`stair_flow`/`prediction_evaluation` already read `NavigationGraph`/`BuildingState` today — never requires `BuildingTopology` itself to change shape. |
| Cross-camera tracking | Already the point of §3.5 — `BuildingTopology.camera_topology` is literally the object `cross_camera_identity.resolver`/`transition_model` already consume; this design is what finally supplies it in production. |
| Digital Twin | Unaffected — `BuildingTopology.building` is a reference to the same `Building`, never a duplicate. Digital Twin *is* `Building`, per this project's own standing rule. |
| Occupancy / route prediction | `predictive_dataset`/`ai_registry`/`prediction_evaluation` already consume `NavigationGraph`-shaped features; nothing about that surface changes. |
| Multi-floor tracking | `NavigationGraph` already spans floors via Stair edges, and `build_topology_from_navigation_graph` already reuses those same cross-floor edges — multi-floor camera transitions become representable the moment a floor actually has calibrated, zone-assigned cameras. (Floor 1 in `p3.syn` has none today — a content gap, not an architecture gap.) |

---

## Exact files that would change

**New:**

| File | Purpose |
|---|---|
| `building_topology/__init__.py` | Package marker. |
| `building_topology/models.py` | `BuildingTopology`, `CameraOverlap`, `CameraOverlapSnapshot`, `OverlapState`. |
| `building_topology/overlap.py` | `compute_camera_overlap()` / `compute_camera_overlap_snapshot()` — the one new geometric computation. |
| `building_topology/builder.py` | `BuildingTopologyBuilder.build(building, calibrations=None)`. |
| `tests/test_building_topology.py` | Unit tests, following this codebase's one-test-file-per-package convention. |
| `docs/architecture/building_topology.md` | Implementation-time architecture doc (this file is the pre-implementation proposal; a real milestone doc follows the same naming pattern as `docs/architecture/cross_camera_identity.md` once built). |

**Modified (minimum only):**

| File | Change |
|---|---|
| `live_runtime/factory.py` | One new optional `build_live_runtime(building_topology=None)` parameter; auto-constructs `RuleBasedCrossCameraIdentityResolver` when supplied and no resolver was already given. |
| `designer/live_runtime_controller.py` | Build one `BuildingTopology` via `BuildingTopologyBuilder.build(building)` and pass it to `build_live_runtime(...)`. |
| `designer/widgets/property_panel.py` | Camera's zone assignment: single `QComboBox` → the existing `_populate_zone_checklist` multi-zone pattern (`Speaker` already uses it). |

**Explicitly NOT changed** — stated so nothing gets duplicated by accident:

- `models/building.py`, `models/floor.py`, `models/zone.py`, `models/camera.py`,
  `models/door.py`, `models/exit.py`, `models/staircase.py`, `models/connectable_space.py` —
  Building stays the single source of truth.
- `navigation/*` — `NavigationGraph`/`NavigationGraphGenerator` already are the zone/building
  topology graph.
- `cross_camera_identity/*` — `CameraTopology`/`CameraTransition`/
  `build_topology_from_navigation_graph` already are the camera topology model; they only
  need a real caller, not new code.
- `camera_coverage/*` — Camera↔Asset coverage stays exactly as scoped; camera↔camera overlap
  is a deliberately separate, new concept, not an extension of this package's own job.

---

## Why this is the smallest architectural seam

- Every piece needed for **Building topology** and **Zone topology** already exists and is
  proven across ~5,000+ passing tests: `Building`/`Floor`/`Zone`/`Door`/`Exit`/`Staircase`
  and `NavigationGraph`. This proposal adds **zero** new model classes for either — it only
  composes what's already there.
- **Camera topology** (adjacency + transition time) already exists as engineering-verified
  code with its own architecture doc (`docs/architecture/cross_camera_identity.md`) — but has
  never been called outside its own test file. The smallest seam is to finally call it from
  one real place, not to rebuild it.
- The **only** genuinely new concept required is camera-to-camera geometric overlap — nothing
  in this codebase computes that today. It reuses the exact polygon-intersection primitives
  `camera_coverage/discovery.py` already proved (`visibility.geometry.point_in_polygon`/
  `segment_intersection`), just pointed at camera-vs-camera instead of camera-vs-asset.
- The shape matches every prior milestone in this codebase's own history: one new,
  narrow-scope top-level package that *reads* (never duplicates) `Building`/`NavigationGraph`,
  wired through **one** new optional constructor parameter on `build_live_runtime()`.
  `building_topology/` is that same shape, and `BuildingTopologyBuilder.build(building)`
  mirrors `NavigationGraphGenerator.build(building)` exactly.

---

## Class diagram (text)

```
Building (existing)
 └─ Floor (existing)
     ├─ Zone (existing)        zone_type: Generic | Room | Corridor | Lobby | Stair Lobby | ...
     ├─ Door (existing)        zone_a_id, zone_b_id
     ├─ Exit (existing)        zone_id
     ├─ Staircase (existing)   from_zone_id/to_zone_id, from_floor_id/to_floor_id
     ├─ AssemblyPoint (existing)
     └─ Camera (existing, EngineeringAsset)   zone_ids: Tuple[str,...], position, rotation, horizontal_fov, max_range

NavigationGraph (existing) ── derived from Building by NavigationGraphGenerator.build()
 ├─ Node (existing)        wraps Zone | AssemblyPoint | "Outside"
 ├─ Edge (existing)        wraps Door | Exit | Staircase; walking_distance
 └─ FlowRegion (existing)  groups related Edges

CameraTopology (existing, cross_camera_identity/topology.py)
 └─ CameraTransition (existing)   from_camera_id, to_camera_id, min/max/expected_transition_time
 -- derived from NavigationGraph + Camera.zone_ids via build_topology_from_navigation_graph()

CameraCoverageSnapshot (existing, camera_coverage/models.py)
 └─ CameraCoverage → AssetCoverage (existing)
 -- derived from CalibrationProfile + observable regions (Camera vs. Asset, unrelated axis)

══════════════════════════ NEW ══════════════════════════
BuildingTopology (NEW, building_topology/models.py)          one read-only facade per loaded Building
 ├─ building: Building                          reference only
 ├─ navigation_graph: NavigationGraph            reused, unchanged
 ├─ camera_topology: CameraTopology              reused, unchanged
 └─ camera_overlap: CameraOverlapSnapshot        NEW
      └─ CameraOverlap (NEW)   camera_a_id, camera_b_id, state (NOT_OVERLAPPING|PARTIAL|SUBSTANTIAL|UNKNOWN), overlap_polygon

BuildingTopologyBuilder (NEW, building_topology/builder.py)
 └─ build(building, calibrations=None) -> BuildingTopology     pure function, mirrors NavigationGraphGenerator.build()
```

## Data-flow diagram (text)

```
p3.syn --Serializer.load()--> Project.building : Building                          [existing, unchanged]
                                    │
                                    ├─> NavigationGraphGenerator.build(building) -> NavigationGraph   [existing, unchanged]
                                    │
                                    └─> BuildingTopologyBuilder.build(building)                       [NEW — orchestrates below]
                                            │
                                            ├─ reuses the SAME NavigationGraph (no second build)
                                            │
                                            ├─ build_topology_from_navigation_graph(
                                            │      navigation_graph, camera_zone_ids)
                                            │      -> CameraTopology                                  [existing — first real caller]
                                            │
                                            ├─ compute_camera_overlap_snapshot(cameras, calibrations)
                                            │      -> CameraOverlapSnapshot                            [NEW]
                                            │
                                            └─> BuildingTopology(building, navigation_graph,
                                                                 camera_topology, camera_overlap)

BuildingTopology
    └─> live_runtime.factory.build_live_runtime(building_topology=...)   [NEW optional param,
                                                                             mirrors cross_camera_identity_resolver's
                                                                             own existing optional-seam pattern]
           └─> cross_camera_identity.resolver.RuleBasedCrossCameraIdentityResolver(
                    building_topology.camera_topology)                   [existing class — first real production caller]

           └─> (future, no BuildingTopology shape change needed)
                  tripwire / camera-coverage-visualization / virtual-zone consumers
                  read the same BuildingTopology instance
```

---

## Open items surfaced by this investigation (not fixed, disclosed)

1. `p3.syn`'s four cameras have no `zone_ids` assigned — until §3.4 ships (or an operator
   manually assigns zones today, via the existing single-zone combo), `CameraTopology` for
   this exact project would be empty.
2. Floor 1 has zero cameras — cross-floor camera-topology transitions can't be exercised
   until at least one camera is placed there.
3. `CameraTopology.is_plausible_transition()` today returns a **boolean** plausibility
   (inside/outside a min/max time window), not a graded confidence score. A continuous
   confidence is a natural, backward-compatible extension of `CameraTransition` later, but
   nothing downstream asks for one yet — not designed here, per this codebase's own
   "don't build for a consumer that doesn't exist yet" discipline.
4. Zone 1 on Ground Floor is a structural hub (every door, the only exit, and the stair
   landing all converge on it) — worth flagging for future flow-region/predictive work, out
   of scope for this milestone.

**Nothing above requires implementation to proceed. Awaiting review/approval before any code
is written.**
