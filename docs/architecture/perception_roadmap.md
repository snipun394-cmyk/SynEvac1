# Perception Layer — Roadmap

Status: living document, tracking what's built vs. what's planned. This revision adds **Perception
V2 — Multi-Region Camera Observation** as an approved future enhancement. No code changes accompany
this revision.

---

## V1 — Complete

Built across four implementation phases, all still exactly as delivered (see
`docs/architecture/perception_layer.md` / `perception_layer_review_2.md` for the frozen design these
phases implement):

1. **Package foundation** (`perception/models/`, `perception/providers/provider.py` +
   `camera_provider.py`/`smoke_detector_provider.py`/`heat_detector_provider.py`) — `BuildingObservation`
   and its component types (`ObservedNodeState`, `ObservedOccupancy`, `ObservedEdgeState`,
   `PerceptionSystemStatus`), plain-Python and AI-framework-agnostic; abstract provider interfaces only.
2. **Ground Truth adapters** (`GroundTruthCameraProvider`, `GroundTruthSmokeDetectorProvider`,
   `GroundTruthHeatDetectorProvider`) — answer "what would each sensor observe," reading only Ground
   Truth Provider interfaces (`HazardProvider`/`OccupancyProvider`) and Building Model geometry, never
   simulator internals.
3. **Occupancy Estimation** (`perception/fusion/occupancy_estimation.py` —
   `OccupancyEstimator`/`EstimatedOccupancy`) — converts raw `CameraFrameObservation`s into per-zone
   estimates, preserving `UNOBSERVED` and resolving duplicate camera coverage via max-not-sum.
4. **Sensor Fusion** (`perception/fusion/sensor_fusion.py` — `SensorFusion`) — the one place a
   `BuildingObservation` is assembled, combining occupancy estimates, camera visibility, and
   smoke/heat detector readings.
5. **Verification tooling** — the Perception Debug Panel (`designer/widgets/perception_debug_panel.py`,
   `designer/perception_debug_runner.py`), a Designer-only visualization/validation surface, not part of
   the Perception Layer itself.

All four phases and the debug panel are implemented, tested, and unchanged by this revision.

---

## Known V1 limitation → motivation for V2

`CameraFrameObservation` (`perception/models/camera_observation.py`) is camera-scoped only — one
`estimated_occupant_count` per camera, no zone breakdown. `GroundTruthCameraProvider` resolves the
*full* list of zones a camera geometrically covers internally, then collapses it by summing into that
single number. Downstream, `OccupancyEstimator` and `SensorFusion` each need a `camera_id -> zone_id`
topology mapping to route that number — but each accepts only **one** zone per camera, and each is
handed its own **independently-injected copy** of that mapping.

Net effect: a camera whose real coverage spans two zones has its full count attributed to whichever one
zone an external caller happens to assign, while the other zone gets no signal at all — despite being
genuinely covered. This was flagged as a V1 limitation in the Occupancy Estimation phase itself,
confirmed structurally while building the Perception Debug Panel, and formally investigated in a
dedicated architecture review that evaluated four candidate designs (a per-zone-breakdown field on the
existing type, per-occupant detections, a new region abstraction, and a rejected "broadcast the
aggregate to every covered zone" variant).

---

## Perception V2 — Multi-Region Camera Observation *(planned, not implemented)*

**Approved architecture**: the investigation's Option C (introduce `CameraObservationRegion`), with one
refinement adopted from this review: rather than `CameraFrameObservation` and `CameraObservationRegion`
being two unrelated, parallel observation streams, **`CameraFrameObservation` becomes the parent object
for one captured frame, containing the regions observed within it.**

```
Camera
   │
   ▼
Frame                    (CameraFrameObservation — camera_id, timestamp)
   │
   ▼
Observed Regions          (List[CameraObservationRegion] — one per zone visible in this frame)
```

This preserves the natural camera → frame → observed-regions hierarchy instead of exposing two
independent, oddly-related API surfaces (a camera-level aggregate method alongside a disconnected
region-level method) — a single observation object, internally structured, rather than a bag of loose
siblings.

### Proposed shape (sketch for the implementation phase to finalize, not locked here)

- **`CameraFrameObservation`**: `camera_id`, `timestamp`, `regions: List[CameraObservationRegion]`
  (default empty). The current top-level `estimated_occupant_count`/`visibility_estimate`/`confidence`
  fields become redundant once a real per-region breakdown exists as the source of truth, and should be
  **removed rather than kept alongside** the regions — a camera-level total, if any consumer still wants
  one, is then a trivial *derived* sum over `regions`, never a second, separately-authored value that
  could drift from the region-level truth. This follows the same "derived, never stored redundantly"
  convention already applied throughout this codebase (`HazardNodeState.severity`, `Node`'s engineering
  properties).
- **`CameraObservationRegion`**: `zone_id`, `estimated_occupant_count: Optional[float]`,
  `confidence: Optional[float]`, `visibility_estimate: Optional[str]`. Deliberately **no** `camera_id` or
  `timestamp` on the region itself — both are inherited from the parent frame, since a region is only
  ever meaningful in the context of the frame containing it. `zone_id` does stay on the region (unlike
  `BuildingObservation`'s own dict-keyed `ObservedNodeState`/`ObservedOccupancy`, a region lives in a
  `List`, not a `Mapping`, so it has no external key to supply that identity instead).

### What changes vs. V1 (for future scheduling — not implemented now)

- `GroundTruthCameraProvider`: instead of summing across `_covered_zone_ids()`, emit one
  `CameraObservationRegion` per covered zone (each reading that zone's own true Ground Truth
  `occupant_count`, no more cross-zone summing) inside a single `CameraFrameObservation` per camera.
  This is a change to the *existing* `frame_observation_at()` return shape, not an additional sibling
  method as the original Option C sketch proposed — the parent/child refinement means there's only one
  observation stream to produce, not two.
- `OccupancyEstimator`: drops the externally-injected `camera_zone_assignments` mapping entirely;
  `estimate()` takes `List[CameraFrameObservation]` and groups by iterating each frame's `.regions` to
  get `(zone_id, count, confidence)` triples. The existing max-not-sum duplicate-resolution rule is
  unchanged — only how zone membership is discovered changes (carried in the data, not injected
  separately).
- `SensorFusion`: the same fix, for the same reason, on its own currently-duplicated
  `camera_zone_assignments` copy used for visibility routing.
- Test impact: Phase 2/3/4 suites need mechanical rework wherever they construct `camera_zone_assignments`
  or assert on today's summed `frame_observation_at` output — real effort, but not a redesign of test
  intent.

### What does NOT change

`BuildingObservation`'s own shape, the `ObservationEncoder` boundary, and the Rule-Based
Engine/RL-facing seam are entirely unaffected — this migration is contained to the
Camera → Occupancy Estimation → Sensor Fusion (visibility) path, upstream of `BuildingObservation`
assembly.

### Explicitly out of scope, even for V2 (future backlog items, not this enhancement)

- **Partial single-zone coverage** (a camera seeing only part of one large Zone) — regions remain
  zone-granular, all-or-nothing per zone; this was out of scope for every option evaluated, not just
  the chosen one.
- **Stairwell-shaft representation** — corridors already work today (they're Zones), but a stairwell
  shaft between landings has no Zone of its own in the Building Model at all. A Building-Model-level
  gap, not something any Perception-side option can fix on its own.
- **Per-occupant cross-camera deduplication** — the investigation's Option B (per-detection
  observations) territory. `CameraObservationRegion` is deliberately a forward-compatible stepping
  stone toward it (a region could later gain an optional `detections` field), not a replacement for it.
  Remains blocked on a prerequisite this repository doesn't have yet: a per-occupant Ground Truth
  position feed (flagged as an open gap since the very first Perception Layer architecture review and
  still unresolved).

### Rejected alternative (for traceability)

A cheaper-looking patch — broadcasting each camera's existing aggregate count as a candidate reading to
*every* zone it covers (`Mapping[camera_id, List[zone_id]]`, reusing today's max-resolution unchanged) —
was considered and rejected: it trades today's silent under-count for a systematic over-count risk
whenever a covered zone genuinely has close to that many people, without actually recovering the true
per-zone split. Not adopted.

---

## Status

Perception V2 — Multi-Region Camera Observation is **approved as a future architectural enhancement**
and recorded here for scheduling. **Not implemented.** The current V1 implementation
(`GroundTruthCameraProvider`, `OccupancyEstimator`, `SensorFusion`, and their existing tests) is
unchanged by this revision.
