# Virtual Camera System — Architecture

Status: **implemented (Simulation mode only)**. No Live CCTV integration accompanies this revision.

## 1. What this is

The Virtual Camera answers one question: *what would this Camera Asset's real CCTV feed produce,
right now, if it existed?* It generates a `Detection` per visible person per camera per tick — the
same shape a real detector (RTSP → YOLO → tracker) would eventually produce — using only
Designer-authored geometry and the existing simulation/Ground-Truth systems. No computer vision, no
video, no network connection exists anywhere in this milestone.

```
Camera Asset (models/camera.py)          -- where/how the camera looks
        +
Visibility Engine (visibility/engine.py) -- what it can geometrically see
        +
HumanObservationProvider (perception/)   -- who's there, and their classification/state
        =
VirtualCamera.detections_at(time) -> Detection stream
```

## 2. Package: `virtual_camera/`

- **`detection.py` — `Detection`** — one camera's sighting of one person at one instant: `camera_id`,
  `timestamp`, `occupant_id`, `floor_id`, `zone_id`, `position`, `confidence`, `classification`
  (reuses `perception.models.human_observation.HumanClassification`), `human_state` (reuses
  `HumanState`), `is_false_positive`. Deliberately **not** the same type as `HumanObservation` — a
  Detection is raw, per-camera, unfused output; `HumanObservation` is already a resolved per-person
  fact. `position` is the occupant's zone center — the same zone-granularity limitation
  `GroundTruthCameraProvider`'s own V1 docstring already discloses (no per-occupant exact-position
  feed exists anywhere in this codebase yet), not a new one introduced here.

- **`imperfections.py` — `DetectionImperfectionModel`** — Phase 4's optional realism controls:
  `detection_probability`, `false_positive_rate`, `confidence_variation`, `tracking_delay`, all
  defaulting to perfect detection (`is_perfect` short-circuits every RNG draw when true, which is
  what makes the stream deterministic by construction whenever imperfections are off). `seed` makes
  an imperfect run reproducible.

- **`camera.py` — `VirtualCamera`** — one Camera Asset's detector. Computes its visibility polygon
  **once**, at construction (a Camera's own geometry doesn't change mid-run; only occupant positions
  do — recomputing per tick would be pure waste given the Visibility Engine's per-call cost).
  `detections_at(time)`: pulls `HumanObservationProvider.observations_at(time - tracking_delay)`,
  keeps only observations on the camera's own floor whose `zone_id` falls in the visibility polygon's
  `visible_zone_ids`/`partially_visible_zone_ids`, applies missed-detection/confidence-variation
  imperfections, optionally appends one false positive.

- **`provider.py` — `DetectionProvider` / `SimulatedDetectionProvider`** — the actual seam (see §3).

## 3. The seam: `DetectionProvider`

```python
class DetectionProvider:
    def detections_at(self, camera_id: str, time: float) -> Tuple[Detection, ...]:
        raise NotImplementedError
```

Same "one interface, raises `NotImplementedError`, plus a Ground-Truth default" shape as every other
provider in this codebase (`CameraProvider`, `HumanObservationProvider`, `HazardProvider`,
`OccupancyProvider`, `PerceptionProvider`). `SimulatedDetectionProvider` is V1's one concrete
implementation — a simulation-backed stand-in, not a vision model.

**Nothing downstream is written against `SimulatedDetectionProvider` or `VirtualCamera` by name.**
Every future consumer (Perception fusion, a Command Center live feed, a dataset exporter) depends on
`DetectionProvider.detections_at()` and the `Detection` type alone.

## 4. Future replacement path (Phase 6)

```
RTSP  →  YOLO  →  Tracking  →  Detection Stream
```

becomes a second `DetectionProvider` implementation — e.g. `LiveDetectionProvider` — that:

1. Pulls frames from a real RTSP stream per Camera Asset (`Camera.connection.rtsp_address`, already
   stored as a placeholder field since the Camera Coverage milestone — see
   `models/engineering_asset.py`).
2. Runs a YOLO-class detector + tracker on each frame.
3. Resolves each tracked person's classification/state itself (a real detector does not have Ground
   Truth to consult) and emits the identical `Detection` shape this module already produces.

**Nothing else changes.** Perception, AI, RL, Advisory System, and Command Center all consume
`DetectionProvider.detections_at()` — swapping `SimulatedDetectionProvider` for
`LiveDetectionProvider` (or a `ReplayDetectionProvider` reading recorded video) is a single
construction-site change, the same substitution
`docs/architecture/perception_layer.md` §6 already proves safe one layer over
(`SimulatedReplaySensorProvider` swapping for a real sensor source with zero engine changes). A
`VirtualCamera`-shaped Simulation Mode and a `LiveDetectionProvider`-shaped Live Mode can also run
side by side per camera (`Camera.mode` already distinguishes them — see
`models/engineering_asset.DeviceMode`), letting a real deployment mix cameras that are and aren't
wired up yet.

## 5. What this milestone deliberately does NOT do

- No computer vision, no YOLO, no pose/action recognition of any kind.
- No RTSP/ONVIF/video decoding — `Camera.connection.*` fields stay inert placeholders.
- No exact sub-zone occupant position — inherits the same zone-granularity limitation the rest of
  this codebase's Ground-Truth-backed Perception adapters already carry.
- No change to `perception/`, `simulator/`, `ground_truth/`, `behavior/`, `behaviour_profile_resolver/`,
  AI Training, RL Training, Advisory System, Command Center, or Hazard Evolution — `virtual_camera/`
  reaches classification/state only through the `HumanObservationProvider` interface it's handed
  (enforced by a regex dependency-direction test, same convention as every other package boundary in
  this codebase — see `tests.test_virtual_camera.VirtualCameraPackageDependencyDirectionTests`).
