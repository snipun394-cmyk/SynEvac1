# Camera Calibration & World Coordinate Projection

> See `docs/architecture/camera_calibration_and_world_projection.md` for the **Real Camera Calibration & World-Coordinate Validation** milestone built on top of this one — the practical calibration workflow, error validation/RMSE, `CalibrationQuality`, and an honest classification of what has (and has not) been validated against a real measured scene.

Status as of this milestone: the geometric layer converting IMAGE-space detections into building-space (world) positions — a person's bounding box now yields a real `(x, y)` floor-plan position, a resolved zone, and (once behavior recognition runs) a world-space velocity in meters/second. No automatic calibration, no OpenCV routines, no pose estimation — "assume feet touch the floor" is the one geometric assumption this milestone makes.

## 1. Pipeline (current)

```
CameraFrame
    -> human_detection.yolo_human_detector.YOLOHumanDetector       (unchanged, existing)
    -> tracking.simple_tracker.SimpleSingleCameraTracker             (unchanged, existing)
    -> tracking.tracked_human.TrackedHuman                          (unchanged, existing)
    -> camera_calibration.projection.WorldProjector                  (NEW -- this milestone)
    -> camera_calibration.projection.WorldProjection                 (NEW)
    -> behavior_recognition.rule_based_recognizer.RuleBasedBehaviorRecognizer  (extended, additive)
    -> behavior_recognition.observation.BehaviorObservation           (extended, additive -- world_metrics)
    -> cross_camera_identity.resolver.RuleBasedCrossCameraIdentityResolver  (unchanged, existing)
    -> [world-space-augmented] RawHumanDetection                      (extended, additive fields)
    -> live_camera_pipeline.identity_resolver.IdentityResolver       (unchanged interface; _to_detection() field mapping updated)
    -> virtual_camera.detection.Detection                             (extended, additive fields)
    -> multi_camera_fusion.engine.MultiCameraFusionEngine              (unchanged, existing)
    -> building_state.estimator.BuildingStateEstimator                 (unchanged, existing)
```

## 2. Investigation findings (Phase 1)

Verified directly against the current source:

1. **Camera coordinate system**: `models.camera.Camera` (via `EngineeringAsset`) already has `position: (x, y)` in floor-plan **meters** — the same 2D coordinate system `models.zone.Zone` uses (`Zone.x/y/width/height`, also meters). `Camera.coverage_polygon()` confirms the axis convention directly: 0° faces `+X`, angle increases clockwise (matching Qt's rotation transform).
2. **Camera rotation representation**: `Camera.rotation` (degrees) is **yaw only** — rotation about the vertical axis, within the horizontal floor plane. No pitch or roll exists anywhere in the codebase before this milestone; `designer/items/camera_item.py` (the placement tool) is purely a 2D `QGraphicsItem` with `setRotation()` — confirmed no 3D orientation UI exists.
3. **Camera elevation representation**: `EngineeringAsset.mount_height: float = 3.0` — a genuine, already-existing field, meters above the camera's own floor surface. Reused directly as `CameraExtrinsics.mount_height`; **not** duplicated.
4. **Camera FOV representation**: `Camera.horizontal_fov` (degrees) drives the 2D visibility coverage wedge (`visibility/engine.py`'s ray-casting). `Camera.resolution` ("1920x1080") is an explicitly **display-only string** — its own docstring states it is "never parsed, validated, or used to configure an actual video pipeline." Neither is a real pinhole-camera intrinsic (focal length in pixels) — `CameraIntrinsics.from_horizontal_fov()` is the one deliberate, explicitly-assumption-documented bridge from an FOV angle to an equivalent focal length (assumes square pixels).
5. **Floor coordinate system**: `models.floor.Floor` stores no absolute elevation at all — floor elevation is always *derived* from cumulative floor heights below it (`Building.floor_elevation()`), never stored. This milestone's own world Z-axis is deliberately **floor-local** (Z=0 at this floor's own surface), matching `Zone`/`Detection.position`'s existing "floor_id + 2D (x,y)" convention — never a building-wide absolute elevation, since nothing else in this codebase carries one either.
6. **Pixel-to-world assumptions already present**: none. Grepped the entire repository for `focal|intrinsic|calibrat|homography|pixel_to_world|ground_plane` — the only hits were unrelated (`confidence` fields, and `cross_camera_identity.observation`'s own comment explicitly noting *"no world-coordinate calibration exists yet"*, confirming the gap this milestone closes).
7. **Existing geometry utilities**: `visibility.geometry.point_in_polygon()` (ray-casting point-in-polygon test) and `Zone.contains(x, y)` (rectangle containment) both already exist and are reused directly for zone lookup — never reimplemented. `navigation.graph.NavigationGraph` (zone-adjacency graph, `Edge.walking_distance` in real meters) is reused for the optional `nearest_navigation_node()` utility.

## 3. Camera model (Phase 3)

`camera_calibration/camera_model.py`:
- `CameraIntrinsics` — `image_width/height` (pixels), `focal_length_x/y` (pixels, independent per axis), `principal_point_x/y` (pixels, defaults to image center). `from_horizontal_fov()` derives an equivalent focal length from `Camera.horizontal_fov` for a caller with no real lens datasheet.
- `CameraExtrinsics` — `position`/`mount_height` reuse `EngineeringAsset`'s own fields exactly; `yaw_degrees` reuses `Camera.rotation`'s own convention exactly. `pitch_degrees` (positive = tilted downward from horizontal) and `roll_degrees` (about the optical axis) are the two genuinely new angles this milestone introduces.
- `CalibrationProfile` — camera_id + floor_id + intrinsics + extrinsics, stored **separately from Camera Assets** (Phase 3's own requirement) in `camera_calibration.calibration.CalibrationRegistry` — the same "new registry, keyed by id, owns this new concern" pattern `cross_camera_identity.identity_registry.IdentityRegistry` already established.

## 4. World projection mathematics (Phase 4)

`camera_calibration/geometry.py` (pure math, no camera_id/Zone lookups) + `camera_calibration/projection.py` (the full pipeline + zone lookup):

1. **Ground contact point**: bottom-center of the bounding box (`((x1+x2)/2, y2)`) — "assume feet touch the floor," never a 3D skeleton/pose estimate.
2. **Pixel → world-space ray**: standard pinhole model. `camera_basis_vectors()` builds an orthonormal `(forward, right, down)` frame from yaw/pitch/roll (verified directly: unit length, mutually orthogonal, for arbitrary poses — `tests/test_camera_calibration.py::ProjectionConsistencyTests::test_11_basis_vectors_are_orthonormal_for_arbitrary_pose`). `pixel_ray_direction()` combines `forward + dx·right + dy·down` where `dx, dy` are the pixel offset from the principal point, scaled by focal length.
3. **Ray/floor-plane intersection**: `intersect_ray_with_floor()` solves for where the ray crosses `Z=0` (this floor's surface). Returns `None` — never a fabricated point — whenever the ray points level with or above the horizon (`direction.z >= 0`), or would only cross the floor plane behind the camera (`t < 0`).
4. **Zone lookup**: `Zone.contains(x, y)` (rectangle) or `visibility.geometry.point_in_polygon()` (when a zone has an explicit `polygon`) — both reused directly, never reimplemented.
5. **Navigation lookup** (a light, additive utility, not a `Detection` field): `nearest_navigation_node()` finds the closest Zone-type `navigation.graph.NavigationGraph` node by center distance.

Verified numerically by hand-computed trigonometry (not just unit-test assertions written after the fact) — e.g. a camera at 3m height, pitched 45° down, projects its exact image-center pixel to `3.0m` straight ahead (`height / tan(pitch)`), confirmed both analytically and by the passing test.

## 5. Calibration loader (Phase 5)

`camera_calibration/calibration_loader.py` supports exactly the three explicitly-requested paths:
- **Manual entry**: direct `CalibrationProfile(...)` construction (already possible via the plain dataclasses), or `calibration_from_camera(camera, pitch_degrees=..., roll_degrees=...)` — reuses an existing `models.camera.Camera`'s own position/mount_height/rotation/floor_id/id, asking only for the two angles Camera has no field for.
- **JSON**: `load_calibration_json()`/`save_calibration_json()`, round-trip verified (`test_11_calibration_round_trips_through_dict_serialization`).
- **Future OpenCV calibration**: explicitly **not implemented** (Phase 5's own instruction). `CameraIntrinsics`/`CameraExtrinsics` are already shaped to receive a real OpenCV camera matrix (`fx, fy, cx, cy`) and `solvePnP()` rotation/translation result without any change to these types — documented as the extension point, not stubbed as dead code.

## 6. Pipeline integration (Phase 6) and behavior_recognition extension

`LiveCameraPipeline` gained one further optional constructor parameter, `world_projector`, consulted only when `tracker` is also supplied. `_process_camera_cycle()` now runs: **Tracker → WorldProjection → BehaviorRecognizer → CrossCameraIdentity**, exactly this milestone's desired order.

"Behavior should now operate using world-space motion instead of pixel-space whenever calibration is available, gracefully fall back to image-space if calibration is unavailable" required a small, **additive** extension to `behavior_recognition` (not `camera_calibration`, which never imports it):
- `behavior_recognition.metrics.WorldTemporalMetrics` + `compute_world_metrics()` — a parallel, separate type from the existing pixel-space `TemporalMetrics`/`compute_metrics()` (zero regression risk — verified against all 21 pre-existing tests, unmodified).
- `BehaviorHistory` gained an optional `world_position` parameter on `append()` and a `recent_world()` accessor — a parallel store, never merged with pixel samples.
- `BehaviorObservation` gained one new optional field, `world_metrics: Optional[WorldTemporalMetrics] = None`.
- `BehaviorRecognizer.recognize()` gained one new optional parameter, `world_positions_by_track_id: Optional[Mapping[str, Tuple[float, float]]] = None` — a plain tuple mapping, **not** a `camera_calibration` type (this package still imports nothing from `camera_calibration`; the pipeline glue is the only bridge).
- `RuleBasedBehaviorRecognizer._classify()` uses `world_metrics.world_velocity` (against new, separate meters/second thresholds — 0.3 m/s stationary, 2.5 m/s running) in **preference** to pixel velocity whenever it's honestly computable, falling back to the existing pixel-space thresholds otherwise.

## 7. Detection enhancement (Phase 7)

`virtual_camera.detection.Detection` already had `position`/`floor_id`/`zone_id` (previously always `None` for the Live/YOLO path — `live_camera_pipeline/identity_resolver.py`'s own prior comment literally said *"no world-coordinate calibration exists yet"*). This milestone populates them rather than duplicating them with a new "world_position" field, and adds exactly two genuinely new fields: `world_velocity: Optional[float] = None`, `projection_confidence: Optional[float] = None`. `RawHumanDetection` gained the matching carrier fields (`world_position`, `world_velocity`, `projection_confidence`) so the pipeline glue can set them via `dataclasses.replace()` before `IdentityResolver.resolve()` — the same established pattern `state_evidence` already uses. All five fields are `None` whenever their upstream stage (calibration/behavior recognition) is not configured — never fabricated, verified directly in `tests/test_live_camera_pipeline_calibration_integration.py::NoWorldProjectorPreservesPriorBehaviorTests`.

## 8. Manual vs. Future OpenCV vs. Future Automatic Calibration

| | Manual (this milestone) | Future OpenCV Calibration | Future Automatic Calibration |
|---|---|---|---|
| Input | Direct construction / `calibration_from_camera()` / JSON | `cv2.calibrateCamera()`/`solvePnP()` output | Learned/self-calibrating |
| Precision | As accurate as the operator's measurements | Sub-pixel, lens-distortion-aware | Model-dependent |
| Status | Implemented | Not implemented (types already shaped to receive it) | Not implemented |

## 9. Files created / modified

**Created:**
- `camera_calibration/{__init__,camera_model,geometry,projection,calibration,calibration_loader}.py`
- `tests/test_camera_calibration.py` — 22 unit tests (Phase 8)
- `tests/test_live_camera_pipeline_calibration_integration.py` — 3 pipeline-integration tests (Phase 6/7)
- `tests/test_camera_calibration_architecture_guards.py` — 2 import-guard tests (Phase 11)
- `scripts/demo_camera_calibration.py` — offline demo (Phase 9)
- `scripts/benchmark_camera_calibration.py` — performance benchmark (Phase 10)
- `docs/architecture/camera_calibration.md` — this document

**Modified (additively, backward-compatible):**
- `live_camera_pipeline/human_detector.py` — `RawHumanDetection` gained `world_position`, `world_velocity`, `projection_confidence` (all `Optional`, default `None`).
- `virtual_camera/detection.py` — `Detection` gained `world_velocity`, `projection_confidence` (both `Optional`, default `None`; `position`/`floor_id`/`zone_id` already existed).
- `live_camera_pipeline/identity_resolver.py` — `_to_detection()` now passes through `raw.world_position`/`world_velocity`/`projection_confidence` instead of hardcoding `position=None`.
- `live_camera_pipeline/pipeline.py` — added `world_projector` constructor parameter; `_track_and_recognize()` renamed `_process_camera_cycle()` and extended with the projection step.
- `behavior_recognition/{metrics,behavior_history,observation,recognizer,rule_based_recognizer}.py` — additive world-space support (§6). Every pre-existing test in `tests/test_behavior_recognition.py` (21 tests) continues to pass unmodified.

**Unchanged (verified, not modified):** `tracking/*`, `human_detection/yolo_human_detector.py`, `cross_camera_identity/*`, `multi_camera_fusion/*`, `building_state/*`, `models/camera.py`, `models/zone.py`, `navigation/*`, `visibility/*`.

## 10. Performance

`scripts/benchmark_camera_calibration.py`, zero YOLO/tracker/behavior-recognizer inference:
- Basis-vector computation: ~0.0019 ms/call.
- Geometry (ray + floor intersection): ~0.0026 ms/call.
- Full projection incl. zone lookup (200 zones): ~0.011 ms/call.

## 11. What still remains

Automatic/OpenCV-based calibration, and pose estimation (a real skeleton would let a future milestone drop the "feet touch the floor" assumption entirely) remain explicitly out of scope. `CameraIntrinsics`/`CameraExtrinsics` are already shaped to receive a real OpenCV calibration result without any type change.
