# Camera Calibration & World-Coordinate Validation

This document covers the **Real Camera Calibration & World-Coordinate Validation** milestone, built on top of the existing Camera Calibration & World Coordinate Projection milestone (`docs/architecture/camera_calibration.md` — unchanged, still the authoritative reference for `CameraIntrinsics`/`CameraExtrinsics`/`WorldProjector`/the pipeline integration). That milestone built the geometry; this one asks the harder question: **has any of it ever been checked against a real, measured place?**

Read this document for: what a practical calibration workflow looks like for one real fixed CCTV camera, how to quantitatively validate a calibration's error, the new `CalibrationQuality` status vocabulary, the Designer UI surfacing it, and — most importantly — an honest, unambiguous classification of what has and has not actually been validated in this environment.

## 1. Four categories that must never be collapsed into one

| Category | What it means | What in this codebase currently qualifies |
|---|---|---|
| **MATHEMATICALLY TESTED** | Hand-computed trigonometry, verified against deterministic unit tests. Proves the *math* is correct. Proves nothing about any real camera. | `camera_calibration/geometry.py`, `projection.py`, `calibration_solver.py`, `validation.py` — all of `tests/test_camera_calibration.py`, `tests/test_camera_calibration_validation.py`, `tests/test_camera_calibration_failure_modes.py` |
| **REAL VIDEO TESTED** | Real YOLO detections from a real recorded video genuinely flow through `WorldProjector` and reach `LiveOccupant`/`BuildingState`/`CrowdIntelligenceEngine`/`TrajectoryIntelligenceEngine`/`EvacuationProgressEngine`. Proves the **wiring**. Does **not** prove the resulting meter values are metrically accurate, because the camera geometry used was assumed, not measured. | §6 below (`vtest.avi` + an illustrative, explicitly-labeled-unvalidated calibration) |
| **REAL METRICALLY CALIBRATED** | A calibration fitted or entered from an *actual, measured* real-world scene, with error quantified against *held-out, genuinely measured* reference points (`CalibrationQuality.rmse_m` genuinely populated). Proves accuracy, in meters, for one specific real camera. | **Not yet achieved anywhere in this codebase.** No physical scene has been measured. §7 explains exactly what closing this gap requires. |
| **PHYSICAL CCTV VALIDATED** | The above, running against a live, physical camera feed rather than a recorded file. | **Not yet achieved.** Requires physical CCTV access, out of scope for this milestone by explicit instruction. |

Every claim below is tagged with exactly one of these four labels. None of the new code in this milestone upgrades the "REAL VIDEO TESTED" row into "REAL METRICALLY CALIBRATED" — that upgrade only ever happens when a real tape-measure/laser-measure scene is fed through the new tooling.

## 2. What already existed (Phase 1 investigation)

Verified directly against source before writing anything:

1. `CameraIntrinsics` (`fx, fy, cx, cy`, resolution) and `CameraExtrinsics` (`position`, `mount_height`, `yaw/pitch/roll`) already fully modeled the pinhole camera this milestone needed — no new competing type was created.
2. `WorldProjector.project(camera_id, bounding_box, confidence)` already implemented bounding-box → ground-contact-point → ray-cast → floor-plane intersection → zone lookup, returning `WorldProjection(world_position, floor_id, zone_id, projection_confidence)`. Never fabricates a value it cannot honestly derive.
3. `CalibrationRegistry` (camera_id → `CalibrationProfile`) and `calibration_loader.py` (`calibration_from_camera()`, JSON round-trip) already existed — the "store/serialize a calibration" mechanism needed no new type.
4. `LiveCameraPipeline`'s `world_projector` seam, and `live_runtime.factory.build_live_runtime()`'s own `world_projector` parameter, were already wired and already proven (offline) to reach `Detection`/`LiveOccupant`.
5. Every downstream consumer already checked — `crowd_intelligence/`, `trajectory_intelligence/`, `evacuation_progress/` — already honestly degrades when `world_position is None` (grepped directly; see e.g. `crowd_intelligence/queue.py`'s own "never fabricate a queue from zone occupancy alone" comment, `evacuation_progress/ledger.py`'s own "no world_position means..." comment). **None of these packages needed to change** — Phase 9/10/11's own instruction not to change them unless a genuine bug was found; none was found.
6. What was genuinely missing: a way to go from "a person with a tape measure standing in the actual scene" to a `CalibrationProfile`, a way to quantify that calibration's error, and any UI/CLI surface for either.

## 3. The practical calibration workflow (Phase 2/3)

Two supported paths, both producing the exact same `CalibrationProfile` type — never a competing calibration model:

**Manual entry** — an operator who already knows (or has a datasheet for) `yaw_degrees`/`pitch_degrees` directly enters them (`calibration_from_camera()`, unchanged from the original milestone, or direct construction).

**Correspondence-fitted** (`camera_calibration/calibration_solver.py`, new) — the practical case for almost any real, already-mounted camera, where nobody knows the exact tilt angle by eye. The operator supplies:
- `mount_height` (tape/laser measure — a GIVEN input, never fitted)
- `camera_position` (floor-plan meters — a GIVEN input, never fitted)
- Image resolution + FOV (or focal length)
- ≥3 `(pixel, world)` correspondences — a paused frame's pixel coordinate, paired with that exact spot's already-measured real-world position

`solve_calibration_from_correspondences()` fits `yaw_degrees`/`pitch_degrees` (roll held at the caller's supplied value, normally `0.0` — fitting a third angle from a handful of points is poorly conditioned) by minimizing squared world-space reprojection error, using the **exact same** `pixel_ray_direction()`/`intersect_ray_with_floor()` forward model `WorldProjector` itself uses — never a second projection algorithm. Uses `scipy.optimize.minimize` (Nelder-Mead) — `scipy` is already a transitive dependency (`scikit-learn` requires it), not a new framework.

`CalibrationSolveError` is raised (never a silent bad fit) when fewer than 3 correspondences are supplied, or when the fitted pose cannot project any of them onto the floor plane at all.

**Status: MATHEMATICALLY TESTED** — `tests/test_camera_calibration_validation.py::SolveCalibrationFromCorrespondencesTests` recovers known yaw/pitch (e.g. 15°/40°, or -20°/35° at a different height/position) from synthetic correspondences to within `1e-2` degrees.

## 4. `scripts/calibrate_camera_scene.py` (Phase 3)

A CLI tool, not a second Designer application. Takes one JSON "scene" file (camera_id, floor_id, resolution, FOV or focal length, `camera_position`, `mount_height`, either `yaw_degrees`/`pitch_degrees` directly OR a `correspondences` list, and an optional, separate `validation_points` list), and:

1. Builds a `CalibrationProfile` (manual entry or correspondence-fit, per §3).
2. If `validation_points` were supplied, validates against them (§5) and attaches the resulting `CalibrationQuality` to the saved profile.
3. Saves via the existing `calibration_loader.save_calibration_json()` — the same file format Designer's own calibration dialog (§8) loads.

```
python scripts/calibrate_camera_scene.py scene.json --out calibration.json
python scripts/calibrate_camera_scene.py scene.json --validate-only
```

A lightweight, optional point-picking helper (`--pick-points <image>`) opens a single `cv2` window — click a pixel, it's recorded — for building the `correspondences`/`validation_points` pixel side directly off a real paused frame. Not an annotation tool, not a second GUI framework: one window, click, `q` to finish.

**Status: MATHEMATICALLY TESTED end-to-end** against a synthetic scene (`scipy`-fitted yaw/pitch recovered to `<1e-4` degrees; saved/reloaded JSON round-trips exactly, including the attached `CalibrationQuality`). **Not yet REAL METRICALLY CALIBRATED** — no real scene has been run through it (§7).

## 5. Calibration validation (Phase 4) — `camera_calibration/validation.py`

`project_pixel_point(pixel, calibration)` — the one new geometric primitive this milestone adds to `camera_calibration` itself: projects an exact, already-known pixel (no bounding box, no "assume feet touch the floor") through the identical ray-cast `WorldProjector` uses. Needed because validating a calibration means projecting a *marked floor point*, which has no bounding box at all.

`validate_calibration(calibration, reference_points)` projects every supplied `ReferencePoint` (pixel + its already-known real-world position) and reports:

- `reference_point_count` — every point supplied, including ones that failed to project at all (a calibration that cannot even project half its own reference points is itself a genuine finding, never silently excluded from the count)
- `validated_point_count` — how many of those actually produced a position
- `mean_error_m` / `median_error_m` / `max_error_m` / `rmse_m` — computed only over the points that *did* project; `None` (never a fabricated `0.0`) when zero points projected at all

**Status: MATHEMATICALLY TESTED** — `tests/test_camera_calibration_validation.py::ValidateCalibrationTests` confirms a perfect calibration reports zero error against its own ground truth, a hand-computed 0.5m offset reports exactly 0.5m mean/RMSE error, and unprojectable points are counted but never averaged in.

## 6. Real video: the honest wiring proof (Phase 5/6/7/9/10/11)

`vtest.avi`'s actual physical camera geometry (mount height, tilt, position in its own courtyard) was **never measured** — no physical access to that scene exists, and this milestone explicitly refuses to fabricate it (per its own Phase 5 instruction). Instead, an **explicitly labeled, illustrative, unvalidated** calibration (`mount_height=3.5m`, `pitch=28°`, `yaw=0°`, assumed `horizontal_fov=60°` — plausible CCTV-mounting values, nothing more) was used purely to prove the **wiring**, run through the exact real commit-`18d5099` YOLO pipeline (`UltralyticsYOLOBackend` → `YOLOHumanDetector` → `SimpleSingleCameraTracker` → `RuleBasedBehaviorRecognizer` → `build_live_runtime()`):

- Real detections' bounding boxes → ground-contact point → `WorldProjector` → real (non-`None`) `world_position` values in meters reached `LiveOccupant` and `BuildingState.occupant_tracks` (a real `FusedTrack` with a real `zone_id` and real `HumanState.WALKING`).
- An illustrative `Zone`/`Exit` (placed where real projected positions happened to cluster, again never surveyed) resolved `zone_id` correctly for every projected position inside it.
- `world_velocity` values genuinely computed from real frame-to-frame world-position history landed in the 0.2–2.5 m/s range — plausible for human walking pace, but **explicitly not claimed as accurate** (Phase 7's own "do not claim precise human speed until calibration error is known" instruction), since the underlying geometry is assumed.
- `CrowdIntelligenceEngine.compute()`, fed the same real projected positions, produced a genuine worked sequence against the illustrative exit: `approaching_count`/`queue_candidate_count` rose to 5 and fell back to 0 repeatedly across the run, driven entirely by real people's real (illustratively-projected) movement — proving the **wiring**, not a validated crowd measurement.
- `TrajectoryIntelligenceEngine.compute()` reported `distance_travelled`/`net_displacement`/`current_speed` values that were consistently meter-scale (single digits), never pixel-scale (hundreds) — confirming genuine world-space arithmetic, not an accidental pixel/meter mix-up.
- `EvacuationProgressEngine.compute()` correctly attributed exit crossings (`unique_exited_count=13` in one run) and computed `evacuation_progress_fraction` honestly from `known_active`/`known_exited` counts. **No code change was made to `evacuation_progress/`** — its existing `LIKELY_EXITED`-vs-`CONFIRMED` honesty boundary (position-based exit attribution never silently upgraded to a stronger claim just because a calibration object happens to exist) is untouched and unaffected by this milestone.

**Status: REAL VIDEO TESTED.** Every number above came from real YOLO detections on real recorded video. **Not REAL METRICALLY CALIBRATED** — the geometry feeding the projection was assumed, so the specific meter values (positions, distances, speeds) carry no accuracy guarantee; only the fact that the pipeline correctly propagates *whatever* calibration it is given is proven.

## 7. What REAL METRICALLY CALIBRATED actually requires

Closing this gap needs one thing this environment could not provide: a physically measured real scene. Concretely, per camera:

1. The camera's real mount height and floor-plan position (tape/laser measure).
2. A paused real frame from that camera.
3. ≥3 (ideally 5+) real, physically identifiable floor points, each with a measured real-world `(x, y)` and its corresponding pixel in the paused frame — split into a fitting set and a **separate, held-out** validation set (validating against the same points used to fit only proves convergence, not generalization).
4. Running `scripts/calibrate_camera_scene.py` against that scene, producing a `CalibrationProfile` with a genuinely populated `CalibrationQuality` (§8).

`docs/architecture/physical_cctv_access_checklist.md` §"Calibration Measurements" now lists exactly this, ready for the day physical site access exists.

## 8. `CalibrationQuality` (Phase 12) — the honest validated/unvalidated distinction

`camera_calibration/camera_model.py::CalibrationQuality` — `reference_point_count`, `validated_point_count`, `mean_error_m`, `median_error_m`, `max_error_m`, `rmse_m`, `validation_timestamp`. Attached to `CalibrationProfile.quality` (`Optional`, defaults to `None`) — **only** ever set by a caller actually running `validate_calibration()` and choosing to record the result (`scripts/calibrate_camera_scene.py` does this automatically when `validation_points` are supplied). Never derived from FOV/geometry alone (explicitly instructed against — a wide FOV or a steep pitch says nothing about whether anyone ever checked the result against reality).

Three, and only three, states:
- `quality is None` → **CALIBRATION: NOT CONFIGURED** (no `CalibrationProfile` registered at all) or **CONFIGURED — UNVALIDATED** (a profile exists, but nobody ever validated it)
- `quality.rmse_m is not None` → **CALIBRATION: VALIDATED — RMSE: X m**

Round-trips through `calibration_loader.calibration_to_dict()`/`calibration_from_dict()` — absent from the JSON entirely when `quality is None` (never a fabricated empty block).

**Status: MATHEMATICALLY TESTED** — `tests/test_camera_calibration_validation.py::CalibrationQualitySerializationTests`.

## 9. Designer integration (Phase 13)

The Camera Property Panel (`designer/widgets/property_panel.py`) gained exactly one status line and one action button, per this milestone's own "minimum useful visibility, not raw matrices in the main panel" instruction:

- A `Calibration` row showing one of the three states from §8 (`_refresh_calibration_status()`, called every time `show_camera()` runs — i.e., every time a different camera is selected).
- A `Calibrate Camera...` button opening a small dialog: **Load Calibration JSON** (the output of `scripts/calibrate_camera_scene.py`, warns — never silently accepts — a `camera_id` mismatch) and **Save Manual (angle-based) Calibration As...** (a quick export of a manual-entry calibration built directly from the Camera Asset's own existing `position`/`mount_height`/`rotation`/`horizontal_fov`, via the pre-existing `calibration_from_camera()` bridge). Correspondence-based fitting is explicitly **not** duplicated into this dialog — the note inside it points back to the CLI tool.

`PropertyPanel.calibration_registry` (a plain `CalibrationRegistry`) is owned **in-memory, per Designer session** — deliberately **not yet persisted** with the project file (no calibration field exists in `models.camera.Camera` or the `.syn` project format; adding one is a real future extension, out of scope here to avoid touching project-file versioning as part of this milestone). Loading/saving a calibration JSON to/from disk (§4/§8) is the durable path today.

**Status: MATHEMATICALLY/OFFLINE TESTED** — `tests/test_property_panel_calibration_status.py` (4 tests: default NOT CONFIGURED, CONFIGURED — UNVALIDATED once registered, VALIDATED shows RMSE, switching cameras shows each camera's own independent status), run headless against the real `MainWindow`/`PropertyPanel`/`CameraItem` classes.

## 10. Failure modes (Phase 14)

`tests/test_camera_calibration_failure_modes.py` (16 tests) plus the pre-existing `tests/test_camera_calibration.py::MissingAndInvalidCalibrationTests`:

| Failure mode | Behavior |
|---|---|
| Wrong/missing `camera_id` | `WorldProjector.project()` returns all-`None` — never raises |
| Invalid FOV (0°, 180°, negative) | **Fixed this milestone** — previously: `0°` raised an opaque `ZeroDivisionError`; `180°` silently returned a near-zero (`~2e-14`) focal length; negative FOV silently returned a negative focal length. Now: `CameraIntrinsics.from_horizontal_fov()` raises a clear `ValueError` for any FOV not strictly between 0° and 180° |
| Invalid/negative mount height | No new validation added — physically nonsensical but geometrically well-defined; never crashes, an operator-measurement responsibility |
| Camera looking above horizon / ray parallel to floor | Pre-existing, reconfirmed: `None`, never fabricated |
| Point behind the camera (negative `t`) | `None`, newly test-covered (a below-floor-plane mount height still pointing away from the floor) |
| Projection outside every known zone | Pre-existing, reconfirmed: `zone_id=None` |
| A zone deleted after `WorldProjector` construction | `WorldProjector` holds an immutable snapshot at construction — a caller must reconstruct it with the updated zone list; documented as expected behavior, not a bug |
| Calibration loaded for the wrong resolution | **Newly exposed** as `camera_calibration.validation.resolution_mismatch(calibration, frame_width, frame_height)` — an opt-in diagnostic (never wired into the live per-cycle path, which would require changing `WorldProjector.project()`'s own signature — explicitly out of scope, "do not redesign the pipeline"). A caller (Designer, an operator pre-flight check) can run it against a real `CameraFrame`'s own `width`/`height` fields |
| Corrupt / incomplete / missing calibration JSON | Pre-existing `CalibrationLoadError`, reconfirmed |

## 11. Performance (Phase 15)

`scripts/benchmark_camera_calibration.py`, extended this milestone with a world-velocity benchmark (`behavior_recognition.metrics.compute_world_metrics`, 50 simultaneous occupants — a realistic count):

| Stage | Mean | p95 |
|---|---|---|
| Basis-vector computation | 0.0018 ms | 0.0020 ms |
| Geometry (ray + floor intersection) | 0.0025 ms | 0.0026 ms |
| Full projection incl. zone lookup (200 zones) | 0.0112 ms | 0.0189 ms |
| World-velocity calculation (50 occupants) | 0.0047 ms | 0.0056 ms |

All four combined remain roughly three orders of magnitude below real YOLO inference latency (~52ms/frame CPU, per `docs/architecture/human_detection.md` §16) — calibration/projection/tracking overhead is not, and is not expected to become, a bottleneck.

## 12. Architecture guards (Phase 16)

`tests/test_camera_calibration_architecture_guards.py` (unchanged, re-verified) mechanically confirms `camera_calibration/` imports nothing from `ai_engine`, `advisory_system`, `command_center`, `building_state`, `camera_manager`, `multi_camera_fusion`, `tracking`, `behavior_recognition`, `cross_camera_identity`, or `cv2`/`torch`/`ultralytics`/`onvif`. The two new modules (`calibration_solver.py`, `validation.py`) import only `scipy` (a plain least-squares dependency, not a forbidden one) and this package's own types — `camera_calibration` remains exactly what it was: a pure "where does this image point correspond to in the modeled world" geometry layer, never making an evacuation decision, never touching a hazard, Decision Policy, voice, control, FACP, or AI/RL system.

## 13. Files created / modified

**Created:**
- `camera_calibration/validation.py` — `project_pixel_point()`, `ReferencePoint`, `PointValidationResult`, `CalibrationValidationReport`, `validate_calibration()`, `resolution_mismatch()`
- `camera_calibration/calibration_solver.py` — `solve_calibration_from_correspondences()`, `SolvedCalibration`, `CalibrationSolveError`
- `scripts/calibrate_camera_scene.py` — the practical CLI calibration workflow (§4), plus an optional lightweight point-picking helper
- `tests/test_camera_calibration_validation.py` — 12 tests (solver + validation + `CalibrationQuality` serialization)
- `tests/test_camera_calibration_failure_modes.py` — 16 tests (§10)
- `tests/test_property_panel_calibration_status.py` — 4 tests (§9)
- `docs/architecture/camera_calibration_and_world_projection.md` — this document

**Modified (additively, backward-compatible):**
- `camera_calibration/camera_model.py` — `CalibrationProfile` gained `quality: Optional[CalibrationQuality] = None`; new `CalibrationQuality` dataclass; `CameraIntrinsics.from_horizontal_fov()` now raises `ValueError` for FOV not strictly between 0°/180° (a genuine bug fix, §10)
- `camera_calibration/calibration_loader.py` — `calibration_to_dict()`/`calibration_from_dict()` additively (de)serialize `quality` when present
- `scripts/benchmark_camera_calibration.py` — added the world-velocity benchmark (§11)
- `designer/widgets/property_panel.py` — `calibration_registry`, calibration status label, `Calibrate Camera...` dialog (§9)
- `.gitignore` — unchanged (calibration JSON files produced by this workflow are ordinary small text files, not gitignored by default; a real deployment's own calibration files are a legitimate project artifact an operator may choose to commit or not)
- `docs/architecture/physical_cctv_access_checklist.md` — added the "Calibration Measurements" section (§7)

**Unchanged (verified, not modified):** `camera_calibration/{camera_model.py's existing fields, geometry.py, projection.py, calibration.py}`, `live_camera_pipeline/pipeline.py`, `behavior_recognition/*`, `crowd_intelligence/*`, `trajectory_intelligence/*`, `evacuation_progress/*`, `live_runtime/factory.py`.

## 14. What still remains

A real, measured physical scene (§7) — the one thing that would upgrade this milestone's own "REAL VIDEO TESTED" row to "REAL METRICALLY CALIBRATED." Physical CCTV access (out of scope by explicit instruction). Persisting a calibration with the project file itself (currently in-memory per Designer session only, or a standalone JSON file on disk). Automatic/OpenCV-based calibration and pose estimation remain out of scope, unchanged from the original milestone's own §11.
