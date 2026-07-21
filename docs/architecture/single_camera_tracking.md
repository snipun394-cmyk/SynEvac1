# Single-Camera Tracking Framework

Status as of this milestone: temporal tracking WITHIN one camera only. `tracking.simple_tracker.SimpleSingleCameraTracker` maintains stable local identities across consecutive frames from the same camera. This is **not** cross-camera re-identification, **not** behavior recognition, and **not** pose estimation.

## 1. Pipeline (current)

```
CameraFrame
    -> human_detection.yolo_human_detector.YOLOHumanDetector   (unchanged, existing)
    -> live_camera_pipeline.human_detector.RawHumanDetection   (unchanged, existing)
    -> tracking.simple_tracker.SimpleSingleCameraTracker        (NEW -- this milestone)
    -> tracking.tracked_human.TrackedHuman                      (NEW -- internal to this seam)
    -> [stabilized] RawHumanDetection                            (same type, local_track_id replaced)
    -> live_camera_pipeline.identity_resolver.IdentityResolver  (unchanged, existing)
    -> virtual_camera.detection.Detection                       (unchanged, existing)
    -> multi_camera_fusion.engine.MultiCameraFusionEngine        (unchanged, existing)
    -> building_state.estimator.BuildingStateEstimator           (unchanged, existing)
```

`IdentityResolver.resolve()` was **not** modified and does not receive `TrackedHuman` objects directly. Instead, `live_camera_pipeline.pipeline.LiveCameraPipeline` (the one integration point, §5) converts each matched/new `TrackedHuman` back into a plain `RawHumanDetection` — identical to the detector's own output except `local_track_id` is now the tracker's stable `track_id` instead of a raw per-frame index. This was a deliberate, investigated design choice (§4): `RawHumanDetection` is exactly the shape `IdentityResolver.resolve()` already requires, and reusing it means `IdentityResolver`, `Detection`, `MultiCameraFusionEngine`, and `BuildingState` needed **zero** changes.

## 2. Tracking ID vs. Occupant ID vs. Cross-Camera ReID

| Concept | Scope | Lifetime | Owner |
|---|---|---|---|
| **Tracking ID** (`TrackedHuman.track_id`) | One camera | Until the track expires (a few missed frames) | `tracking/` (this milestone) |
| **Occupant ID** (`Detection.occupant_id`) | Global (all cameras) | As long as `IdentityResolver`'s mapping/strategy says so | `live_camera_pipeline.identity_resolver.IdentityResolver` (unchanged, existing) |
| **Cross-Camera ReID** | Global, appearance-based | Future | Not implemented — remains `IdentityResolver`'s own named future seam |

A tracking ID is **local and temporary**: `"CAM-001-T7"` on one camera has no relationship whatsoever to any id on another camera, and is deleted the moment the tracked person is not re-matched within `max_missing_frames` cycles. `SimpleSingleCameraTracker` never invents, reads, or writes an `occupant_id` anywhere in its code (verified — `tracking/` does not import `virtual_camera.detection.Detection` at all).

## 3. Investigation findings (Phase 1)

Verified directly against the current source, not against any prior milestone's report:

1. **`RawHumanDetection` lifecycle**: created fresh every `LiveCameraPipeline.run_cycle()` by `HumanDetector.detect()`, passed once to `IdentityResolver.resolve()`, then converted to `Detection` and discarded. Nothing previously persisted a `RawHumanDetection` across cycles.
2. **`local_track_id` usage (before this milestone)**: `YOLOHumanDetector` assigned it as a bare per-frame index (`"0", "1", "2", ...` in detection order) — **not stable across frames**. `IdentityResolver` (`MappingIdentityResolver`/`SimulationIdentityResolver`) used `(camera_id, local_track_id)` as its resolution key. This instability is exactly what this milestone fixes — see the demonstration in `tests/test_live_camera_pipeline_tracking_integration.py::TrackerCorrectsForRawIndexInstabilityTests`, which shows a person's raw index shifting from `"0"` to `"1"` purely because a second person entered and was listed first by the detector, and how the tracker prevents this from corrupting the resolved `occupant_id`.
3. **Where `IdentityResolver` assigns occupant identity**: entirely inside `resolve()` (`MappingIdentityResolver._resolve_one` / `SimulationIdentityResolver.resolve`) — untouched by this milestone.
4. **Does `Detection` assume temporal information?** No — `virtual_camera.detection.Detection` is a single-instant, per-camera sighting with no age/history fields. All temporal state lives in `multi_camera_fusion.track.TrackHistory`, entirely downstream of `Detection`, keyed by `occupant_id` — unrelated to, and unaffected by, single-camera tracking.
5. **Does `CameraManager` store tracking state?** No — confirmed by reading `camera_manager/manager.py` in full: its only state is the camera registry, per-mode `DetectionProvider` registrations, and runtime connection status. No tracking-related state of any kind.
6. **Does `MultiCameraFusionEngine` assume detector stability?** No — confirmed by reading `multi_camera_fusion/engine.py`: it associates purely by `Detection.occupant_id` equality and has no notion of, or dependency on, how a `local_track_id` was produced upstream. This is exactly why the tracker could be inserted with zero change to fusion: fusion only ever sees the fully-resolved `occupant_id`.
7. **Does `ReplayFrameSource` preserve timestamps/frame numbers?** Yes — `timestamp` is passed through unchanged from whatever the caller supplied; `frame_sequence` is a monotonically increasing integer assigned by the source's own internal counter, incremented once per successful `read_frame()` call. `RTSPFrameSource` follows the identical convention (its own `_next_index` counter).
8. **CameraFrame ordering guarantees**: `frame_sequence` is guaranteed monotonically increasing per frame-source *instance*, never across different cameras/instances (irrelevant here — tracking is single-camera by design). `SingleCameraTracker.update()` does **not** read `frame_sequence` at all — only the explicit `timestamp` argument and call order matter (verified by `tests/test_single_camera_tracker.py::TimestampAndFrameOrderingTests`).

## 4. Design decision: why `IdentityResolver` was not touched

Phase 7 of this milestone left it as a judgment call ("`IdentityResolver` should now receive `TrackedHuman` ... if appropriate"). After investigating `IdentityResolver.resolve()`'s actual field usage (`camera_id`, `local_track_id`, `timestamp`, `floor_id`, `zone_id`, `confidence`, `classification_evidence`, `state_evidence`, `is_false_positive` — every one of those a `RawHumanDetection` field, and every one of them **absent** from `TrackedHuman` by Phase 4's own explicit design), the smallest, least-risky integration is: keep feeding `IdentityResolver` plain `RawHumanDetection` objects, with the tracker only responsible for supplying a better `local_track_id`. This preserves the "IdentityResolver remains separate" / "MultiCameraFusion remains unchanged" guarantees stated as this milestone's own required starting state, with **zero** modification to `identity_resolver.py`, `detection.py` (Detection), `multi_camera_fusion/`, or `building_state/`.

## 5. Pipeline integration point

`live_camera_pipeline/pipeline.py`'s `LiveCameraPipeline` gained one new, optional constructor parameter: `tracker: Optional[SingleCameraTracker] = None`. Omitting it (every existing caller/test) reproduces the pipeline's exact pre-tracking behavior — proven by `tests/test_live_camera_pipeline_tracking_integration.py::NoTrackerPreservesExactPriorBehaviorTests` and by every pre-existing pipeline/RTSP/YOLO test continuing to pass completely unmodified (`tests/test_live_camera_pipeline.py`, `tests/test_rtsp_offline_e2e.py`, `tests/test_yolo_rtsp_live_runtime_compatibility.py`).

When a tracker is supplied, `run_cycle()`'s only change is:

```python
raw = self.human_detector.detect(frame)
if self.tracker is not None:
    raw = self._stabilize(camera_id, frame.timestamp, raw)
raw_detections.extend(raw)
```

`_stabilize()` calls `tracker.update(camera_id, frame.timestamp, raw)` and `dataclasses.replace()`s each original `RawHumanDetection`'s `local_track_id` with the corresponding `TrackedHuman.track_id` — a plain `zip()`, made unambiguous by `SimpleSingleCameraTracker`'s own output contract (§6). Only the positional, per-detection prefix of the tracker's output is used; the trailing `MISSING`/`EXPIRED` remainder (tracks not matched this cycle) is deliberately **not** forwarded to `IdentityResolver` — there is no real observation to report for someone not currently seen, the same honest "nothing to report" convention already established throughout this codebase.

## 6. `SimpleSingleCameraTracker` — matching algorithm and output contract

A clean, deterministic engineering baseline — **not** DeepSORT/ByteTrack/StrongSORT/OCSORT (all explicitly out of scope). Matching (`tracking/cost_functions.py`): IoU first (`iou_threshold`), falling back to centroid distance (`max_centroid_distance`) when IoU does not qualify — Phase 5's "IoU, or centroid distance, or both." Assignment is greedy: every geometrically-qualifying (detection, existing-track) pair is scored, best-first, one-to-one, with fully deterministic tie-breaking (by detection index, then track id) — no randomness anywhere.

**Output contract** (stronger than the abstract `SingleCameraTracker.update()` requires, but always true for this implementation): the returned tuple's first `len(detections)` entries correspond **positionally**, one-to-one, to the input `detections` — regardless of confidence or geometry, every input detection produces exactly one output entry. Any further entries are pre-existing tracks not matched this cycle (`MISSING`, still coasting; or `EXPIRED`, reported exactly once on the cycle it is deleted, then never again).

## 7. Track lifecycle (Phase 6)

- **NEW** — first cycle a track exists.
- **TRACKED** — matched again on a later cycle.
- **MISSING** — not matched this cycle, but `frames_missing <= max_missing_frames`; still coasting on its last known bounding box (no motion prediction — the box simply does not move while missing), so a reappearing detection near that position resumes the *same* `track_id`.
- **EXPIRED** — `frames_missing` just exceeded `max_missing_frames`; reported exactly once, then deleted from internal state permanently (a later detection at the same location becomes a brand-new track).

Configurable, never hardcoded: `max_missing_frames` (default 5), `minimum_confidence` (default 0.0), `iou_threshold` (default 0.3), `max_centroid_distance` (default 50.0 px) — all named module-level constants in `tracking/simple_tracker.py`, all constructor overrides.

`minimum_confidence` does not drop a detection's output entry (the positional contract in §6 always holds) — it only prevents that detection from touching *persistent* track state: a below-threshold detection gets a disposable, one-off `NEW` track that is never stored and can never be matched again, so a low-confidence blip cannot fabricate stable identity or silently refresh a real track's `frames_missing` budget.

## 8. Package boundaries (Phase 2/12)

New package: `tracking/` (`__init__.py`, `tracker.py`, `tracked_human.py`, `simple_tracker.py`, `cost_functions.py`, `track_state.py`). Deliberately **not** inside `human_detection/`, `multi_camera_fusion/`, or `camera_manager/` — a separate responsibility from detection, fusion, and camera lifecycle management.

`tracking/` depends only on `RawHumanDetection` (one import: `live_camera_pipeline.human_detector`), plain geometry (its own `cost_functions.py`), and time (plain floats — no clock dependency). It imports no AI (`ai_engine`/`reinforcement_learning`), no `BuildingState`, no `advisory_system`, no `command_center`, no `multi_camera_fusion`, no `camera_manager`, no RTSP (`live_camera_pipeline.rtsp_frame_source`/`.rtsp_backend`), and no YOLO backend directly (`human_detection.yolo_backend`/`.yolo_human_detector`, `cv2`, `torch`, `ultralytics`). Enforced mechanically by `tests/test_tracking_architecture_guards.py`, the same regex-source-scan convention `tests/test_no_cv_dependencies.py` and the `camera_manager`/`multi_camera_fusion` package-dependency tests already use.

## 9. `TrackedHuman` — deliberately narrow (Phase 4)

Fields: `track_id`, `camera_id`, `bounding_box`, `confidence`, `state`, `age`, `frames_seen`, `frames_missing`, `last_timestamp`. Deliberately **excludes** `occupant_id`, building zone/floor, behavior/human_state, AI state, and any `BuildingState` reference — tracking stays local to one camera and one responsibility.

## 10. Files created / modified

**Created:**
- `tracking/__init__.py`, `tracking/track_state.py`, `tracking/tracked_human.py`, `tracking/cost_functions.py`, `tracking/tracker.py`, `tracking/simple_tracker.py`
- `tests/test_single_camera_tracker.py` — 24 unit tests (Phase 9)
- `tests/test_live_camera_pipeline_tracking_integration.py` — 3 pipeline-integration tests (Phase 7)
- `tests/test_tracking_architecture_guards.py` — 2 import-guard tests (Phase 12)
- `scripts/demo_single_camera_tracking.py` — offline demo (Phase 10)
- `scripts/benchmark_single_camera_tracker.py` — performance benchmark (Phase 11)
- `docs/architecture/single_camera_tracking.md` — this document

**Modified:**
- `live_camera_pipeline/pipeline.py` — added one optional constructor parameter (`tracker`) and one small `_stabilize()` helper; default behavior (`tracker=None`) is byte-for-byte unchanged.

**Unchanged (verified, not modified):** `live_camera_pipeline/human_detector.py`, `live_camera_pipeline/identity_resolver.py`, `live_camera_pipeline/detection_provider.py`, `live_camera_pipeline/frame_source.py`, `virtual_camera/detection.py`, `human_detection/*`, `camera_manager/*`, `multi_camera_fusion/*`, `building_state/*`.

## 11. Performance

`scripts/benchmark_single_camera_tracker.py` measures, separately, with zero detector/YOLO inference involved anywhere:

- Steady-state matching + update (20 people/camera): ~0.5 ms/cycle.
- Track creation (20 brand-new tracks/cycle): ~0.06 ms/cycle.
- Track deletion/expiry (20 tracks expiring/cycle): ~0.03 ms/cycle.

See `scripts/benchmark_yolo_human_detector.py` for detector-side overhead, measured completely separately — the two are never combined into one number, so neither hides the other's cost.

## 12. What still remains

Stable single-camera tracking across occlusion/re-entry is now implemented; cross-camera re-identification, genuine human classification/behavior-state, and pose estimation remain explicitly out of scope, matching `docs/architecture/human_detection.md`'s own capability boundary. A future tracker (DeepSORT/ByteTrack/StrongSORT/OCSORT, or any appearance/motion-model-based strategy) can replace `SimpleSingleCameraTracker` entirely — `tracking.tracker.SingleCameraTracker`, `live_camera_pipeline.pipeline.LiveCameraPipeline`, `IdentityResolver`, and everything downstream would need no changes at all.
