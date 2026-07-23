# Real Human Detection Pipeline

Status as of the **Real YOLO Person Detection Validation & Productionization** milestone (which audited, validated, and wired together the original Real Human Detection Pipeline milestone below with the already-committed Single-Camera Tracking Framework, Human Behavior Recognition Framework, Cross-Camera Identity Resolution Framework, Camera Calibration, and Live Occupant Digital Twin milestones — all of which were built *after* this package was originally written but never looped back into it): `human_detection.yolo_human_detector.YOLOHumanDetector` is a concrete `live_camera_pipeline.human_detector.HumanDetector` that detects people in individual camera frames using an injectable YOLO inference backend, and its output now provably composes with the full, real, already-committed tracking/identity/behavior/occupant chain. No physical CCTV access exists, and none was used or required by either milestone.

## 0. What changed in the productionization pass

The original milestone (§1-12 below, preserved as written) explicitly named single-camera tracking as "investigated only, not built" and stopped at `IdentityResolver`. Investigation for the productionization pass found that gap had, in the meantime, already been filled by four separate, already-committed milestones this package simply never got wired back into: `tracking/` (`SimpleSingleCameraTracker`), `behavior_recognition/` (`RuleBasedBehaviorRecognizer`), `cross_camera_identity/`, `camera_calibration/`, and `live_occupants/` (`LiveOccupantManager`) — and that `live_camera_pipeline.pipeline.LiveCameraPipeline` **already** accepts every one of them as optional, additive constructor seams (`tracker`/`behavior_recognizer`/`cross_camera_identity_resolver`/`world_projector`/`live_occupant_manager`), and that `live_runtime.factory.build_live_runtime()` **already** exposes `human_detector`/`tracker`/etc. as plain injection parameters. Nothing in any of those packages needed to change. The productionization pass added:

- `tests/test_yolo_tracking_integration.py` — proves `YOLOHumanDetector` → `SimpleSingleCameraTracker` produces stable track ids across consecutive frames (detection-order changes, temporary occlusion, entering, leaving), that the real `LiveCameraPipeline` (with tracker + behavior recognizer + `LiveOccupantManager` all configured) consumes YOLO's output correctly, that the same composition works over `RTSPFrameSource`, that `build_live_runtime()` composes a real YOLO+tracker configuration through its existing seams, and a battery of failure-mode tests (missing/invalid weights, corrupt/empty frames, inference exceptions, one camera failing while another stays healthy, stale-detection replay).
- `tests/test_human_detection_architecture_guards.py` — mechanical proof `human_detection/` cannot reach decision/execution layers (AI, Advisory, Command Center, voice, building control, FACP, signage, hazard, BuildingState) and stays confined to producing `RawHumanDetection`.
- `scripts/demo_real_yolo_tracking.py` — the full real chain (video → YOLO → tracker → behavior → occupant manager), with an `--annotate-out` visualization mode.
- `scripts/benchmark_yolo_human_detector.py` gained two more separated measurements: tracking overhead and downstream pipeline overhead (still reporting real YOLO inference as **NOT RUN** rather than fabricating a number, when no `--weights` is supplied).

No real YOLO weights are bundled with this repository and none were downloaded in either milestone (no network access) — real-model inference remains **structurally proven, not numerically measured**, exactly as honestly disclosed in §11 below.

## 1. Current chain (implemented, offline-testable, proven by test)

```
Replay/local frame source (or RTSPFrameSource -- proven identical, see §9)
    -> live_camera_pipeline.replay_frame_source.ReplayFrameSource  (unchanged, existing)
    -> live_camera_pipeline.frame_source.CameraFrame               (unchanged, existing)
    -> human_detection.yolo_human_detector.YOLOHumanDetector       (Real Human Detection Pipeline milestone)
         delegates to an injected human_detection.yolo_backend.YOLOInferenceBackend:
             human_detection.yolo_backend.UltralyticsYOLOBackend       (real, lazy-loaded)
             tests.human_detection_fixtures.FakeYOLOBackend            (deterministic, offline)
    -> live_camera_pipeline.human_detector.RawHumanDetection        (unchanged, existing)
    -> tracking.simple_tracker.SimpleSingleCameraTracker            (Single-Camera Tracking Framework
         milestone, already committed -- NOW PROVEN compatible with YOLO's own output shape,
         see tests/test_yolo_tracking_integration.py) -- STABLE local track ids across frames
    -> behavior_recognition.rule_based_recognizer.RuleBasedBehaviorRecognizer  (optional, additive --
         same proof)
    -> live_camera_pipeline.identity_resolver.IdentityResolver      (unchanged, existing --
         SimulationIdentityResolver/MappingIdentityResolver both proven)
    -> live_occupants.manager.LiveOccupantManager                  (optional, additive -- same proof)
    -> virtual_camera.detection.Detection                          (unchanged, existing)
    -> multi_camera_fusion.engine.MultiCameraFusionEngine           (unchanged, existing)
    -> building_state.estimator.BuildingStateEstimator              (unchanged, existing)
```

`YOLOHumanDetector` and `YOLOInferenceBackend`/`UltralyticsYOLOBackend` live in a **new top-level package, `human_detection/`** — deliberately outside `live_camera_pipeline/`, which `tests/test_no_cv_dependencies.py` mechanically forbids from ever importing `cv2`/`torch`/`ultralytics`/`onvif`. `human_detection/` is not in that guard's `TARGETS` list; its entire purpose is to hold real computer-vision code. `live_camera_pipeline/`, `camera_manager/`, `multi_camera_fusion/`, `building_state/`, `virtual_camera/`, `advisory_system/`, `command_center/` needed **zero changes** to add this detector — the `HumanDetector` seam already existed for exactly this purpose (§8 below).

## 2. Future chain (unchanged plan, now one step closer)

```
Physical CCTV Camera
    -> live_camera_pipeline.rtsp_frame_source.RTSPFrameSource        (existing, production-ready)
    -> [FUTURE] real FrameDecoderBackend                             (needs a real decode/transport
         library -- explicitly out of scope for this milestone, same gap docs/architecture/
         cctv_integration_readiness.md already names)
    -> live_camera_pipeline.frame_source.CameraFrame
    -> human_detection.yolo_human_detector.YOLOHumanDetector          (SAME class, SAME instance
         shape as §1 -- proven directly, see §9)
    -> live_camera_pipeline.identity_resolver.IdentityResolver / [FUTURE] real cross-camera ReID
    -> multi_camera_fusion.engine.MultiCameraFusionEngine
    -> building_state.estimator.BuildingStateEstimator
```

The only future work this milestone identifies as still missing: a real `FrameDecoderBackend`, and real cross-camera ReID. `YOLOHumanDetector` itself needs no changes to work with either, once they exist.

## 3. Capability boundary — what is honestly implemented

- **PERSON DETECTION — implemented here.** `YOLOHumanDetector` filters an underlying model's raw class predictions down to exactly the configured `person_class_name` (default `"person"`), matched by name (not a hardcoded class index), and reports each surviving detection's bounding box and confidence.
- **HUMAN CLASSIFICATION — not implemented.** `classification_evidence` is always `HumanClassification.UNKNOWN`. A generic person detector has no honest basis for Adult/Child/Elderly/Wheelchair User/Firefighter/Fire Warden.
- **HUMAN STATE/BEHAVIOR RECOGNITION — not implemented.** `state_evidence` is always `None` (`RawHumanDetection`'s own established "no evidence" convention). A single bounding box in one frame cannot honestly support Walking/Running/Fallen/Crawling/Helping.
- **SINGLE-CAMERA TRACKING — IMPLEMENTED, proven compatible by test.** `tracking.simple_tracker.SimpleSingleCameraTracker` (already committed by a separate, later milestone) is now proven directly against YOLO's own detection output shape — see §7 (updated) and `tests/test_yolo_tracking_integration.py`.
- **CROSS-CAMERA REID — seam exists, not exercised with YOLO by this pass.** `YOLOHumanDetector` never invents a global `occupant_id`; `cross_camera_identity.resolver.CrossCameraIdentityResolver` is a real, already-committed, optional `LiveCameraPipeline` seam (see §8) that composes with YOLO+tracker exactly like every other seam, but exercising it needs a real multi-camera topology/calibration this milestone did not construct (single-camera and two-independent-camera scenarios were proven instead, see §9).
- **PHYSICAL CCTV VALIDATION — not performed.** No physical camera exists to test against; every test and demo in either milestone runs offline, against synthetic content or an injected fake backend.

## 4. Detector-result model — `RawHumanDetection` (unchanged, already sufficient)

Investigated before writing any code (Phase 1): `live_camera_pipeline.human_detector.RawHumanDetection` already carried `bounding_box: Optional[Tuple[float, float, float, float]]`, `confidence: float`, and `local_track_id: Optional[str]` before this milestone touched anything. Nothing needed to be added to it. The canonical, fused `virtual_camera.detection.Detection` type deliberately has **no** bounding-box field at all (by design — a fused, cross-camera-resolved occupant record has no single frame's box to report) and was correctly left untouched; there was nothing to "pollute."

`human_detection.yolo_backend.BoundingBoxDetection` is the one new type this milestone adds, and it is intentionally **not** `RawHumanDetection` — it is the smaller, pre-filtering shape a raw model prediction takes (`class_id`, `class_name`, `confidence`, `bounding_box`), before `YOLOHumanDetector` decides which of those are people at all. Keeping it separate from `RawHumanDetection` keeps "what the model said" and "what `HumanDetector`'s contract promises" as two distinct, honestly-scoped types.

## 5. `CameraFrame.payload_ref` — investigated, not changed

`CameraFrame.payload_ref` was already typed `Optional[Any]` before this milestone, deliberately uninterpreted by `live_camera_pipeline/` itself. By this milestone's own established convention, a decoded frame is a numpy `ndarray` (HxWx3, BGR) — the same shape `human_detection.video_source.load_video_frames`/`load_image_frame` (both real `cv2` decode calls) produce, and the same shape a real `FrameDecoderBackend`/`RTSPFrameSource` would eventually populate `CameraFrame.payload_ref` with. `YOLOHumanDetector` itself never inspects or type-checks `payload_ref` beyond a `None` check — it is handed directly to the injected `YOLOInferenceBackend.infer()`, keeping the detector agnostic to whether that `ndarray` came from a replayed video, a hand-built test string, or eventually a real RTSP stream.

## 6. Identity boundary — investigated, confirmed already correct

`YOLOHumanDetector.detect()` assigns `local_track_id = "0", "1", "2", ...` in detection order, purely so `IdentityResolver` has a distinct `(camera_id, local_track_id)` key per person **within one frame** — without it, every person detected in the same frame would collapse to `local_track_id=None` and incorrectly fuse into one occupant via `identity_resolver.py`'s own `f"{camera_id}:"` fallback. This index is **not** stable across frames: the person who was index `"0"` in frame *N* has no guaranteed relationship to whoever is index `"0"` in frame *N+1*. This is an honest limitation, not a bug — real single-camera tracking (a *stable* local id across frames for the same physical person) is explicitly out of scope (§7).

No global `occupant_id` is ever assigned by `YOLOHumanDetector` — that remains entirely `IdentityResolver`'s job (`live_camera_pipeline/identity_resolver.py`, unchanged). `tests/human_detection_fixtures.py` and `tests/test_yolo_rtsp_live_runtime_compatibility.py` both reuse `MappingIdentityResolver`, the same production `IdentityResolver` implementation the RTSP milestone already established — no new resolver was needed or written.

## 7. Single-camera tracking — now IMPLEMENTED and proven compatible

Originally investigated only (a detector-local index that is *stable across frames* was identified as needing either a tracking algorithm layered on YOLO detections, or `ultralytics`' own `model.track()` API). By the time of the productionization pass, `tracking.simple_tracker.SimpleSingleCameraTracker` had already been built and committed by a separate milestone (`docs/architecture` — Single-Camera Tracking Framework) — a clean, deterministic IoU/centroid-matching tracker, deliberately not ByteTrack/DeepSORT (that remains a legitimate future replacement; nothing importing `tracking.tracker.SingleCameraTracker`, the seam, would need to change if it were swapped in later). `YOLOHumanDetector.detect()` itself remains deliberately stateless — the tracker is a separate, composed object (`live_camera_pipeline.pipeline.LiveCameraPipeline`'s own optional `tracker` parameter), never folded into the detector.

`tests/test_yolo_tracking_integration.py::TrackerStabilityWithYOLODetectionsTests` proves directly: the same physical person keeps the same tracker `track_id` across consecutive frames even when YOLO's own per-frame `local_track_id` ("0", "1", ...) would not (detection-order changes between frames); a temporarily occluded person's track survives as `MISSING` and resumes the same id once re-detected; a newly entering person gets a genuinely `NEW` track; a person who leaves eventually reaches `EXPIRED` after `max_missing_frames`. The tracker's own `track_id` (e.g. `"CAM-001-T1"`) is a different, camera-namespaced string shape from YOLO's own `local_track_id`, never confusable with it.

## 8. Files created / modified

**Created (Real Human Detection Pipeline milestone):**
- `human_detection/__init__.py`
- `human_detection/yolo_backend.py` — `BoundingBoxDetection`, `YOLOInferenceBackend` (ABC), `ModelWeightsNotFoundError`, `UltralyticsYOLOBackend`
- `human_detection/yolo_human_detector.py` — `YOLOHumanDetector`
- `human_detection/video_source.py` — `load_video_frames`, `load_image_frame`, `load_image_frames`, `VideoSourceError` (offline local video/image loading, feeding `ReplayFrameSource`)
- `tests/human_detection_fixtures.py` — `FakeYOLOBackend`, `person()`, `non_person()` (test doubles)
- `tests/test_yolo_human_detector.py` — 20 unit tests (Phase 7)
- `tests/test_yolo_rtsp_live_runtime_compatibility.py` — 5 integration tests (Phase 9)
- `scripts/demo_yolo_human_detection.py` — offline local-video demo (Phase 8)
- `scripts/benchmark_yolo_human_detector.py` — performance benchmark (Phase 10)
- `docs/architecture/human_detection.md` — this document

**Created (Real YOLO Person Detection Validation & Productionization milestone):**
- `tests/test_yolo_tracking_integration.py` — 23 tests: tracker stability with YOLO detections, real `LiveCameraPipeline` full-chain composition, RTSP+tracker compatibility, `build_live_runtime()` composition, failure-mode coverage, device configuration
- `tests/test_human_detection_architecture_guards.py` — 5 mechanical layer-separation guard tests
- `scripts/demo_real_yolo_tracking.py` — full real chain demo (video → YOLO → tracker → behavior → `LiveOccupantManager`), with `--annotate-out` visualization

**Modified (both milestones):**
- `requirements.txt` — added `opencv-python==4.13.0.92`, `ultralytics==8.4.30` (`torch` was already a dependency)
- `scripts/benchmark_yolo_human_detector.py` — added tracking-overhead and downstream-pipeline-overhead measurements (productionization pass)

**Unchanged (verified, not modified):** `live_camera_pipeline/human_detector.py`, `live_camera_pipeline/frame_source.py`, `live_camera_pipeline/identity_resolver.py`, `live_camera_pipeline/detection_provider.py`, `live_camera_pipeline/pipeline.py`, `virtual_camera/detection.py`, `camera_manager/`, `multi_camera_fusion/`, `building_state/`, `live_runtime/`.

## 9. Live Runtime / future-RTSP compatibility proof

`tests/test_yolo_rtsp_live_runtime_compatibility.py` proves the same `YOLOHumanDetector` architecture (one instance per camera, each with its own injected fake inference backend) works unchanged through the full production chain: `FakeRTSPBackend` → `RTSPFrameSource` (real, production class) → `CameraFrame` → `YOLOHumanDetector` (real, production class, fake backend only) → `MappingIdentityResolver` → `Detection` → `CameraManager` → `MultiCameraFusionEngine` → `BuildingStateEstimator`. Zero network access, zero CCTV access. `CameraManager`, `MultiCameraFusionEngine`, and `BuildingStateEstimator` are exercised exactly as-is, with no changes.

`tests/test_yolo_tracking_integration.py::RTSPWithTrackerCompatibilityTests` extends this exact proof one seam further: `FakeRTSPBackend` → `RTSPFrameSource` → `CameraFrame` → `YOLOHumanDetector` → `SimpleSingleCameraTracker` → `LiveOccupantManager`, confirming the tracker-stabilized identity stays the same occupant across two consecutive cycles over the RTSP-shaped transport, not just over a plain replay source.

`tests/test_yolo_tracking_integration.py::LiveRuntimeFactoryCompositionTests` proves `live_runtime.factory.build_live_runtime()` — investigated directly — **already** exposes `human_detector`/`tracker`/`behavior_recognizer`/`cross_camera_identity_resolver`/`world_projector`/`live_occupant_manager` as plain, independently-optional injection parameters (no new composition class was needed): a real `YOLOHumanDetector` + `SimpleSingleCameraTracker` configuration composes cleanly through that existing mechanism and runs a full `LiveOrchestrator` cycle, while calling `build_live_runtime()` with none of them supplied (the offline/demo default) continues to work with zero YOLO/ultralytics involvement — the production seam and the offline default were never made to conflict.

## 10. Dependency / model-weights policy

`ultralytics==8.4.30` and `opencv-python==4.13.0.92` were added to `requirements.txt` (`torch==2.11.0` was already present). Both are confined to `human_detection/` — the one package this codebase's own architecture guard (`tests/test_no_cv_dependencies.py`) deliberately does not cover, because holding real CV code is this package's whole purpose.

- Importing `human_detection/` never touches the network or loads a model: `ultralytics`/`torch` are imported lazily, inside `UltralyticsYOLOBackend._ensure_loaded()`, called only on the first real `infer()` call.
- Constructing `UltralyticsYOLOBackend(weights_path)` performs zero I/O beyond a local `Path.exists()` check — it raises `ModelWeightsNotFoundError` immediately if the given `.pt` file does not already exist locally, **before** `ultralytics.YOLO(...)` ever gets a chance to interpret a bare model name (e.g. `"yolov8n.pt"`) as something to download.
- No model weights are bundled with this repository, and none were downloaded during this milestone. Every test and the default demo/benchmark run entirely against `FakeYOLOBackend`.
- To use a real model: supply the path to an already-downloaded `.pt` file, e.g. `UltralyticsYOLOBackend("C:/models/yolov8n.pt")`, or `python scripts/demo_yolo_human_detection.py --weights C:/models/yolov8n.pt`.

## 11. Performance

`scripts/benchmark_yolo_human_detector.py` measures, separately:

- Frame preparation (`human_detection.video_source.load_video_frames`, real `cv2` decode, synthetic-content local video): ~0.7 ms/frame on a development machine.
- `YOLOHumanDetector.detect()`'s own adapter overhead (class filtering + `RawHumanDetection` construction, against `FakeYOLOBackend` — zero real model time): ~0.01 ms/frame, 8 people/frame.
- Tracking overhead (`SimpleSingleCameraTracker.update()`, zero real model time): ~0.09 ms/frame, 8 people/frame.
- Downstream pipeline overhead (`LiveCameraPipeline.run_cycle()` with tracker + identity resolution, zero real model time): ~0.14 ms/frame, 8 people/frame.
- Real `ultralytics` model inference latency: **NOT RUN** by default — no local `.pt` weights file is bundled with this repository, and none was downloaded (no network access in either milestone). Pass `--weights <path>` to measure honestly against a real local model; the script never fabricates this number in the flag's absence.

Physical RTSP decode latency and physical network latency remain unmeasured and unclaimed, same disclosure as `docs/architecture/cctv_integration_readiness.md` §19.9 already makes for the RTSP transport seam itself — no physical CCTV stream exists yet to measure either honestly.

## 12. What still remains

A real `FrameDecoderBackend` (needs a real decode/transport library, out of scope here, same gap named in `docs/architecture/cctv_integration_readiness.md`), exercising cross-camera ReID together with a real YOLO+tracker configuration against a genuine multi-camera topology (the seam exists and composes, see §3/§9, but no real topology/calibration was constructed here), and any genuine human classification/behavior-state model beyond the existing rule-based velocity heuristics (explicitly out of scope — §3). Physical CCTV validation of `YOLOHumanDetector` itself has not been performed and cannot be until real camera access exists. Real local YOLO weights were not available in either milestone's environment (no network access) — real-model inference is structurally proven runnable (§13) but was not numerically measured.

## 13. GPU / device support

`UltralyticsYOLOBackend(weights_path, device=..., confidence_threshold=...)` exposes `device` as a plain, explicit string parameter (default `"cpu"`) passed straight through to `self._model.predict(image, device=self._device, ...)` — `ultralytics` itself already handles CPU/CUDA/MPS device selection and validation; this codebase adds no custom `torch` device management on top of it, and never hard-requires CUDA. Passing `device="cuda"` on a machine without a CUDA-capable GPU/`torch` build is a plain `ultralytics`/`torch` runtime error at `infer()` time, not something this backend catches or silently downgrades — an honest failure, not a fabricated CPU fallback. `torch.cuda.is_available()` was checked directly in this milestone's own development environment and returned `False` (CPU-only `torch` build) — GPU inference was therefore never exercised, only confirmed to be a supported, uncustomized passthrough.

## 14. Layer separation (mechanically guarded)

`tests/test_human_detection_architecture_guards.py` proves `human_detection/` stays confined to the Detection layer only: it cannot import `ai_decision`/`ai_registry`/`ai_inference`/`ai_training`/`rl_training`, `decision_policy`, `advisory_system`/`command_center`, `voice_evacuation`/`speaker_manager`, `building_control`, `facp`, `sign_manager`/`dynamic_signage`, `hazard`/`hazard_evolution`/`fire_growth`/`smoke_propagation`, or `building_state` — and it cannot reach up into the Tracking/Identity/Behavior/Intelligence layers either (`tracking`, `cross_camera_identity`, `behavior_recognition`, `camera_calibration`, `live_occupants`, `live_perception`, or any of the crowd/evacuation intelligence engines). Composition of all of those layers happens exactly one level up, in `LiveCameraPipeline`/`live_runtime.factory`, never inside this package. `YOLOHumanDetector.detect()`'s own return type is mechanically confirmed to be `Tuple[RawHumanDetection, ...]`, and no operator-action verb (`.acknowledge(`, `.broadcast(`, `.execute_control(`, ...) appears anywhere in the package.

## 15. Real-world path classification

| Piece | Status |
|---|---|
| Local video/image decode (`human_detection.video_source`) | **IMPLEMENTED + OFFLINE TESTED** |
| `YOLOHumanDetector` + `UltralyticsYOLOBackend` (person detection) | **IMPLEMENTED + OFFLINE TESTED** (structurally proven runnable against real weights; no weights bundled/downloaded, so real-model inference itself is untested numerically) |
| `FakeYOLOBackend` deterministic double | **IMPLEMENTED + OFFLINE TESTED** |
| `SimpleSingleCameraTracker` composed with YOLO's output | **IMPLEMENTED + OFFLINE TESTED** |
| `RuleBasedBehaviorRecognizer` composed with YOLO+tracker | **IMPLEMENTED + OFFLINE TESTED** |
| `LiveOccupantManager` composed with YOLO+tracker | **IMPLEMENTED + OFFLINE TESTED** |
| `RTSPFrameSource` + YOLO + tracker composition | **IMPLEMENTED + OFFLINE TESTED** (`FakeRTSPBackend` transport; no physical stream) |
| `build_live_runtime()` composition with real YOLO+tracker | **IMPLEMENTED + OFFLINE TESTED** |
| Cross-camera ReID (`CrossCameraIdentityResolver`) with a real YOLO+tracker, real multi-camera topology | **FUTURE WORK** (seam exists and composes structurally; no real topology/calibration constructed) |
| Real RTSP transport (`FrameDecoderBackend` for an actual camera stream) | **FUTURE WORK — IMPLEMENTED BUT REQUIRES PHYSICAL CCTV** for the decode layer itself; everything downstream of a decoded `CameraFrame` is already proven |
| Physical CCTV validation of `YOLOHumanDetector` against a real camera | **IMPLEMENTED BUT REQUIRES PHYSICAL CCTV** — cannot be performed until real camera access exists |
| GPU/CUDA inference | **FUTURE WORK** — supported as a plain passthrough parameter, never exercised (no CUDA-capable environment available) |
