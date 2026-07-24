# Physical CCTV Field Validation Readiness

Status as of this milestone: still **no physical CCTV/NVR access.** This milestone does NOT add new CCTV architecture (explicitly out of scope by its own instruction) — it freezes the exact software baseline from commit `8120180` (see `docs/architecture/cctv_connection_and_calibration_readiness.md`) and builds the ONE orchestration tool needed to make the actual field day controlled and reproducible: `scripts/run_physical_camera_validation.py`.

## 1. Frozen production baseline (Phase 1)

The exact objects/functions the first physical camera will instantiate, verified directly against commit `8120180` — nothing redesigned:

```
Camera(id=..., name=..., floor_id=...)                          models.camera.Camera
  -> CameraManager().register_camera(camera)                    camera_manager.manager.CameraManager
  -> LocalFileCredentialStore()                                 credential_store.local_file_store
  -> OpenCVFrameDecoderBackend(open_timeout_ms=...)              human_detection.opencv_decoder_backend
  -> RTSPFrameSource(camera_id, endpoint, decoder_backend,       live_camera_pipeline.rtsp_frame_source
       username, password, credential_ref, credential_store)
  -> UltralyticsYOLOBackend(weights_path, device="cpu")          human_detection.yolo_backend
  -> YOLOHumanDetector(backend)                                  human_detection.yolo_human_detector
  -> SimpleSingleCameraTracker()                                 tracking.simple_tracker
  -> SimulationIdentityResolver()                                live_camera_pipeline.identity_resolver
       (Phase 10 -- ONE camera only; tracker's own stable track id
        IS the occupant id, no cross-camera resolver wired yet)
  -> LiveCameraPipelineDetectionProvider()                       live_camera_pipeline.detection_provider
  -> LiveCameraPipeline(frame_sources, human_detector,            live_camera_pipeline.pipeline
       identity_resolver, detection_provider, tracker,
       world_projector, live_occupant_manager)
  -> WorldProjector(calibrations={camera_id: profile},            camera_calibration.projection
       zones_by_floor)                                            (profile from load_calibration_json() --
                                                                     REQUIRED, never fabricated)
  -> LiveOccupantManager()                                       live_occupants.manager
  -> build_live_runtime(building, frame_sources=..., ...)         live_runtime.factory
  -> LiveRuntime.start() / .run_cycle(time) / .stop()             live_runtime.runtime
  -> LiveOrchestrator.latest_building_state / .latest_crowd_       live_system.orchestrator
       intelligence / .latest_trajectory_intelligence / .latest_
       evacuation_progress / .latest_evacuation_recommendation /
       .latest_evacuation_guidance / .latest_advisory_report
```

No new class was introduced anywhere in this chain. `build_live_runtime()` is never given a `voice_output_provider` or `building_control_provider` during field validation — per the factory's own pre-existing "`None` means NO_PROVIDER, never fabricated" convention, this means zero automatic voice broadcast and zero automatic building control execution, mechanically, not by a new guard this milestone had to add.

## 2. `scripts/run_physical_camera_validation.py` (Phase 2)

One script, six progressive modes (`--connection-only` → `--frames` → `--detect` → `--track` → `--project` → `--full-runtime`), each a strict superset of the previous. Every stage orchestrates the exact objects in §1 — the script itself contains no detection/tracking/projection/runtime logic of its own, only construction, sequencing, and reporting. Connection-stage bookkeeping (`ConnectionDiagnostic`, password resolution) is imported directly from `scripts/test_camera_connection.py` (the prior milestone's own diagnostic), never reimplemented.

- `--connection-only`: builds and starts a real `RTSPFrameSource`/`OpenCVFrameDecoderBackend`, reports connection state only.
- `--frames`: + reads real frames, measures time-to-first-frame/resolution/codec/FPS.
- `--detect`: + real `YOLOHumanDetector`, reports frame/detection/confidence/inference-latency statistics (one-time model load excluded from every average).
- `--track`: + real `SimpleSingleCameraTracker`, reports unique/active track counts, longest continuous track, creation/expiry counts; `--preview` opens an annotated OpenCV window (confidence + local track id per box — no face/appearance overlay).
- `--project`: + **requires** `--calibration <profile.json>` — fails honestly with `CALIBRATION_REQUIRED` if omitted or unfittable, never substitutes an illustrative calibration. Reports world position/zone/confidence/provenance for every detection.
- `--full-runtime`: + wires everything into `build_live_runtime()`, runs real cycles, reports whether `BuildingState`/Crowd/Trajectory/Evacuation Progress/Recommendation/Guidance/Advisory were reached, and explicitly confirms `voice_broadcast_attempted=false`/`building_control_executed=false`.

## 3. Failure vocabulary (Phase 13)

Ten script-level outcome labels (`NETWORK_UNREACHABLE`, `AUTHENTICATION_FAILED`, `STREAM_UNAVAILABLE`, `NO_FRAMES`, `DECODER_FAILURE`, `YOLO_UNAVAILABLE`, `CALIBRATION_REQUIRED`, `CALIBRATION_INVALID`, `PROJECTION_UNAVAILABLE`, `RUNTIME_DEGRADED`) layered on top of — never replacing — the existing `RTSPFrameSource.status`/`CameraConnectionState`/`WORLD_POSITION_PROVENANCE_*` vocabularies. `classify_connection_outcome()` is a best-effort heuristic over `RTSPFrameSource.last_error` text, deliberately conservative: a genuine finding during this milestone's own testing was that `OpenCVFrameDecoderBackend`'s generic open-failure message ("stream unreachable, wrong path, or unsupported by this OpenCV build") uses the word "unreachable" for **every** open failure, including a plain bad local path with no network involved at all — the classifier was corrected to require a genuinely network-specific phrase (`"resolve hostname"`, `"no route to host"`, etc.) before reporting `NETWORK_UNREACHABLE`, falling back to the honest, generic `STREAM_UNAVAILABLE` otherwise. Caught by `tests/test_physical_camera_validation_field_runner.py`'s own CLI-level test before this doc was written.

## 4. Second-camera / ReID readiness (Phase 11 — investigation only)

`cross_camera_identity.resolver.RuleBasedCrossCameraIdentityResolver` (unchanged, re-verified) composes `IdentityRegistry` (storage) + `TransitionModel` (adjacency/expiry policy, built from `CameraTopology`) + `RuleBasedCrossCameraMatcher` (scoring). **Confirmed: topology/time based, zero appearance embeddings, zero facial recognition.** No deep ReID was added or investigated for addition. The field validation harness deliberately does not wire this resolver at all — Phase 10's own "one camera first" instruction — `SimulationIdentityResolver` is used instead (the tracker's own stable local track id serves directly as the occupant id, correct and honest for a single camera with no cross-camera evidence).

## 5. Tests (Phase 14)

`tests/test_physical_camera_validation_field_runner.py` — 20 tests: pure classifier/report-shape tests (unconditional), CLI-level tests against real `vtest.avi` (connection-only, frames, report JSON shape, no-secrets check — skipped if the video is absent), and CLI-level tests against real `vtest.avi` + real `yolov8n.pt` (detect/track/project-without-calibration/full-runtime — skipped if either artifact is absent). No test requires network access or a real camera; every physical-network scenario remains explicitly out of scope until physical access exists.

## 6. Final component classification

| Component | Classification |
|---|---|
| `scripts/run_physical_camera_validation.py` (all 6 modes) | REAL IMPLEMENTATION + REAL DATA TESTED (offline, `vtest.avi` + `yolov8n.pt`) |
| Failure classification vocabulary | REAL IMPLEMENTATION + OFFLINE TESTED (one real classifier bug found and fixed during this milestone) |
| Calibration-required gate for `--project` | REAL IMPLEMENTATION + OFFLINE TESTED — never fabricates |
| `--full-runtime` reaching BuildingState/Crowd/Trajectory/Evacuation/Advisory | REAL IMPLEMENTATION + REAL DATA TESTED (offline) |
| No automatic voice/building-control during field validation | REAL IMPLEMENTATION + OFFLINE TESTED |
| Real network RTSP camera behavior | REQUIRES PHYSICAL CCTV ACCESS |
| Real measured calibration (RMSE against a physically measured scene) | REQUIRES PHYSICAL CCTV ACCESS |
| Real-world position sanity test (measured vs. projected) | REQUIRES PHYSICAL CCTV ACCESS |
| Second-camera topology/time identity resolution against real overlap | REQUIRES PHYSICAL CCTV ACCESS (two cameras) |
| Deep/appearance-based ReID | FUTURE WORK — explicitly not started |

## 7. Files created / modified

**New:** `scripts/run_physical_camera_validation.py`, `tests/test_physical_camera_validation_field_runner.py`, this document.

**Modified:** `docs/architecture/physical_cctv_access_checklist.md` (literal 20-step runbook centered on the new field runner, field-session-recording section, second-camera readiness section).

**Unchanged:** every production component in §1 — `RTSPFrameSource`, `OpenCVFrameDecoderBackend`, `YOLOHumanDetector`, `SimpleSingleCameraTracker`, `WorldProjector`, `LiveOccupantManager`, `build_live_runtime()`, `LiveRuntime`, `LiveOrchestrator`, `RuleBasedCrossCameraIdentityResolver`.
