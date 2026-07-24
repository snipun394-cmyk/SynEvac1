# CCTV Connection & Calibration Readiness

Status as of this milestone: still **no physical CCTV/NVR access.** This milestone's goal was narrow and explicit: make the software side completely ready for the day physical college CCTV access exists, so that the *only* missing input on that day is real network/credential/measurement data — never more code. Read this alongside `docs/architecture/cctv_integration_readiness.md` (the architecture-hardening milestone that built `RTSPFrameSource` and named the real decoder backend as the one missing piece) and `docs/architecture/camera_calibration_and_world_projection.md` (the calibration/validation math and workflow).

## 1. What this milestone found missing (Phase 1 investigation)

Before writing anything, the exact gap named by `cctv_integration_readiness.md` §18.4/§19.11 was re-verified: `RTSPFrameSource`, `CameraFrame`, `CameraManager`, `MultiCameraFusionEngine`, `BuildingStateEstimator`, `CredentialStore`, `CalibrationProfile`/`WorldProjector`/`calibration_solver`/`validation`, and `scripts/calibrate_camera_scene.py` all already existed and were already offline-tested. The genuinely missing pieces were:

1. A real `FrameDecoderBackend` implementation (only `FakeRTSPBackend` existed).
2. A connection diagnostic tool usable on physical-access day.
3. A way to distinguish an unvalidated calibration's `world_position` from a validated one anywhere downstream of `WorldProjector` — this existed only as a Designer status label, not as data flowing through `RawHumanDetection`/`Detection`/`LiveOccupant`.
4. A calibration-frame-capture step using the *real* production frame-acquisition path (previously only illustrated via `--pick-points` on an already-existing image).
5. A rehearsable, ordered dry-run proving the whole chain end-to-end offline.
6. A per-camera Calibration status visible anywhere outside the Property Panel.

## 2. The real decoder backend (Phase 2/3)

`human_detection/opencv_decoder_backend.py::OpenCVFrameDecoderBackend` — a real, `cv2.VideoCapture`-backed `FrameDecoderBackend`. Lives in `human_detection/`, not `live_camera_pipeline/` (the one package `tests/test_no_cv_dependencies.py` does NOT forbid `cv2`/`ultralytics`/`torch` imports in — the same convention `yolo_backend.py`/`video_source.py` already established). `RTSPFrameSource` itself required zero changes.

Key implementation details, each verified directly against real behavior on this machine:

- **Zero I/O in `__init__`** — `cv2` itself is imported lazily inside `open()`.
- **Credentials embedded via URL netloc** (`rtsp://user:pass@host:port/path`) for RTSP endpoints only — a local file path (used by every offline test) is left untouched.
- **Bounded connect time requires constructor `params`, not `.set()` after construction.** `cv2.VideoCapture(url, ...)` performs the actual connect *synchronously inside the constructor* — a property set afterward is already too late. Verified directly: an unroutable host otherwise blocks for the OS's own TCP retransmission timeout (tens of seconds). `CAP_PROP_OPEN_TIMEOUT_MSEC`/`CAP_PROP_READ_TIMEOUT_MSEC` must be passed as constructor `params`.
- **`params` requires pinning `apiPreference` explicitly.** Verified directly: `cv2.CAP_ANY` (the default OpenCV would otherwise choose) probes multiple backend candidates in turn when `params` are present, and any candidate that doesn't understand these two properties ignores them and blocks for its own much longer default — silently defeating the bound. `CAP_FFMPEG` is pinned by default.
- **A refused connection is NOT fast**, verified directly against a real closed local TCP port: FFMPEG's own RTSP client still waits out the full configured timeout rather than surfacing an OS-level "connection refused" immediately. Documented in `docs/architecture/physical_cctv_access_checklist.md` so an operator sets a short timeout and tests reachability with `ping`/VLC first.
- **`read()` never raises for end-of-stream** (honest `None`, matching `RTSPFrameSource`'s own "no frame this cycle" convention) but **does raise** if the underlying capture closed out from under it (a genuine drop) — proven with a real forced mid-stream release, not just simulated via a fake.
- **`close()` is idempotent and never raises.**
- Codec is decoded from the real FourCC integer OpenCV reports; width/height come from `CAP_PROP_FRAME_WIDTH`/`HEIGHT` — `None` for anything genuinely unreportable, never fabricated.

**Status: REAL IMPLEMENTATION, OFFLINE TESTED.** `tests/test_opencv_decoder_backend.py` (23 tests): lifecycle, credential URL construction, real local-file decode, real mid-stream drop + reconnect, connection-refused timing, repeated-failure non-hang, and a real two-camera "one fails while the other keeps working" proof at the `LiveCameraPipeline` level.

## 3. Full chain proof with NO fakes in frame acquisition (Phase 3)

`tests/test_real_decoder_full_chain_e2e.py` — the one test in this codebase proving:

```
real vtest.avi
  -> OpenCVFrameDecoderBackend (real cv2.VideoCapture)
  -> RTSPFrameSource (real, unmodified)
  -> CameraFrame
  -> YOLOHumanDetector(UltralyticsYOLOBackend)  (real inference, real local weights)
  -> SimpleSingleCameraTracker (real, stable track ids)
  -> SimulationIdentityResolver -> Detection
  -> CameraManager -> MultiCameraFusionEngine -> BuildingStateEstimator -> BuildingState
  -> LiveOccupantManager
```

`FakeRTSPBackend` is never imported by this file. Every earlier "real YOLO on real video" proof (`tests/test_real_yolo_model_validation.py`, `scripts/demo_real_yolo_tracking.py`, the Real Camera Calibration milestone's own §6) fed frames through `ReplayFrameSource`, never `RTSPFrameSource` — this closes that specific, previously-open gap. World projection/calibration is deliberately not exercised here (no physically measured scene exists for `vtest.avi`'s real courtyard — fabricating one would be dishonest, unchanged from the prior milestone's own position).

**Status: REAL IMPLEMENTATION + REAL DATA TESTED.**

## 4. Live camera configuration (Phase 4 — investigation only, no change needed)

`models.engineering_asset.ConnectionInfo` already provides one clean production path: `rtsp_address`, `ip_address`, `username`, `credential_ref` (password resolved from `credential_store` at connect time, never stored in project JSON — unchanged from `cctv_integration_readiness.md` §7). No separate `port` field exists — a port is expected embedded in `rtsp_address` (`rtsp://host:554/path`), the same convention every RTSP client uses. No vendor-specific settings were added, per this milestone's own instruction not to invent them ahead of a real camera to test against.

**Status: ALREADY SUFFICIENT — no code change made.**

## 5. Camera connection diagnostic (Phase 5)

`scripts/test_camera_connection.py` — the physical-access-day tool. Given `--camera-id`/`--endpoint` (and, for authenticated cameras, `--username` + `--credential-ref` or `--prompt-password`, never a bare `--password` CLI argument — that would leak into shell history), it reports: connection status transitions, first-frame success, measured resolution/codec, measured FPS, reconnect attempts observed, and a sanitized failure reason. An optional `--detect --weights <path>` mode also runs the real `YOLOHumanDetector` against received frames. The password is never printed, logged, or included in any report line — mechanically re-verified in `tests/test_credential_store.py::RealDecoderBackendCredentialSafetyTests`.

Verified directly against both a real local-file dry run (`Connection OK`, correct resolution/FPS) and a real unreachable-address failure (bounded, sanitized `Stream Unavailable` report).

**Status: REAL IMPLEMENTATION, OFFLINE TESTED.**

## 6. Calibration data-collection harness (Phase 6)

`scripts/calibrate_camera_scene.py` gained `--capture-frame <endpoint> --capture-out <path.png>`, using the *same* production `RTSPFrameSource`/`OpenCVFrameDecoderBackend` as the rest of the live path — never a second, ad-hoc `cv2.VideoCapture` call. Combined with the pre-existing `--pick-points` helper, the full on-site sequence is now: capture a real frame off the real camera → click floor reference points → measure their real-world positions → build a scene JSON → fit/validate. No new GUI, no second Designer application, per this milestone's own "not a giant calibration application" instruction.

**Status: REAL IMPLEMENTATION, OFFLINE TESTED** (verified against `vtest.avi` as a dry run; the correspondence-fitting/validation math itself was already MATHEMATICALLY TESTED by the prior calibration milestone, unchanged here).

## 7. Calibration quality runtime policy (Phase 7)

Investigated whether production runtime treated `CONFIGURED — UNVALIDATED` and `VALIDATED` calibration identically: **it did** — `WorldProjector.project()` used a `CalibrationProfile`'s intrinsics/extrinsics regardless of whether `quality` was ever set, and nothing downstream could tell the difference.

**Policy established** (no accuracy threshold invented — none is defensible yet, matching `camera_calibration_and_world_projection.md` §7's own position): an `UNVALIDATED` calibration's `world_position` is still produced (never blocked — this milestone's own explicit instruction against arbitrarily blocking useful diagnostics), but is now clearly marked via a new `provenance` field (§8 below) as one of `no_calibration` / `unvalidated` / `validated`. A validation attempt that could not project any reference point at all (`quality.rmse_m is None` despite `quality` existing) is honestly treated as `unvalidated`, not `validated` — an attempted-but-failed check earns no more trust than never checking.

**Status: REAL IMPLEMENTATION, OFFLINE TESTED.**

## 8. World-position provenance (Phase 8)

The smallest additive field necessary, threaded through the existing chain with no redesign:

```
WorldProjector.project() -> WorldProjection.provenance          (new field, always set)
  -> LiveCameraPipeline._process_camera_cycle()
  -> RawHumanDetection.world_position_provenance                (new field, default None)
  -> live_camera_pipeline.identity_resolver._to_detection()
  -> Detection.world_position_provenance                        (new field, default None)
  -> LiveCameraPipeline.run_cycle() -> LiveOccupantManager.update(world_position_provenance=...)
  -> LiveOccupant.world_position_provenance                     (new field, default None)
```

Three string constants (`camera_calibration.camera_model.WORLD_POSITION_PROVENANCE_{NONE,UNVALIDATED,VALIDATED}`), matching the plain-string-vocabulary convention `RTSPFrameSource.status` already established (no forced cross-package enum import). `BuildingState` itself was not touched — the objective (Command Center/debugging can never present an assumed position as physically validated) is met at the `LiveOccupant` layer, one hop before `BuildingState`, which is where per-occupant detail already lives.

**Status: REAL IMPLEMENTATION, OFFLINE TESTED.** `tests/test_world_position_provenance.py` (11 tests) plus zero regressions across the full `live_camera_pipeline`/`live_occupants`/`camera_calibration` suites (91+ pre-existing tests still passing).

## 9. Command Center / camera status (Phase 9)

A genuine architectural finding: this codebase's `command_center/` package reads only `BuildingState` (aggregate, cross-camera) — no per-camera connectivity/calibration view exists there, and `BuildingState.active_camera_ids`/`offline_camera_ids` are derived from `Camera.active` (configuration), not real connectivity (a known, pre-existing, documented gap — `cctv_integration_readiness.md` §6 — this milestone does not redesign `building_state/estimator.py` to fix, per its own Phase 15 guard against redesigning `BuildingState`).

The one place per-camera operational status already lived is `designer.widgets.camera_manager_panel.CameraManagerPanel` (Connection status, mode-aware detail). This milestone added a **Calibration** column there (`NOT CONFIGURED` / `CONFIGURED — UNVALIDATED` / `VALIDATED — RMSE: X m`), reading the *same* `CalibrationRegistry` instance the Property Panel's own "Calibrate Camera..." dialog writes into (`designer/windows/main_window.py` now shares one registry between both panels — previously each `PropertyPanel` instance owned its own, unreachable from `CameraManagerPanel`). The duplicate status-text formatting logic that existed twice inside `property_panel.py` was factored into one shared function, `camera_calibration.camera_model.calibration_status_text()`, now used in three places instead of two independently-maintained copies.

**Honest limitation, not fabricated:** "Frames: RECEIVING" and "Detector: ACTIVE" (named in this milestone's own Phase 9 prompt) have no real runtime signal anywhere in this codebase to back them — nothing currently wires per-cycle frame/detector activity into any status object in production (only `scripts/test_camera_connection.py`'s own diagnostic run measures those, live, for the duration of that one manual test). Fabricating a status field with no real signal behind it would violate this codebase's own "never fabricate" discipline throughout `camera_calibration`/`live_camera_pipeline`/`RTSPFrameSource`. Connection + Calibration + RMSE are shown because they are genuinely backed by real data; Frames/Detector are not, and are not shown.

**Status: REAL IMPLEMENTATION, OFFLINE TESTED.**

## 10. Physical CCTV dry-run (Phase 10)

`scripts/dry_run_physical_cctv.py` — rehearses the exact sequence for physical-access day using a local video file, reporting each stage as `READY NOW` or `REQUIRES PHYSICAL CCTV ACCESS`:

```
Camera configuration [READY NOW]
  -> Credential lookup [READY NOW]
  -> Decoder startup [READY NOW]
  -> Frame acquisition [READY NOW]
  -> YOLO [READY NOW if --weights given, else REQUIRES PHYSICAL... only in the sense
           that real weights are needed, not physical camera access]
  -> Tracking [READY NOW]
  -> Calibration lookup [REQUIRES PHYSICAL CCTV ACCESS -- no measured scene exists]
  -> World projection [REQUIRES PHYSICAL CCTV ACCESS -- same reason]
  -> LiveOccupant [READY NOW]
  -> BuildingState [READY NOW]
```

Run against real `vtest.avi` + real `yolov8n.pt` weights, this genuinely reaches `BuildingState` with real, non-zero occupant tracks (verified directly: 7 occupants reached `LiveOccupantManager`, 6 occupant tracks reached `BuildingState`, in one real run of this script).

**Status: REAL IMPLEMENTATION + REAL DATA TESTED.**

## 11. Failure modes tested (Phase 11)

| Failure mode | Test | Result |
|---|---|---|
| Wrong/unreachable host | `OpenFailureTests` (`test_opencv_decoder_backend.py`) | Bounded failure, sanitized message |
| Connection refused (closed local port) | `ConnectionRefusedTests` | Bounded by configured timeout (NOT instant — see §2) |
| Bad path/endpoint | `OpenFailureTests` | `FrameDecoderError`, never a crash |
| Timeout | `OpenFailureTests`, `ConnectionRefusedTests` | Bounded via constructor `params`, verified |
| Stream opens but returns no frame | `RealLocalVideoDecodeTests::test_reading_past_end_of_file_returns_none_not_an_error` | Honest `None`, never raises |
| Stream drops mid-run | `RealDecoderThroughRTSPFrameSourceTests::test_mid_stream_drop_is_detected_and_reconnect_recovers` | Detected, `RTSPFrameSource`'s existing bounded reconnect recovers, using the real backend |
| Reconnect succeeds | Same test above | Verified |
| Reconnect fails / exhausted retries | Pre-existing `test_rtsp_failure_modes.py` (unmodified, still passing against the interface this backend also satisfies) | `Stream Unavailable`, never an infinite loop |
| Repeated failed opens | `OpenFailureTests::test_repeated_failed_opens_never_hang_or_loop` | Deterministic, no internal retry loop of its own |
| One camera fails while another continues | `OneCameraFailsWhileAnotherContinuesTests` | Verified at `LiveCameraPipeline` level, both real backends |
| Credentials missing | Pre-existing `RTSPFrameSource._resolve_password()` tests (unmodified) | `FrameDecoderError`, converted to `Stream Unavailable` |
| Credential-store unavailable | Pre-existing `test_credential_store.py` (unmodified) | Never crashes |
| YOLO weights missing | Pre-existing `ModelWeightsNotFoundError` (unmodified) | Raised at construction, never a silent download |
| Calibration missing | `test_world_position_provenance.py` | `provenance=no_calibration`, `world_position=None` |
| Calibration unvalidated | `test_world_position_provenance.py` | `provenance=unvalidated`, `world_position` still produced |
| Corrupt calibration | Pre-existing `CalibrationLoadError` (unmodified) | Raised, never silently accepted |
| Camera resolution differs from calibration | Pre-existing `camera_calibration.validation.resolution_mismatch()` (unmodified) — now has real `frame.width`/`height` to compare against, from the real decoder | Opt-in diagnostic, unchanged |

No failure crashed the full `LiveRuntime` or `LiveCameraPipeline` in any test above.

## 12. Security re-audit (Phase 12)

Re-audited the complete new path (`OpenCVFrameDecoderBackend`, `scripts/test_camera_connection.py`, `scripts/calibrate_camera_scene.py --capture-frame`) for password leakage:

- `repr(RTSPFrameSource(...))` with a real password, connected against a real (failing) endpoint via the real backend: password never appears — `tests/test_credential_store.py::RealDecoderBackendCredentialSafetyTests`.
- Every exception `OpenCVFrameDecoderBackend.open()` raises is built from `_redact(endpoint)`, never the credentialed URL `build_authenticated_url()` constructs internally (that value is used for exactly one `cv2.VideoCapture(...)` call and immediately falls out of scope).
- `scripts/test_camera_connection.py`'s password resolution is `getpass`-hidden or `credential_store`-resolved, never a bare CLI argument, never printed — mechanically re-verified by a static source-text guard.
- `scripts/calibrate_camera_scene.py --capture-frame` handles no credentials itself and redacts the endpoint it prints.
- Real credentials remain outside the repository/project file — unchanged from the pre-existing `credential_store` architecture, re-verified by the full pre-existing `test_credential_store.py` suite (23 tests, all still passing).

**Status: RE-AUDITED, MECHANICAL TESTS ADDED.**

## 13. Performance (Phase 13)

`scripts/benchmark_real_decoder_pipeline.py`, run against real `vtest.avi` + real `yolov8n.pt` on this development machine (CPU):

| Stage | Mean | p95 |
|---|---|---|
| Decode latency (real `OpenCVFrameDecoderBackend.read()`) | 1.262 ms | 1.880 ms |
| One-time YOLO model load + first inference (excluded from all averages) | 5441 ms | — |
| YOLO inference latency (real `UltralyticsYOLOBackend`, CPU) | 51.195 ms | 55.075 ms |
| Tracking latency (real `SimpleSingleCameraTracker.update()`) | 0.137 ms | 0.198 ms |
| Complete perception cycle (`LiveCameraPipeline.run_cycle()`, full chain) | 54.275 ms | 65.231 ms |

**Effective FPS (this machine, this benchmark only, CPU): 18.42.** World-projection latency was not measured in this run (no `--calibration` supplied — honestly reported as NOT RUN rather than fabricated; the projection math itself was already benchmarked at ~0.01ms/detection by the prior calibration milestone and is not expected to change this number materially). These numbers must not be extrapolated to real network/RTSP transport latency or to different hardware — unmeasured and unmeasurable until physical access exists.

## 14. Physical access runbook (Phase 14)

`docs/architecture/physical_cctv_access_checklist.md` now contains a literal, 19-step ordered procedure (topology → reachability → RTSP endpoint → external stream test → Camera Asset configuration → `scripts/test_camera_connection.py` → YOLO → mounting/floor measurements → `--capture-frame` → `--pick-points` → `calibrate_camera_scene.py` fit/validate → world projection → zone localization → one-camera `LiveRuntime` → only then multi-camera), each naming the exact tool/command to run, plus the connection-refused timing finding from §2 above.

## 15. Architecture guards (Phase 15) — held throughout

This milestone did not: give AI execution authority, change Decision Policy, auto-broadcast Voice Evacuation, auto-execute Building Controls, modify hazard physics, add passive-fire assets, retrain AI, add pose estimation, or add face recognition. The camera path remains perception-only — `OpenCVFrameDecoderBackend` decodes frames and nothing else; `YOLOHumanDetector` classifies persons only (unchanged, `HumanClassification.UNKNOWN`/no `state_evidence`, per the pre-existing Real Human Detection Pipeline milestone's own boundary). `tests/test_no_cv_dependencies.py` (unmodified) still passes — no `cv2`/`torch`/`ultralytics`/`onvif` import anywhere in `models/camera.py`, `camera_manager/`, `multi_camera_fusion/`, `virtual_camera/`, `building_state/`, `advisory_system/`, `command_center/`, `live_camera_pipeline/`, `credential_store/`, or `live_system/`.

## 16. Tests (Phase 16)

Focused suites run first (all passing), then the complete suite:

```
python -m unittest discover -s tests
```

**4121 / 4121 tests passing** (up from the 4076 baseline at commit `eb8fddf` — 45 new tests added by this milestone). One genuine regression surfaced and was fixed during this milestone: `tests/test_human_detection_architecture_guards.py`'s own pre-existing guard forbade `human_detection/` from importing `live_camera_pipeline.rtsp_backend` at all — written before this milestone, when no real `FrameDecoderBackend` implementation existed to need it. Narrowed to exempt only `opencv_decoder_backend.py` (the one file whose entire job is implementing that seam), with a new companion test (`test_opencv_decoder_backend_only_imports_the_decoder_seam_not_pipeline_layers`) re-confirming it still cannot reach `identity_resolver`/`pipeline`/`detection_provider`. One flaky, pre-existing, unrelated test (`test_zone_usability.py::test_zone_id_is_visible_and_copyable_in_property_panel`, an OS clipboard round-trip) failed once under full-suite contention and passed on every other run, isolated or not — not caused by, and not fixed by, this milestone.

## 17. Final component classification

| Component | Classification |
|---|---|
| `OpenCVFrameDecoderBackend` | REAL IMPLEMENTATION + REAL DATA TESTED |
| `RTSPFrameSource` + real backend, full chain to `BuildingState` | REAL IMPLEMENTATION + REAL DATA TESTED |
| `scripts/test_camera_connection.py` | REAL IMPLEMENTATION + OFFLINE TESTED (real local-file dry run; real network camera REQUIRES PHYSICAL CCTV) |
| `scripts/calibrate_camera_scene.py --capture-frame` | REAL IMPLEMENTATION + OFFLINE TESTED |
| Live camera configuration (`ConnectionInfo`) | ALREADY SUFFICIENT (no change) |
| Calibration quality runtime policy + provenance | REAL IMPLEMENTATION + OFFLINE TESTED |
| Command Center / Camera Manager Panel calibration status | REAL IMPLEMENTATION + OFFLINE TESTED |
| `scripts/dry_run_physical_cctv.py` | REAL IMPLEMENTATION + REAL DATA TESTED |
| Failure-mode handling (network/credential/calibration) | REAL IMPLEMENTATION + OFFLINE TESTED |
| Credential security | RE-AUDITED, MECHANICAL TESTS ADDED |
| Performance (decode/YOLO/tracking/cycle latency) | REAL DATA TESTED (CPU, local file only) |
| Real RTSP network transport against a physical camera | REQUIRES PHYSICAL CCTV ACCESS |
| Real, measured calibration (RMSE against a real scene) | REQUIRES PHYSICAL CCTV ACCESS |
| Real cross-camera ReID (`LiveReIDIdentityResolver`) | FUTURE WORK (unchanged from prior milestones) |
| Pose estimation / face recognition | FUTURE WORK (explicitly out of scope, unchanged) |

## 18. Files created / modified

**New:** `human_detection/opencv_decoder_backend.py`, `scripts/test_camera_connection.py`, `scripts/dry_run_physical_cctv.py`, `scripts/benchmark_real_decoder_pipeline.py`, `tests/test_opencv_decoder_backend.py`, `tests/test_real_decoder_full_chain_e2e.py`, `tests/test_world_position_provenance.py`, this document.

**Modified (additively):** `camera_calibration/camera_model.py` (`WORLD_POSITION_PROVENANCE_*`, `calibration_provenance()`, `calibration_status_text()`), `camera_calibration/projection.py` (`WorldProjection.provenance`), `live_camera_pipeline/human_detector.py` (`RawHumanDetection.world_position_provenance`), `live_camera_pipeline/identity_resolver.py` (pass-through), `live_camera_pipeline/pipeline.py` (sets/threads provenance), `virtual_camera/detection.py` (`Detection.world_position_provenance`), `live_occupants/occupant.py` (`LiveOccupant.world_position_provenance`), `live_occupants/manager.py` (`update(world_position_provenance=...)`), `designer/widgets/property_panel.py` (reuses shared `calibration_status_text()`), `designer/widgets/camera_manager_panel.py` (Calibration column), `designer/windows/main_window.py` (shares one `CalibrationRegistry`), `scripts/calibrate_camera_scene.py` (`--capture-frame`), `tests/test_human_detection_architecture_guards.py` (narrowed the pre-existing `rtsp_backend` import guard to exempt `opencv_decoder_backend.py` specifically, plus a new companion test), `tests/test_credential_store.py` (new `RealDecoderBackendCredentialSafetyTests`), `tests/test_camera_manager_panel.py` (new Calibration-column tests), `docs/architecture/cctv_integration_readiness.md` (pointer update), `docs/architecture/physical_cctv_access_checklist.md` (literal runbook).

**Unchanged, re-verified:** `RTSPFrameSource`, `CameraManager`, `MultiCameraFusionEngine`, `BuildingStateEstimator`, `BuildingState`, `CredentialStore`/`LocalFileCredentialStore`, `WorldProjector`'s own projection math, `calibration_solver.py`, `validation.py`.
