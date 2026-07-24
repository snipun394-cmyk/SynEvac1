# Physical CCTV Access Checklist

Practical, fill-in-the-blanks checklist for the day physical college CCTV/NVR access becomes available. Scope: **one test camera only** — see `docs/architecture/cctv_integration_readiness.md` §18.2 for why the first milestone must stay narrowly scoped to `RTSPFrameSource` → `CameraFrame` alone, with no detection or identity resolution attempted yet.

**Never write real credentials into this file, into any file committed to the repository, or into any project `.syn` file.** Passwords belong in `credential_store` (`credential_store/local_file_store.py`, `Path.home() / ".synevac" / "credentials.json"`, outside the repo) only, captured via the Property Panel exactly as `docs/architecture/cctv_integration_readiness.md` §7 describes. Where this checklist needs to record *how* a credential is stored, record a reference (e.g. "stored under camera's own id"), never the value itself.

## Camera Identity

- [ ] Digital Twin `camera_id` (the exact `Camera.id` already registered in the Building Designer / Camera Manager Panel — copy it, don't retype it)
- [ ] Camera display name
- [ ] Floor
- [ ] Zone(s) — ideally exactly one (see readiness doc §4: a camera spanning multiple zones can't honestly localize a detection without calibration this codebase doesn't yet do)
- [ ] Physical camera location (room / hallway / stairwell — human-readable description for whoever is on-site during setup)

## Network

- [ ] Camera IP address
- [ ] NVR IP address, if applicable
- [ ] Is direct camera access allowed, or must it go through the NVR?
- [ ] Network/VLAN restrictions (is the SynEvac machine on the same network segment? firewall rules needed?)
- [ ] Does the camera/NVR allow multiple simultaneous RTSP clients? (SynEvac reading the stream must not disrupt any existing recording/monitoring system already attached to it)

## Stream

- [ ] RTSP URL or stream path
- [ ] Main stream vs. substream (which one SynEvac should use — substream may be preferable for lower bandwidth during early testing)
- [ ] Codec (H.264 / H.265 / other)
- [ ] Resolution
- [ ] FPS
- [ ] Transport requirements (TCP / UDP, or whatever the camera/NVR documentation specifies)

## Authentication

- [ ] Username
- [ ] Password — **do not write the value here.** Record only that it exists and where it will be stored (`credential_store` under the camera's own `credential_ref`, per the Property Panel flow).
- [ ] Authentication type (Basic / Digest / other, if known)

## Device

- [ ] Manufacturer
- [ ] Model
- [ ] Firmware version, if available
- [ ] ONVIF support, if known (not assumed — see readiness doc §18, no ONVIF code exists or is planned until this is confirmed)

## Tests (run outside SynEvac first)

- [ ] Can VLC (or an equivalent RTSP client) open the stream?
- [ ] Can OpenCV/FFmpeg read frames from it? (a standalone script, not SynEvac — SynEvac has no frame-decoding library wired in yet by design, see readiness doc §14/§18.1)
- [ ] Average frame latency
- [ ] Dropped frames observed
- [ ] Reconnect behavior (what happens if the stream is interrupted — does the camera/NVR recover on its own, or does the client need to re-initiate?)

## Calibration Measurements (Real Camera Calibration & World-Coordinate Validation milestone)

Collect these for one test camera **in addition** to the Stream/Device sections above — this is the input `scripts/calibrate_camera_scene.py` needs to produce a real, metrically-validated `CalibrationProfile` (`docs/architecture/camera_calibration_and_world_projection.md` has the full field reference and a worked example). Bring a laptop, a tape measure or laser measure, and a way to pause/screenshot a live frame on-site.

- [ ] `camera_id` (same Digital Twin id recorded above — never retyped, copy it)
- [ ] `floor_id` (same Digital Twin floor id recorded above)
- [ ] Actual configured stream resolution (`image_width` × `image_height`, pixels — must match the Stream section's own Resolution box exactly; a mismatch here is a genuine, previously-undetectable failure mode, see `camera_calibration.validation.resolution_mismatch()`)
- [ ] Camera mounting height above the floor (`mount_height`, meters — tape/laser measure from the floor directly below the lens up to the lens itself; this is a GIVEN, measured input, never fitted)
- [ ] Camera's floor-plan position (`camera_position` = (x, y) meters, in this floor's own Digital Twin coordinate system — read off the Building Designer's own placement, or measure from a known reference point already on the floor plan)
- [ ] Horizontal field of view, if known from a datasheet (`horizontal_fov_degrees`) — OR, if unknown, plan to fit `yaw_degrees`/`pitch_degrees` from correspondences instead (below); FOV itself is still needed either way
- [ ] **At least 3, ideally 5+, floor reference points**, each as a (pixel, world) correspondence:
  - [ ] A paused/screenshotted real frame from this exact camera, saved locally (`scripts/calibrate_camera_scene.py --pick-points <image>` can record the pixel side by clicking)
  - [ ] For each reference point: a real, identifiable floor mark (a tile corner, a floor-mounted sign, a doorway threshold, a wall base — anything visible in the frame AND physically measurable) — record its pixel coordinate in the paused frame AND its measured real-world (x, y) floor-plan position (tape/laser measure from the same reference point the camera's own `camera_position` was measured from)
  - [ ] Split these into a **fitting set** (the correspondences `scripts/calibrate_camera_scene.py` solves yaw/pitch from) and a **separate, held-out validation set** (points NOT used for fitting — this is the only honest way to get a real RMSE; validating against the same points used to fit only ever proves the fit converged, not that it generalizes)
- [ ] Camera roll, if visibly tilted in the image (`roll_degrees` — 0.0 for a normally, levelly mounted camera; only worth measuring if the image horizon is visibly not level)

## Handoff to SynEvac integration (after the above is filled in)

Once every box above is checked for one camera, proceed per `docs/architecture/cctv_integration_readiness.md` §13 (steps 1-7) and §18.2 (Milestone A): match this camera to its Digital Twin `Camera` asset, configure `ConnectionInfo` in the Property Panel, save the password once (captured into `credential_store`), and connect using `human_detection.opencv_decoder_backend.OpenCVFrameDecoderBackend` — the real `FrameDecoderBackend` implementation built by the CCTV Connection & Calibration Readiness milestone (see `docs/architecture/cctv_connection_and_calibration_readiness.md`) — behind the already-built, already-offline-tested `RTSPFrameSource` (`live_camera_pipeline/rtsp_frame_source.py`, see readiness doc §19).

**IMPORTANT, verified directly against a real closed local port during this milestone:** the FFMPEG-backed RTSP client this backend uses does NOT fail fast on a refused connection the way a raw TCP connect would — it waits out the full configured `open_timeout_ms` regardless. Set a short timeout (2-5 seconds is enough for a genuinely reachable camera) and, per the Tests section above, always confirm reachability with `ping`/VLC/an equivalent RTSP client FIRST — do not rely on SynEvac itself to fail fast on a wrong IP/path.

## Literal physical-access-day procedure

This is the exact, ordered sequence to follow standing in front of the real CCTV/NVR system. Each step is marked with what tool to use.

1. **Identify recorder/camera/network topology.** Determine whether this is a direct-to-camera setup or an NVR-mediated one (see Network section above).
2. **Determine whether the SynEvac laptop can reach the camera/NVR.** `ping <camera or NVR IP>` from the SynEvac machine, outside SynEvac.
3. **Record the camera IP/device identity** into the Network/Device sections above — never a credential value.
4. **Determine the RTSP endpoint** (URL/path) — check the camera/NVR's own documentation or admin UI; record into the Stream section above.
5. **Test the stream outside SynEvac first** — VLC (`Media > Open Network Stream`) or `ffplay rtsp://...` against the real endpoint. If this doesn't work, nothing below will either — debug at this layer, not inside SynEvac.
6. **Configure the Camera Asset** in the Building Designer: `Mode = Live`, `ConnectionInfo.rtsp_address`/`ip_address`/`username` in the Property Panel; type the password once (captured into `credential_store`, never written to the project file — see readiness doc §7).
7. **Run SynEvac's own connection diagnostic:**
   ```
   python scripts/test_camera_connection.py --camera-id <CAM-ID> --endpoint <rtsp URL> \
       --username <username> --credential-ref <CAM-ID> --open-timeout-ms 3000
   ```
   Confirms: connection status, frame resolution, measured FPS, reconnect behavior — all without touching YOLO. A sanitized failure reason is printed if this fails; the password is never printed.
8. **Confirm real frames** — the diagnostic's own `RESULT: Connection OK` line, plus a sane, non-zero `Frame resolution`/`Measured FPS`.
9. **Run YOLO against the real stream** — re-run step 7 with `--detect --weights weights/yolov8n.pt` and confirm plausible detection counts (0 is honest if nobody is in frame; a crash or an exception is not).
10. **Measure camera mounting geometry** — mount height, floor-plan position (see Calibration Measurements section above).
11. **Measure floor reference points** — at least 3, ideally 5+, split into a fitting set and a held-out validation set.
12. **Capture a calibration frame:**
    ```
    python scripts/calibrate_camera_scene.py --capture-frame <rtsp URL or endpoint> --capture-out frame.png
    ```
13. **Pick pixel points** off the captured frame:
    ```
    python scripts/calibrate_camera_scene.py --pick-points frame.png --points-out clicked_pixels.json
    ```
14. **Fit calibration** — build a scene JSON (camera_id, floor_id, resolution, FOV, camera_position, mount_height, correspondences, validation_points — see `docs/architecture/camera_calibration_and_world_projection.md` §4 for the full field reference) and run:
    ```
    python scripts/calibrate_camera_scene.py scene.json --out calibration.json
    ```
15. **Validate against the held-out points** — the same command above already reports RMSE if `validation_points` were supplied; record the RMSE here, do not treat any specific number as pass/fail (no accuracy threshold is established yet — see camera_calibration_and_world_projection.md §7).
16. **Run world projection** — load `calibration.json` into the Property Panel's "Calibrate Camera..." dialog (or pass `--calibration calibration.json` to `scripts/dry_run_physical_cctv.py`) and confirm non-`None` `world_position` values for real detections.
17. **Confirm zone localization** — projected positions resolve to the expected `zone_id` for known real locations in the frame.
18. **Run the full `LiveRuntime` for ONE camera** — wire the real `RTSPFrameSource`/`OpenCVFrameDecoderBackend`/`YOLOHumanDetector`/calibration into `live_runtime.factory.build_live_runtime()` and confirm `BuildingState.occupant_tracks` reflects the one real camera correctly in Command Center.
19. **Only then consider a second camera / cross-camera ReID** — per readiness doc §18.2 Milestone C, not before.

**Rehearse this whole sequence offline first** with:
```
python scripts/dry_run_physical_cctv.py --weights weights/yolov8n.pt
```
This runs steps 6-18 above (minus the physical network/measurement parts) against a local video file, using the exact same production code, and reports which stages are `READY NOW` versus `REQUIRES PHYSICAL CCTV ACCESS`.
