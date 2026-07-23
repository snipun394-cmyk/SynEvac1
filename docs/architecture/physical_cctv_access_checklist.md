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

Once every box above is checked for one camera, proceed per `docs/architecture/cctv_integration_readiness.md` §13 (steps 1-7) and §18.2 (Milestone A): match this camera to its Digital Twin `Camera` asset, configure `ConnectionInfo` in the Property Panel, save the password once (captured into `credential_store`), and only then write the one remaining piece — a real `FrameDecoderBackend` implementation behind the already-built, already-offline-tested `RTSPFrameSource` (`live_camera_pipeline/rtsp_frame_source.py`, see readiness doc §19) — against the confirmed RTSP URL/codec/transport recorded here.

## First physical test procedure (Milestone A only — stop here)

`RTSPFrameSource` itself is implemented and fully offline-tested (readiness doc §19); the only missing piece is a real `FrameDecoderBackend`. Once that backend exists and every box above is filled in for one test camera, run exactly this sequence and no further:

1. Camera Asset `CAM-001` already exists in the Digital Twin (or create it).
2. Configure the real endpoint (`ConnectionInfo.rtsp_address`/`ip_address`/`username`) in the Property Panel.
3. Configure the credential reference (save the password once — captured into `credential_store`, never written to the project file).
4. Construct `RTSPFrameSource(camera_id="CAM-001", endpoint=..., decoder_backend=<the new real backend>, credential_ref=..., credential_store=...)` and call `start()`.
5. Confirm the source reaches status `Online` (directly, or via a wired `status_callback` reporting `CameraManager.connection_status("CAM-001") == CameraConnectionState.ONLINE`).
6. Call `read_frame()` once.
7. Confirm `CameraFrame.camera_id == "CAM-001"` — the one non-negotiable check (readiness doc §19.6).
8. Confirm whatever resolution/frame metadata (`width`/`height`/`codec`) the real backend reports, if any — `None` for anything it genuinely cannot report, never a fabricated value.
9. Call `stop()`.

**STOP THERE.** Do not, in this same test, wire a real `HumanDetector` (YOLO or equivalent — Milestone B, per readiness doc §18.2) or attempt cross-camera identity resolution (Milestone C). This first test proves exactly one claim: *SynEvac can receive and identify frames from one physical CCTV camera while preserving the Digital Twin Camera Asset identity* — nothing more.
