# Pre-CCTV Live Integration Readiness

Status as of this milestone: **no real CCTV camera is connected, and none is required by anything in this document.** This is an architecture-hardening milestone only. It exists so that when real camera access becomes available, connecting the first stream is a narrowly-scoped adapter task, not a redesign.

## 1. Current architecture: the Simulation data flow (confirmed, working, unchanged by this milestone)

```
Scenario Occupant
    -> simulation_runtime.human_observation_bridge.GroundTruthHumanObservationProvider
         (produces HumanObservation, person_id = occupant.id)
    -> virtual_camera.camera.VirtualCamera
         (visibility.engine.VisibilityEngine computes each camera's visible zones once;
          detection imperfection model applies missed-detection / false-positive noise)
    -> virtual_camera.detection.Detection
         (occupant_id passed straight through as the global identity)
    -> virtual_camera.provider.SimulatedDetectionProvider  (a DetectionProvider)
    -> camera_manager.manager.CameraManager
         (routes detections_at(camera_id, time) to whichever DetectionProvider
          is registered for that camera's DeviceMode)
    -> multi_camera_fusion.engine.MultiCameraFusionEngine
         (groups Detections into a FusedTrack purely by occupant_id equality)
    -> building_state.estimator.BuildingStateEstimator
    -> building_state.models.BuildingState
    -> advisory_system (civilian announcements, firefighter intelligence,
                         building recommendations, commander dashboard)
    -> command_center
```

This chain is complete, tested (~19 existing test modules covering every stage), and untouched by this milestone.

## 2. The future Live equivalent

```
Physical CCTV Camera
    -> [FUTURE] RTSPFrameSource        (live_camera_pipeline.frame_source.CameraFrameSource)
    -> [FUTURE] real HumanDetector     (live_camera_pipeline.human_detector.HumanDetector)
    -> live_camera_pipeline.human_detector.RawHumanDetection
    -> [FUTURE] LiveReIDIdentityResolver, or a manual/mapping-based one today
         (live_camera_pipeline.identity_resolver.IdentityResolver)
    -> virtual_camera.detection.Detection   (same type Simulation already produces)
    -> live_camera_pipeline.detection_provider.LiveCameraPipelineDetectionProvider
         (a DetectionProvider, registered for DeviceMode.LIVE)
    -> camera_manager.manager.CameraManager     <-- UNCHANGED
    -> multi_camera_fusion.engine.MultiCameraFusionEngine     <-- UNCHANGED
    -> building_state.estimator.BuildingStateEstimator     <-- UNCHANGED
    -> BuildingState -> advisory_system -> command_center     <-- UNCHANGED
```

Everything from `CameraManager` downward is **exactly the same code Simulation already runs through today.** The only genuinely new pieces this milestone had to build were the three boxes above `LiveCameraPipelineDetectionProvider`: a frame source seam, a detector seam, and an identity-resolution seam. All three exist today as interfaces plus honest, non-ML reference implementations — no vision code, no RTSP, no ReID model.

`live_camera_pipeline.pipeline.LiveCameraPipeline` is the orchestrator that wires these three seams together (`run_cycle()`); it never calls `CameraManager`/`MultiCameraFusionEngine`/`BuildingStateEstimator` itself — a caller (the future live orchestration entry point) wires those the same way Simulation's own composition root already does.

**Note on `live_system/`:** a separate, already-built orchestration skeleton (`LiveOrchestrator.run_cycle() -> LiveBuildingSnapshot`) exists in this codebase, but it does not instantiate `CameraManager`/`MultiCameraFusionEngine` and produces a different state type (`LiveBuildingSnapshot`, not `BuildingState`). This is a pre-existing architectural duplication, **not created or touched by this milestone**, and reconciling it is out of scope here — flagged as a named future-work item, not a blocker.

## 3. The central gap this milestone closes: identity resolution

`Detection.occupant_id` has always been treated as a **global** identity, and `MultiCameraFusionEngine` fuses detections purely by string equality on it (`multi_camera_fusion/engine.py:_group_by_occupant`). In Simulation this is exactly correct — ground truth already knows the global identity. A real camera's local tracker never will: two cameras independently tracking the same physical person will produce two *different* local track ids, and two cameras that happen to both emit local id `"5"` are almost certainly tracking two *different* people.

`live_camera_pipeline.identity_resolver.IdentityResolver` is the seam that sits between a camera-local `RawHumanDetection` and a globally-resolved `Detection`:

- `RawHumanDetection.local_track_id` is **namespaced by `camera_id`** — `(camera_id, local_track_id)` is the only safe key.
- `MappingIdentityResolver` is a real, non-ML strategy: an explicit `{(camera_id, local_track_id): global_id}` table. Any unmapped pair gets a synthesized id derived from the pair itself (`f"{camera_id}:{local_track_id}"`) — so a coincidental local-id collision across two cameras **never silently fuses two different people** by default. Proven directly against the real `MultiCameraFusionEngine` in `tests/test_identity_resolver.py` (scenarios A–D from the milestone spec) and end-to-end through a full fake pipeline including a camera handover in `tests/test_live_camera_pipeline.py`.
- `SimulationIdentityResolver` is a deterministic pass-through proving parity with Simulation's existing behavior. **It is not wired into the live-running Simulation pipeline** — `VirtualCamera`/`SimulatedDetectionProvider` keep doing that job exactly as before; this class exists only to exercise the `IdentityResolver` interface under test and as a resolver a future `ReplayDetectionProvider` could reuse as-is.
- A real `LiveReIDIdentityResolver` (actual cross-camera re-identification via appearance matching) is explicitly **not built** — it is the one remaining piece of computer vision this milestone deliberately leaves for when a real camera and a real detector exist to feed it.

## 4. Zone localization strategy (Phase 6 — documentation only, no code)

`RawHumanDetection.zone_id` and `Detection.zone_id` are both `Optional[str]`. A camera assigned to exactly one zone can localize a detection to that zone with confidence. A camera assigned to multiple zones cannot, honestly, without either per-camera calibration (homography from image space to world/zone space) or a per-zone sub-region rule — neither of which this milestone implements, per its own explicit instruction not to fabricate precise world coordinates.

**V1 recommendation for the first real camera integration:** assign each camera to exactly one zone wherever physically possible; leave `zone_id=None` (explicitly uncertain, never a guess) for any camera whose FOV genuinely spans more than one zone until calibration work is scoped separately. Nothing in `Detection`, `MultiCameraFusionEngine`, or `BuildingState` needs to change to support that — `zone_id` already tolerates `None` everywhere it's consumed.

## 5. Occupancy deduplication (Phase 7 — proven by test)

- Same person, two cameras, resolver declares the relationship → one `FusedTrack`, `occupancy = 1`.
- Two different people, two cameras, no relationship declared → two `FusedTrack`s, `occupancy = 2`.
- Same local track id string on two *different* cameras, no relationship declared → stays two people (never silently merged).
- A resolver *can* explicitly declare two different (camera, local id) pairs as one global person, including across a camera handover (t=0 only CAM-A sees them, t=1 both cameras overlap, t=2 only CAM-B sees them) — occupancy stays 1 throughout, and `MultiCameraFusionEngine`'s own `TrackHistory` records the handover (`previous_camera_id` / `current_camera_id`).

See `tests/test_identity_resolver.py` and `tests/test_live_camera_pipeline.py` for the full proof.

## 6. Camera runtime health model (Phase 9)

`Camera.active` (configuration: "is this asset enabled at all?") and camera *connectivity* ("is the live stream actually reachable right now?") are different questions that were previously conflated — `CameraStatus.active` is a direct passthrough of `Camera.active`, and `building_state/estimator.py:216` derives `offline_camera_ids` from `not status.active`, i.e. today a *disabled* camera and an *unreachable* camera look identical to `BuildingState`.

This milestone adds `camera_manager.connection_status.CameraConnectionState` (`CONFIGURED, CONNECTING, ONLINE, OFFLINE, DEGRADED, STREAM_UNAVAILABLE`) as a **runtime-only**, in-memory value tracked separately on `CameraManager` (`set_connection_status()` / `connection_status()`), never persisted on the `Camera` asset itself, and **not** wired into `BuildingState`/`CameraStatus` yet — doing so would mean changing `building_state/estimator.py`'s existing `active`-based derivation, which this milestone treats as a completed subsystem it should not redesign. Flagged here as the specific site future work should reconcile once a real Live adapter can actually report connectivity.

## 7. Credential architecture (Phase 10)

**Before this milestone:** `ConnectionInfo.password` was written into every saved `.syn` project file in plaintext, with no redaction anywhere (`repr()`, debug panels, logs all printed it verbatim).

**After this milestone:**

- `ConnectionInfo.to_dict()` **never** includes `password`, unconditionally — only `rtsp_address`, `ip_address`, `username`, and a new `credential_ref` field are persisted.
- `ConnectionInfo.__repr__` redacts `password` to `<redacted>`/`<unset>` — and because `Camera`/`EngineeringAsset` use Python's auto-generated dataclass `repr()`, which calls `repr()` on the nested `connection` field, `repr(Camera(...))` is safe transitively with no separate override needed.
- A new `credential_store/` package holds the actual secret, completely outside any project file: `CredentialStore` is a small interface (`save_credential`, `get_credential`, `delete_credential`, `has_credential`); `LocalFileCredentialStore` is the one concrete implementation for this milestone, backed by a JSON file at `Path.home() / ".synevac" / "credentials.json"` — outside the repository entirely, so it cannot be committed regardless of `.gitignore` correctness, and created lazily (no file exists until a password is actually saved).
- `credential_store.project_credentials.capture_and_clear_camera_credentials(project, store)` is the one function that captures any in-memory plaintext password into the store, assigns `credential_ref` (defaulting to `camera.id`), and clears the in-memory `password` field. It is called from the one real disk-I/O boundary, `serialization.serializer.Serializer.save()`/`.load()` (both now accept an optional `credential_store` parameter, defaulting to `None` so every existing caller is unaffected), which `designer/windows/main_window.py` now passes a real `LocalFileCredentialStore` into on every project save/open.
- **Legacy compatibility:** an old project file saved before this milestone still loads without error — `ConnectionInfo.from_dict()` still tolerates a bare `"password"` key. The very next save (or the load itself, when a `credential_store` is supplied) migrates that plaintext into the store and clears it from memory, so the plaintext never gets written back into the project file again. No credential is silently lost — it lands in the store either way.
- **Known limitation (by design, not a bug):** the property panel currently has no "reveal password" UI. After any save (or a legacy-project load), the in-memory `password` field is empty — only the `CredentialStore` has the real value. Re-opening the Property Panel for that camera will show an empty password field even though a credential does exist and is resolvable via `store.get_credential(camera.connection.credential_ref)`. A future adapter (e.g. an `RTSPFrameSource`) is expected to resolve the real password this way at connect time, never by reading `Camera.connection.password` directly.
- **Explicitly not built:** any OS keychain / Windows Credential Manager / environment-based secret store integration. `CredentialStore` is deliberately an interface for exactly this reason — `LocalFileCredentialStore` can be swapped for one of those later with zero change to anything that calls `CredentialStore`.
- Simulation mode never touches the credential store at all — proven directly in `tests/test_credential_store.py` with a store whose every method raises, wired through a full simulated tick.

## 8. Detection contract (Phase 2)

`virtual_camera.detection.Detection` already reuses `perception.models.human_observation`'s existing `HumanClassification`/`HumanState` enums (no duplicate vocabulary), and every field on it (`camera_id`, `timestamp`, `occupant_id`, `floor_id`, `zone_id`, `position`, `confidence`, `classification`, `human_state`, `is_false_positive`) is something a real detector could honestly populate — **except** that `occupant_id` was, and structurally still is, defined as a *global* identity. This milestone does not change `Detection`'s shape (every existing consumer already depends on that exact contract); instead, `live_camera_pipeline.identity_resolver.IdentityResolver` is what's responsible for actually earning the right to call whatever it puts in `occupant_id` a global identity, before a `Detection` is ever constructed from a real camera's output.

`live_camera_pipeline.human_detector.RawHumanDetection` is the honest, pre-resolution counterpart: `local_track_id` (namespaced by `camera_id`), `bounding_box`, raw `classification_evidence`/`state_evidence` (still just evidence, not fact, until fused/resolved), and an honestly-uncertain `zone_id`.

## 9. Observed vs. inferred (Phase 5)

`perception.human_inference.InferenceFlag` already establishes the precedent this milestone leans on rather than duplicating: `POSSIBLE_PRE_MOVEMENT_DELAY`, `POSSIBLE_INJURY`, `HIGH_RESCUE_PRIORITY` are all *derived*, never a field a detector sets directly. The same discipline applies to `RawHumanDetection`/`Detection`: `classification_evidence`/`state_evidence`/`classification`/`human_state` are always what was *observed* (a person appears fallen), never a diagnosis (possible injury) — that inference step is `derive_inference_flags()`'s job, unchanged and untouched by this milestone.

## 10. Simulation / Replay / Live mode contract (Phase 11)

`Camera.mode` (`DeviceMode.SIMULATION` / `REPLAY` / `LIVE`) already existed before this milestone, and `CameraManager.register_detection_provider(mode, provider)` / `set_camera_mode()` already routed generically by mode with **zero code change required** to support Live: `CameraManager.register_detection_provider(DeviceMode.LIVE, LiveCameraPipelineDetectionProvider(...))` works today. `tests/test_camera_mode_identity_stability.py` proves `camera.id`, `floor_id`, `zone_ids`, `position`, `rotation` (orientation), and `horizontal_fov` are all unchanged by a `SIMULATION -> LIVE` mode switch, and that the same `Camera` object (not a replacement) stays registered throughout — only the registered provider changes.

## 11. Files created / modified

**New packages:** `live_camera_pipeline/` (`frame_source.py`, `human_detector.py`, `identity_resolver.py`, `detection_provider.py`, `pipeline.py`), `credential_store/` (`store.py`, `local_file_store.py`, `project_credentials.py`).

**New file:** `camera_manager/connection_status.py`.

**Edited (additive only):** `models/engineering_asset.py` (`credential_ref` field, redacted `to_dict()`/`__repr__`), `serialization/serializer.py` (optional `credential_store` parameter), `designer/windows/main_window.py` (constructs and passes a `LocalFileCredentialStore`), `camera_manager/manager.py` (`set_connection_status`/`connection_status`).

**New tests:** `tests/test_identity_resolver.py`, `tests/test_live_camera_pipeline.py`, `tests/test_camera_mode_identity_stability.py`, `tests/test_camera_connection_status.py`, `tests/test_credential_store.py`, `tests/test_no_cv_dependencies.py`.

**Edited tests:** `tests/test_engineering_asset.py` (new credential contract), `tests/test_camera_manager.py`/`test_virtual_camera.py`/`test_multi_camera_fusion.py` (extended dependency-direction guard regex).

**New script:** `scripts/benchmark_live_camera_pipeline.py`.

## 12. Per-camera information checklist (Phase 15)

When real CCTV access becomes available, gather the following **per physical camera** before starting integration:

- [ ] Camera ID to assign in SynEvac (or which existing Digital Twin `Camera` asset it matches)
- [ ] Camera display name
- [ ] Floor
- [ ] Zone(s) — ideally exactly one, per §4 above
- [ ] IP address
- [ ] RTSP URL / stream endpoint (main stream and substream, if both exist)
- [ ] Camera manufacturer / model (if known — affects which vendor quirks a future `RTSPFrameSource` needs to handle)
- [ ] Stream codec (H.264 / H.265 / other)
- [ ] Resolution
- [ ] FPS
- [ ] Authentication required? Username? Password/credential method
- [ ] Network accessibility from the SynEvac computer (same LAN? VPN? firewall rules needed?)
- [ ] Whether the camera/NVR allows multiple simultaneous RTSP clients (SynEvac reading the stream must not disrupt any existing recording/monitoring system)

## 13. First real integration procedure

1. Select one camera.
2. Match it to an existing Digital Twin `Camera` asset (or create one) — the `Camera.id` becomes the permanent Digital Twin identity; nothing about it changes when the physical camera or its RTSP address changes later.
3. Configure its endpoint (`ConnectionInfo.rtsp_address`/`ip_address`/`username`) in the Property Panel; enter its password once — it is captured into `LocalFileCredentialStore` on the next save and never appears in the saved project file.
4. Test network reachability (a plain `ping`/port check, outside SynEvac).
5. Open the stream (this is where the first real `RTSPFrameSource` implementation gets written — implementing `live_camera_pipeline.frame_source.CameraFrameSource`).
6. Read frames.
7. Display/debug frames (a minimal viewer, to confirm the stream is being read correctly, before any detection logic is written).
8. Run a real human detector (the first real `HumanDetector` implementation — YOLO or equivalent) on those frames, producing `RawHumanDetection`s.
9. Run a local tracker (assigning `local_track_id`, namespaced by this camera).
10. Confirm the resulting `RawHumanDetection`s look correct.
11. Map to zone (per §4 — start with `zone_id=None` if the camera's FOV spans more than one zone).
12. Resolve identity — start with a `MappingIdentityResolver` with an empty mapping (every detection gets its own namespaced synthetic id) until a second camera with real overlap exists to test cross-camera resolution against.
13. Fuse detections — register a `LiveCameraPipelineDetectionProvider` for `DeviceMode.LIVE`, no `CameraManager`/`MultiCameraFusionEngine` changes needed.
14. Build `BuildingState` — no `BuildingStateEstimator` changes needed.
15. Verify Command Center reflects the live camera's occupant correctly.
16. Only then scale to additional cameras, and only then does cross-camera identity resolution actually need to do real work.

## 14. Architecture guards (Phase 16) — enforced by test

`tests/test_no_cv_dependencies.py` enforces, by scanning source files, that none of `models/camera.py`, `camera_manager/`, `multi_camera_fusion/`, `virtual_camera/`, `building_state/`, `advisory_system/`, `command_center/`, `live_camera_pipeline/`, or `credential_store/` import `cv2`, `torch`, `ultralytics`, or `onvif`. `tests/test_live_camera_pipeline.py` enforces that `live_camera_pipeline/` never imports `camera_manager`/`multi_camera_fusion`/`building_state` directly. Each package's own pre-existing `*PackageDependencyDirectionTests` guard (in `test_camera_manager.py`, `test_virtual_camera.py`, `test_multi_camera_fusion.py`) was extended with the same forbidden-import list rather than duplicated.

## 15. What remains once real camera access exists

Only the pieces this document explicitly deferred: a real `RTSPFrameSource`, a real `HumanDetector` (YOLO/tracking/pose estimation/fallen-person detection/etc.), and eventually a real `LiveReIDIdentityResolver` for genuine cross-camera re-identification once more than one camera has real overlapping coverage. Nothing else in this document, and nothing downstream of `LiveCameraPipelineDetectionProvider`, should require any change.

## 16. CCTV Pipeline End-to-End Offline Validation milestone

Status as of this milestone: still **no real CCTV camera connected** — physical access was expected within 1-2 days of this milestone but had not arrived yet, so this milestone deliberately proves the entire data path using only offline/deterministic sources instead of waiting. Everything below was built and test-proven with zero network I/O, zero video decoding, and zero real detection model.

**What this milestone added:**

- `live_camera_pipeline.replay_frame_source.ReplayFrameSource` — the first concrete, production `CameraFrameSource` (previously only a test-only `FakeFrameSource` existed, deliberately kept out of the production package per Phase 16's "pure interface" rule). Plays back a fixed, ordered, in-memory sequence of `CameraFrame`s deterministically (`read_frame()`/`reset()`), gated by `start()`/`stop()` with no background thread. `is_source_available` distinguishes "frames provided" / "a `source_path` genuinely exists on disk" from "nothing to play" — the honest signal Phase 8's Camera Manager status now surfaces as `Replay — Source Missing` vs `Replay — Source Loaded`. It deliberately never opens or decodes `source_path` itself (no image-decoding library is imported anywhere in `live_camera_pipeline/`, enforced by `tests/test_no_cv_dependencies.py`) — a real video file's frames still require a real decoder, which remains explicitly out of scope (see §15).
- `tests/live_camera_pipeline_fixtures.MockHumanDetector` — a deterministic `HumanDetector` stand-in (test-only, same "production package stays pure interface" convention as `FakeHumanDetector` before it), reused across every new test in this milestone instead of duplicated per file.
- `designer/widgets/camera_manager_panel.py` — the Status column now appends a truthful, mode-aware detail (`Simulation — Ready/Not Configured`, `Replay — No Source/Source Missing/Source Loaded`, `Live — Credentials Missing/Not Connected/Online`), derived entirely from fields that already existed (`CameraStatus.mode`/`has_detection_provider`, `CameraManager.connection_status()`, `Camera.connection.credential_ref`/`password`). `Live` never reports `Online` unless `connection_status()` was explicitly told `ONLINE` by something — nothing in this codebase does that yet, so every Live camera honestly shows `Not Connected` until a real adapter exists.
- `tests/test_replay_frame_source.py`, `tests/test_cctv_offline_pipeline_validation.py`, and additions to `tests/test_camera_manager_panel.py` — see §16.1 below for exactly what each proves.

**What this milestone deliberately did not touch:** `CameraManager`, `MultiCameraFusionEngine`, `BuildingStateEstimator`, `Detection`, `RawHumanDetection`, `IdentityResolver`, `LiveCameraPipeline`, `LiveCameraPipelineDetectionProvider` — every one of those already existed and already worked (proven by the earlier milestone's own `tests/test_live_camera_pipeline.py`); this milestone only needed to add the one missing concrete class (`ReplayFrameSource`) and prove the full chain end-to-end through it.

### 16.1 The end-to-end chain now proven (offline, deterministic, no CCTV)

```
Camera(id="CAM-001")                              <- Digital Twin identity, stable
    -> CameraManager.register_camera / set_camera_mode(REPLAY)
    -> ReplayFrameSource(camera_id="CAM-001", frames=[...])   <- Camera Source Adapter
    -> CameraFrame(camera_id, timestamp, frame_sequence, payload_ref)   <- Frame Packet
    -> MockHumanDetector.detect(frame)             <- Detector Adapter seam
    -> RawHumanDetection(camera_id, local_track_id, ...)
    -> MappingIdentityResolver.resolve(...)         <- Identity Resolution seam
    -> Detection(camera_id, occupant_id, ...)
    -> LiveCameraPipelineDetectionProvider.publish/detections_at   <- Detection Provider
    -> CameraManager.detections_for_camera(camera_id, time)
    -> MultiCameraFusionEngine.fuse(detections, time)   <- Multi-Camera Fusion
    -> FusionResult.tracks[*].source_camera_ids includes "CAM-001"
    -> BuildingStateEstimator.estimate(fusion_result=...)   <- Building State
    -> BuildingState.occupant_tracks[occupant_id].source_camera_ids includes "CAM-001"
```

Proven explicitly, by test:

- **Camera identity continuity** (`CameraIdentityContinuityTests`): `camera_id` survives every stage above unchanged; the same `Camera` object stays registered when its source or its `ConnectionInfo` (RTSP/IP/credentials) changes.
- **Multi-camera deduplication** (`MultiCameraDeduplicationTests`, `BuildingStateNoDoubleCountingTests`): two cameras, three physical people (one seen only by CAM-A, one only by CAM-B, one by both) produce **4 raw detections but exactly 3 unique `BuildingState.occupant_tracks`** — never 4. The shared occupant's `FusedTrack.source_camera_ids` contains both cameras.
- **Mode independence** (`ModeIndependenceTests`): the identical `Detection`/`FusedTrack`/`BuildingState` shape results regardless of whether the camera's registered provider was reached via `DeviceMode.SIMULATION`, `REPLAY`, or `LIVE` — only the upstream provider differs; `CameraManager`, `MultiCameraFusionEngine`, and `BuildingStateEstimator` never branch on mode.
- **No automatic network connection**: setting `mode=LIVE` and populating real-looking `ConnectionInfo` (RTSP URL, IP, username) never changes `CameraManager.connection_status()` away from its default `CONFIGURED`, and `detections_for_camera()` returns `()` rather than attempting anything, when no provider is registered.
- **Graceful failure**: a `ReplayFrameSource` with no frames and no existing `source_path` reports `is_source_available=False` and `read_frame()` returns `None` forever (never raises); resolving a Live camera's credential before any password was ever saved returns `None` from `CredentialStore.get_credential()` (never raises, never auto-creates the credential file).
- **Credential safety**: a real-looking password never appears in `repr(Camera(...))` or `Camera.to_dict()`, re-proven directly in this milestone's own tests (`LiveCredentialSafetyTests`), on top of the earlier milestone's existing coverage.

## 17. Physical CCTV Connection Procedure

The procedure for the day physical CCTV access actually arrives, with every step marked against what is **ALREADY IMPLEMENTED** (built and test-proven, today), what **REQUIRES PHYSICAL CCTV ACCESS** (cannot be done, tested, or even meaningfully attempted before then), and what is **FUTURE COMPUTER-VISION WORK** (explicitly out of scope of every milestone so far, real vision/ML work for later).

1. **Open the Digital Twin.** — *ALREADY IMPLEMENTED.* No change needed.
2. **Select the already-placed Camera Asset.** — *ALREADY IMPLEMENTED.* `Camera.id` is the permanent Digital Twin identity (§13 step 2, re-proven end-to-end offline in §16.1); the Camera Manager Panel already lists every camera building-wide.
3. **Set Mode = Live.** — *ALREADY IMPLEMENTED.* `CameraManagerPanel`'s mode dropdown already calls `CameraManager.set_camera_mode(camera_id, DeviceMode.LIVE)`; proven to change nothing else about the Camera Asset (§16.1 "Camera identity continuity").
4. **Enter/configure RTSP/IP endpoint.** — *ALREADY IMPLEMENTED.* `ConnectionInfo.rtsp_address`/`ip_address`/`username` already exist on `Camera.connection`, editable via the Property Panel, persisted (password excluded) via `Serializer.save()`.
5. **Associate credential reference.** — *ALREADY IMPLEMENTED.* Typing a password once captures it into `LocalFileCredentialStore` and assigns `credential_ref` (§7); the Camera Manager Panel now honestly reports `Live — Credentials Missing` until this step is done (§16, Phase 8).
6. **Test connection.** — *REQUIRES PHYSICAL CCTV ACCESS.* No code exists to attempt a real connection at all today (by design — see §14's guard test forbidding `cv2`/`torch`/`ultralytics`/`onvif` imports in every camera-adjacent package). This is where a real network reachability check gets written, and the one place `CameraManager.set_connection_status()` would first ever be called with something other than a test's own value.
7. **Receive `FramePacket`s carrying the existing `camera_id`.** — *PARTIALLY IMPLEMENTED / REQUIRES PHYSICAL CCTV ACCESS.* The `FramePacket` type (`CameraFrame`) and the `CameraFrameSource` interface it flows through already exist and are already proven end-to-end via `ReplayFrameSource` (§16). What's missing is the one real implementation: an `RTSPFrameSource` that actually opens a stream and decodes real frames — this needs a real image-decoding library and a real stream to test against, neither available yet.
8. **Feed frames into the detector adapter.** — *PARTIALLY IMPLEMENTED / FUTURE COMPUTER-VISION WORK.* The adapter seam (`HumanDetector.detect(frame) -> RawHumanDetection`) already exists and is already proven end-to-end via `MockHumanDetector` (§16). The real detector behind it (YOLO or equivalent) is genuine computer-vision work, not started.
9. **Produce `Detection`s.** — *ALREADY IMPLEMENTED.* `IdentityResolver.resolve(...)` -> `Detection` already works end-to-end today, offline (§16.1) — `MappingIdentityResolver` is a real, honest, non-ML strategy usable as-is on day one (start with an empty mapping, per §13 step 12).
10. **Fuse overlapping-camera detections.** — *ALREADY IMPLEMENTED.* `MultiCameraFusionEngine` needs zero changes — already proven against real multi-camera duplicate suppression offline (§16.1).
11. **Update `BuildingState`.** — *ALREADY IMPLEMENTED.* `BuildingStateEstimator` needs zero changes — already proven to receive unique, non-double-counted occupant tracks offline (§16.1).

**In one sentence:** every step except "physically reach a real camera and decode its real video" (steps 6-8's real implementations) is already built and already test-proven; the only genuinely new work still ahead is a real `RTSPFrameSource`, a real `HumanDetector`, and — once more than one camera has real overlapping coverage — a real `LiveReIDIdentityResolver` (unchanged from §15).

## 18. Live CCTV Connection Readiness Audit

Status as of this milestone: still **no physical CCTV/NVR access.** This milestone adds no new code — it is a static audit of the existing seam, so that the day access arrives, connecting one real camera is a scoped adapter task with a known checklist, not fresh exploration. Everything below was determined by reading `live_camera_pipeline/`, `credential_store/`, `camera_manager/`, `models/camera.py`, `models/engineering_asset.py`, and the existing test suite — no guesses about RTSP URL format, authentication, NVR architecture, vendor, codec, resolution, FPS, transport, or ONVIF support are made anywhere in this section, per this milestone's own instruction.

### 18.1 Exact `RTSPFrameSource` integration seam

A future `RTSPFrameSource` is a concrete subclass of `live_camera_pipeline.frame_source.CameraFrameSource` (`live_camera_pipeline/frame_source.py:24`) — the same interface `ReplayFrameSource` (§16) already implements today. Nothing downstream of it changes.

**The interface it must satisfy, method by method:**

| Method | Signature | Contract established by `CameraFrameSource`/`ReplayFrameSource`/`FakeFrameSource` |
|---|---|---|
| `start()` | `() -> None` | Transitions the source to a state where `read_frame()` may yield frames. `ReplayFrameSource` treats this as a pure flag flip (no I/O); a real implementation is expected to open the actual connection here — the interface does not forbid `start()` from doing real network I/O, it only requires that after it returns, `is_running` reflects the result. |
| `stop()` | `() -> None` | Transitions the source out of the running state. `ReplayFrameSource` again just flips the flag. A real implementation must release whatever `start()` acquired (socket/connection) — not specified beyond "must exist," since no concrete network implementation exists yet to establish the convention. |
| `is_running` | property `-> bool` | Must reflect current start/stop state truthfully. Nothing in the interface ties this to actual stream health — `is_running=True` only means "not stopped," not "frames are currently arriving." (See §18's connection-status gap below — that's a separate concern, deliberately not folded into this property.) |
| `read_frame()` | `() -> Optional[CameraFrame]` | Called once per source per pipeline cycle by `LiveCameraPipeline.run_cycle()` (`live_camera_pipeline/pipeline.py:51`), synchronously, no polling thread built into the seam. Must return `None` — never raise — when not running or no frame is currently available (`ReplayFrameSource.read_frame()`, `FakeFrameSource.read_frame()` both do this). Whether a real implementation buffers on a background thread or blocks briefly is entirely its own decision; the seam does not require either. |

**Constructor requirements:** the `CameraFrameSource` ABC imposes none — `__init__` is not part of the interface. The established convention (from `ReplayFrameSource.__init__` and `FakeFrameSource.__init__`) is: accept `camera_id: str` as the first argument, and do **no I/O in `__init__`** (`ReplayFrameSource`'s own docstring calls this out explicitly — constructing a source must never touch a disk or network by itself). A future `RTSPFrameSource.__init__` should follow this: accept the endpoint/credential data it needs as plain constructor arguments, connect only inside `start()`.

**`camera_id` identity requirement (critical, not optional):** every `CameraFrame` a source produces must carry the **same `camera_id`** as the pre-existing Digital Twin `Camera.id` already registered in `CameraManager` (`models/engineering_asset.py`'s `BaseObject.id`, discovered via `CameraManager.discover_cameras()`). This is not enforced by any runtime check — it is enforced by convention and by how routing works: `LiveCameraPipeline.frame_sources` is a `Mapping[str, CameraFrameSource]` keyed by `camera_id` (`live_camera_pipeline/pipeline.py:32`), `LiveCameraPipelineDetectionProvider.detections_at(camera_id, time)` is looked up by that same string (`camera_manager/manager.py:252`, via `camera.mode`'s registered provider), and `CameraManager.detections_for_camera()` calls it with `camera.id`. A mismatched `camera_id` anywhere in this chain produces detections that silently never reach the camera they came from — there is no error, only missing data. **Rule for the first real integration: the string handed to `RTSPFrameSource(camera_id=...)` must be copy-pasted from the exact `Camera.id` already shown in the Camera Manager Panel, never invented fresh.**

**`CameraFrame` structure** (`live_camera_pipeline/frame_source.py:6`, frozen dataclass):

- `camera_id: str` — see above.
- `timestamp: float` — no wall-clock/monotonic-clock convention is established anywhere in this codebase; `ReplayFrameSource` takes a caller-supplied timestamp per frame with no interpretation. A real `RTSPFrameSource` is free to use `time.time()`, a stream's own PTS, or anything else — nothing downstream currently branches on units or epoch, only on ordering for display purposes.
- `frame_sequence: int` — `ReplayFrameSource` uses a simple 0-based ordinal, incrementing once per `read_frame()` call, **local to that one source's own stream** (not a global/cross-camera counter). No consumer currently treats gaps in this sequence as meaningful (no drop-detection logic reads it today) — a real implementation is free to leave gaps where frames were dropped, but nothing currently surfaces that as a "frames dropped" signal (see §18.3, `TESTS` row).
- `payload_ref: Optional[Any] = None` — deliberately untyped/opaque. Nothing in `live_camera_pipeline/` ever inspects it (enforced indirectly by `tests/test_no_cv_dependencies.py`'s import guard, since inspecting it as an image would require an image library). Interpreting it is entirely `HumanDetector.detect()`'s job — a real `RTSPFrameSource` can put a decoded numpy frame, a raw byte buffer, or a file handle here; the choice only needs to match whatever the paired `HumanDetector` implementation expects to receive.

**Error handling / connection-status reporting — the one real gap:** `CameraFrameSource` has **no exception contract** and **no connection-status reporting method** of its own. Connection health is tracked entirely externally, on `CameraManager` (`camera_manager/connection_status.py`'s `CameraConnectionState`: `CONFIGURED, CONNECTING, ONLINE, OFFLINE, DEGRADED, STREAM_UNAVAILABLE`, set via `CameraManager.set_connection_status()` / read via `connection_status()`). **Nothing in the codebase today ever calls `set_connection_status()` with anything other than a test's own value** — there is no wiring from a real (or even fake) frame source's actual state into `CameraManager`'s connection-status registry. A future `RTSPFrameSource` (or whatever orchestrates it) is expected to call `camera_manager.set_connection_status(camera_id, state)` itself as connection attempts succeed/fail/degrade; this call site does not exist yet anywhere.

**Reconnect expectations:** undefined by the interface and unexercised by any existing test — no code anywhere calls `start()` a second time after `stop()`. Whether `start()` is idempotent, whether it must be re-callable after a `stop()`, and what happens if `read_frame()` is called before the first `start()` (both `ReplayFrameSource` and `FakeFrameSource` simply return `None`, never raise) are the only precedents to follow; genuine reconnect-after-network-failure behavior (retry/backoff) is unspecified and left to the future implementation.

**Shutdown behavior:** `stop()` is the only shutdown hook the interface defines. `ReplayFrameSource.stop()` does nothing beyond flipping `is_running` because it holds no real resource. A real `RTSPFrameSource.stop()` would need to close its socket/decoder — not specified further since no concrete implementation exists to establish the convention yet.

**Multiplicity / threading:** `LiveCameraPipeline.run_cycle()` iterates every registered source once per call, in the order of `frame_sources.values()`, and always calls `human_detector.detect(frame)` synchronously right after `read_frame()` returns a non-`None` frame (`live_camera_pipeline/pipeline.py:49-56`). There is no concurrency in the orchestrator itself — a slow `read_frame()` on one camera blocks every other camera's cycle. This is an existing, unaddressed scaling limit, not something this audit fixes.

### 18.2 Staged first-camera milestone sequence

Per this milestone's own instruction, the first real-camera work must **not** combine RTSP integration, detection, and ReID into a single step. The dependency chain (confirmed above) supports exactly this staging:

- **Milestone A** — One physical camera → `RTSPFrameSource` → `CameraFrame`. Success condition: `CameraFrame.camera_id` matches the Digital Twin `Camera.id`, frames arrive at a measurable rate, `payload_ref` holds *something* (raw bytes/decoded frame — format is this milestone's own choice, only needs to match what Milestone B's detector expects). No detection, no identity resolution.
- **Milestone B** — `CameraFrame` → real `HumanDetector` → `RawHumanDetection`. Consumes Milestone A's frames; produces `RawHumanDetection`s with `local_track_id` populated by whatever local tracker accompanies the detector. Can be developed and tested against `ReplayFrameSource`-recorded frames from Milestone A before wiring live, exactly as `MockHumanDetector` already proves the seam offline (§16).
- **Milestone C** — Two overlapping physical cameras → `LiveReIDIdentityResolver` → `MultiCameraFusionEngine`. Only meaningful once two cameras with real overlapping coverage exist; until then, `MappingIdentityResolver` with an empty mapping (§13 step 12) is the correct, honest interim resolver — every detection gets its own namespaced synthetic id, never silently fused.
- **Milestone D** — Full `CameraManager` → `BuildingState` → live system integration. This is largely **already done** (§16.1's offline chain proves `CameraManager`→`MultiCameraFusionEngine`→`BuildingStateEstimator`→`BuildingState` end-to-end); what Milestone D actually adds is wiring a real `LiveCameraPipeline` instance (with A/B/C's real components) into whatever composition root constructs `CameraManager` today (`designer/windows/main_window.py`), registering it via `register_detection_provider(DeviceMode.LIVE, provider)`.

The first live-camera milestone (Milestone A alone) proves exactly one claim: **"SynEvac can receive and identify frames from one physical CCTV camera while preserving the Digital Twin Camera Asset identity."** Detection and identity resolution are explicitly out of scope for it.

### 18.3 Diagnostic tool design (not implemented)

A future connection diagnostic utility belongs at **`scripts/diagnose_camera_connection.py`**, alongside the existing `scripts/benchmark_live_camera_pipeline.py` (§11) — both are manual, non-pytest operational tools, not part of the test suite, run directly by a developer.

It should be implementable **today**, entirely against `CameraFrameSource` (i.e. runnable right now against `ReplayFrameSource`, with zero RTSP-specific assumptions), since everything it needs to report is already expressible through the existing interface:

| Report field | Derived from |
|---|---|
| Camera Asset (`CAM-001`) | The `camera_id` passed to the diagnostic tool, matched against `CameraManager.get_camera()` |
| Connection: Connected / Failed | `source.is_running` after `start()`, plus whether `read_frame()` ever returned non-`None` within a timeout |
| Frame received: Yes / No | Whether any `read_frame()` call returned a `CameraFrame` |
| Resolution | Not derivable from `CameraFrame` today — `payload_ref` is opaque; a real `RTSPFrameSource` would need to expose this itself (e.g. as a property alongside `read_frame()`), which is new surface not yet designed |
| Measured FPS | Frame count / elapsed wall-clock time across a fixed sampling window |
| Frame latency | Time between successive `read_frame()` calls returning a frame, or (if available) `time.time() - frame.timestamp` |
| Codec | Not derivable from `CameraFrame` at all — same gap as Resolution; only a real source implementation could report this, and only if it chooses to expose it |
| Frames dropped | Gaps in `frame_sequence` between successive frames, *if* the source's `frame_sequence` convention leaves gaps on drop (not guaranteed — see §18.1) |
| Digital Twin identity preserved | `frame.camera_id == expected_camera_id`, checked on every frame received |

**Must never print `connection.password` or any resolved credential** — only `credential_ref` (a reference id, not a secret) may appear in output, matching `ConnectionInfo.__repr__`'s existing redaction discipline (§7).

**Why not implemented now:** Resolution and Codec are not derivable from the current `CameraFrameSource`/`CameraFrame` contract at all — building the tool now would mean inventing new interface surface (e.g., new properties on `CameraFrameSource`) speculatively, ahead of any real source needing them, which is exactly the kind of premature design this milestone's own instructions caution against. The tool's design is recorded here so it can be built in an afternoon once `RTSPFrameSource` (Milestone A) exists to give those two fields real values.

### 18.4 Final static audit — what's missing for one real camera

**ALREADY IMPLEMENTED:**
- `CameraFrameSource` interface + one concrete non-network implementation (`ReplayFrameSource`) proving the seam end-to-end.
- `CameraFrame`, `RawHumanDetection`, `Detection` data contracts.
- `HumanDetector`/`IdentityResolver` interfaces + honest reference implementations (`MappingIdentityResolver`, `SimulationIdentityResolver`).
- `LiveCameraPipeline` orchestrator (`run_cycle()`), `LiveCameraPipelineDetectionProvider`.
- `CameraManager` registration/routing/mode-switching — proven mode-independent (§16.1).
- `MultiCameraFusionEngine` → `BuildingStateEstimator` → `BuildingState` — proven with real multi-camera dedup, zero changes needed for Live.
- `ConnectionInfo` (RTSP/IP/username fields) on `Camera.connection`, editable in the Property Panel, persisted (password excluded).
- `CredentialStore` interface + `LocalFileCredentialStore`, wired into project save/load.
- `CameraConnectionState` enum + `CameraManager.set_connection_status()`/`connection_status()` storage (nothing calls the setter yet — see below).
- Camera Manager Panel status column, mode-aware and truthful about what isn't connected.

**MISSING CODE (buildable today, no camera access required):**
- The one real `RTSPFrameSource` implementation (Milestone A) — does not exist in any form, not even a stub.
- Any call site that actually invokes `CameraManager.set_connection_status()` outside of tests — nothing today ever reports a camera as `CONNECTING`/`ONLINE`/`OFFLINE`/`DEGRADED` from real (or even attempted) activity.
- The diagnostic script (`scripts/diagnose_camera_connection.py`, §18.3) — buildable against `ReplayFrameSource` today, not yet written.
- Any property for a frame source to report resolution/codec/dropped-frame count — no such surface exists on `CameraFrameSource` yet; would need to be added if a diagnostic tool is expected to show these before real streams exist to inform the design.

**REQUIRES PHYSICAL CCTV ACCESS:**
- Writing and testing the real `RTSPFrameSource` against an actual stream (the implementation can be *started* without access, but cannot be *proven correct* without one).
- Populating `docs/architecture/physical_cctv_access_checklist.md` (§19) with real values for one test camera.
- The real network-reachability "Test connection" step (§17 step 6) — no code exists to attempt this today, by design (guarded by `tests/test_no_cv_dependencies.py`'s forbidden-import list).

**FUTURE COMPUTER VISION (explicitly out of scope of every milestone so far):**
- Real `HumanDetector` (YOLO or equivalent) — Milestone B.
- Real `LiveReIDIdentityResolver` (cross-camera appearance-based re-identification) — Milestone C.
- Any pose estimation, fallen-person detection, or wheelchair detection beyond what `HumanClassification`/`HumanState` already model as evidence types.

**One-sentence answer to "what's missing to display/read frames from one real camera tomorrow":** everything below `RTSPFrameSource` in the chain already works and needs no changes; the only missing code is the `RTSPFrameSource` implementation itself (plus, optionally, wiring `set_connection_status()` calls into it) — nothing else in this codebase blocks that.

See `docs/architecture/physical_cctv_access_checklist.md` for the practical per-camera information checklist to fill out the moment physical access arrives.

## 19. RTSP Frame Source — offline-testable production implementation

Status as of this milestone: still **no physical CCTV/NVR access.** This milestone builds the one piece of code §18.4 named as genuinely missing: a real `RTSPFrameSource`. It remains fully offline-testable — no network I/O, no real decode library, no real RTSP server — by introducing one new seam (`FrameDecoderBackend`) that the real network/decode work plugs into later, exactly the way `CameraFrameSource` itself let `RTSPFrameSource` be built without a real camera.

### 19.1 What was added

- **`live_camera_pipeline/rtsp_backend.py`** — `FrameDecoderBackend` (ABC: `open(endpoint, username, password)`, `read() -> Optional[DecodedFrame]`, `close()`, `is_open`), `DecodedFrame` (`payload_ref`, `width`, `height`, `codec` — the same "opaque payload plus honestly-optional metadata" shape as `CameraFrame` itself), and `FrameDecoderError` (used for `RTSPFrameSource`'s own non-backend failures, e.g. an unresolved credential).
- **`live_camera_pipeline/rtsp_frame_source.py`** — `RTSPFrameSource`, a concrete `CameraFrameSource`, plus `redact_endpoint()`. Always driven by an injected `FrameDecoderBackend` — no production backend implementation exists yet (writing one needs a real decode library, out of scope here), so `decoder_backend` is a required constructor argument, not optional.
- **`live_camera_pipeline/frame_source.py`** — `CameraFrame` gained three additive, backward-compatible fields: `width`, `height`, `codec` (all `Optional`, default `None`). `ReplayFrameSource` and every existing test constructing `CameraFrame` are unaffected.
- **`tests/live_camera_pipeline_fixtures.py`** — `FakeRTSPBackend`, a deterministic offline `FrameDecoderBackend` (configurable `open()`/`read()` failures, an in-memory frame queue), reused across every new RTSP test the same way `MockHumanDetector` already is.
- **New tests:** `tests/test_rtsp_frame_source.py` (construction/lifecycle/redaction/identity), `tests/test_rtsp_failure_modes.py` (Phase 7 reconnect scenarios + Phase 12's 15 failure scenarios), `tests/test_rtsp_camera_manager_status_integration.py` (Phase 8 status wiring), `tests/test_rtsp_offline_e2e.py` (Phase 10/11 full chain + camera-replacement proof).
- **New script:** `scripts/benchmark_rtsp_frame_source.py` (Phase 13 — routing/construction/status overhead only; see §19.5).

### 19.2 Constructor performs zero network I/O

`RTSPFrameSource(camera_id=..., endpoint=..., decoder_backend=..., ...)` never calls `decoder_backend.open()`. Every constructor argument is a plain assignment — proven directly in `tests/test_rtsp_frame_source.py::NoNetworkIOOnConstructionTests`. A Digital Twin Camera Asset can be fully configured for Live mode (endpoint, username, `credential_ref`) with zero risk of ever touching the physical network until `start()` is explicitly called.

### 19.3 Status vocabulary — reused, not duplicated, without an import

`RTSPFrameSource.status` is one of five plain strings: `"Configured"`, `"Connecting"`, `"Online"`, `"Degraded"`, `"Stream Unavailable"` — chosen to match `camera_manager.connection_status.CameraConnectionState`'s own `.value` strings exactly. This is deliberate, not a competing enum: `live_camera_pipeline/` is forbidden from importing `camera_manager` at all (the existing `LiveCameraPipelineDependencyDirectionTests` guard, re-asserted for the two new RTSP files in `tests/test_rtsp_frame_source.py::DependencyDirectionTests`), so it cannot import `CameraConnectionState` directly. A composition root converts with one line: `CameraConnectionState(status)`.

Mapping from the milestone's conceptual states: `STOPPED → Configured`, `CONNECTING → Connecting`, `CONNECTED → Online`, `RECONNECTING → Degraded` (a source that was previously online and is now retrying after a drop is exactly what `CameraConnectionState.DEGRADED`'s own docstring already means), `FAILED → Stream Unavailable` (retries exhausted, not permanent — calling `start()` again retries). `OFFLINE` is deliberately never produced by `RTSPFrameSource` itself.

### 19.4 CameraManager status integration (Phase 8's named gap, now closed for RTSP)

`RTSPFrameSource` accepts an optional `status_callback: Callable[[camera_id, status, detail], None]`, invoked on every status transition. It never imports `CameraManager` itself (Phase 8's own requirement) — a composition root wires it:

```python
def on_status_changed(camera_id, status, detail):
    camera_manager.set_connection_status(camera_id, CameraConnectionState(status))

source = RTSPFrameSource(..., status_callback=on_status_changed)
```

Proven end-to-end in `tests/test_rtsp_camera_manager_status_integration.py`: successful connect → `ONLINE`, exhausted retries → `STREAM_UNAVAILABLE`, `stop()` → `CONFIGURED`, a mid-stream drop transitions through `DEGRADED` and back to `ONLINE` on successful reconnect. `CameraManager` and `designer/widgets/camera_manager_panel.py` still import nothing from `live_camera_pipeline` — Command Center and the Camera Manager Panel remain transport-agnostic, structurally proven by that same test file.

A callback that itself raises can never crash a connect/read cycle — `RTSPFrameSource` swallows exceptions from `status_callback` (it is UI/logging code, not this class's own correctness boundary).

### 19.5 Reconnection strategy — bounded, synchronous, injectable timing

`max_retries` / `retry_delay` / `backoff_factor` control a bounded exponential-backoff loop (`retry_delay * backoff_factor ** attempt`), used identically for the very first connection attempt (`start()`) and for recovering from a mid-stream drop detected inside `read_frame()` — there is exactly one retry mechanism, not two. Never an infinite loop: once `max_retries` is exhausted, the source reports `Stream Unavailable` and stops attempting until `start()` is called again.

This is synchronous and inline — consistent with `live_camera_pipeline.pipeline.LiveCameraPipeline.run_cycle()`'s own pre-existing, undocumented-as-a-problem "no concurrency in the orchestrator" design (§18.1). A slow reconnect on one camera still blocks that cycle, same pre-existing limit as any slow `read_frame()`.

`sleep_fn` (replacing `time.sleep`) and `clock_fn` (replacing `time.time`) are both injectable, defaulting to the real functions in production but overridden with zero-delay stand-ins in every test — no test in this milestone sleeps for a real duration. `stop()` called concurrently mid-retry (or a supervisor translating "camera disabled" into `stop()`) is proven deterministically by having a test's injected `sleep_fn` call `source.stop()` as a side effect at the exact point a real concurrent stop would land — the retry loop checks `is_running` immediately after every sleep and aborts rather than fighting a deliberate stop with one more attempt.

### 19.6 Camera identity guarantee (Phase 4/11 — re-proven for RTSP specifically)

`DecodedFrame` (what a `FrameDecoderBackend` returns) has no `camera_id` field at all — structurally, nothing a decoder backend produces can influence `CameraFrame.camera_id`; `RTSPFrameSource.read_frame()` always constructs `CameraFrame(camera_id=self.camera_id, ...)` from the value it was given at construction. Proven directly, including an explicit "malicious/misconfigured backend tries to smuggle a different id through `payload_ref`" case, in `tests/test_rtsp_frame_source.py::CameraIdGuaranteeTests`.

The camera-replacement scenario (`tests/test_rtsp_offline_e2e.py::CameraReplacementPreservesIdentityTests`) proves the full claim end-to-end: a `Camera(id="CAM-001")` Digital Twin asset, an `RTSPFrameSource` pointed at `rtsp://old-camera/stream` producing detections that reach `BuildingState.occupant_tracks[...].source_camera_ids`, then — simulating the physical camera being swapped — a **new** `RTSPFrameSource` instance pointed at `rtsp://new-camera/stream` but constructed with the identical `camera_id="CAM-001"`. `Camera.id`, the registered `Camera` object identity in `CameraManager`, its `floor_id`/`zone_ids`/mode registration, and every downstream `Detection`/`FusedTrack`/`BuildingState` key all stay `"CAM-001"` — no downstream configuration changes.

### 19.7 Credential handling

`_resolve_password()` resolves at connect time only, every time — through `CredentialStore.get_credential(credential_ref)` if both `credential_ref` and `credential_store` are supplied (the same "never read `Camera.connection.password` directly" discipline §7 already establishes), or a directly-supplied `password` otherwise. Never cached onto `self` beyond the single connect attempt that needed it.

Every failure path — no store configured, no credential saved under the reference, the store itself raising (Phase 12 items 11/12) — is caught by the same generic `except Exception` the connect/reconnect loop already uses, converted into an honest `Stream Unavailable` status with a sanitized detail message, never a crash.

`redact_endpoint()` strips any `user:pass@` embedded directly in an endpoint string (`rtsp://***:***@host`), applied to `repr()` and to any exception text a backend might echo the endpoint back through. `_sanitize_error()` additionally scrubs the literal resolved password value (whether directly supplied or freshly resolved from the store) out of any exception message before it ever becomes `last_error`/a status detail/a log line — proven with a password that deliberately appears inside a fake backend's own exception text (`tests/test_rtsp_frame_source.py::CredentialSafetyTests`). `__repr__` never prints the password even redacted-inline, only `<redacted>`/`<unset>`, matching `ConnectionInfo.__repr__`'s own convention exactly.

### 19.8 Offline RTSP end-to-end result

`tests/test_rtsp_offline_e2e.py::RTSPOfflineEndToEndTests` reruns §16.1's exact multi-camera-deduplication proof (two cameras, three physical people, one seen by both) through **real `RTSPFrameSource` instances** (only the decoder backend is fake) instead of `ReplayFrameSource` — four raw detections fuse to exactly three `BuildingState.occupant_tracks`, never four, with correct multi-camera provenance on the shared occupant. This is the proof that the production RTSP source architecture works end-to-end without ever touching a network.

### 19.9 Performance

`scripts/benchmark_rtsp_frame_source.py` measures `RTSPFrameSource`'s own overhead against `FakeRTSPBackend`-supplied fake decoded frames: frame-routing/`CameraFrame`-construction overhead (~1-2 µs/frame on a development machine) and status-callback dispatch overhead (a fraction of a µs/call). **Not measured, and not claimable, until physical CCTV access exists:** real decode latency, real network latency, real frame-drop rate, real stream stability — every one of those requires an actual RTSP stream to measure honestly.

### 19.10 Answers to the milestone's own explicit questions

- **Is `RTSPFrameSource` implemented?** Yes — `live_camera_pipeline/rtsp_frame_source.py`, a concrete `CameraFrameSource`.
- **Has it been tested against a real physical CCTV stream?** No.
- **Can `RTSPFrameSource` be tested without network access?** Yes — every test in `tests/test_rtsp_frame_source.py`, `tests/test_rtsp_failure_modes.py`, `tests/test_rtsp_camera_manager_status_integration.py`, and `tests/test_rtsp_offline_e2e.py` runs against `FakeRTSPBackend`, zero sockets, zero real decode library.
- **If a physical camera is replaced but the Digital Twin Camera ID stays `CAM-001`, does downstream SynEvac continue using `CAM-001`?** Yes — §19.6.
- **Does configuring a Live camera automatically attempt a network connection?** No — §19.2; re-confirmed by `tests/test_cctv_offline_pipeline_validation.py::ModeIndependenceTests::test_configuring_live_mode_performs_no_automatic_connection`, unaffected by this milestone.

### 19.11 What still remains once real camera access exists

Only what §15/§18.4 already named: a real `FrameDecoderBackend` implementation (needs a real decode/transport library — the one thing this milestone deliberately does not add), a real `HumanDetector` (YOLO/tracking), and eventually a real `LiveReIDIdentityResolver`. Nothing about `RTSPFrameSource` itself, `CameraManager`, `MultiCameraFusionEngine`, or `BuildingState` should need to change — the real backend plugs into exactly the `FrameDecoderBackend` seam `FakeRTSPBackend` already proves out. See `docs/architecture/physical_cctv_access_checklist.md` for the scoped first-physical-test procedure.
