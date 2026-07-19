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
