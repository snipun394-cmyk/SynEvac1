"""Physical CCTV Field Validation milestone -- the ONE script to run
standing at the real college CCTV/NVR system. Orchestrates the already-
existing, already-tested production components (never reimplements
them) in progressive, layered modes, so a failure can be isolated to
the exact layer that caused it instead of debugging the entire
SynEvac stack blindly.

Production components this script constructs, in order, per the
already-frozen CCTV software baseline (commit 8120180, docs/
architecture/cctv_connection_and_calibration_readiness.md):

    Camera (models.camera.Camera)
    -> CredentialStore (credential_store.local_file_store.LocalFileCredentialStore)
    -> OpenCVFrameDecoderBackend (human_detection.opencv_decoder_backend)
    -> RTSPFrameSource (live_camera_pipeline.rtsp_frame_source)
    -> YOLOHumanDetector(UltralyticsYOLOBackend) (human_detection.yolo_human_detector/yolo_backend)
    -> SimpleSingleCameraTracker (tracking.simple_tracker)
    -> LiveCameraPipeline (live_camera_pipeline.pipeline)
    -> WorldProjector (camera_calibration.projection) -- REQUIRES a real,
       already-fitted CalibrationProfile; never fabricated
    -> LiveOccupantManager (live_occupants.manager)
    -> live_runtime.factory.build_live_runtime() -> LiveRuntime -> BuildingState
       + Crowd/Trajectory/Evacuation Progress/Recommendation/Guidance/Advisory

Modes (each builds on the previous -- pass the highest one you need):

    --connection-only   Test the RTSP/decoder connection only. No frames read.
    --frames             + read real frames, measure resolution/FPS/reconnects.
    --detect             + run the real YOLOHumanDetector on received frames.
    --track               + run the real SimpleSingleCameraTracker.
    --project              + project tracked detections through a REQUIRED,
                             already-fitted CalibrationProfile (FAILS HONESTLY
                             with CALIBRATION_REQUIRED if none is supplied --
                             never fabricates an illustrative one).
    --full-runtime          + wire everything into the real production
                             live_runtime.factory.build_live_runtime() and run
                             real cycles, reading BuildingState/Crowd
                             Intelligence/Trajectory Intelligence/Evacuation
                             Progress/Recommendation/Guidance/Advisory. Never
                             wires a voice_output_provider or
                             building_control_provider -- no automatic voice
                             broadcast or building control execution.

Examples:

    # Isolate the connection layer only:
    python scripts/run_physical_camera_validation.py --camera-id CAM-001 \\
        --endpoint rtsp://192.168.1.50:554/stream1 --username admin \\
        --credential-ref CAM-001 --connection-only

    # Full chain, real weights, real calibration, saved report:
    python scripts/run_physical_camera_validation.py --camera-id CAM-001 \\
        --endpoint rtsp://192.168.1.50:554/stream1 --username admin \\
        --credential-ref CAM-001 --weights weights/yolov8n.pt \\
        --calibration calibration.json --full-runtime \\
        --report-out field_validation_CAM-001.json

    # Offline rehearsal against a local file (no physical camera needed):
    python scripts/run_physical_camera_validation.py --camera-id CAM-DRYRUN \\
        --endpoint validation_media/vtest.avi --weights weights/yolov8n.pt --track

Never prints a resolved password. Usernames are printed (this codebase's
existing redaction policy -- ConnectionInfo/RTSPFrameSource -- already
treats username as non-secret, only password/credential values are
redacted); the endpoint itself is always redacted via the existing
live_camera_pipeline.rtsp_frame_source.redact_endpoint().
"""

import argparse
import json
import statistics
import subprocess
import sys
import time as time_module
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource, redact_endpoint
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.pipeline import LiveCameraPipeline

from human_detection.opencv_decoder_backend import OpenCVFrameDecoderBackend

from tracking.simple_tracker import SimpleSingleCameraTracker

from live_occupants.manager import LiveOccupantManager

from camera_calibration.calibration_loader import CalibrationLoadError, load_calibration_json
from camera_calibration.projection import WorldProjector

from models.camera import Camera
from models.building import Building
from models.engineering_asset import DeviceMode

from camera_manager.manager import CameraManager

from multi_camera_fusion.engine import MultiCameraFusionEngine

from live_runtime.factory import build_live_runtime

# Reused, not reimplemented -- the exact connection-diagnostic bookkeeping
# and password-resolution policy this milestone's own Phase 5 predecessor
# already built and proved (scripts/test_camera_connection.py). This
# script's job is to ORCHESTRATE it into progressively deeper modes, never
# to duplicate its logic.
import test_camera_connection as connection_diagnostic


# =====================================================
# Field-report failure vocabulary (Phase 13) -- a SCRIPT-LEVEL
# classification only, layered on top of (never replacing) the existing
# production status vocabularies (RTSPFrameSource.status/
# camera_manager.connection_status.CameraConnectionState,
# camera_calibration.camera_model.WORLD_POSITION_PROVENANCE_*). No new
# production status type is introduced anywhere in the real codebase --
# this exists only so a human reading a field-validation JSON report
# gets one unambiguous top-level word for "what actually went wrong,"
# without having to re-derive it from a raw status string + free-text
# error detail every time.
# =====================================================

OUTCOME_OK = "OK"
OUTCOME_NETWORK_UNREACHABLE = "NETWORK_UNREACHABLE"
OUTCOME_AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
OUTCOME_STREAM_UNAVAILABLE = "STREAM_UNAVAILABLE"
OUTCOME_NO_FRAMES = "NO_FRAMES"
OUTCOME_DECODER_FAILURE = "DECODER_FAILURE"
OUTCOME_YOLO_UNAVAILABLE = "YOLO_UNAVAILABLE"
OUTCOME_CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"
OUTCOME_CALIBRATION_INVALID = "CALIBRATION_INVALID"
OUTCOME_PROJECTION_UNAVAILABLE = "PROJECTION_UNAVAILABLE"
OUTCOME_RUNTIME_DEGRADED = "RUNTIME_DEGRADED"


def classify_connection_outcome(status: str, last_error: Optional[str]) -> str:

    # Best-effort, HEURISTIC classification of RTSPFrameSource's own
    # honest status/last_error into the vocabulary above -- never a
    # certainty (FFMPEG's own error text is not a structured API this
    # codebase controls), always logged alongside the raw status/detail
    # in the report so a human can re-judge it. Deliberately conservative:
    # falls back to the generic STREAM_UNAVAILABLE rather than guessing
    # wrong into a more specific, misleading category.

    if status == "Online":
        return OUTCOME_OK

    detail = (last_error or "").lower()

    # Deliberately NOT matching the bare word "unreachable" -- OpenCVFrameDecoderBackend's
    # own generic open() failure message ("stream unreachable, wrong path, or unsupported by
    # this OpenCV build") uses that word for EVERY open failure, network or not (verified
    # directly: a plain nonexistent local file path produces the identical wording). Only
    # genuinely network-specific phrasing is trusted here.
    if any(term in detail for term in ("resolve hostname", "no route to host", "network is down", "name or service not known")):
        return OUTCOME_NETWORK_UNREACHABLE

    if any(term in detail for term in ("401", "403", "unauthorized", "authentication", "auth failed", "forbidden")):
        return OUTCOME_AUTHENTICATION_FAILED

    if status in ("Configured", "Connecting", "Degraded", "Stream Unavailable"):
        return OUTCOME_STREAM_UNAVAILABLE

    return OUTCOME_STREAM_UNAVAILABLE


# =====================================================


class FieldValidationReport:

    # Phase 12 -- one machine-readable (JSON) + human-readable (console)
    # record per field session. Every section defaults to None ("this
    # stage was never reached/attempted this run"), never a fabricated
    # empty-but-present block -- the same "never fabricate" discipline
    # every other seam in this codebase already holds to.

    def __init__(self, camera_id: str, endpoint: str, mode: str):

        self.data = {
            "camera_id": camera_id,
            "endpoint_redacted": redact_endpoint(endpoint),
            "requested_mode": mode,
            "software_commit": _current_commit(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "connection": None,
            "frames": None,
            "detection": None,
            "tracking": None,
            "calibration": None,
            "projection": None,
            "runtime": None,
            "final_outcome": None,
            "warnings": [],
            "failures": [],
        }

    def warn(self, message: str) -> None:
        self.data["warnings"].append(message)
        print(f"WARNING: {message}")

    def fail(self, outcome: str, message: str) -> None:
        self.data["failures"].append({"outcome": outcome, "message": message})
        self.data["final_outcome"] = outcome
        print(f"FAILED [{outcome}]: {message}")

    def succeed(self, outcome: str = OUTCOME_OK) -> None:
        if self.data["final_outcome"] is None or self.data["final_outcome"] == OUTCOME_OK:
            self.data["final_outcome"] = outcome

    def save(self, path: Optional[str]) -> None:

        if path is None:
            return

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)

        print(f"\nField validation report saved to: {path}")

    def print_summary(self) -> None:

        print()
        print("=== Field Validation Summary ===")
        print(f"Camera ID:       {self.data['camera_id']}")
        print(f"Endpoint:        {self.data['endpoint_redacted']}")
        print(f"Software commit: {self.data['software_commit']}")
        print(f"Requested mode:  {self.data['requested_mode']}")
        print(f"Final outcome:   {self.data['final_outcome']}")
        if self.data["warnings"]:
            print(f"Warnings:        {len(self.data['warnings'])}")
        if self.data["failures"]:
            print(f"Failures:        {len(self.data['failures'])}")
        print()


def _current_commit() -> str:

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return "unknown"


def _percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


# =====================================================
# Stage 1 -- Connection (always run; every deeper mode depends on it)
# =====================================================


def run_connection_stage(args, report: FieldValidationReport):

    password = connection_diagnostic.resolve_password(args)

    diagnostic = connection_diagnostic.ConnectionDiagnostic(args.camera_id)

    backend = OpenCVFrameDecoderBackend(open_timeout_ms=args.open_timeout_ms)

    source = RTSPFrameSource(
        camera_id=args.camera_id, endpoint=args.endpoint, decoder_backend=backend,
        username=args.username, password=password, max_retries=args.max_retries,
        status_callback=diagnostic.on_status_changed,
    )

    print(f"[connection] connecting to {args.camera_id} @ {redact_endpoint(args.endpoint)!r} ...")
    source.start()

    outcome = classify_connection_outcome(source.status, source.last_error)

    report.data["connection"] = {
        "camera_id": args.camera_id,
        "endpoint_redacted": redact_endpoint(args.endpoint),
        "username": args.username,
        "final_status": source.status,
        "sanitized_last_error": source.last_error,
        "classified_outcome": outcome,
        "status_transitions": [
            {"status": status, "detail": detail} for _, status, detail in diagnostic.status_transitions
        ],
        "reconnect_attempts_observed": diagnostic.reconnect_events(),
    }

    if outcome != OUTCOME_OK:
        report.fail(outcome, f"Connection did not reach Online: {source.status} -- {source.last_error}")
        source.stop()
        return None, None

    print(f"[connection] OK -- status={source.status}")
    report.succeed()
    return source, backend


# =====================================================
# Stage 2 -- Frames
# =====================================================


def run_frames_stage(args, report: FieldValidationReport, source: RTSPFrameSource):

    frame_timestamps: List[float] = []
    first_frame = None
    first_frame_wall_time = None
    start_wall_time = time_module.monotonic()

    deadline = time_module.monotonic() + args.duration_seconds

    frames = []
    while time_module.monotonic() < deadline and len(frames) < args.max_frames:

        frame = source.read_frame()

        if frame is None:
            time_module.sleep(0.01)
            continue

        if first_frame is None:
            first_frame = frame
            first_frame_wall_time = time_module.monotonic() - start_wall_time

        frames.append(frame)
        frame_timestamps.append(time_module.monotonic())

    if not frames:
        report.data["frames"] = {"frames_received": 0}
        report.fail(OUTCOME_NO_FRAMES, "Connected, but no frames were received within the sampling window.")
        return []

    deltas = [b - a for a, b in zip(frame_timestamps, frame_timestamps[1:]) if b > a]
    mean_delta = statistics.mean(deltas) if deltas else None
    fps = (1.0 / mean_delta) if mean_delta else None

    report.data["frames"] = {
        "time_to_first_frame_seconds": round(first_frame_wall_time, 4),
        "frame_resolution": f"{first_frame.width}x{first_frame.height}" if first_frame.width and first_frame.height else None,
        "frame_codec": first_frame.codec,
        "frames_received": len(frames),
        "approximate_fps": round(fps, 2) if fps is not None else None,
    }

    print(
        f"[frames] {len(frames)} frame(s) received, "
        f"resolution={report.data['frames']['frame_resolution']}, fps={report.data['frames']['approximate_fps']}"
    )

    report.succeed()
    return frames


# =====================================================
# Stage 3 -- Detection
# =====================================================


def run_detect_stage(args, report: FieldValidationReport, frames):

    if not args.weights or not Path(args.weights).exists():
        report.fail(OUTCOME_YOLO_UNAVAILABLE, f"--weights not supplied or not found: {args.weights!r}")
        return None, []

    from human_detection.yolo_backend import UltralyticsYOLOBackend
    from human_detection.yolo_human_detector import YOLOHumanDetector

    backend = UltralyticsYOLOBackend(args.weights, device="cpu", confidence_threshold=args.confidence_threshold)
    detector = YOLOHumanDetector(backend)

    # One-time model load + first inference is excluded from every
    # per-frame statistic below (same "warm up, then measure" discipline
    # scripts/benchmark_yolo_human_detector.py/benchmark_real_decoder_
    # pipeline.py already establish) -- otherwise a single ~5s model-load
    # outlier would dominate the mean while leaving detections from that
    # warmup frame uncounted, an honest but confusing mismatch.
    warmup_detections = detector.detect(frames[0]) if frames else ()

    per_frame_detections = [warmup_detections]
    inference_ms = []
    confidences = list(d.confidence for d in warmup_detections)

    for frame in frames[1:]:
        start = time_module.perf_counter()
        detections = detector.detect(frame)
        inference_ms.append((time_module.perf_counter() - start) * 1000)
        per_frame_detections.append(detections)
        confidences.extend(d.confidence for d in detections)

    frames_with_people = sum(1 for d in per_frame_detections if len(d) > 0)
    total_detections = sum(len(d) for d in per_frame_detections)

    report.data["detection"] = {
        "frames_processed": len(frames),
        "frames_with_people": frames_with_people,
        "total_person_detections": total_detections,
        "average_people_per_frame": round(total_detections / len(frames), 3) if frames else None,
        "confidence_mean": round(statistics.mean(confidences), 4) if confidences else None,
        "confidence_min": round(min(confidences), 4) if confidences else None,
        "confidence_max": round(max(confidences), 4) if confidences else None,
        "inference_latency_mean_ms": round(statistics.mean(inference_ms), 3) if inference_ms else None,
        "inference_latency_median_ms": round(statistics.median(inference_ms), 3) if inference_ms else None,
        "inference_latency_p95_ms": round(_percentile(inference_ms, 0.95), 3) if inference_ms else None,
        "effective_processing_fps": round(1000.0 / statistics.mean(inference_ms), 2) if inference_ms and statistics.mean(inference_ms) > 0 else None,
    }

    print(
        f"[detect] {total_detections} detection(s) across {len(frames)} frame(s), "
        f"{frames_with_people} frame(s) with people"
    )

    report.succeed()
    return detector, per_frame_detections


# =====================================================
# Stage 4 -- Tracking
# =====================================================


def run_track_stage(args, report: FieldValidationReport, frames, per_frame_detections):

    tracker = SimpleSingleCameraTracker()

    all_track_ids = set()
    track_history_frames = {}  # track_id -> frames seen
    creations = 0
    expiries = 0

    preview_window_name = "run_physical_camera_validation -- tracking preview"
    preview_enabled = args.preview

    if preview_enabled:
        import cv2

    for frame, raw in zip(frames, per_frame_detections):

        tracked = tracker.update(args.camera_id, frame.timestamp, raw)

        from tracking.track_state import TrackState

        for t in tracked:

            if t.state == TrackState.EXPIRED:
                expiries += 1
                continue

            if t.track_id not in all_track_ids:
                all_track_ids.add(t.track_id)
                creations += 1
                track_history_frames[t.track_id] = 0

            if t.state in (TrackState.NEW, TrackState.TRACKED):
                track_history_frames[t.track_id] = track_history_frames.get(t.track_id, 0) + 1

        if preview_enabled and frame.payload_ref is not None:

            annotated = frame.payload_ref.copy() if hasattr(frame.payload_ref, "copy") else frame.payload_ref

            for detection, t in zip(raw, tracked[: len(raw)]):
                if detection.bounding_box is None:
                    continue
                x1, y1, x2, y2 = (int(v) for v in detection.bounding_box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = f"conf={detection.confidence:.2f} id={t.track_id}"
                cv2.putText(annotated, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

            cv2.imshow(preview_window_name, annotated)
            cv2.waitKey(1)

    if preview_enabled:
        import cv2
        cv2.destroyAllWindows()

    active_tracks = tracker._tracks.get(args.camera_id, {})

    report.data["tracking"] = {
        "unique_track_ids": len(all_track_ids),
        "currently_active_tracks": len(active_tracks),
        "longest_continuous_track_frames": max(track_history_frames.values()) if track_history_frames else 0,
        "track_creations": creations,
        "track_expiries": expiries,
    }

    print(
        f"[track] {len(all_track_ids)} unique track id(s), "
        f"longest continuous track: {report.data['tracking']['longest_continuous_track_frames']} frame(s)"
    )

    report.succeed()
    return tracker


# =====================================================
# Stage 5 -- Calibration / Projection
# =====================================================


def run_project_stage(args, report: FieldValidationReport, frames, per_frame_detections):

    if not args.calibration:

        report.data["calibration"] = None
        report.fail(
            OUTCOME_CALIBRATION_REQUIRED,
            "CALIBRATION REQUIRED -- no --calibration supplied. An illustrative calibration is "
            "never silently fabricated. Run scripts/calibrate_camera_scene.py against a real, "
            "physically measured scene first (see docs/architecture/"
            "camera_calibration_and_world_projection.md), then pass --calibration <output.json>.",
        )
        return None

    if not Path(args.calibration).exists():
        report.fail(OUTCOME_CALIBRATION_REQUIRED, f"--calibration file not found: {args.calibration!r}")
        return None

    try:
        profile = load_calibration_json(args.calibration)
    except CalibrationLoadError as exc:
        report.fail(OUTCOME_CALIBRATION_INVALID, f"Calibration file could not be loaded: {exc}")
        return None

    if profile.camera_id != args.camera_id:
        report.warn(
            f"Calibration camera_id {profile.camera_id!r} does not match --camera-id {args.camera_id!r} "
            f"-- proceeding, but double-check this is the right file."
        )

    quality = profile.quality
    report.data["calibration"] = {
        "camera_id": profile.camera_id,
        "floor_id": profile.floor_id,
        "quality": (
            {
                "reference_point_count": quality.reference_point_count,
                "validated_point_count": quality.validated_point_count,
                "mean_error_m": quality.mean_error_m,
                "median_error_m": quality.median_error_m,
                "max_error_m": quality.max_error_m,
                "rmse_m": quality.rmse_m,
                "validation_timestamp": quality.validation_timestamp,
            }
            if quality is not None else None
        ),
    }

    if quality is None:
        report.warn("Calibration has NEVER been validated against held-out measured points (CONFIGURED -- UNVALIDATED).")

    zones_by_floor = {}
    if args.project_file and Path(args.project_file).exists():

        from serialization.serializer import Serializer

        try:
            project = Serializer.load(args.project_file)
            for floor in project.building.ordered_floors():
                zones_by_floor[floor.id] = list(floor.zones)
        except Exception as exc:
            report.warn(f"Could not load --project-file for zone lookup: {exc}")

    else:
        report.warn("No --project-file supplied -- zone_id will be None for every projection (no zone geometry to check against).")

    world_projector = WorldProjector(calibrations={args.camera_id: profile}, zones_by_floor=zones_by_floor)

    projections = []
    for frame, raw in zip(frames, per_frame_detections):
        for detection in raw:
            projection = world_projector.project(args.camera_id, detection.bounding_box, detection.confidence)
            projections.append(projection)

    projected_count = sum(1 for p in projections if p.world_position is not None)

    report.data["projection"] = {
        "detections_considered": len(projections),
        "detections_with_world_position": projected_count,
        "sample_projections": [
            {
                "world_position": p.world_position,
                "floor_id": p.floor_id,
                "zone_id": p.zone_id,
                "projection_confidence": p.projection_confidence,
                "provenance": p.provenance,
            }
            for p in projections[:10]
        ],
    }

    print(
        f"[project] {projected_count}/{len(projections)} detection(s) projected to a real world position "
        f"(provenance={projections[0].provenance if projections else 'n/a'})"
    )

    if len(projections) > 0 and projected_count == 0:
        report.fail(OUTCOME_PROJECTION_UNAVAILABLE, "Calibration loaded, but zero detections could be projected onto the floor plane.")
        return None

    report.succeed()
    return world_projector


# =====================================================
# Stage 6 -- Full runtime
# =====================================================


def run_full_runtime_stage(args, report: FieldValidationReport, world_projector):

    building = Building(name="Field Validation Building")
    floor = building.create_floor(name="Field Validation Floor")
    camera = Camera(id=args.camera_id, name=args.camera_id, floor_id=floor.id)
    floor.add_camera(camera)

    if args.project_file and Path(args.project_file).exists():

        from serialization.serializer import Serializer

        try:
            project = Serializer.load(args.project_file)
            building = project.building
        except Exception as exc:
            report.warn(f"Could not load --project-file for --full-runtime; using a minimal one-camera building instead: {exc}")

    backend = OpenCVFrameDecoderBackend(open_timeout_ms=args.open_timeout_ms)
    password = connection_diagnostic.resolve_password(args)

    source = RTSPFrameSource(
        camera_id=args.camera_id, endpoint=args.endpoint, decoder_backend=backend,
        username=args.username, password=password, max_retries=args.max_retries,
    )

    from human_detection.yolo_backend import UltralyticsYOLOBackend
    from human_detection.yolo_human_detector import YOLOHumanDetector

    detector = YOLOHumanDetector(UltralyticsYOLOBackend(args.weights, device="cpu", confidence_threshold=args.confidence_threshold))
    tracker = SimpleSingleCameraTracker()

    # Phase 10 -- ONE camera only. SimulationIdentityResolver treats the
    # tracker's own stable local track id as already-global, the correct,
    # honest strategy for a single camera with no cross-camera evidence
    # to fuse yet (Phase 11 -- RuleBasedCrossCameraIdentityResolver is
    # deliberately NOT wired here; see this milestone's own doc note).
    identity_resolver = SimulationIdentityResolver()

    runtime = build_live_runtime(
        building,
        frame_sources={args.camera_id: source},
        human_detector=detector,
        identity_resolver=identity_resolver,
        tracker=tracker,
        world_projector=world_projector,
        interval_seconds=1.0,
        # Deliberately NOT supplying voice_output_provider or
        # building_control_provider -- Phase 9's own explicit requirement:
        # no automatic voice broadcast, no automatic building control
        # execution, ever, during a field validation run.
    )

    print("[full-runtime] starting LiveRuntime ...")
    runtime.start()

    try:

        cycle_count = min(args.max_frames, 60)
        for index in range(cycle_count):
            runtime.run_cycle(float(index))

    finally:
        runtime.stop()

    building_state = runtime.orchestrator.latest_building_state
    crowd = runtime.orchestrator.latest_crowd_intelligence
    trajectory = runtime.orchestrator.latest_trajectory_intelligence
    evac_progress = runtime.orchestrator.latest_evacuation_progress
    evac_recommendation = runtime.orchestrator.latest_evacuation_recommendation
    evac_guidance = runtime.orchestrator.latest_evacuation_guidance
    advisory = runtime.orchestrator.latest_advisory_report

    occupant_count = len(building_state.occupant_tracks) if building_state is not None else 0

    report.data["runtime"] = {
        "cycles_run": cycle_count,
        "building_state_reached": building_state is not None,
        "occupant_track_count": occupant_count,
        "crowd_intelligence_reached": crowd is not None,
        "trajectory_intelligence_reached": trajectory is not None,
        "evacuation_progress_reached": evac_progress is not None,
        "evacuation_recommendation_reached": evac_recommendation is not None,
        "evacuation_guidance_reached": evac_guidance is not None,
        "advisory_reached": advisory is not None,
        "voice_broadcast_attempted": False,
        "building_control_executed": False,
    }

    print(
        f"[full-runtime] BuildingState reached: {building_state is not None}, "
        f"occupant tracks: {occupant_count}"
    )

    if building_state is None:
        report.fail(OUTCOME_RUNTIME_DEGRADED, "LiveRuntime ran, but BuildingState was never produced.")
        return

    report.succeed()


# =====================================================


MODE_ORDER = ["connection-only", "frames", "detect", "track", "project", "full-runtime"]


def _resolve_mode(args) -> str:

    for mode in reversed(MODE_ORDER):
        flag = "full_runtime" if mode == "full-runtime" else mode.replace("-", "_")
        if getattr(args, flag):
            return mode

    return "connection-only"


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--endpoint", required=True, help="RTSP URL, or a local video file for an offline rehearsal.")
    parser.add_argument("--username", default=None)
    parser.add_argument("--credential-ref", default=None)
    parser.add_argument("--credential-store-path", default=None)
    parser.add_argument("--prompt-password", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--open-timeout-ms", type=int, default=3000)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--weights", default=None, help="Path to real YOLO weights (required for --detect and deeper).")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--calibration", default=None, help="Path to a real, fitted CalibrationProfile JSON (required for --project and deeper).")
    parser.add_argument("--project-file", default=None, help="Optional .syn project file, for real zone geometry during --project/--full-runtime.")
    parser.add_argument("--preview", action="store_true", help="Show an annotated OpenCV preview window during --track.")
    parser.add_argument("--report-out", default=None, help="Path to save the JSON field validation report.")

    parser.add_argument("--connection-only", action="store_true")
    parser.add_argument("--frames", action="store_true")
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--full-runtime", action="store_true")

    args = parser.parse_args()

    mode = _resolve_mode(args)
    mode_index = MODE_ORDER.index(mode)

    report = FieldValidationReport(args.camera_id, args.endpoint, mode)

    source, backend = run_connection_stage(args, report)

    if source is None:
        report.print_summary()
        report.save(args.report_out)
        sys.exit(1)

    if mode_index == 0:
        source.stop()
        report.print_summary()
        report.save(args.report_out)
        sys.exit(0 if report.data["final_outcome"] == OUTCOME_OK else 1)

    frames = run_frames_stage(args, report, source)

    if not frames:
        source.stop()
        report.print_summary()
        report.save(args.report_out)
        sys.exit(1)

    if mode_index == 1:
        source.stop()
        report.print_summary()
        report.save(args.report_out)
        sys.exit(0 if report.data["final_outcome"] == OUTCOME_OK else 1)

    detector, per_frame_detections = run_detect_stage(args, report, frames)

    if detector is None:
        source.stop()
        report.print_summary()
        report.save(args.report_out)
        sys.exit(1)

    if mode_index == 2:
        source.stop()
        report.print_summary()
        report.save(args.report_out)
        sys.exit(0 if report.data["final_outcome"] == OUTCOME_OK else 1)

    run_track_stage(args, report, frames, per_frame_detections)

    if mode_index == 3:
        source.stop()
        report.print_summary()
        report.save(args.report_out)
        sys.exit(0 if report.data["final_outcome"] == OUTCOME_OK else 1)

    world_projector = run_project_stage(args, report, frames, per_frame_detections)

    if world_projector is None:
        source.stop()
        report.print_summary()
        report.save(args.report_out)
        sys.exit(1)

    if mode_index == 4:
        source.stop()
        report.print_summary()
        report.save(args.report_out)
        sys.exit(0 if report.data["final_outcome"] == OUTCOME_OK else 1)

    # --full-runtime constructs its OWN RTSPFrameSource/backend (the real
    # LiveRuntime needs to own its own frame source lifecycle) -- stop the
    # one used for the earlier stages first, cleanly.
    source.stop()

    run_full_runtime_stage(args, report, world_projector)

    report.print_summary()
    report.save(args.report_out)
    sys.exit(0 if report.data["final_outcome"] == OUTCOME_OK else 1)


if __name__ == "__main__":
    main()
