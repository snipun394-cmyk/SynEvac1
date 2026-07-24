"""CCTV Connection & Calibration Readiness milestone, Phase 13 --
performance report for the REAL decoder + REAL YOLO + REAL tracker
pipeline, using validation_media/vtest.avi and weights/yolov8n.pt (the
same real artifacts tests/test_real_decoder_full_chain_e2e.py proves
correctness with). Every number below comes from genuinely executed
code on THIS machine's CPU -- never extrapolated to real network/RTSP
conditions or to different hardware (physical CCTV decode/network
latency remains unmeasured and unmeasurable until physical access
exists -- see scripts/benchmark_rtsp_frame_source.py's own identical
disclosure for the transport seam this reuses unmodified).

Five stages measured, deliberately kept separate (one-time YOLO model
load is its own number, excluded from every per-frame average):

1. Decode latency       -- OpenCVFrameDecoderBackend.read() against a
                            real, already-open video file.
2. YOLO inference latency -- YOLOHumanDetector.detect() with a real
                              UltralyticsYOLOBackend (requires --weights).
3. Tracking latency      -- SimpleSingleCameraTracker.update().
4. Projection latency    -- camera_calibration.projection.WorldProjector.
                             project() (only if --calibration is supplied
                             -- otherwise reported NOT RUN, never faked).
5. Complete cycle latency -- LiveCameraPipeline.run_cycle() end to end,
                              and the resulting effective FPS.

Not a pytest test -- run manually:
    python scripts/benchmark_real_decoder_pipeline.py --weights weights/yolov8n.pt
    python scripts/benchmark_real_decoder_pipeline.py --weights weights/yolov8n.pt --calibration calibration.json
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.camera import Camera
from models.engineering_asset import DeviceMode

from camera_manager.manager import CameraManager

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline
from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

from human_detection.opencv_decoder_backend import OpenCVFrameDecoderBackend

from tracking.simple_tracker import SimpleSingleCameraTracker

from live_occupants.manager import LiveOccupantManager

from camera_calibration.calibration_loader import load_calibration_json
from camera_calibration.projection import WorldProjector


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = REPO_ROOT / "validation_media" / "vtest.avi"
CAMERA_ID = "CAM-BENCH-REAL-DECODER"
CYCLE_COUNT = 60


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _report(name, samples_ms):

    print(
        f"{name}: {len(samples_ms)} samples, mean {statistics.mean(samples_ms):.3f} ms, "
        f"p95 {_percentile(samples_ms, 0.95):.3f} ms, max {max(samples_ms):.3f} ms"
    )


def run(video_path: Path, weights_path: str, calibration_path: str) -> None:

    # ---- 1. Decode latency ----------------------------------------------
    backend = OpenCVFrameDecoderBackend()
    backend.open(str(video_path), None, None)

    decode_ms = []
    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        frame = backend.read()
        decode_ms.append((time.perf_counter() - start) * 1000)
        if frame is None:
            break

    backend.close()
    _report("1. Decode latency (real OpenCVFrameDecoderBackend.read())", decode_ms)
    print()

    # ---- Build the real pipeline for stages 2-5 --------------------------
    camera = Camera(id=CAMERA_ID, name="Benchmark Camera", floor_id="floor-1")
    camera_manager = CameraManager()
    camera_manager.register_camera(camera)

    source_backend = OpenCVFrameDecoderBackend()
    source = RTSPFrameSource(camera_id=CAMERA_ID, endpoint=str(video_path), decoder_backend=source_backend)
    source.start()

    yolo_ran = False
    if weights_path and Path(weights_path).exists():

        from human_detection.yolo_backend import UltralyticsYOLOBackend
        from human_detection.yolo_human_detector import YOLOHumanDetector

        yolo_backend_real = UltralyticsYOLOBackend(weights_path, device="cpu")
        detector = YOLOHumanDetector(yolo_backend_real)

        # One-time model load/warmup, excluded from every per-frame number
        # below -- the same "warm up, then measure" discipline scripts/
        # benchmark_yolo_human_detector.py already establishes.
        warmup_frame = source.read_frame()
        if warmup_frame is not None:
            warmup_start = time.perf_counter()
            detector.detect(warmup_frame)
            warmup_ms = (time.perf_counter() - warmup_start) * 1000
            print(f"One-time YOLO model load + first inference (excluded from all averages below): {warmup_ms:.1f} ms")
            print()

        yolo_ran = True

    else:

        from tests.human_detection_fixtures import FakeYOLOBackend, person
        from human_detection.yolo_human_detector import YOLOHumanDetector

        fake_backend = FakeYOLOBackend()
        for _ in range(CYCLE_COUNT):
            fake_backend.set_default_result(person(confidence=0.9, box=(10.0, 10.0, 40.0, 90.0)))
        detector = YOLOHumanDetector(fake_backend)

    tracker = SimpleSingleCameraTracker()

    world_projector = None
    if calibration_path and Path(calibration_path).exists():
        profile = load_calibration_json(calibration_path)
        world_projector = WorldProjector(calibrations={CAMERA_ID: profile}, zones_by_floor={})

    identity_resolver = SimulationIdentityResolver()
    detection_provider = LiveCameraPipelineDetectionProvider()
    occupant_manager = LiveOccupantManager()

    pipeline = LiveCameraPipeline(
        frame_sources={CAMERA_ID: source},
        human_detector=detector,
        identity_resolver=identity_resolver,
        detection_provider=detection_provider,
        tracker=tracker,
        world_projector=world_projector,
        live_occupant_manager=occupant_manager,
    )

    camera_manager.register_detection_provider(DeviceMode.LIVE, detection_provider)
    camera_manager.set_camera_mode(CAMERA_ID, DeviceMode.LIVE)

    # ---- 2. Isolated YOLO inference latency ------------------------------
    if yolo_ran:

        inference_ms = []
        for _ in range(min(CYCLE_COUNT, 30)):
            frame = source.read_frame()
            if frame is None:
                break
            start = time.perf_counter()
            detector.detect(frame)
            inference_ms.append((time.perf_counter() - start) * 1000)

        if inference_ms:
            _report("2. YOLO inference latency (real UltralyticsYOLOBackend, CPU)", inference_ms)
        print()

    else:
        print("2. YOLO inference latency: NOT RUN -- no --weights supplied. Never fabricated.")
        print()

    # ---- 3. Isolated tracking latency -------------------------------------
    tracking_ms = []
    for cycle in range(min(CYCLE_COUNT, 30)):
        frame = source.read_frame()
        if frame is None:
            break
        raw = detector.detect(frame)
        start = time.perf_counter()
        tracker.update(CAMERA_ID, frame.timestamp, raw)
        tracking_ms.append((time.perf_counter() - start) * 1000)

    if tracking_ms:
        _report("3. Tracking latency (real SimpleSingleCameraTracker.update())", tracking_ms)
    print()

    # ---- 4. Isolated projection latency ------------------------------------
    if world_projector is not None:

        projection_ms = []
        box = (10.0, 10.0, 40.0, 90.0)
        for _ in range(CYCLE_COUNT):
            start = time.perf_counter()
            world_projector.project(CAMERA_ID, box, 0.9)
            projection_ms.append((time.perf_counter() - start) * 1000)

        _report("4. World projection latency (real WorldProjector.project())", projection_ms)
        print()

    else:
        print("4. World projection latency: NOT RUN -- no --calibration supplied. Never fabricated.")
        print()

    source.stop()

    # ---- 5. Complete cycle latency (fresh source, full pipeline) -----------
    fresh_backend = OpenCVFrameDecoderBackend()
    fresh_source = RTSPFrameSource(camera_id=CAMERA_ID, endpoint=str(video_path), decoder_backend=fresh_backend)
    fresh_source.start()

    pipeline.frame_sources[CAMERA_ID] = fresh_source

    cycle_ms = []
    for index in range(CYCLE_COUNT):
        start = time.perf_counter()
        pipeline.run_cycle(float(index))
        cycle_ms.append((time.perf_counter() - start) * 1000)

    fresh_source.stop()

    _report("5. Complete perception cycle latency (LiveCameraPipeline.run_cycle(), full chain)", cycle_ms)

    mean_cycle_ms = statistics.mean(cycle_ms)
    effective_fps = 1000.0 / mean_cycle_ms if mean_cycle_ms > 0 else float("inf")

    print()
    print(f"Effective FPS (this machine, this benchmark only, CPU): {effective_fps:.2f}")
    print()
    print(
        "NOTE: this measures real decode + (real or fake, per --weights) detection + real "
        "tracking + (real or skipped, per --calibration) projection on a LOCAL video file. "
        "Real network/RTSP transport latency and real NVR/camera hardware behavior remain "
        "UNMEASURED and must not be extrapolated from these numbers -- see docs/architecture/"
        "physical_cctv_access_checklist.md."
    )


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Local video file to benchmark against.")
    parser.add_argument("--weights", default=None, help="Path to real YOLO weights (omit to skip real inference timing).")
    parser.add_argument("--calibration", default=None, help="Path to a calibration JSON (omit to skip projection timing).")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video file not found: {video_path}")

    run(video_path, args.weights, args.calibration)


if __name__ == "__main__":
    main()
