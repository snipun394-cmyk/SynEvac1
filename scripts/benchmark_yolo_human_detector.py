"""Real Human Detection Pipeline milestone, Phase 10 performance
readiness benchmark.

Benchmarks YOLOHumanDetector's OWN overhead -- separate from real model
inference latency, which requires real local YOLO weights to measure
honestly (none are bundled with this repository; see --weights below).

Real YOLO Person Detection Validation & Productionization milestone,
Phase 12 -- extended with two further separated measurements: tracking
overhead (SimpleSingleCameraTracker.update()) and downstream pipeline
overhead (LiveCameraPipeline.run_cycle() with tracker + identity
resolution wired in), both against FakeYOLOBackend (zero real model
time). If real weights are unavailable, this script reports "REAL YOLO
INFERENCE: NOT RUN" rather than reporting FakeYOLOBackend timing as if
it were real YOLO inference performance.

Five things are measured, deliberately kept separate:

1. Frame preparation latency -- decoding a local (synthetic) video file
   via human_detection.video_source.load_video_frames, real cv2 decode
   work, just on generated-noise content rather than real footage.
2. Detector adapter overhead -- YOLOHumanDetector.detect()'s own
   per-frame work (class filtering, RawHumanDetection construction)
   using an injected FakeYOLOBackend, so this number contains ZERO real
   model inference time.
3. Real model inference latency -- ONLY measured if --weights points at
   an existing local .pt file. Otherwise printed as NOT RUN, never
   fabricated (this milestone's own explicit requirement).

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_yolo_human_detector.py`) and read
the printed report. Physical RTSP decode/network latency remains
unknown until real CCTV access exists -- never claimed here.
"""

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrame
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline

from human_detection.video_source import load_video_frames
from human_detection.yolo_backend import UltralyticsYOLOBackend
from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from tests.human_detection_fixtures import FakeYOLOBackend, person


FRAME_COUNT = 500
PEOPLE_PER_FRAME = 8
VIDEO_FRAME_COUNT = 60
VIDEO_SIZE = (320, 240)


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def benchmark_frame_preparation() -> dict:

    with tempfile.TemporaryDirectory() as tmp_dir:

        video_path = Path(tmp_dir) / "bench.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, VIDEO_SIZE)

        rng = np.random.default_rng(seed=0)
        for _ in range(VIDEO_FRAME_COUNT):
            writer.write(rng.integers(0, 255, (VIDEO_SIZE[1], VIDEO_SIZE[0], 3), dtype=np.uint8))
        writer.release()

        start = time.perf_counter()
        frames = load_video_frames(video_path)
        elapsed_seconds = time.perf_counter() - start

    return {
        "frame_count": len(frames),
        "total_ms": elapsed_seconds * 1000,
        "per_frame_ms": (elapsed_seconds * 1000) / max(len(frames), 1),
    }


def benchmark_detector_adapter_overhead() -> dict:

    backend = FakeYOLOBackend()
    for _ in range(FRAME_COUNT):
        backend.set_default_result(*(person(confidence=0.9) for _ in range(PEOPLE_PER_FRAME)))

    detector = YOLOHumanDetector(backend)

    frame = CameraFrame(
        camera_id="CAM-BENCH", timestamp=0.0, frame_sequence=0, payload_ref="fake-image",
    )

    per_call_ms = []

    for _ in range(FRAME_COUNT):
        start = time.perf_counter()
        detector.detect(frame)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {
        "frame_count": FRAME_COUNT,
        "people_per_frame": PEOPLE_PER_FRAME,
        "mean_ms": statistics.mean(per_call_ms),
        "p95_ms": _percentile(per_call_ms, 0.95),
        "max_ms": max(per_call_ms),
    }


def benchmark_tracking_overhead() -> dict:

    # Real YOLO Person Detection Validation & Productionization
    # milestone, Phase 12 -- tracking time kept SEPARATE from detector
    # adapter overhead above: SimpleSingleCameraTracker.update()'s own
    # per-cycle cost (IoU/centroid matching + track bookkeeping),
    # against a fixed set of already-produced RawHumanDetection objects
    # (zero YOLO/model time whatsoever).

    from tests.human_detection_fixtures import person as _person

    backend = FakeYOLOBackend()
    for _ in range(FRAME_COUNT):
        backend.set_default_result(*(_person(confidence=0.9, box=(i * 5.0, 0.0, i * 5.0 + 10.0, 10.0)) for i in range(PEOPLE_PER_FRAME)))

    detector = YOLOHumanDetector(backend)
    tracker = SimpleSingleCameraTracker()

    frame = CameraFrame(camera_id="CAM-BENCH", timestamp=0.0, frame_sequence=0, payload_ref="fake-image")

    per_call_ms = []

    for cycle in range(FRAME_COUNT):
        raw = detector.detect(frame)
        start = time.perf_counter()
        tracker.update("CAM-BENCH", float(cycle), raw)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {
        "frame_count": FRAME_COUNT,
        "people_per_frame": PEOPLE_PER_FRAME,
        "mean_ms": statistics.mean(per_call_ms),
        "p95_ms": _percentile(per_call_ms, 0.95),
        "max_ms": max(per_call_ms),
    }


def benchmark_downstream_pipeline_overhead() -> dict:

    # Downstream pipeline time: LiveCameraPipeline.run_cycle()'s own
    # per-cycle cost with tracker + identity resolution wired in (zero
    # real YOLO time -- FakeYOLOBackend stands in, same as every other
    # adapter-overhead measurement in this script).

    class _RepeatingSource:
        def read_frame(self):
            return CameraFrame(camera_id="CAM-BENCH", timestamp=0.0, frame_sequence=0, payload_ref="fake-image")

    backend = FakeYOLOBackend()
    for _ in range(FRAME_COUNT):
        backend.set_default_result(*(person(confidence=0.9) for _ in range(PEOPLE_PER_FRAME)))

    pipeline = LiveCameraPipeline(
        frame_sources={"CAM-BENCH": _RepeatingSource()},
        human_detector=YOLOHumanDetector(backend),
        identity_resolver=SimulationIdentityResolver(),
        detection_provider=LiveCameraPipelineDetectionProvider(),
        tracker=SimpleSingleCameraTracker(),
    )

    per_call_ms = []

    for cycle in range(FRAME_COUNT):
        start = time.perf_counter()
        pipeline.run_cycle(float(cycle))
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {
        "frame_count": FRAME_COUNT,
        "people_per_frame": PEOPLE_PER_FRAME,
        "mean_ms": statistics.mean(per_call_ms),
        "p95_ms": _percentile(per_call_ms, 0.95),
        "max_ms": max(per_call_ms),
    }


def benchmark_real_model_inference(weights_path: str) -> dict:

    backend = UltralyticsYOLOBackend(weights_path)
    detector = YOLOHumanDetector(backend)

    rng = np.random.default_rng(seed=0)
    image = rng.integers(0, 255, (VIDEO_SIZE[1], VIDEO_SIZE[0], 3), dtype=np.uint8)
    frame = CameraFrame(camera_id="CAM-BENCH", timestamp=0.0, frame_sequence=0, payload_ref=image)

    # First call pays one-time model load cost -- excluded from the
    # per-frame inference numbers below, same "warm up, then measure"
    # discipline any honest model benchmark needs.
    detector.detect(frame)

    per_call_ms = []
    inference_frame_count = 30

    for _ in range(inference_frame_count):
        start = time.perf_counter()
        detector.detect(frame)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {
        "frame_count": inference_frame_count,
        "mean_ms": statistics.mean(per_call_ms),
        "p95_ms": _percentile(per_call_ms, 0.95),
        "max_ms": max(per_call_ms),
    }


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default=None,
        help="Path to an existing local .pt YOLO weights file, to also "
             "measure real ultralytics inference latency. Omit to skip "
             "(reported as NOT RUN, never fabricated).",
    )
    args = parser.parse_args()

    prep = benchmark_frame_preparation()
    print(
        f"Frame preparation (real cv2 decode, synthetic video content): "
        f"{prep['frame_count']} frames, {prep['total_ms']:.3f} ms total, "
        f"{prep['per_frame_ms']:.4f} ms/frame"
    )

    adapter = benchmark_detector_adapter_overhead()
    print(
        f"Detector adapter overhead (FakeYOLOBackend, {adapter['people_per_frame']} "
        f"people/frame, zero real model time): {adapter['frame_count']} frames, "
        f"mean {adapter['mean_ms']:.4f} ms, p95 {adapter['p95_ms']:.4f} ms, "
        f"max {adapter['max_ms']:.4f} ms"
    )

    tracking = benchmark_tracking_overhead()
    print(
        f"Tracking overhead (SimpleSingleCameraTracker.update(), {tracking['people_per_frame']} "
        f"people/frame, zero real model time): {tracking['frame_count']} frames, "
        f"mean {tracking['mean_ms']:.4f} ms, p95 {tracking['p95_ms']:.4f} ms, "
        f"max {tracking['max_ms']:.4f} ms"
    )

    downstream = benchmark_downstream_pipeline_overhead()
    print(
        f"Downstream pipeline overhead (LiveCameraPipeline.run_cycle() with tracker + "
        f"identity resolution, {downstream['people_per_frame']} people/frame, zero real "
        f"model time): {downstream['frame_count']} frames, mean {downstream['mean_ms']:.4f} ms, "
        f"p95 {downstream['p95_ms']:.4f} ms, max {downstream['max_ms']:.4f} ms"
    )

    total_per_frame_ms = prep["per_frame_ms"] + adapter["mean_ms"] + tracking["mean_ms"]
    estimated_fps = 1000.0 / total_per_frame_ms if total_per_frame_ms > 0 else float("inf")

    print()

    if args.weights:
        inference = benchmark_real_model_inference(args.weights)
        print(
            f"Real model inference (ultralytics, weights={args.weights!r}): "
            f"{inference['frame_count']} frames, mean {inference['mean_ms']:.3f} ms, "
            f"p95 {inference['p95_ms']:.3f} ms, max {inference['max_ms']:.3f} ms"
        )
        total_per_frame_ms += inference["mean_ms"]
        estimated_fps = 1000.0 / total_per_frame_ms if total_per_frame_ms > 0 else float("inf")
    else:
        print(
            "Real model inference latency: NOT RUN -- no local YOLO weights "
            "supplied via --weights. Never fabricated in its absence."
        )

    print()
    print(f"Total per-frame latency (frame prep + adapter overhead + tracking"
          f"{' + real inference' if args.weights else ' -- excludes real inference, not run'}): "
          f"{total_per_frame_ms:.4f} ms")
    print(f"Estimated FPS (this machine, this benchmark only): {estimated_fps:.1f}")
    print()
    print(
        "NOTE: physical RTSP decode latency and physical network latency are "
        "NOT measured here -- no real CCTV stream exists yet to measure them "
        "honestly (see scripts/benchmark_rtsp_frame_source.py for the same "
        "disclosure on the RTSP transport seam)."
    )


if __name__ == "__main__":
    main()
