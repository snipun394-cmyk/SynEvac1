"""Real Human Detection Pipeline milestone, Phase 8 -- offline local
video demo.

Proves the full, real chain end-to-end with zero network access and
zero CCTV access:

    local video file
    -> human_detection.video_source.load_video_frames (real cv2 decode)
    -> live_camera_pipeline.replay_frame_source.ReplayFrameSource (real,
       production CameraFrameSource -- not a test double)
    -> CameraFrame
    -> human_detection.yolo_human_detector.YOLOHumanDetector (real,
       production class)
    -> live_camera_pipeline.identity_resolver.MappingIdentityResolver
       (real, production class)
    -> Detection

No local video/CCTV footage exists in this repository, so this script
generates a small synthetic local .mp4 (random-noise frames written by
cv2.VideoWriter, decoded back by cv2.VideoCapture -- both real,
genuine local disk I/O, never fabricated) purely so there is a real
video FILE for load_video_frames to decode. This is honestly reported
below: the video's *content* is synthetic noise, not real footage of
real people.

No YOLO weights are supplied by default, so real ultralytics inference
is NOT run by default -- an injected FakeYOLOBackend (the same
deterministic double tests/test_yolo_human_detector.py uses) stands in
for it, and this script prints "REAL MODEL BENCHMARK: NOT RUN" rather
than fabricating a number. Pass --weights <path/to/model.pt> to also
run genuine ultralytics inference against a real local weights file
(never downloaded -- must already exist on disk).

Not a pytest test: run manually --
    python scripts/demo_yolo_human_detection.py
    python scripts/demo_yolo_human_detection.py --weights C:/models/yolov8n.pt
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

from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from human_detection.video_source import load_video_frames
from human_detection.yolo_backend import UltralyticsYOLOBackend
from human_detection.yolo_human_detector import YOLOHumanDetector

from tests.human_detection_fixtures import FakeYOLOBackend, person


FRAME_COUNT = 30
FRAME_SIZE = (320, 240)  # width, height
CAMERA_ID = "CAM-DEMO"


def _generate_synthetic_video(path: Path) -> None:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, FRAME_SIZE)

    rng = np.random.default_rng(seed=0)

    for _ in range(FRAME_COUNT):
        frame = rng.integers(0, 255, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()


def _build_fake_backend() -> FakeYOLOBackend:

    # Deterministic stand-in for real inference: every third frame "sees"
    # one person, so the demo exercises both the zero-people and
    # one-person paths over a realistic frame sequence, without needing
    # a real model to interpret the synthetic noise frames.
    backend = FakeYOLOBackend()

    for index in range(FRAME_COUNT):

        if index % 3 == 0:
            backend.queue_result(person(confidence=0.75 + (index % 5) * 0.01))
        else:
            backend.queue_result()

    return backend


def run_demo(weights_path: str = None) -> None:

    with tempfile.TemporaryDirectory() as tmp_dir:

        video_path = Path(tmp_dir) / "synthetic_demo.mp4"
        _generate_synthetic_video(video_path)

        frames = load_video_frames(video_path)

        source = ReplayFrameSource(camera_id=CAMERA_ID, frames=frames)
        source.start()

        real_model_ran = weights_path is not None

        if real_model_ran:
            backend = UltralyticsYOLOBackend(weights_path)
        else:
            backend = _build_fake_backend()

        detector = YOLOHumanDetector(backend)
        resolver = MappingIdentityResolver()

        frame_prep_latencies_ms = []
        detect_latencies_ms = []
        person_detection_count = 0
        frames_processed = 0

        overall_start = time.perf_counter()

        while True:

            prep_start = time.perf_counter()
            frame = source.read_frame()
            frame_prep_latencies_ms.append((time.perf_counter() - prep_start) * 1000)

            if frame is None:
                break

            frames_processed += 1

            detect_start = time.perf_counter()
            raw_detections = detector.detect(frame)
            detect_latencies_ms.append((time.perf_counter() - detect_start) * 1000)

            person_detection_count += len(raw_detections)

            resolver.resolve(raw_detections, time=frame.timestamp)

        overall_elapsed_seconds = time.perf_counter() - overall_start
        source.stop()

    fps = frames_processed / overall_elapsed_seconds if overall_elapsed_seconds > 0 else float("inf")

    print("=== Real Human Detection Pipeline -- Offline Local Video Demo ===")
    print()
    print(f"Video content: SYNTHETIC (random noise, generated locally) -- "
          f"no real CCTV/recorded-person footage was available in this repository.")
    print(f"Frames processed: {frames_processed}")
    print(f"Person detections (raw, pre-identity-resolution): {person_detection_count}")
    print(f"Frame read/prep latency: mean {statistics.mean(frame_prep_latencies_ms):.4f} ms, "
          f"max {max(frame_prep_latencies_ms):.4f} ms")
    print(f"Detector adapter latency: mean {statistics.mean(detect_latencies_ms):.4f} ms, "
          f"max {max(detect_latencies_ms):.4f} ms")
    print(f"Total wall time: {overall_elapsed_seconds * 1000:.2f} ms")
    print(f"Approximate FPS (this demo, this machine): {fps:.1f}")
    print()

    if real_model_ran:
        print(f"REAL MODEL BENCHMARK: RAN -- weights={weights_path!r} "
              f"(detector adapter latency above includes real ultralytics inference)")
    else:
        print("REAL MODEL BENCHMARK: NOT RUN -- no local YOLO weights supplied "
              "(pass --weights <path/to/model.pt> to run genuine ultralytics "
              "inference; a number is never fabricated in its absence).")

    print()
    print("Network access performed: NO")
    print("Physical CCTV accessed: NO")


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default=None,
        help="Path to an existing local .pt YOLO weights file. If omitted, "
             "a deterministic FakeYOLOBackend is used instead and the "
             "real-model benchmark is reported as NOT RUN.",
    )
    args = parser.parse_args()

    run_demo(weights_path=args.weights)


if __name__ == "__main__":
    main()
