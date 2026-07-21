"""Human Behavior Recognition Framework milestone, Phase 10 -- offline
local-video demo.

Proves the full, real chain end-to-end with zero network access and
zero CCTV access:

    local video file
    -> human_detection.video_source.load_video_frames (real cv2 decode)
    -> live_camera_pipeline.replay_frame_source.ReplayFrameSource
    -> CameraFrame
    -> human_detection.yolo_human_detector.YOLOHumanDetector
    -> RawHumanDetection
    -> tracking.simple_tracker.SimpleSingleCameraTracker
    -> TrackedHuman
    -> behavior_recognition.rule_based_recognizer.RuleBasedBehaviorRecognizer  (NEW)
    -> BehaviorObservation -- printed below (track ID, velocity, behavior, confidence)

Same synthetic-content local video generation approach as
scripts/demo_yolo_human_detection.py and
scripts/demo_single_camera_tracking.py -- a scripted FakeYOLOBackend
(not real ultralytics inference) reports a person accelerating from a
stationary start, into a walk, into a run.

Not a pytest test: run manually --
    python scripts/demo_behavior_recognition.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from human_detection.video_source import load_video_frames
from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from tests.human_detection_fixtures import FakeYOLOBackend, person


FRAME_COUNT = 10
FRAME_SIZE = (320, 240)
CAMERA_ID = "CAM-DEMO"


def _generate_synthetic_video(path: Path) -> None:

    # Written at 1 fps deliberately -- load_video_frames() derives each
    # frame's timestamp from the container's own reported fps
    # (index / fps), and the scripted backend's per-cycle pixel
    # increments below are tuned for a 1-SECOND cadence between frames
    # (matching the velocity thresholds documented in
    # behavior_recognition/rule_based_recognizer.py). A higher fps here
    # would compress those increments into a smaller time delta and
    # inflate every velocity reading proportionally.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 1.0, FRAME_SIZE)

    rng = np.random.default_rng(seed=0)

    for _ in range(FRAME_COUNT):
        frame = rng.integers(0, 255, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()


def _build_scripted_backend() -> FakeYOLOBackend:

    # One person, one camera: stationary for frames 0-2, walking
    # (20 px/s) for frames 3-6, running (150 px/s) for frames 7-9 --
    # deterministic, so the demo shows behavior recognition transition
    # cleanly through all three recognized states.
    backend = FakeYOLOBackend()

    x = 0.0

    for frame_index in range(FRAME_COUNT):

        backend.queue_result(person(confidence=0.9, box=(x, 0.0, x + 20.0, 60.0)))

        if frame_index < 2:
            x += 0.0
        elif frame_index < 6:
            x += 20.0
        else:
            x += 150.0

    return backend


def run_demo() -> None:

    with tempfile.TemporaryDirectory() as tmp_dir:

        video_path = Path(tmp_dir) / "synthetic_demo.mp4"
        _generate_synthetic_video(video_path)

        frames = load_video_frames(video_path)[:FRAME_COUNT]

        source = ReplayFrameSource(camera_id=CAMERA_ID, frames=frames)
        source.start()

        detector = YOLOHumanDetector(_build_scripted_backend())
        # max_centroid_distance sized generously for the 150 px/s
        # running segment -- see tests/
        # test_live_camera_pipeline_behavior_integration.py's own note
        # on tuning this to the expected frame rate/speed.
        tracker = SimpleSingleCameraTracker(max_centroid_distance=200.0)
        recognizer = RuleBasedBehaviorRecognizer()

        print("=== Behavior Recognition -- Offline Local Video Demo ===")
        print()
        print(f"{'frame':>5}  {'track_id':<12} {'velocity(px/s)':>15} {'behavior':<14} {'confidence':>10}")

        for frame_index in range(FRAME_COUNT):

            frame = source.read_frame()
            if frame is None:
                break

            raw_detections = detector.detect(frame)
            tracked_humans = tracker.update(CAMERA_ID, frame.timestamp, raw_detections)
            observations = recognizer.recognize(CAMERA_ID, frame.timestamp, tracked_humans)

            for obs in observations:
                velocity = f"{obs.supporting_metrics.velocity:.1f}" if obs.supporting_metrics.velocity is not None else "-"
                print(
                    f"{frame_index:>5}  {obs.track_id:<12} {velocity:>15} "
                    f"{obs.recognized_behavior.name:<14} {obs.confidence:>10.2f}"
                )

        source.stop()

    print()
    print("Video content: SYNTHETIC (random noise) -- detections are from a scripted FakeYOLOBackend, not real inference.")
    print("Network access performed: NO")
    print("Physical CCTV accessed: NO")


if __name__ == "__main__":
    run_demo()
