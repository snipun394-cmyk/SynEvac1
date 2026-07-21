"""Single-Camera Tracking Framework milestone, Phase 10 -- offline
local-video demo.

Proves the full, real chain end-to-end with zero network access and
zero CCTV access:

    local video file
    -> human_detection.video_source.load_video_frames (real cv2 decode)
    -> live_camera_pipeline.replay_frame_source.ReplayFrameSource
    -> CameraFrame
    -> human_detection.yolo_human_detector.YOLOHumanDetector
    -> RawHumanDetection
    -> tracking.simple_tracker.SimpleSingleCameraTracker  (NEW)
    -> TrackedHuman -- printed below, showing stable track IDs

Same synthetic-content local video generation approach as
scripts/demo_yolo_human_detection.py (no real CCTV/recorded-person
footage exists in this repository) -- the video's content is
irrelevant here since a FakeYOLOBackend (not real ultralytics
inference) is what actually reports "detections", deterministically
scripted to walk one bounding box slowly across the frame so the demo
can show a track ID staying stable while its position changes, one
person leaving, and a second person entering partway through.

Not a pytest test: run manually --
    python scripts/demo_single_camera_tracking.py
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from human_detection.video_source import load_video_frames
from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from tests.human_detection_fixtures import FakeYOLOBackend, person


FRAME_COUNT = 12
FRAME_SIZE = (320, 240)
CAMERA_ID = "CAM-DEMO"


def _generate_synthetic_video(path: Path) -> None:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, FRAME_SIZE)

    rng = np.random.default_rng(seed=0)

    for _ in range(FRAME_COUNT):
        frame = rng.integers(0, 255, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()


def _build_scripted_backend() -> FakeYOLOBackend:

    # Deterministic scenario: Person A walks steadily left-to-right for
    # frames 0-6, is briefly occluded on frame 7 (no detection), then
    # reappears at the expected next position on frame 8 (proving
    # temporary-occlusion continuity). Person B enters at frame 4 and
    # stays. Frames 10-11: Person A leaves (never seen again).
    backend = FakeYOLOBackend()

    person_a_positions = [0, 10, 20, 30, 40, 50, 60, None, 80, 90, None, None]
    person_b_start_frame = 4

    for frame_index in range(FRAME_COUNT):

        boxes = []

        x = person_a_positions[frame_index]
        if x is not None:
            boxes.append(person(confidence=0.9, box=(float(x), 0.0, float(x) + 20.0, 60.0)))

        if frame_index >= person_b_start_frame:
            boxes.append(person(confidence=0.85, box=(200.0, 100.0, 220.0, 160.0)))

        backend.queue_result(*boxes)

    return backend


def run_demo() -> None:

    with tempfile.TemporaryDirectory() as tmp_dir:

        video_path = Path(tmp_dir) / "synthetic_demo.mp4"
        _generate_synthetic_video(video_path)

        frames = load_video_frames(video_path)[:FRAME_COUNT]

        source = ReplayFrameSource(camera_id=CAMERA_ID, frames=frames)
        source.start()

        detector = YOLOHumanDetector(_build_scripted_backend())
        tracker = SimpleSingleCameraTracker(max_missing_frames=2)

        print("=== Single-Camera Tracking -- Offline Local Video Demo ===")
        print()
        print(f"{'frame':>5}  {'track_id':<12} {'state':<8} {'age':>4} {'seen':>5} {'missing':>8} bbox")

        matching_latencies_ms = []

        for frame_index in range(FRAME_COUNT):

            frame = source.read_frame()
            if frame is None:
                break

            raw_detections = detector.detect(frame)

            start = time.perf_counter()
            tracked_humans = tracker.update(CAMERA_ID, frame.timestamp, raw_detections)
            matching_latencies_ms.append((time.perf_counter() - start) * 1000)

            for tracked in tracked_humans:
                box = tuple(round(v, 1) for v in tracked.bounding_box) if tracked.bounding_box else None
                print(
                    f"{frame_index:>5}  {tracked.track_id:<12} {tracked.state.name:<8} "
                    f"{tracked.age:>4} {tracked.frames_seen:>5} {tracked.frames_missing:>8} {box}"
                )

        source.stop()

    print()
    print(
        f"Tracker update latency: mean {sum(matching_latencies_ms) / len(matching_latencies_ms):.4f} ms "
        f"over {len(matching_latencies_ms)} cycles."
    )
    print()
    print("Video content: SYNTHETIC (random noise) -- detections are from a scripted FakeYOLOBackend, not real inference.")
    print("Network access performed: NO")
    print("Physical CCTV accessed: NO")


if __name__ == "__main__":
    run_demo()
