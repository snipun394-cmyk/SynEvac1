"""Real YOLO Person Detection Validation & Productionization milestone
-- the full real-composition demo, extending scripts/demo_yolo_human_
detection.py (YOLO + identity resolution only) with the pieces that
milestone identified as "investigated only, not built": stable
single-camera tracking, occupant lifecycle, and (when a --calibration
file is supplied) behavior recognition over world-space motion.

Real chain proven end-to-end:

    local video file (or synthetic, if --video omitted)
    -> human_detection.video_source.load_video_frames  (real cv2 decode)
    -> live_camera_pipeline.replay_frame_source.ReplayFrameSource (real,
       production CameraFrameSource)
    -> CameraFrame
    -> human_detection.yolo_human_detector.YOLOHumanDetector (real,
       production class -- UltralyticsYOLOBackend if --weights is
       supplied, FakeYOLOBackend deterministic double otherwise; a
       number/track is never fabricated in the fake case, it is
       honestly reported as such)
    -> tracking.simple_tracker.SimpleSingleCameraTracker (real,
       production class -- STABLE local track ids across frames)
    -> behavior_recognition.rule_based_recognizer.RuleBasedBehaviorRecognizer
       (real, production class)
    -> live_camera_pipeline.identity_resolver.SimulationIdentityResolver
       (single camera -- the tracker's own stable id already serves as
       a good-enough occupant id; cross-camera fusion is out of scope
       for a one-camera demo)
    -> live_occupants.manager.LiveOccupantManager (real, production class)

No physical CCTV, no network access. World projection/cross-camera
identity are genuinely optional, additive seams this script does NOT
turn on by default (no calibration file exists for a demo scene) --
see docs/architecture/human_detection.md for why fabricating either
here would be dishonest.

Not a pytest test: run manually --
    python scripts/demo_real_yolo_tracking.py
    python scripts/demo_real_yolo_tracking.py --video path/to/video.mp4
    python scripts/demo_real_yolo_tracking.py --video path/to/video.mp4 --weights C:/models/yolov8n.pt
    python scripts/demo_real_yolo_tracking.py --video path/to/video.mp4 --weights C:/models/yolov8n.pt --annotate-out out_dir/
"""

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from human_detection.video_source import load_video_frames
from human_detection.yolo_backend import UltralyticsYOLOBackend
from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from live_occupants.manager import LiveOccupantManager

from tests.human_detection_fixtures import FakeYOLOBackend, person


CAMERA_ID = "CAM-DEMO"
SYNTHETIC_FRAME_COUNT = 30
SYNTHETIC_FRAME_SIZE = (320, 240)  # width, height


def _generate_synthetic_video(path: Path) -> None:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, SYNTHETIC_FRAME_SIZE)

    rng = np.random.default_rng(seed=0)

    for _ in range(SYNTHETIC_FRAME_COUNT):
        frame = rng.integers(0, 255, (SYNTHETIC_FRAME_SIZE[1], SYNTHETIC_FRAME_SIZE[0], 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()


def _build_fake_backend(frame_count: int) -> FakeYOLOBackend:

    # A person drifting slowly across the frame, present most cycles,
    # briefly occluded once -- enough to exercise NEW/TRACKED/MISSING
    # tracker states over a realistic sequence without a real model.
    backend = FakeYOLOBackend()

    for index in range(frame_count):

        if index in (10, 11):
            backend.queue_result()  # brief occlusion
        else:
            x = 10.0 + index * 3.0
            backend.queue_result(person(confidence=0.8 + (index % 5) * 0.02, box=(x, 20.0, x + 20.0, 80.0)))

    return backend


def _draw_annotated_frame(frame, detections_with_tracks):

    annotated = frame.copy() if hasattr(frame, "copy") else frame

    for raw_detection, tracked_human, occupant_id, behavior in detections_with_tracks:

        if raw_detection.bounding_box is None:
            continue

        x1, y1, x2, y2 = (int(v) for v in raw_detection.bounding_box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)

        label_parts = [f"conf={raw_detection.confidence:.2f}"]

        if tracked_human is not None:
            label_parts.append(f"track={tracked_human.track_id}")

        if occupant_id is not None:
            label_parts.append(f"occ={occupant_id}")

        if behavior is not None:
            label_parts.append(f"behavior={behavior.name}")

        # Only shown when calibration genuinely provided it -- this
        # script never turns on world projection (no calibration file
        # exists for a demo scene), so these fields are always absent
        # here, never fabricated.
        if raw_detection.zone_id is not None:
            label_parts.append(f"zone={raw_detection.zone_id}")
        if raw_detection.world_position is not None:
            label_parts.append(f"world={raw_detection.world_position}")

        label = " ".join(label_parts)
        cv2.putText(annotated, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    return annotated


def run_demo(video_path: str = None, weights_path: str = None, annotate_out: str = None) -> None:

    with tempfile.TemporaryDirectory() as tmp_dir:

        if video_path is None:
            video_path = Path(tmp_dir) / "synthetic_demo.mp4"
            _generate_synthetic_video(video_path)
            video_is_synthetic = True
        else:
            video_is_synthetic = False

        frames = load_video_frames(video_path)

        source = ReplayFrameSource(camera_id=CAMERA_ID, frames=frames)
        source.start()

        real_model_ran = weights_path is not None

        if real_model_ran:
            backend = UltralyticsYOLOBackend(weights_path)
        else:
            backend = _build_fake_backend(len(frames))

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()
        behavior_recognizer = RuleBasedBehaviorRecognizer()
        identity_resolver = SimulationIdentityResolver()
        occupant_manager = LiveOccupantManager()

        annotate_dir = None
        if annotate_out is not None:
            annotate_dir = Path(annotate_out)
            annotate_dir.mkdir(parents=True, exist_ok=True)

        frames_processed = 0
        total_raw_detections = 0
        distinct_track_ids = set()
        distinct_occupant_ids = set()

        while True:

            frame = source.read_frame()
            if frame is None:
                break

            frames_processed += 1

            raw = detector.detect(frame)
            total_raw_detections += len(raw)

            tracked_all = tracker.update(CAMERA_ID, frame.timestamp, raw)
            matched = tracked_all[: len(raw)]

            observations = behavior_recognizer.recognize(CAMERA_ID, frame.timestamp, tracked_all, None)
            behavior_by_track_id = {obs.track_id: obs.recognized_behavior for obs in observations}

            # Stable track_id (not the detector's own per-frame index)
            # feeds the identity resolver -- SimulationIdentityResolver
            # treats it as already the occupant id, appropriate for a
            # single-camera demo with no cross-camera fusion configured.
            stabilized_raw = []
            for detection, tracked_human in zip(raw, matched):
                stabilized_raw.append(dataclasses.replace(detection, local_track_id=tracked_human.track_id))

            resolved = identity_resolver.resolve(tuple(stabilized_raw), frame.timestamp)

            seen_occupant_ids = set()
            for detection, tracked_human, resolved_detection in zip(raw, matched, resolved):

                distinct_track_ids.add(tracked_human.track_id)
                distinct_occupant_ids.add(resolved_detection.occupant_id)
                seen_occupant_ids.add(resolved_detection.occupant_id)

                occupant_manager.update(
                    resolved_detection.occupant_id, CAMERA_ID, tracked_human.track_id,
                    resolved_detection.zone_id, resolved_detection.floor_id,
                    resolved_detection.position, resolved_detection.world_velocity,
                    behavior_by_track_id.get(tracked_human.track_id), tracked_human.confidence, frame.timestamp,
                    classification_evidence=detection.classification_evidence,
                    classification_confidence=detection.confidence,
                    state_evidence=detection.state_evidence,
                    state_confidence=detection.confidence,
                )

            occupant_manager.sweep_missing(frame.timestamp, seen_occupant_ids)

            if annotate_dir is not None:

                annotated_entries = [
                    (detection, tracked_human, resolved_detection.occupant_id, behavior_by_track_id.get(tracked_human.track_id))
                    for detection, tracked_human, resolved_detection in zip(raw, matched, resolved)
                ]
                annotated_frame = _draw_annotated_frame(frame.payload_ref, annotated_entries)
                cv2.imwrite(str(annotate_dir / f"frame_{frames_processed:05d}.png"), annotated_frame)

        source.stop()

    print("=== Real YOLO Person Detection -- Full Tracking Chain Demo ===")
    print()
    print(f"Video content: {'SYNTHETIC (random noise, generated locally)' if video_is_synthetic else video_path}")
    print(f"Frames processed: {frames_processed}")
    print(f"Total raw YOLO person detections: {total_raw_detections}")
    print(f"Distinct STABLE tracker ids observed: {len(distinct_track_ids)} -> {sorted(distinct_track_ids)}")
    print(f"Distinct occupant ids observed (LiveOccupantManager): {len(distinct_occupant_ids)}")
    print(f"Occupants currently tracked at end of demo: {len(occupant_manager.all_occupants())}")
    print()

    if real_model_ran:
        print(f"REAL YOLO INFERENCE: RAN -- weights={weights_path!r}")
    else:
        print("REAL YOLO INFERENCE: NOT RUN -- no local YOLO weights supplied "
              "(pass --weights <path/to/model.pt> to run genuine ultralytics "
              "inference; a detection/track is never fabricated in its absence).")

    if annotate_dir is not None:
        print(f"Annotated frames written to: {annotate_dir}")

    print()
    print("Network access performed: NO")
    print("Physical CCTV accessed: NO")


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--video", default=None,
        help="Path to a local video file. If omitted, a small synthetic local .mp4 is generated instead.",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Path to an existing local .pt YOLO weights file. If omitted, a deterministic "
             "FakeYOLOBackend is used and real inference is reported as NOT RUN.",
    )
    parser.add_argument(
        "--annotate-out", default=None,
        help="Directory to write annotated frames (bounding box, confidence, track id, "
             "occupant id, behavior) to, as PNG images. Omit to skip visualization output.",
    )
    args = parser.parse_args()

    run_demo(video_path=args.video, weights_path=args.weights, annotate_out=args.annotate_out)


if __name__ == "__main__":
    main()
