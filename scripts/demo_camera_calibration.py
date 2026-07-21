"""Camera Calibration & World Coordinate Projection milestone,
Phase 9 -- offline synthetic-building demo.

Synthetic building: one calibrated camera, mounted 3m up, tilted 45
degrees down, looking along +X across a 10x10m Lobby zone. A scripted
person walks steadily away from directly beneath the camera.

Prints, per frame: pixel position (bounding-box ground contact point),
projected world position, resolved zone, and world-space velocity.

No CCTV, no network -- FakeYOLOBackend stands in for a real model.

Not a pytest test: run manually --
    python scripts/demo_camera_calibration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_camera_pipeline.frame_source import CameraFrame

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from models.zone import Zone

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector

from tests.human_detection_fixtures import FakeYOLOBackend, person


CAMERA_ID = "CAM-LOBBY"
FLOOR_ID = "floor-1"


def build_projector() -> WorldProjector:

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=45.0)
    calibration = CalibrationProfile(camera_id=CAMERA_ID, floor_id=FLOOR_ID, intrinsics=intrinsics, extrinsics=extrinsics)

    lobby = Zone(name="Lobby", x=0.0, y=-5.0, width=10.0, height=10.0, floor_id=FLOOR_ID)
    lobby.id = "zone-lobby"

    return WorldProjector(calibrations={CAMERA_ID: calibration}, zones_by_floor={FLOOR_ID: [lobby]})


def build_scripted_backend() -> FakeYOLOBackend:

    # A person's ground-contact pixel drifts steadily upward in the
    # image (v decreasing) as they walk AWAY from the camera along its
    # optical axis -- exactly what a real receding person's bounding
    # box does (further away -> smaller, ground point rises toward the
    # vanishing point/horizon).
    backend = FakeYOLOBackend()

    for i in range(6):
        v = 240.0 - i * 15.0  # rising ground-contact point -> increasing world distance
        backend.queue_result(person(confidence=0.9, box=(310.0, v - 40.0, 330.0, v)))

    return backend


def run_demo() -> None:

    detector = YOLOHumanDetector(build_scripted_backend())
    tracker = SimpleSingleCameraTracker(max_centroid_distance=200.0)
    recognizer = RuleBasedBehaviorRecognizer()
    projector = build_projector()

    print("=== Camera Calibration & World Projection -- Offline Demo ===")
    print()
    print(f"{'frame':>5}  {'pixel(u,v)':<14} {'world(x,y)':<18} {'zone':<12} {'velocity(m/s)':>14}")

    for i in range(6):

        frame = CameraFrame(camera_id=CAMERA_ID, timestamp=float(i), frame_sequence=i, payload_ref="frame")

        raw = detector.detect(frame)
        tracked = tracker.update(CAMERA_ID, float(i), raw)
        matched = tracked[:len(raw)]

        projections = {t.track_id: projector.project(CAMERA_ID, t.bounding_box, t.confidence) for t in matched}
        world_positions = {tid: p.world_position for tid, p in projections.items() if p.world_position is not None}

        observations = recognizer.recognize(CAMERA_ID, float(i), tracked, world_positions or None)
        by_track = {obs.track_id: obs for obs in observations}

        for tracked_human in matched:

            box = tracked_human.bounding_box
            pixel = ((box[0] + box[2]) / 2.0, box[3]) if box else None

            projection = projections.get(tracked_human.track_id)
            world = projection.world_position if projection else None
            zone = projection.zone_id if projection else None

            obs = by_track.get(tracked_human.track_id)
            velocity = obs.world_metrics.world_velocity if obs and obs.world_metrics else None

            pixel_str = f"({pixel[0]:.0f},{pixel[1]:.0f})" if pixel else "-"
            world_str = f"({world[0]:.2f},{world[1]:.2f})" if world else "-"
            velocity_str = f"{velocity:.2f}" if velocity is not None else "-"

            print(f"{i:>5}  {pixel_str:<14} {world_str:<18} {str(zone):<12} {velocity_str:>14}")

    print()
    print("Network access performed: NO")
    print("Physical CCTV accessed: NO")


if __name__ == "__main__":
    run_demo()
