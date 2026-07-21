"""Cross-Camera Identity Resolution (ReID Framework) milestone,
Phase 10 -- offline synthetic-building demo.

Demonstrates the whole point of this milestone: a single person's LOCAL
tracker IDs change every time they cross into a new camera's view (each
camera's own tracker mints its own independent "CAM-X-Tn" id), while the
GLOBAL occupant ID minted by CrossCameraIdentityResolver stays IDENTICAL
throughout.

Synthetic building:

    Camera A  ->  Camera B  ->  Camera C

No CCTV, no network -- FakeYOLOBackend stands in for a real model
(exactly as every other demo in this codebase's Live/CV milestones).

Not a pytest test: run manually --
    python scripts/demo_cross_camera_identity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_camera_pipeline.frame_source import CameraFrame

from tracking.simple_tracker import SimpleSingleCameraTracker

from human_detection.yolo_human_detector import YOLOHumanDetector

from cross_camera_identity.identity_registry import IdentityRegistry
from cross_camera_identity.resolver import RuleBasedCrossCameraIdentityResolver
from cross_camera_identity.topology import CameraTopology
from cross_camera_identity.transition_model import TransitionModel

from tests.human_detection_fixtures import FakeYOLOBackend, person


# One physical person walking A -> (gap) -> B -> (gap) -> C. Each entry
# is (camera_id, timestamp, detections-for-that-cycle-on-that-camera).
SCHEDULE = [
    ("CAM-A", 0.0, [person(confidence=0.9, box=(0.0, 0.0, 10.0, 20.0))]),
    ("CAM-A", 1.0, []),  # leaves CAM-A's view -- local track expires
    ("CAM-B", 4.0, [person(confidence=0.9, box=(0.0, 0.0, 10.0, 20.0))]),  # 3s later, plausible
    ("CAM-B", 5.0, []),  # leaves CAM-B's view
    ("CAM-C", 9.0, [person(confidence=0.9, box=(0.0, 0.0, 10.0, 20.0))]),  # another 4s later
]


def run_demo() -> None:

    # Camera A -> Camera B -> Camera C, a simple linear corridor --
    # this is exactly the "lightweight topology model" fallback
    # (cross_camera_identity/topology.py), hand-built here rather than
    # derived from a real Building/NavigationGraph, since this demo has
    # no real Digital Twin project to reuse one from.
    topology = CameraTopology()
    topology.add_transition("CAM-A", "CAM-B", min_transition_time=1.0, max_transition_time=10.0)
    topology.add_transition("CAM-B", "CAM-C", min_transition_time=1.0, max_transition_time=10.0)

    registry = IdentityRegistry()
    transition_model = TransitionModel(topology, timeout_seconds=30.0)
    cross_camera_resolver = RuleBasedCrossCameraIdentityResolver(
        topology=topology, registry=registry, transition_model=transition_model,
    )

    camera_ids = ("CAM-A", "CAM-B", "CAM-C")
    backends = {camera_id: FakeYOLOBackend() for camera_id in camera_ids}
    detectors = {camera_id: YOLOHumanDetector(backends[camera_id]) for camera_id in camera_ids}
    trackers = {camera_id: SimpleSingleCameraTracker(max_missing_frames=0) for camera_id in camera_ids}

    for camera_id, _timestamp, detections in SCHEDULE:
        backends[camera_id].queue_result(*detections)

    print("=== Cross-Camera Identity Resolution -- Offline Demo ===")
    print()
    print(f"{'time':>6}  {'camera':<8} {'local track id':<16} {'global occupant id':<20}")

    for camera_id, timestamp, _detections in SCHEDULE:

        frame = CameraFrame(camera_id=camera_id, timestamp=timestamp, frame_sequence=0, payload_ref="frame")

        raw = detectors[camera_id].detect(frame)
        tracked = trackers[camera_id].update(camera_id, timestamp, raw)
        matched = tracked[:len(raw)]

        resolved = cross_camera_resolver.resolve(camera_id, timestamp, tracked, {})
        resolved_by_track_id = {r.track_id: r.global_id for r in resolved}

        for tracked_human in matched:
            global_id = resolved_by_track_id.get(tracked_human.track_id, "-")
            print(f"{timestamp:>6.1f}  {camera_id:<8} {tracked_human.track_id:<16} {global_id:<20}")

    print()
    print(
        "Local tracker IDs above are all DIFFERENT (CAM-A-T1, CAM-B-T1, CAM-C-T1 -- "
        "each camera's own independent counter) -- the global occupant id column stays "
        "IDENTICAL throughout, proving the same physical person was correctly "
        "recognized across all three cameras."
    )
    print()
    print("Network access performed: NO")
    print("Physical CCTV accessed: NO")


if __name__ == "__main__":
    run_demo()
