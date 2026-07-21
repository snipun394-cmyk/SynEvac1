"""Live Occupant Digital Twin milestone, Phase 11 -- offline demo.

Replay -> YOLO -> Tracker -> WorldProjection -> Behavior ->
CrossCameraIdentity -> LiveOccupantManager

Prints, per cycle: occupant ID, camera, zone, world position, velocity,
behavior, and lifecycle status -- the ONE canonical runtime view this
milestone introduces.

No CCTV, no network -- FakeYOLOBackend stands in for a real model.

Not a pytest test: run manually --
    python scripts/demo_live_occupants.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrame, CameraFrameSource
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from cross_camera_identity.identity_registry import IdentityRegistry
from cross_camera_identity.resolver import RuleBasedCrossCameraIdentityResolver
from cross_camera_identity.topology import CameraTopology
from cross_camera_identity.transition_model import TransitionModel

from models.zone import Zone

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector

from live_occupants.manager import LiveOccupantManager

from tests.human_detection_fixtures import FakeYOLOBackend, person


CAMERA_ID = "CAM-LOBBY"
FLOOR_ID = "floor-1"


class SequencedFrameSource(CameraFrameSource):

    def __init__(self, camera_id, frames):
        self.camera_id = camera_id
        self._frames = list(frames)
        self._index = 0
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    @property
    def is_running(self):
        return self._running

    def read_frame(self):
        if not self._running or self._index >= len(self._frames):
            return None
        timestamp, payload_ref = self._frames[self._index]
        frame = CameraFrame(camera_id=self.camera_id, timestamp=timestamp, frame_sequence=self._index, payload_ref=payload_ref)
        self._index += 1
        return frame


def build_world_projector() -> WorldProjector:

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=45.0)
    calibration = CalibrationProfile(camera_id=CAMERA_ID, floor_id=FLOOR_ID, intrinsics=intrinsics, extrinsics=extrinsics)

    lobby = Zone(name="Lobby", x=0.0, y=-5.0, width=10.0, height=10.0, floor_id=FLOOR_ID)
    lobby.id = "zone-lobby"

    return WorldProjector(calibrations={CAMERA_ID: calibration}, zones_by_floor={FLOOR_ID: [lobby]})


def run_demo() -> None:

    backend = FakeYOLOBackend()
    for i in range(6):
        v = 240.0 - i * 15.0
        backend.queue_result(person(confidence=0.9, box=(310.0, v - 40.0, 330.0, v)))

    detector = YOLOHumanDetector(backend)
    tracker = SimpleSingleCameraTracker(max_centroid_distance=200.0)
    recognizer = RuleBasedBehaviorRecognizer()
    world_projector = build_world_projector()

    topology = CameraTopology()
    registry = IdentityRegistry()
    transition_model = TransitionModel(topology, timeout_seconds=30.0)
    cross_camera_resolver = RuleBasedCrossCameraIdentityResolver(topology=topology, registry=registry, transition_model=transition_model)

    occupant_manager = LiveOccupantManager()

    detection_provider = LiveCameraPipelineDetectionProvider()
    source = SequencedFrameSource(CAMERA_ID, [(float(i), f"frame-{i}") for i in range(6)])
    source.start()

    pipeline = LiveCameraPipeline(
        frame_sources={CAMERA_ID: source},
        human_detector=detector,
        identity_resolver=SimulationIdentityResolver(),
        detection_provider=detection_provider,
        tracker=tracker,
        behavior_recognizer=recognizer,
        cross_camera_identity_resolver=cross_camera_resolver,
        world_projector=world_projector,
        live_occupant_manager=occupant_manager,
    )

    print("=== Live Occupant Digital Twin -- Offline Demo ===")
    print()
    print(
        f"{'t':>4}  {'occupant':<10} {'camera':<10} {'zone':<12} "
        f"{'world(x,y)':<16} {'v(m/s)':>8} {'behavior':<12} {'status':<12}"
    )

    for t in range(6):
        pipeline.run_cycle(float(t))

        for occupant in occupant_manager.active_occupants():

            world_str = f"({occupant.world_position[0]:.2f},{occupant.world_position[1]:.2f})" if occupant.world_position else "-"
            velocity_str = f"{occupant.world_velocity:.2f}" if occupant.world_velocity is not None else "-"
            behavior_str = occupant.behavior.name if occupant.behavior else "-"

            print(
                f"{t:>4}  {occupant.occupant_id:<10} {occupant.current_camera_id:<10} "
                f"{str(occupant.current_zone_id):<12} {world_str:<16} {velocity_str:>8} "
                f"{behavior_str:<12} {occupant.status.name:<12}"
            )

    print()
    print("Network access performed: NO")
    print("Physical CCTV accessed: NO")


if __name__ == "__main__":
    run_demo()
