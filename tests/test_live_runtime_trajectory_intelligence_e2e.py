import unittest

from models.building import Building
from models.camera import Camera
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from live_runtime.factory import build_live_runtime

from trajectory_intelligence.models import RouteProgressStatus

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone, Phase 28 -- deterministic offline end-to-end
# proof, driven through the COMPLETE production chain (ReplayFrameSource
# -> Fake YOLO -> Tracker -> World Projection -> Behavior Recognition ->
# Cross-Camera Identity (via MappingIdentityResolver) -> LiveOccupants ->
# Sensor Fusion -> BuildingState -> Crowd Intelligence -> Evacuation
# Progress -> Trajectory Intelligence -> Emergency Response -> Advisory
# -> Command Center), across multiple cycles.
#
# Topology (single floor "f1"): z1 (Lobby, EXIT-1) -- DOOR-1 -- z2 (Hall)
# -- DOOR-2 -- z4 (Annex, EXIT-2), one camera per zone.
#
# Zero network, zero physical CCTV, zero automatic voice/building-
# control/firefighter-dispatch/FACP-mutation execution anywhere in this
# file (Phase 26).
# =====================================================


CAMERA_Z1, CAMERA_Z2, CAMERA_Z4 = "CAM-Z1", "CAM-Z2", "CAM-Z4"


def make_building():

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[
            Zone(id="z1", name="Lobby", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            Zone(id="z2", name="Hall", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            Zone(id="z4", name="Annex", x=40.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
        ],
        cameras=[
            Camera(id=CAMERA_Z1, name="Z1 Camera", floor_id="f1", zone_ids=("z1",), position=(5.0, 5.0), mount_height=3.0),
            Camera(id=CAMERA_Z2, name="Z2 Camera", floor_id="f1", zone_ids=("z2",), position=(25.0, 5.0), mount_height=3.0),
            Camera(id=CAMERA_Z4, name="Z4 Camera", floor_id="f1", zone_ids=("z4",), position=(45.0, 5.0), mount_height=3.0),
        ],
        doors=[
            Door(id="DOOR-1", name="Lobby-Hall Door", floor_id="f1", zone_a_id="z1", zone_b_id="z2"),
            Door(id="DOOR-2", name="Hall-Annex Door", floor_id="f1", zone_a_id="z2", zone_b_id="z4"),
        ],
        exits=[
            Exit(id="EXIT-1", name="Main Exit", floor_id="f1", zone_id="z1"),
            Exit(id="EXIT-2", name="Annex Exit", floor_id="f1", zone_id="z4"),
        ],
    )

    return Building(id="trajectory-e2e-building", name="Trajectory E2E Building", floors=[floor])


def make_world_projector():

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)

    zones = []
    for zone_id, x in (("z1", 0.0), ("z2", 20.0), ("z4", 40.0)):
        zone = Zone(name=zone_id, x=x, y=0.0, width=10.0, height=10.0, floor_id="f1")
        zone.id = zone_id
        zones.append(zone)

    calibrations = {
        CAMERA_Z1: CalibrationProfile(
            camera_id=CAMERA_Z1, floor_id="f1", intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(position=(5.0, 5.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0),
        ),
        CAMERA_Z2: CalibrationProfile(
            camera_id=CAMERA_Z2, floor_id="f1", intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(position=(25.0, 5.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0),
        ),
        CAMERA_Z4: CalibrationProfile(
            camera_id=CAMERA_Z4, floor_id="f1", intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(position=(45.0, 5.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0),
        ),
    }

    return WorldProjector(calibrations=calibrations, zones_by_floor={"f1": zones})


_CENTER_BOX = (310.0, 230.0, 330.0, 250.0)


class ProgressAndCrossCameraHandoverTests(unittest.TestCase):

    # OCC-1 starts in z2 (Hall, no exit of its own), then hands over to
    # z1 (Lobby, EXIT-1) across a camera boundary -- proving both graph-
    # aware PROGRESSING_TOWARD_EXIT and cross-camera trajectory
    # continuity through the real pipeline (Phase 27 tests 1/23).

    def setUp(self):

        self.building = make_building()

        backend_z2 = FakeYOLOBackend()
        backend_z2.queue_result(person(confidence=0.9, box=_CENTER_BOX))  # t=0
        backend_z2.queue_result(person(confidence=0.9, box=_CENTER_BOX))  # t=1
        backend_z2.queue_result()  # t=2 -- occupant has left z2's view

        backend_z1 = FakeYOLOBackend()
        backend_z1.queue_result()  # t=0
        backend_z1.queue_result()  # t=1
        backend_z1.queue_result(person(confidence=0.9, box=_CENTER_BOX))  # t=2 -- occupant now in z1

        frame_sources = {
            CAMERA_Z1: ReplayFrameSource(camera_id=CAMERA_Z1, frames=[(0.0, "f0"), (1.0, "f1"), (2.0, "f2")]),
            CAMERA_Z2: ReplayFrameSource(camera_id=CAMERA_Z2, frames=[(0.0, "f0"), (1.0, "f1"), (2.0, "f2")]),
            CAMERA_Z4: ReplayFrameSource(camera_id=CAMERA_Z4, frames=[(0.0, "f0"), (1.0, "f1"), (2.0, "f2")]),
        }
        for source in frame_sources.values():
            source.start()

        class _DispatchingDetector:
            def __init__(self, detectors_by_camera):
                self._by_camera = detectors_by_camera

            def detect(self, frame):
                return self._by_camera[frame.camera_id].detect(frame)

        human_detector = _DispatchingDetector({
            CAMERA_Z1: YOLOHumanDetector(backend_z1),
            CAMERA_Z2: YOLOHumanDetector(backend_z2),
            CAMERA_Z4: YOLOHumanDetector(FakeYOLOBackend()),
        })

        identity_resolver = MappingIdentityResolver({
            (CAMERA_Z2, f"{CAMERA_Z2}-T1"): "OCC-1",
            (CAMERA_Z1, f"{CAMERA_Z1}-T1"): "OCC-1",
        })

        self.runtime = build_live_runtime(
            self.building,
            frame_sources=frame_sources,
            human_detector=human_detector,
            identity_resolver=identity_resolver,
            tracker=SimpleSingleCameraTracker(),
            behavior_recognizer=RuleBasedBehaviorRecognizer(),
            world_projector=make_world_projector(),
        )

        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_progressing_toward_exit_survives_cross_camera_handover(self):

        self.runtime.run_cycle(0.0)
        self.runtime.run_cycle(1.0)
        self.runtime.run_cycle(2.0)

        snapshot = self.runtime.orchestrator.latest_trajectory_intelligence
        result = snapshot.occupant("OCC-1")

        self.assertIsNotNone(result)
        self.assertEqual(result.zone_id, "z1")
        self.assertEqual(result.route_progress_status, RouteProgressStatus.PROGRESSING_TOWARD_EXIT)
        self.assertEqual(result.nearest_safe_exit_id, "EXIT-1")
        self.assertFalse(result.stale)

    def test_no_automatic_execution_or_dispatch(self):

        self.runtime.run_cycle(0.0)
        self.runtime.run_cycle(1.0)
        self.runtime.run_cycle(2.0)

        self.assertIsNone(self.runtime.voice_evacuation_controller)
        self.assertIsNone(self.runtime.building_control_controller)
        self.assertIsNone(self.runtime.facp)


class SafetyPrecedenceRecomputeTests(unittest.TestCase):

    # OCC-2 remains stationary in z2 throughout -- proving Trajectory
    # Intelligence recomputes its safe-exit candidate the moment z1
    # becomes hazardous, and never continues recommending the now-
    # unsafe EXIT-1 (Phase 23/28's own required proof).

    def setUp(self):

        self.building = make_building()

        backend_z2 = FakeYOLOBackend()
        for _ in range(4):
            backend_z2.queue_result(person(confidence=0.9, box=_CENTER_BOX))

        frame_sources = {
            CAMERA_Z2: ReplayFrameSource(
                camera_id=CAMERA_Z2, frames=[(0.0, "f0"), (1.0, "f1"), (2.0, "f2"), (3.0, "f3")],
            ),
        }
        for source in frame_sources.values():
            source.start()

        identity_resolver = MappingIdentityResolver({(CAMERA_Z2, f"{CAMERA_Z2}-T1"): "OCC-2"})

        self.runtime = build_live_runtime(
            self.building,
            frame_sources=frame_sources,
            human_detector=YOLOHumanDetector(backend_z2),
            identity_resolver=identity_resolver,
            tracker=SimpleSingleCameraTracker(),
            behavior_recognizer=RuleBasedBehaviorRecognizer(),
            world_projector=make_world_projector(),
        )

        self.runtime.start()

        # Test-only seam: z1 becomes hazardous starting t=2.0 -- reaches
        # the SAME hazard_snapshot_provider constructor parameter
        # live_system.building_state_gateway.BuildingStateGateway
        # already documents as its own public seam (build_live_runtime()
        # itself always wires the sensor-fusion-derived one internally
        # and exposes no override parameter for it, so this test reaches
        # the already-constructed gateway's own attribute directly,
        # rather than reimplementing SensorFusionEngine/detector-reading
        # machinery this milestone does not need to exercise).
        def hazard_snapshot_provider(time):

            if time < 2.0:
                return HazardSnapshot()

            return HazardSnapshot(node_states={"z1": HazardNodeState(hazard_score=0.9)})

        self.runtime.orchestrator.building_state_gateway._hazard_snapshot_provider = hazard_snapshot_provider

    def tearDown(self):
        self.runtime.stop()

    def test_recomputes_safe_exit_once_z1_becomes_hazardous(self):

        self.runtime.run_cycle(0.0)
        self.runtime.run_cycle(1.0)

        before = self.runtime.orchestrator.latest_trajectory_intelligence.occupant("OCC-2")
        self.assertEqual(before.nearest_safe_exit_id, "EXIT-1")
        self.assertNotIn("NO_SAFE_ROUTE", before.anomaly_flags)

        self.runtime.run_cycle(2.0)
        self.runtime.run_cycle(3.0)

        after = self.runtime.orchestrator.latest_trajectory_intelligence.occupant("OCC-2")

        self.assertNotEqual(after.nearest_safe_exit_id, "EXIT-1")
        self.assertEqual(after.nearest_safe_exit_id, "EXIT-2")

    def test_no_automatic_action_taken_when_hazard_appears(self):

        for time in (0.0, 1.0, 2.0, 3.0):
            self.runtime.run_cycle(time)

        self.assertIsNone(self.runtime.voice_evacuation_controller)
        self.assertIsNone(self.runtime.building_control_controller)

        # Deterministic safety status (structural + hazard-derived) is
        # never overridden by trajectory evidence -- z1 remains excluded
        # from every occupant's safe-route candidates for as long as the
        # hazard snapshot says so, never silently "recovered" by this
        # package on its own.
        result = self.runtime.orchestrator.latest_trajectory_intelligence.occupant("OCC-2")
        self.assertNotEqual(result.nearest_safe_exit_id, "EXIT-1")


if __name__ == "__main__":
    unittest.main()
