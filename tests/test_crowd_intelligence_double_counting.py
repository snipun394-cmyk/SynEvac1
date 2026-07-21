import unittest

from models.building import Building
from models.camera import Camera
from models.floor import Floor
from models.zone import Zone

from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector

from live_runtime.factory import build_live_runtime

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Live Occupancy, Crowd Density & Congestion Intelligence milestone,
# Phase 12 -- THE required proof: Camera A sees occupants 1,2; Camera B
# sees occupants 2,3; raw detections = 4; global occupants = 3; Crowd
# Intelligence occupant count/density MUST be based on 3, never 4.
#
# Directly extends tests/test_live_perception_double_counting.py's own
# already-proven fixture (2 cameras, 3 physical occupants, 4 raw
# detections, MappingIdentityResolver) one layer further, into
# crowd_intelligence's own zone occupant_count/density.
# =====================================================


def make_building():

    floor = Floor(
        id="floor-1", name="Ground Floor",
        zones=[Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=20.0, height=20.0, floor_id="floor-1")],
        cameras=[
            Camera(id="CAM-A", name="A", floor_id="floor-1", zone_ids=("zone-1",), position=(5.0, 5.0), mount_height=3.0),
            Camera(id="CAM-B", name="B", floor_id="floor-1", zone_ids=("zone-1",), position=(15.0, 15.0), mount_height=3.0),
        ],
    )

    return Building(id="crowd-double-count-building", name="Crowd Double Count Building", floors=[floor])


def make_world_projector():

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)

    zone = Zone(name="zone-1", x=0.0, y=0.0, width=20.0, height=20.0, floor_id="floor-1")
    zone.id = "zone-1"

    calibrations = {
        "CAM-A": CalibrationProfile(
            camera_id="CAM-A", floor_id="floor-1", intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(position=(5.0, 5.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0),
        ),
        "CAM-B": CalibrationProfile(
            camera_id="CAM-B", floor_id="floor-1", intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(position=(15.0, 15.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0),
        ),
    }

    return WorldProjector(calibrations=calibrations, zones_by_floor={"floor-1": [zone]})


class TwoCamerasThreeOccupantsCrowdDensityTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()

        backend_a = FakeYOLOBackend()
        backend_a.queue_result(
            person(confidence=0.9, box=(310.0, 230.0, 330.0, 250.0)),
            person(confidence=0.9, box=(320.0, 240.0, 340.0, 260.0)),
        )

        backend_b = FakeYOLOBackend()
        backend_b.queue_result(
            person(confidence=0.9, box=(310.0, 230.0, 330.0, 250.0)),
            person(confidence=0.9, box=(320.0, 240.0, 340.0, 260.0)),
        )

        frame_sources = {
            "CAM-A": ReplayFrameSource(camera_id="CAM-A", frames=[(0.0, "frame")]),
            "CAM-B": ReplayFrameSource(camera_id="CAM-B", frames=[(0.0, "frame")]),
        }
        for source in frame_sources.values():
            source.start()

        class _DispatchingDetector:
            def __init__(self, detectors_by_camera):
                self._by_camera = detectors_by_camera

            def detect(self, frame):
                return self._by_camera[frame.camera_id].detect(frame)

        human_detector = _DispatchingDetector({
            "CAM-A": YOLOHumanDetector(backend_a),
            "CAM-B": YOLOHumanDetector(backend_b),
        })

        identity_resolver = MappingIdentityResolver({
            ("CAM-A", "CAM-A-T1"): "OCC-ONLY-A",
            ("CAM-A", "CAM-A-T2"): "OCC-SHARED",
            ("CAM-B", "CAM-B-T1"): "OCC-SHARED",
            ("CAM-B", "CAM-B-T2"): "OCC-ONLY-B",
        })

        self.runtime = build_live_runtime(
            self.building,
            frame_sources=frame_sources,
            human_detector=human_detector,
            identity_resolver=identity_resolver,
            tracker=SimpleSingleCameraTracker(),
            world_projector=make_world_projector(),
        )

        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_crowd_intelligence_occupant_count_and_density_reflect_three_not_four(self):

        self.runtime.run_cycle(0.0)

        snapshot = self.runtime.crowd_intelligence_engine.compute(0.0)
        zone_metrics = snapshot.zone("zone-1")

        self.assertEqual(zone_metrics.occupant_count, 3)
        self.assertAlmostEqual(zone_metrics.density_people_per_m2, 3 / 400.0)

        self.assertEqual(snapshot.building_summary.total_observed_occupants, 3)
        self.assertEqual(snapshot.building_summary.calibrated_occupant_count, 3)


if __name__ == "__main__":
    unittest.main()
