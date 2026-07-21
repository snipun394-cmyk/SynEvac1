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
# Live Perception -> BuildingState Integration Bridge milestone, Phase
# 8 -- THE critical proof: 2 cameras, 3 physical occupants, 4 raw
# detections (one person visible in both cameras' overlapping view),
# expected result 3 -- never 4, 6, or 7 -- everywhere occupancy is
# represented.
#
# Uses MappingIdentityResolver (NOT cross_camera_identity_resolver) --
# investigated deliberately: cross_camera_identity.resolver.
# RuleBasedCrossCameraIdentityResolver's own matching (Cross-Camera
# Identity Resolution milestone) is a SEQUENTIAL departure/arrival
# transition matcher; it has no mechanism for reconciling two cameras
# that see the SAME person SIMULTANEOUSLY from the very first cycle
# (there is no "departure" from either camera for the resolver to match
# an "arrival" against). MappingIdentityResolver is the existing,
# already-proven mechanism for exactly this overlapping-camera-view
# scenario (see tests/test_rtsp_offline_e2e.py::
# RTSPOfflineEndToEndTests, "two cameras, three physical people, one
# seen by both" -> "four raw detections fuse to exactly three
# BuildingState.occupant_tracks", the direct precedent this test
# extends one layer further, into LiveOccupantManager/SensorFusionEngine).
# This is an honest, investigated limitation of cross_camera_identity's
# CURRENT design, documented here and in docs/architecture/
# live_perception_building_state_integration.md Sec 8 -- not something
# this milestone redesigns (out of scope: "Do NOT redesign ...
# SensorFusionEngine").
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

    return Building(id="double-count-building", name="Double Count Building", floors=[floor])


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


class TwoCamerasThreeOccupantsNoDoubleCountingTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()

        backend_a = FakeYOLOBackend()
        # CAM-A sees: person-only-A, person-shared -- both near image
        # center (straight-down camera at (5,5), so both project close
        # to (5,5), inside zone-1).
        backend_a.queue_result(
            person(confidence=0.9, box=(310.0, 230.0, 330.0, 250.0)),
            person(confidence=0.9, box=(320.0, 240.0, 340.0, 260.0)),
        )

        backend_b = FakeYOLOBackend()
        # CAM-B sees: person-shared, person-only-B -- both project near
        # (15,15), also inside zone-1.
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

        # A single shared tracker/human_detector pair is not possible
        # (YOLOHumanDetector wraps ONE backend) -- but LiveCameraPipeline
        # itself only ever takes ONE human_detector for every camera it
        # serves. Two independent, per-camera-scripted backends behind
        # a small dispatching detector reproduces exactly what a real
        # deployment's single real YOLO model would do (detect within
        # whichever frame it was actually given), without changing
        # anything about the tracker/resolver/occupant-manager wiring
        # under test.
        class _DispatchingDetector:
            def __init__(self, detectors_by_camera):
                self._by_camera = detectors_by_camera

            def detect(self, frame):
                return self._by_camera[frame.camera_id].detect(frame)

        human_detector = _DispatchingDetector({
            "CAM-A": YOLOHumanDetector(backend_a),
            "CAM-B": YOLOHumanDetector(backend_b),
        })

        # Tracker-assigned local_track_id is deterministic within one
        # fresh cycle: first detection in queue order -> "<CAM>-T1",
        # second -> "<CAM>-T2" (tracking.simple_tracker.
        # SimpleSingleCameraTracker's own incrementing-counter format).
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

    def test_four_raw_detections_resolve_to_exactly_three_occupants_everywhere(self):

        self.runtime.run_cycle(0.0)

        building_state = self.runtime.orchestrator.latest_building_state

        # 1. BuildingState.occupant_tracks (MultiCameraFusion) -- 3, not 4.
        self.assertEqual(len(building_state.occupant_tracks), 3)
        for occupant_id in ("OCC-ONLY-A", "OCC-SHARED", "OCC-ONLY-B"):
            self.assertIn(occupant_id, building_state.occupant_tracks)

        # 2. LiveOccupantManager -- 3 LiveOccupants, not 4.
        active = self.runtime.live_occupant_manager.active_occupants()
        self.assertEqual(len(active), 3)
        self.assertEqual({o.occupant_id for o in active}, {"OCC-ONLY-A", "OCC-SHARED", "OCC-ONLY-B"})

        # 3. SensorFusionEngine-derived BuildingState.zone_occupancy --
        # summed headcount is 3, not 4, 6, or 7.
        occupancy = building_state.zone_occupancy.observation_at("zone-1").occupant_count
        self.assertEqual(occupancy, 3.0)

        # 4. The shared FusedTrack genuinely reflects both cameras as
        # sources -- confirming it really is ONE person seen twice, not
        # a coincidental id collision.
        shared_track = building_state.occupant_track("OCC-SHARED")
        self.assertEqual(set(shared_track.source_camera_ids), {"CAM-A", "CAM-B"})


if __name__ == "__main__":
    unittest.main()
