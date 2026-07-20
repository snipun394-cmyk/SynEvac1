import unittest

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot

from models.camera import Camera
from models.engineering_asset import DeviceMode

from camera_manager.manager import CameraManager

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline
from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

from multi_camera_fusion.engine import MultiCameraFusionEngine

from building_state.estimator import BuildingStateEstimator

from tests.live_camera_pipeline_fixtures import FakeRTSPBackend, MockHumanDetector


# RTSP Frame Source milestone, Phase 10/11: the full production chain
# -- Digital Twin Camera Asset -> CameraManager -> RTSPFrameSource (not
# ReplayFrameSource) -> CameraFrame -> MockHumanDetector ->
# MappingIdentityResolver -> Detection -> LiveCameraPipelineDetection
# Provider -> CameraManager -> MultiCameraFusionEngine ->
# BuildingStateEstimator -- proven end-to-end with zero network I/O.
# The only thing "fake" here is the decoder backend (FakeRTSPBackend);
# RTSPFrameSource itself is the real, production class.


def no_delay(_seconds):
    pass


def _run_one_cycle(pipeline, detection_provider, camera_ids, fusion_engine, time):

    pipeline.run_cycle(time)

    detections = []
    for camera_id in camera_ids:
        detections.extend(detection_provider.detections_at(camera_id, time))

    fusion_result = fusion_engine.fuse(detections, time)

    building_state = BuildingStateEstimator().estimate(
        time,
        hazard_snapshot=HazardSnapshot(),
        occupancy_snapshot=OccupancySnapshot(),
        fusion_result=fusion_result,
    )

    return tuple(detections), fusion_result, building_state


class RTSPOfflineEndToEndTests(unittest.TestCase):

    # Phase 10: two cameras, real RTSPFrameSource instances (fake
    # decoder backends only), through to BuildingState.

    def setUp(self):

        self.cam_1 = Camera(id="CAM-001", name="North Corridor", floor_id="floor-1")
        self.cam_2 = Camera(id="CAM-002", name="South Corridor", floor_id="floor-1")

        self.manager = CameraManager()
        self.manager.register_camera(self.cam_1)
        self.manager.register_camera(self.cam_2)

        self.backend_1 = FakeRTSPBackend()
        self.backend_2 = FakeRTSPBackend()

        self.source_1 = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://endpoint-a/stream",
            decoder_backend=self.backend_1, sleep_fn=no_delay,
        )
        self.source_2 = RTSPFrameSource(
            camera_id="CAM-002", endpoint="rtsp://endpoint-b/stream",
            decoder_backend=self.backend_2, sleep_fn=no_delay,
        )
        self.source_1.start()
        self.source_2.start()

        self.backend_1.queue_frame(payload_ref=[
            {"local_track_id": "1"},   # only CAM-001
            {"local_track_id": "9"},   # shared with CAM-002
        ])
        self.backend_2.queue_frame(payload_ref=[
            {"local_track_id": "2"},   # only CAM-002
            {"local_track_id": "5"},   # shared with CAM-001
        ])

        self.resolver = MappingIdentityResolver({
            ("CAM-001", "1"): "OCC-ONLY-1",
            ("CAM-002", "2"): "OCC-ONLY-2",
            ("CAM-001", "9"): "OCC-SHARED",
            ("CAM-002", "5"): "OCC-SHARED",
        })

        self.detection_provider = LiveCameraPipelineDetectionProvider()

        self.pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": self.source_1, "CAM-002": self.source_2},
            human_detector=MockHumanDetector(),
            identity_resolver=self.resolver,
            detection_provider=self.detection_provider,
        )

        self.manager.register_detection_provider(DeviceMode.LIVE, self.detection_provider)
        self.manager.set_camera_mode("CAM-001", DeviceMode.LIVE)
        self.manager.set_camera_mode("CAM-002", DeviceMode.LIVE)

        self.fusion_engine = MultiCameraFusionEngine()

    def test_four_raw_detections_fuse_to_three_unique_occupants(self):

        detections, fusion_result, building_state = _run_one_cycle(
            self.pipeline, self.detection_provider, ["CAM-001", "CAM-002"], self.fusion_engine, time=0.0,
        )

        self.assertEqual(len(detections), 4)
        self.assertEqual(len(fusion_result.tracks), 3)
        self.assertEqual(len(building_state.occupant_tracks), 3)

        for occupant_id in ("OCC-ONLY-1", "OCC-ONLY-2", "OCC-SHARED"):
            self.assertIn(occupant_id, building_state.occupant_tracks)

    def test_shared_occupant_has_provenance_from_both_cameras(self):

        _, _, building_state = _run_one_cycle(
            self.pipeline, self.detection_provider, ["CAM-001", "CAM-002"], self.fusion_engine, time=0.0,
        )

        shared_track = building_state.occupant_track("OCC-SHARED")

        self.assertIsNotNone(shared_track)
        self.assertEqual(set(shared_track.source_camera_ids), {"CAM-001", "CAM-002"})

    def test_camera_manager_routing_still_resolves_the_real_camera_asset(self):

        _run_one_cycle(
            self.pipeline, self.detection_provider, ["CAM-001", "CAM-002"], self.fusion_engine, time=0.0,
        )

        self.assertIs(self.manager.get_camera("CAM-001"), self.cam_1)
        self.assertIs(self.manager.get_camera("CAM-002"), self.cam_2)


class CameraReplacementPreservesIdentityTests(unittest.TestCase):

    # Phase 11 -- the most important architectural proof: a physical
    # camera being swapped (new RTSP endpoint) must never change what
    # SynEvac calls that camera downstream. Only the RTSPFrameSource
    # instance changes (a new endpoint means a new physical
    # connection, hence a new object -- endpoint is immutable per-
    # instance, matching every other frame source in this codebase);
    # camera_id, the Camera Asset itself, and everything registered
    # against camera_id in CameraManager/the pipeline's frame_sources
    # mapping stay exactly the same key.

    def setUp(self):

        self.camera = Camera(
            id="CAM-001", name="Cafeteria Camera", floor_id="floor-1", zone_ids=("zone-cafeteria",),
        )

        self.manager = CameraManager()
        self.manager.register_camera(self.camera)
        self.manager.set_camera_mode("CAM-001", DeviceMode.LIVE)

        self.resolver = MappingIdentityResolver({("CAM-001", "17"): "OCC-1"})
        self.detection_provider = LiveCameraPipelineDetectionProvider()
        self.manager.register_detection_provider(DeviceMode.LIVE, self.detection_provider)
        self.fusion_engine = MultiCameraFusionEngine()

    def _tick_with_source(self, source, time):

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=MockHumanDetector(),
            identity_resolver=self.resolver,
            detection_provider=self.detection_provider,
        )
        pipeline.run_cycle(time)

        detections = self.detection_provider.detections_at("CAM-001", time)
        fusion_result = self.fusion_engine.fuse(detections, time)

        building_state = BuildingStateEstimator().estimate(
            time,
            hazard_snapshot=HazardSnapshot(),
            occupancy_snapshot=OccupancySnapshot(),
            fusion_result=fusion_result,
        )

        return detections, fusion_result, building_state

    def test_old_camera_endpoint_produces_detections_under_cam_001(self):

        old_backend = FakeRTSPBackend()
        old_source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://old-camera/stream",
            decoder_backend=old_backend, sleep_fn=no_delay,
        )
        old_source.start()
        old_backend.queue_frame(payload_ref=[{"local_track_id": "17"}])

        detections, fusion_result, building_state = self._tick_with_source(old_source, time=0.0)

        self.assertEqual(detections[0].camera_id, "CAM-001")
        self.assertIn("OCC-1", building_state.occupant_tracks)
        self.assertIn("CAM-001", building_state.occupant_tracks["OCC-1"].source_camera_ids)

        old_source.stop()

    def test_new_camera_endpoint_still_produces_detections_under_the_same_cam_001(self):

        old_backend = FakeRTSPBackend()
        old_source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://old-camera/stream",
            decoder_backend=old_backend, sleep_fn=no_delay,
        )
        old_source.start()
        old_backend.queue_frame(payload_ref=[{"local_track_id": "17"}])
        self._tick_with_source(old_source, time=0.0)
        old_source.stop()

        # The physical camera at CAM-001's position was replaced --
        # only the endpoint changes; the Digital Twin Camera Asset's
        # own id is never touched.
        original_camera_id = self.camera.id

        new_backend = FakeRTSPBackend()
        new_source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://new-camera/stream",
            decoder_backend=new_backend, sleep_fn=no_delay,
        )
        new_source.start()
        new_backend.queue_frame(payload_ref=[{"local_track_id": "17"}])

        detections, fusion_result, building_state = self._tick_with_source(new_source, time=1.0)

        self.assertEqual(self.camera.id, original_camera_id)
        self.assertIs(self.manager.get_camera("CAM-001"), self.camera)

        self.assertEqual(detections[0].camera_id, "CAM-001")

        track = fusion_result.track("OCC-1")
        self.assertIsNotNone(track)
        self.assertIn("CAM-001", track.source_camera_ids)

        self.assertIn("OCC-1", building_state.occupant_tracks)
        self.assertIn("CAM-001", building_state.occupant_tracks["OCC-1"].source_camera_ids)

        new_source.stop()

    def test_zone_and_floor_assignment_unaffected_by_endpoint_replacement(self):

        # No downstream configuration (zone assignment, floor, mode
        # registration) needs to change when only the endpoint changes
        # -- re-asserted directly against the still-registered Camera
        # Asset.

        old_backend = FakeRTSPBackend()
        old_source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://old-camera/stream",
            decoder_backend=old_backend, sleep_fn=no_delay,
        )
        old_source.start()
        old_source.stop()

        new_backend = FakeRTSPBackend()
        new_source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://new-camera/stream",
            decoder_backend=new_backend, sleep_fn=no_delay,
        )
        new_source.start()
        new_source.stop()

        camera_after = self.manager.get_camera("CAM-001")

        self.assertEqual(camera_after.floor_id, "floor-1")
        self.assertEqual(camera_after.zone_ids, ("zone-cafeteria",))
        self.assertEqual(self.manager.camera_mode("CAM-001"), DeviceMode.LIVE)


if __name__ == "__main__":
    unittest.main()
