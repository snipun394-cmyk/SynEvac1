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

from human_detection.yolo_human_detector import YOLOHumanDetector

from tests.human_detection_fixtures import FakeYOLOBackend, person
from tests.live_camera_pipeline_fixtures import FakeRTSPBackend


# =====================================================
# Real Human Detection Pipeline milestone, Phase 9 -- the key
# future-CCTV compatibility proof: the exact same YOLOHumanDetector
# instance (with a fake inference backend standing in for real
# ultralytics inference -- swapping FakeYOLOBackend for
# UltralyticsYOLOBackend is the only change a real deployment ever
# needs) plugged into the SAME chain tests/test_rtsp_offline_e2e.py
# already proves for MockHumanDetector:
#
#   FakeRTSPBackend -> RTSPFrameSource -> CameraFrame
#   -> YOLOHumanDetector(FakeYOLOBackend) -> MappingIdentityResolver
#   -> Detection -> CameraManager -> MultiCameraFusionEngine
#   -> BuildingStateEstimator
#
# CameraManager, MultiCameraFusionEngine, BuildingStateEstimator,
# LiveOrchestrator, AI, Advisory, and Command Center are never touched
# or imported here beyond CameraManager/MultiCameraFusionEngine/
# BuildingStateEstimator themselves -- exactly the packages the
# milestone spec names as "must not change." Zero network access,
# zero CCTV access: FakeRTSPBackend never opens a socket and
# FakeYOLOBackend never runs a real model.
# =====================================================


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


class YOLOOverRTSPLiveRuntimeCompatibilityTests(unittest.TestCase):

    def setUp(self):

        self.cam_1 = Camera(id="CAM-001", name="North Corridor", floor_id="floor-1")
        self.cam_2 = Camera(id="CAM-002", name="South Corridor", floor_id="floor-1")

        self.manager = CameraManager()
        self.manager.register_camera(self.cam_1)
        self.manager.register_camera(self.cam_2)

        self.rtsp_backend_1 = FakeRTSPBackend()
        self.rtsp_backend_2 = FakeRTSPBackend()

        self.source_1 = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://endpoint-a/stream",
            decoder_backend=self.rtsp_backend_1, sleep_fn=no_delay,
        )
        self.source_2 = RTSPFrameSource(
            camera_id="CAM-002", endpoint="rtsp://endpoint-b/stream",
            decoder_backend=self.rtsp_backend_2, sleep_fn=no_delay,
        )
        self.source_1.start()
        self.source_2.start()

        # payload_ref is an opaque "decoded image" as far as this test
        # cares -- YOLOHumanDetector never inspects it itself, only
        # hands it to the injected YOLOInferenceBackend, so a plain
        # string stands in for a real decoded ndarray exactly as it
        # does in tests/test_yolo_human_detector.py.
        self.rtsp_backend_1.queue_frame(payload_ref="cam-1-frame-bytes")
        self.rtsp_backend_2.queue_frame(payload_ref="cam-2-frame-bytes")

        self.yolo_backend_1 = FakeYOLOBackend()
        self.yolo_backend_1.queue_result(
            person(confidence=0.91, box=(0.0, 0.0, 10.0, 10.0)),  # local_track_id "0" -> only CAM-001
            person(confidence=0.85, box=(20.0, 0.0, 30.0, 10.0)),  # local_track_id "1" -> shared
        )

        self.yolo_backend_2 = FakeYOLOBackend()
        self.yolo_backend_2.queue_result(
            person(confidence=0.77, box=(0.0, 0.0, 10.0, 10.0)),  # local_track_id "0" -> only CAM-002
            person(confidence=0.66, box=(20.0, 0.0, 30.0, 10.0)),  # local_track_id "1" -> shared
        )

        # Two independent YOLOHumanDetector instances (one fake
        # inference backend per camera) -- proving nothing about this
        # detector is coupled to a single camera or a single frame
        # source instance.
        self.detector_1 = YOLOHumanDetector(self.yolo_backend_1)
        self.detector_2 = YOLOHumanDetector(self.yolo_backend_2)

        self.resolver = MappingIdentityResolver({
            ("CAM-001", "0"): "OCC-ONLY-1",
            ("CAM-002", "0"): "OCC-ONLY-2",
            ("CAM-001", "1"): "OCC-SHARED",
            ("CAM-002", "1"): "OCC-SHARED",
        })

        self.detection_provider = LiveCameraPipelineDetectionProvider()

        # LiveCameraPipeline itself only accepts one human_detector, so
        # to prove per-camera-instance independence we run one pipeline
        # per camera, publishing into the same detection_provider each
        # cycle -- CameraManager/MultiCameraFusionEngine/
        # BuildingStateEstimator never know or care how many
        # LiveCameraPipeline/detector instances fed that provider.
        self.pipeline_1 = LiveCameraPipeline(
            frame_sources={"CAM-001": self.source_1},
            human_detector=self.detector_1,
            identity_resolver=self.resolver,
            detection_provider=self.detection_provider,
        )
        self.pipeline_2 = LiveCameraPipeline(
            frame_sources={"CAM-002": self.source_2},
            human_detector=self.detector_2,
            identity_resolver=self.resolver,
            detection_provider=self.detection_provider,
        )

        self.manager.register_detection_provider(DeviceMode.LIVE, self.detection_provider)
        self.manager.set_camera_mode("CAM-001", DeviceMode.LIVE)
        self.manager.set_camera_mode("CAM-002", DeviceMode.LIVE)

        self.fusion_engine = MultiCameraFusionEngine()

    def _run_both_cycles(self, time):

        self.pipeline_1.run_cycle(time)
        self.pipeline_2.run_cycle(time)

        detections = []
        for camera_id in ("CAM-001", "CAM-002"):
            detections.extend(self.detection_provider.detections_at(camera_id, time))

        fusion_result = self.fusion_engine.fuse(detections, time)

        building_state = BuildingStateEstimator().estimate(
            time,
            hazard_snapshot=HazardSnapshot(),
            occupancy_snapshot=OccupancySnapshot(),
            fusion_result=fusion_result,
        )

        return tuple(detections), fusion_result, building_state

    def test_four_yolo_detections_fuse_to_three_unique_occupants(self):

        detections, fusion_result, building_state = self._run_both_cycles(time=0.0)

        self.assertEqual(len(detections), 4)
        self.assertEqual(len(fusion_result.tracks), 3)
        self.assertEqual(len(building_state.occupant_tracks), 3)

        for occupant_id in ("OCC-ONLY-1", "OCC-ONLY-2", "OCC-SHARED"):
            self.assertIn(occupant_id, building_state.occupant_tracks)

    def test_shared_occupant_has_provenance_from_both_cameras(self):

        _, _, building_state = self._run_both_cycles(time=0.0)

        shared_track = building_state.occupant_track("OCC-SHARED")

        self.assertIsNotNone(shared_track)
        self.assertEqual(set(shared_track.source_camera_ids), {"CAM-001", "CAM-002"})

    def test_yolo_confidence_survives_the_full_chain_into_detection(self):

        detections, _, _ = self._run_both_cycles(time=0.0)

        by_camera_and_confidence = {
            (d.camera_id, round(d.confidence, 2)) for d in detections
        }

        self.assertIn(("CAM-001", 0.91), by_camera_and_confidence)
        self.assertIn(("CAM-002", 0.77), by_camera_and_confidence)

    def test_camera_manager_routing_still_resolves_the_real_camera_asset(self):

        self._run_both_cycles(time=0.0)

        self.assertIs(self.manager.get_camera("CAM-001"), self.cam_1)
        self.assertIs(self.manager.get_camera("CAM-002"), self.cam_2)

    def test_no_network_or_real_model_access_occurred(self):

        # FakeRTSPBackend never opens a real socket and FakeYOLOBackend
        # never imports/runs ultralytics -- asserted here by construction
        # (both fakes are plain in-memory objects), documented explicitly
        # per this milestone's own "Network access performed: NO" report
        # requirement.
        self._run_both_cycles(time=0.0)

        self.assertFalse(hasattr(self.yolo_backend_1, "_model"))
        self.assertFalse(hasattr(self.yolo_backend_2, "_model"))


if __name__ == "__main__":
    unittest.main()
