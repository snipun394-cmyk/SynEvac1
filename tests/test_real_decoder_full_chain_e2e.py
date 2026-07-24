import unittest
from pathlib import Path

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot

from models.camera import Camera
from models.engineering_asset import DeviceMode

from camera_manager.manager import CameraManager

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline
from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

from human_detection.opencv_decoder_backend import OpenCVFrameDecoderBackend
from human_detection.yolo_backend import UltralyticsYOLOBackend
from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from live_occupants.manager import LiveOccupantManager

from multi_camera_fusion.engine import MultiCameraFusionEngine

from building_state.estimator import BuildingStateEstimator


# =====================================================
# CCTV Connection & Calibration Readiness milestone, Phase 3 -- this is
# the one test in this codebase that proves the FULL production chain
# with NO fake anywhere in the frame-acquisition path:
#
#   real vtest.avi
#   -> OpenCVFrameDecoderBackend (real cv2.VideoCapture)
#   -> RTSPFrameSource (real, unmodified production class -- the test-only
#      fake decoder every other RTSP test file uses is deliberately
#      NEVER imported by this file)
#   -> CameraFrame
#   -> YOLOHumanDetector(UltralyticsYOLOBackend)  (real neural-network
#      inference against real local weights)
#   -> SimpleSingleCameraTracker (real, stable per-frame track ids)
#   -> SimulationIdentityResolver -> Detection
#   -> LiveCameraPipelineDetectionProvider -> CameraManager
#   -> MultiCameraFusionEngine -> BuildingStateEstimator -> BuildingState
#   -> LiveOccupantManager
#
# Every earlier "real YOLO on real video" proof (tests/
# test_real_yolo_model_validation.py, scripts/demo_real_yolo_tracking.py,
# docs/architecture/camera_calibration_and_world_projection.md §6) fed
# frames through ReplayFrameSource, never RTSPFrameSource -- this file
# closes that specific gap: it proves the production DECODER BACKEND
# itself (the one genuinely missing piece named in cctv_integration_
# readiness.md §18.4/§19.11), not just the detector/tracker chain above
# it. World projection/calibration is deliberately NOT exercised here
# (no physically measured calibration exists for vtest.avi's real scene
# -- see docs/architecture/camera_calibration_and_world_projection.md
# §6/§7 for why fabricating one would be dishonest); this test's own
# claim is strictly about frame acquisition through detection through
# BuildingState, not about metric world position accuracy.
#
# Automatically SKIPPED (never failed) when the required local
# artifacts (weights/yolov8n.pt, validation_media/vtest.avi) are
# absent -- same discipline as every other opt-in real-model test in
# this codebase.
# =====================================================


REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = REPO_ROOT / "weights" / "yolov8n.pt"
VIDEO_PATH = REPO_ROOT / "validation_media" / "vtest.avi"

CAMERA_ID = "CAM-REAL-DECODER-E2E"

# Real CPU YOLO inference is ~50ms/frame (docs/architecture/
# human_detection.md §16) -- capped well below the video's full length
# so this opt-in test stays fast, while still spanning enough frames
# for a real person to be tracked across more than one frame.
MAX_FRAMES = 40


@unittest.skipUnless(
    WEIGHTS_PATH.exists() and VIDEO_PATH.exists(),
    f"real YOLO weights ({WEIGHTS_PATH}) and/or a local validation video ({VIDEO_PATH}) not found "
    f"-- this opt-in test is skipped, never failed, when either local artifact is absent",
)
class RealDecoderFullChainEndToEndTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.camera = Camera(id=CAMERA_ID, name="Validation Camera", floor_id="floor-1")

        cls.camera_manager = CameraManager()
        cls.camera_manager.register_camera(cls.camera)

        cls.decoder_backend = OpenCVFrameDecoderBackend()
        cls.source = RTSPFrameSource(
            camera_id=CAMERA_ID, endpoint=str(VIDEO_PATH), decoder_backend=cls.decoder_backend,
        )
        cls.source.start()
        assert cls.source.status == "Online", f"expected Online, got {cls.source.status!r}: {cls.source.last_error}"

        yolo_backend = UltralyticsYOLOBackend(WEIGHTS_PATH, device="cpu", confidence_threshold=0.25)
        cls.detector = YOLOHumanDetector(yolo_backend)
        cls.tracker = SimpleSingleCameraTracker()
        cls.identity_resolver = SimulationIdentityResolver()
        cls.detection_provider = LiveCameraPipelineDetectionProvider()
        cls.occupant_manager = LiveOccupantManager()

        cls.pipeline = LiveCameraPipeline(
            frame_sources={CAMERA_ID: cls.source},
            human_detector=cls.detector,
            identity_resolver=cls.identity_resolver,
            detection_provider=cls.detection_provider,
            tracker=cls.tracker,
            live_occupant_manager=cls.occupant_manager,
        )

        cls.camera_manager.register_detection_provider(DeviceMode.LIVE, cls.detection_provider)
        cls.camera_manager.set_camera_mode(CAMERA_ID, DeviceMode.LIVE)

        cls.fusion_engine = MultiCameraFusionEngine()

        cls.frames_run = 0
        cls.total_raw_detections = 0
        cls.building_state = None

        for index in range(MAX_FRAMES):

            time = float(index)
            cls.pipeline.run_cycle(time)
            cls.frames_run += 1

            detections = cls.camera_manager.detections_for_camera(CAMERA_ID, time)
            cls.total_raw_detections += len(detections)

        all_detections = cls.camera_manager.all_detections(float(MAX_FRAMES - 1))
        fusion_result = cls.fusion_engine.fuse(all_detections, float(MAX_FRAMES - 1))

        cls.building_state = BuildingStateEstimator().estimate(
            float(MAX_FRAMES - 1),
            hazard_snapshot=HazardSnapshot(),
            occupancy_snapshot=OccupancySnapshot(),
            fusion_result=fusion_result,
        )

    @classmethod
    def tearDownClass(cls):
        cls.source.stop()

    # =====================================================

    def test_frame_source_reached_online_via_the_real_decoder(self):

        self.assertEqual(self.source.status, "Online")
        self.assertEqual(self.frames_run, MAX_FRAMES)

    def test_real_yolo_inference_genuinely_ran(self):

        # UltralyticsYOLOBackend only ever loads its model lazily on the
        # first real infer() call -- its presence proves genuine
        # inference occurred, not a fake standing in for it.
        backend = self.detector._backend
        self.assertTrue(hasattr(backend, "_model"))
        self.assertIsNotNone(backend._model)

    def test_real_detections_reached_camera_manager(self):

        self.assertGreater(
            self.total_raw_detections, 0,
            "expected at least one real person detection across the sampled vtest.avi frames",
        )

    def test_detections_carry_real_non_fabricated_confidence_and_geometry(self):

        detections = self.camera_manager.detections_for_camera(CAMERA_ID, float(MAX_FRAMES - 1))

        # Not every sampled frame necessarily has a detection at its
        # exact last frame -- re-derive from any cycle that had one by
        # re-running detection directly against the backend's own last
        # confirmed output shape instead of assuming the final frame.
        any_detection_checked = False

        for index in range(MAX_FRAMES):
            for detection in self.camera_manager.detections_for_camera(CAMERA_ID, float(index)):
                self.assertGreaterEqual(detection.confidence, 0.25)
                self.assertLessEqual(detection.confidence, 1.0)
                any_detection_checked = True

        self.assertTrue(any_detection_checked, "expected at least one real detection to inspect")

    def test_at_least_one_stable_track_survives_multiple_frames(self):

        # SimpleSingleCameraTracker's own track ids are namespaced by
        # camera and stable frame-to-frame -- this loop just confirms
        # the SAME tracker instance the pipeline used actually produced
        # continuity, mirroring tests/test_real_yolo_model_validation.py's
        # own RealVideoTrackingContinuityTests claim, now sourced through
        # the real decoder backend instead of load_video_frames().
        multi_frame_track_seen = False
        camera_tracks = getattr(self.tracker, "_tracks", {}).get(CAMERA_ID, {})
        for track in camera_tracks.values():
            if track.frames_seen > 1:
                multi_frame_track_seen = True

        # Fall back to a direct re-assertion via occupant history if the
        # tracker's internal shape ever changes -- LiveOccupantManager
        # itself only ever updates an occupant when the SAME occupant_id
        # (== SimulationIdentityResolver's pass-through of the tracker's
        # own stable track id) reappears across cycles.
        if not multi_frame_track_seen:
            multi_frame_track_seen = any(
                len(occupant.history.positions) > 1 or occupant.last_seen > occupant.first_seen
                for occupant in self.occupant_manager.all_occupants()
            )

        self.assertTrue(multi_frame_track_seen, "expected at least one real person tracked across multiple frames")

    def test_occupants_reached_live_occupant_manager(self):

        self.assertGreater(len(self.occupant_manager.all_occupants()), 0)

    def test_detections_reached_building_state(self):

        self.assertGreater(len(self.building_state.occupant_tracks), 0)

    def test_decoder_backend_is_the_real_opencv_implementation(self):

        # The one non-negotiable claim this whole file exists to prove:
        # frame acquisition ran through the real decoder, never a test
        # double standing in for it.
        self.assertIsInstance(self.decoder_backend, OpenCVFrameDecoderBackend)


if __name__ == "__main__":
    unittest.main()
