import unittest

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrame, CameraFrameSource
from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from models.zone import Zone

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Camera Calibration & World Coordinate Projection milestone, Phase 6/7
# -- proves the actual integration point: LiveCameraPipeline.run_cycle(),
# between SingleCameraTracker.update() and BehaviorRecognizer.recognize(),
# with the resulting world_position/zone_id/floor_id/projection_confidence/
# world_velocity reaching Detection through the UNMODIFIED IdentityResolver
# interface (only _to_detection()'s own field mapping changed, in the
# prior commit).
# =====================================================


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
        frame = CameraFrame(
            camera_id=self.camera_id, timestamp=timestamp, frame_sequence=self._index, payload_ref=payload_ref,
        )
        self._index += 1
        return frame


def make_projector(camera_id="CAM-001", floor_id="floor-1"):

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0)
    calibration = CalibrationProfile(camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics)

    zone = Zone(name="zone-a", x=-5.0, y=-5.0, width=10.0, height=10.0, floor_id=floor_id)
    zone.id = "zone-a"

    return WorldProjector(calibrations={camera_id: calibration}, zones_by_floor={floor_id: [zone]})


class WorldPositionReachesDetectionTests(unittest.TestCase):

    def test_world_position_zone_id_floor_id_and_confidence_reach_detection(self):

        backend = FakeYOLOBackend()
        # Straight-down camera (pitch=90) at (0,0) -- center pixel (320,240)
        # projects to world (0,0), inside the zone [-5,-5]..[5,5].
        backend.queue_result(person(confidence=0.9, box=(315.0, 200.0, 325.0, 240.0)))

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()
        resolver = MappingIdentityResolver()
        detection_provider = LiveCameraPipelineDetectionProvider()

        source = SequencedFrameSource("CAM-001", [(0.0, "frame-0")])
        source.start()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=resolver,
            detection_provider=detection_provider,
            tracker=tracker,
            world_projector=make_projector(),
        )

        pipeline.run_cycle(0.0)
        detections = detection_provider.detections_at("CAM-001", 0.0)

        self.assertEqual(len(detections), 1)
        detection = detections[0]

        self.assertIsNotNone(detection.position)
        self.assertAlmostEqual(detection.position[0], 0.0, places=3)
        self.assertAlmostEqual(detection.position[1], 0.0, places=3)
        self.assertEqual(detection.zone_id, "zone-a")
        self.assertEqual(detection.floor_id, "floor-1")
        self.assertIsNotNone(detection.projection_confidence)


class WorldVelocityReachesDetectionTests(unittest.TestCase):

    def test_world_velocity_populated_once_behavior_recognition_has_enough_history(self):

        backend = FakeYOLOBackend()
        # A person standing still directly below the camera across
        # several cycles -- world position stays (0,0) each time, so
        # world_velocity should resolve to ~0 (STATIONARY), not None,
        # once at least 2 world samples exist.
        for _ in range(3):
            backend.queue_result(person(confidence=0.9, box=(315.0, 200.0, 325.0, 240.0)))

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()
        recognizer = RuleBasedBehaviorRecognizer()
        resolver = MappingIdentityResolver()
        detection_provider = LiveCameraPipelineDetectionProvider()

        source = SequencedFrameSource("CAM-001", [(0.0, "f0"), (1.0, "f1"), (2.0, "f2")])
        source.start()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=resolver,
            detection_provider=detection_provider,
            tracker=tracker,
            behavior_recognizer=recognizer,
            world_projector=make_projector(),
        )

        last_detections = None
        for t in (0.0, 1.0, 2.0):
            pipeline.run_cycle(t)
            last_detections = detection_provider.detections_at("CAM-001", t)

        self.assertIsNotNone(last_detections[0].world_velocity)
        self.assertAlmostEqual(last_detections[0].world_velocity, 0.0, places=3)


class NoWorldProjectorPreservesPriorBehaviorTests(unittest.TestCase):

    # Backward compatibility -- omitting `world_projector` (the default)
    # must leave Detection.position/world_velocity/projection_confidence
    # exactly as the prior milestones left them (None), and zone_id/
    # floor_id untouched (still None, since YOLOHumanDetector never sets
    # either on its own).

    def test_without_world_projector_position_and_world_fields_stay_none(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(315.0, 200.0, 325.0, 240.0)))

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()
        resolver = MappingIdentityResolver()
        detection_provider = LiveCameraPipelineDetectionProvider()

        source = SequencedFrameSource("CAM-001", [(0.0, "f0")])
        source.start()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=resolver,
            detection_provider=detection_provider,
            tracker=tracker,
        )  # world_projector omitted entirely

        pipeline.run_cycle(0.0)
        detections = detection_provider.detections_at("CAM-001", 0.0)

        self.assertIsNone(detections[0].position)
        self.assertIsNone(detections[0].zone_id)
        self.assertIsNone(detections[0].floor_id)
        self.assertIsNone(detections[0].world_velocity)
        self.assertIsNone(detections[0].projection_confidence)


if __name__ == "__main__":
    unittest.main()
