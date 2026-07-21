import unittest

from perception.models.human_observation import HumanState

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrame, CameraFrameSource
from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Human Behavior Recognition Framework milestone, Phase 8 -- proves the
# actual integration point: LiveCameraPipeline.run_cycle(), between
# SingleCameraTracker.update() and IdentityResolver.resolve().
# IdentityResolver itself is the unmodified, existing
# MappingIdentityResolver, and Detection is the unmodified, existing
# type -- this test's whole point is that Detection.human_state (via
# RawHumanDetection.state_evidence) now honestly reflects WALKING/
# RUNNING recognized purely from tracking geometry, with zero change to
# IdentityResolver, Detection, MultiCameraFusion, or BuildingState.
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


class WalkingBehaviorReachesDetectionHumanStateTests(unittest.TestCase):

    def test_steady_moderate_pace_sets_detection_human_state_to_walking(self):

        backend = FakeYOLOBackend()
        for i in range(5):
            x = i * 20.0  # 20 px/s -- WALKING, not STATIONARY or RUNNING
            backend.queue_result(person(confidence=0.9, box=(x, 0.0, x + 10.0, 20.0)))

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()
        recognizer = RuleBasedBehaviorRecognizer()
        resolver = MappingIdentityResolver()
        detection_provider = LiveCameraPipelineDetectionProvider()

        source = SequencedFrameSource("CAM-001", [(float(i), f"frame-{i}") for i in range(5)])
        source.start()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=resolver,
            detection_provider=detection_provider,
            tracker=tracker,
            behavior_recognizer=recognizer,
        )

        last_detections = None
        for i in range(5):
            pipeline.run_cycle(float(i))
            last_detections = detection_provider.detections_at("CAM-001", float(i))

        self.assertEqual(last_detections[0].human_state, HumanState.WALKING)


class RunningBehaviorReachesDetectionHumanStateTests(unittest.TestCase):

    def test_large_displacement_sets_detection_human_state_to_running(self):

        backend = FakeYOLOBackend()
        for i in range(5):
            x = i * 150.0  # well above the running threshold
            backend.queue_result(person(confidence=0.9, box=(x, 0.0, x + 10.0, 20.0)))

        detector = YOLOHumanDetector(backend)
        # A person covering 150px/cycle needs a tracker whose own
        # max_centroid_distance is sized for that displacement -- the
        # tracker's default (50.0, tuned for a much slower typical walk
        # in the Single-Camera Tracking Framework milestone's own
        # examples) would otherwise treat each cycle as a brand new
        # track and never accumulate the history behavior recognition
        # needs. This mirrors a real deployment tuning tracker
        # thresholds to its actual frame rate/expected speeds.
        tracker = SimpleSingleCameraTracker(max_centroid_distance=200.0)
        recognizer = RuleBasedBehaviorRecognizer()
        resolver = MappingIdentityResolver()
        detection_provider = LiveCameraPipelineDetectionProvider()

        source = SequencedFrameSource("CAM-001", [(float(i), f"frame-{i}") for i in range(5)])
        source.start()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=resolver,
            detection_provider=detection_provider,
            tracker=tracker,
            behavior_recognizer=recognizer,
        )

        last_detections = None
        for i in range(5):
            pipeline.run_cycle(float(i))
            last_detections = detection_provider.detections_at("CAM-001", float(i))

        self.assertEqual(last_detections[0].human_state, HumanState.RUNNING)


class StationaryAndUnknownNeverFabricateHumanStateTests(unittest.TestCase):

    def test_stationary_person_leaves_detection_human_state_none(self):

        backend = FakeYOLOBackend()
        for _ in range(5):
            backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 20.0)))  # never moves

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()
        recognizer = RuleBasedBehaviorRecognizer()
        resolver = MappingIdentityResolver()
        detection_provider = LiveCameraPipelineDetectionProvider()

        source = SequencedFrameSource("CAM-001", [(float(i), f"frame-{i}") for i in range(5)])
        source.start()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=resolver,
            detection_provider=detection_provider,
            tracker=tracker,
            behavior_recognizer=recognizer,
        )

        last_detections = None
        for i in range(5):
            pipeline.run_cycle(float(i))
            last_detections = detection_provider.detections_at("CAM-001", float(i))

        # STATIONARY is recognized internally (proven directly in
        # tests/test_behavior_recognition.py) but is NOT one of the two
        # honest HumanState mappings (Phase 8) -- Detection.human_state
        # stays None rather than fabricating STANDING/WAITING/
        # NEVER_MOVING_YET.
        self.assertIsNone(last_detections[0].human_state)

    def test_first_cycle_unknown_behavior_leaves_detection_human_state_none(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 20.0)))

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()
        recognizer = RuleBasedBehaviorRecognizer()
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
            behavior_recognizer=recognizer,
        )

        pipeline.run_cycle(0.0)
        detections = detection_provider.detections_at("CAM-001", 0.0)

        self.assertIsNone(detections[0].human_state)


class NoBehaviorRecognizerPreservesTrackerOnlyBehaviorTests(unittest.TestCase):

    # Backward compatibility -- omitting `behavior_recognizer` (the
    # default) must reproduce the Single-Camera Tracking Framework
    # milestone's exact behavior: local_track_id stabilized, but
    # state_evidence untouched (always None, exactly what YOLOHumanDetector
    # itself already sets).

    def test_tracker_without_behavior_recognizer_leaves_human_state_none(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 20.0)))

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
        )  # behavior_recognizer omitted entirely

        pipeline.run_cycle(0.0)
        detections = detection_provider.detections_at("CAM-001", 0.0)

        self.assertIsNone(detections[0].human_state)


if __name__ == "__main__":
    unittest.main()
