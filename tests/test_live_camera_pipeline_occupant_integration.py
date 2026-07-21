import unittest

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

from live_occupants.manager import LiveOccupantManager
from live_occupants.state import OccupantStatus

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Live Occupant Digital Twin milestone, Phase 7 -- proves the actual
# integration point: LiveCameraPipeline.run_cycle(), where a
# LiveOccupantManager observes the SAME per-cycle data flowing into
# Detection, without altering Detection/BuildingState construction at
# all. Uses SimulationIdentityResolver deliberately (the established
# "local_track_id is already the global identity" pairing from the
# Cross-Camera Identity Resolution milestone) so occupant_id in
# LiveOccupantManager is the SAME string as Detection.occupant_id.
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


class LiveOccupantManagerObservesPipelineTests(unittest.TestCase):

    def test_occupant_created_and_updated_matches_detection_occupant_id(self):

        backend = FakeYOLOBackend()
        for i in range(3):
            backend.queue_result(person(confidence=0.9, box=(float(i) * 5.0, 0.0, float(i) * 5.0 + 10.0, 20.0)))

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()
        recognizer = RuleBasedBehaviorRecognizer()

        topology = CameraTopology()
        registry = IdentityRegistry()
        transition_model = TransitionModel(topology, timeout_seconds=30.0)
        cross_camera_resolver = RuleBasedCrossCameraIdentityResolver(topology=topology, registry=registry, transition_model=transition_model)

        occupant_manager = LiveOccupantManager()

        detection_provider = LiveCameraPipelineDetectionProvider()
        source = SequencedFrameSource("CAM-001", [(0.0, "f0"), (1.0, "f1"), (2.0, "f2")])
        source.start()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=SimulationIdentityResolver(),
            detection_provider=detection_provider,
            tracker=tracker,
            behavior_recognizer=recognizer,
            cross_camera_identity_resolver=cross_camera_resolver,
            live_occupant_manager=occupant_manager,
        )

        last_detections = None
        for t in (0.0, 1.0, 2.0):
            pipeline.run_cycle(t)
            last_detections = detection_provider.detections_at("CAM-001", t)

        detection_occupant_id = last_detections[0].occupant_id
        occupant = occupant_manager.get(detection_occupant_id)

        self.assertIsNotNone(occupant)
        self.assertEqual(occupant.status, OccupantStatus.ACTIVE)
        self.assertEqual(occupant.current_camera_id, "CAM-001")
        self.assertEqual(occupant.first_seen, 0.0)
        self.assertEqual(occupant.last_seen, 2.0)

    def test_occupant_leaving_frame_becomes_temporarily_lost_via_sweep(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 20.0)))
        backend.queue_result()  # gone next cycle

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker(max_missing_frames=5)

        occupant_manager = LiveOccupantManager(expire_after_seconds=100.0)

        detection_provider = LiveCameraPipelineDetectionProvider()
        source = SequencedFrameSource("CAM-001", [(0.0, "f0"), (1.0, "f1")])
        source.start()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=SimulationIdentityResolver(),
            detection_provider=detection_provider,
            tracker=tracker,
            live_occupant_manager=occupant_manager,
        )

        pipeline.run_cycle(0.0)
        self.assertEqual(len(occupant_manager.active_occupants()), 1)

        pipeline.run_cycle(1.0)  # tracker MISSES this cycle -- no update() call reaches the manager

        self.assertEqual(len(occupant_manager.active_occupants()), 0)
        self.assertEqual(len(occupant_manager.all_occupants()), 1)  # still tracked, just TEMPORARILY_LOST


class NoLiveOccupantManagerPreservesPriorBehaviorTests(unittest.TestCase):

    def test_pipeline_without_occupant_manager_behaves_exactly_as_before(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 20.0)))

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()

        detection_provider = LiveCameraPipelineDetectionProvider()
        source = SequencedFrameSource("CAM-001", [(0.0, "f0")])
        source.start()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=SimulationIdentityResolver(),
            detection_provider=detection_provider,
            tracker=tracker,
        )  # live_occupant_manager omitted entirely

        pipeline.run_cycle(0.0)
        detections = detection_provider.detections_at("CAM-001", 0.0)

        self.assertEqual(len(detections), 1)


if __name__ == "__main__":
    unittest.main()
