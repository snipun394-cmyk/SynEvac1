import unittest

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrame, CameraFrameSource
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from cross_camera_identity.identity_registry import IdentityRegistry
from cross_camera_identity.resolver import RuleBasedCrossCameraIdentityResolver
from cross_camera_identity.topology import CameraTopology
from cross_camera_identity.transition_model import TransitionModel

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Cross-Camera Identity Resolution (ReID Framework) milestone, Phase 7
# -- proves the actual integration point: LiveCameraPipeline.run_cycle(),
# feeding a stable GLOBAL occupant id into Detection.occupant_id via
# SimulationIdentityResolver's own EXISTING, unmodified "local_track_id
# IS already the global identity" passthrough strategy. Two separate
# LiveCameraPipeline instances (one per camera, since LiveCameraPipeline
# itself takes one human_detector/tracker each) SHARE one
# CrossCameraIdentityResolver instance and one LiveCameraPipelineDetection
# Provider -- exactly how a real multi-camera deployment would wire it.
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


class CrossCameraOccupantIdReachesDetectionTests(unittest.TestCase):

    def test_same_person_on_two_cameras_gets_the_same_detection_occupant_id(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", min_transition_time=0.0, max_transition_time=100.0)

        registry = IdentityRegistry()
        transition_model = TransitionModel(topology, timeout_seconds=60.0)
        cross_camera_resolver = RuleBasedCrossCameraIdentityResolver(
            topology=topology, registry=registry, transition_model=transition_model,
        )

        detection_provider = LiveCameraPipelineDetectionProvider()

        # --- Camera A: person appears, then leaves ---
        backend_a = FakeYOLOBackend()
        backend_a.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 20.0)))
        backend_a.queue_result()  # gone -- tracker will eventually expire this local track

        source_a = SequencedFrameSource("CAM-A", [(0.0, "f0"), (1.0, "f1")])
        source_a.start()

        pipeline_a = LiveCameraPipeline(
            frame_sources={"CAM-A": source_a},
            human_detector=YOLOHumanDetector(backend_a),
            identity_resolver=SimulationIdentityResolver(),
            detection_provider=detection_provider,
            tracker=SimpleSingleCameraTracker(max_missing_frames=0),  # expire immediately on first miss
            cross_camera_identity_resolver=cross_camera_resolver,
        )

        pipeline_a.run_cycle(0.0)
        cam_a_detections = detection_provider.detections_at("CAM-A", 0.0)
        global_id = cam_a_detections[0].occupant_id

        pipeline_a.run_cycle(1.0)  # local track expires -- releases the binding

        # --- Camera B: a different physical camera, person arrives ---
        backend_b = FakeYOLOBackend()
        backend_b.queue_result(person(confidence=0.85, box=(0.0, 0.0, 10.0, 20.0)))

        source_b = SequencedFrameSource("CAM-B", [(5.0, "f0")])
        source_b.start()

        pipeline_b = LiveCameraPipeline(
            frame_sources={"CAM-B": source_b},
            human_detector=YOLOHumanDetector(backend_b),
            identity_resolver=SimulationIdentityResolver(),
            detection_provider=detection_provider,
            tracker=SimpleSingleCameraTracker(),
            cross_camera_identity_resolver=cross_camera_resolver,
        )

        pipeline_b.run_cycle(5.0)
        cam_b_detections = detection_provider.detections_at("CAM-B", 5.0)

        self.assertEqual(cam_b_detections[0].occupant_id, global_id)
        self.assertTrue(global_id.startswith("OCC-"))
        # Never a raw tracker-local id (which would look like
        # "CAM-A-T1") leaking through as the occupant_id.
        self.assertNotIn("CAM-A-T", global_id)
        self.assertNotIn("CAM-B-T", global_id)


class NoCrossCameraResolverPreservesTrackerOnlyBehaviorTests(unittest.TestCase):

    # Backward compatibility -- omitting `cross_camera_identity_resolver`
    # (the default) must reproduce the Single-Camera Tracking Framework
    # milestone's exact behavior: Detection.occupant_id derived from the
    # tracker's own LOCAL track_id only (via MappingIdentityResolver's
    # synthetic fallback), never a global id.

    def test_without_cross_camera_resolver_occupant_id_is_camera_local(self):

        from live_camera_pipeline.identity_resolver import MappingIdentityResolver

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
        )  # cross_camera_identity_resolver omitted entirely

        pipeline.run_cycle(0.0)
        detections = detection_provider.detections_at("CAM-001", 0.0)

        self.assertEqual(detections[0].occupant_id, "CAM-001:CAM-001-T1")


if __name__ == "__main__":
    unittest.main()
