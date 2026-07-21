import unittest

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrame, CameraFrameSource
from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Single-Camera Tracking Framework milestone, Phase 7 -- proves the
# tracker's actual integration point: LiveCameraPipeline.run_cycle(),
# between YOLOHumanDetector.detect() and IdentityResolver.resolve().
# IdentityResolver itself is the unmodified, existing
# MappingIdentityResolver -- this test's whole point is that a stable
# track_id (not a per-frame detector index) is what now reaches it.
# =====================================================


class SequencedFrameSource(CameraFrameSource):

    # A minimal, deterministic, in-memory CameraFrameSource -- avoids
    # any dependency on ReplayFrameSource/RTSPFrameSource specifics;
    # this test only needs read_frame() to hand back pre-built frames
    # in order.

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


class TrackerStabilizesIdentityAcrossCyclesTests(unittest.TestCase):

    def setUp(self):

        self.backend = FakeYOLOBackend()
        # Same one person, same approximate position, across 3 frames --
        # YOLOHumanDetector on its own would assign local_track_id "0"
        # every single cycle (a coincidence of always being the first/
        # only detection), so this scenario alone would not prove
        # anything about instability. The real proof is in the second
        # test below, with a person who ENTERS after cycle 1 pushes the
        # raw per-frame index around.
        self.backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)))
        self.backend.queue_result(person(confidence=0.85, box=(2.0, 0.0, 12.0, 10.0)))
        self.backend.queue_result(person(confidence=0.8, box=(4.0, 0.0, 14.0, 10.0)))

        self.detector = YOLOHumanDetector(self.backend)
        self.tracker = SimpleSingleCameraTracker()
        self.resolver = MappingIdentityResolver()
        self.detection_provider = LiveCameraPipelineDetectionProvider()

        self.source = SequencedFrameSource("CAM-001", [
            (0.0, "frame-0"), (1.0, "frame-1"), (2.0, "frame-2"),
        ])
        self.source.start()

        self.pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": self.source},
            human_detector=self.detector,
            identity_resolver=self.resolver,
            detection_provider=self.detection_provider,
            tracker=self.tracker,
        )

    def test_stable_track_id_reaches_identity_resolver_across_cycles(self):

        self.pipeline.run_cycle(0.0)
        first = self.detection_provider.detections_at("CAM-001", 0.0)

        self.pipeline.run_cycle(1.0)
        second = self.detection_provider.detections_at("CAM-001", 1.0)

        self.pipeline.run_cycle(2.0)
        third = self.detection_provider.detections_at("CAM-001", 2.0)

        # Same physical person, same resolved occupant_id, all 3 cycles --
        # the tracker's stable track_id, not a raw per-frame index, is
        # what MappingIdentityResolver's fallback synthesized this from.
        self.assertEqual(first[0].occupant_id, second[0].occupant_id)
        self.assertEqual(second[0].occupant_id, third[0].occupant_id)


class TrackerCorrectsForRawIndexInstabilityTests(unittest.TestCase):

    # The scenario that actually demonstrates WHY this milestone
    # matters: without a tracker, YOLOHumanDetector's own local_track_id
    # is nothing more than "0, 1, 2, ... in this frame's detection
    # order" -- if a second person enters and happens to be listed
    # before the first person by the (fake, but representative) model
    # backend, the ORIGINAL person's raw local_track_id would shift
    # from "0" to "1" between cycles, and MappingIdentityResolver's
    # synthetic fallback (f"{camera_id}:{local_track_id}") would
    # incorrectly treat them as two different people. With the tracker
    # in place, geometry-based matching keeps the original person's
    # track_id -- and therefore resolved occupant_id -- the same
    # regardless of detection-list order.

    def test_occupant_identity_survives_a_raw_detection_order_shuffle(self):

        backend = FakeYOLOBackend()
        # Cycle 1: only the original person, box near (0,0)-(10,10).
        backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)))
        # Cycle 2: a NEW person (box near (200,0)) is listed FIRST by
        # the backend, pushing the original person to raw index "1"
        # instead of "0".
        backend.queue_result(
            person(confidence=0.9, box=(200.0, 0.0, 210.0, 10.0)),
            person(confidence=0.85, box=(1.0, 0.0, 11.0, 10.0)),
        )

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()
        resolver = MappingIdentityResolver()
        detection_provider = LiveCameraPipelineDetectionProvider()

        source = SequencedFrameSource("CAM-001", [(0.0, "f0"), (1.0, "f1")])
        source.start()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=resolver,
            detection_provider=detection_provider,
            tracker=tracker,
        )

        pipeline.run_cycle(0.0)
        first_cycle = detection_provider.detections_at("CAM-001", 0.0)
        original_occupant_id = first_cycle[0].occupant_id

        pipeline.run_cycle(1.0)
        second_cycle = detection_provider.detections_at("CAM-001", 1.0)

        occupant_ids = {d.occupant_id for d in second_cycle}

        self.assertEqual(len(second_cycle), 2)
        self.assertIn(
            original_occupant_id, occupant_ids,
            "the original person's resolved occupant_id must survive a raw "
            "detection-order shuffle once a tracker stabilizes local_track_id",
        )


class NoTrackerPreservesExactPriorBehaviorTests(unittest.TestCase):

    # Backward compatibility -- omitting `tracker` (the default) must
    # reproduce this class's exact pre-tracking-milestone behavior, so
    # every existing caller/test (tests/test_live_camera_pipeline.py,
    # tests/test_rtsp_offline_e2e.py, tests/
    # test_yolo_rtsp_live_runtime_compatibility.py) continues to pass
    # completely unmodified.

    def test_pipeline_without_a_tracker_uses_the_raw_detector_local_track_id(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)))

        detector = YOLOHumanDetector(backend)
        resolver = MappingIdentityResolver()
        detection_provider = LiveCameraPipelineDetectionProvider()

        source = SequencedFrameSource("CAM-001", [(0.0, "f0")])
        source.start()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=resolver,
            detection_provider=detection_provider,
        )  # tracker omitted entirely

        pipeline.run_cycle(0.0)
        detections = detection_provider.detections_at("CAM-001", 0.0)

        # MappingIdentityResolver's synthetic fallback for an unmapped
        # (camera_id, local_track_id) pair is f"{camera_id}:{local_track_id}"
        # -- YOLOHumanDetector's own raw index for the first (only)
        # detection in a frame is always "0".
        self.assertEqual(detections[0].occupant_id, "CAM-001:0")


if __name__ == "__main__":
    unittest.main()
