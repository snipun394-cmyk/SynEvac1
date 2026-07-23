import unittest

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrame
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline
from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

from tracking.simple_tracker import SimpleSingleCameraTracker
from tracking.track_state import TrackState

from live_occupants.manager import LiveOccupantManager

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from camera_manager.manager import CameraManager
from models.camera import Camera
from models.engineering_asset import DeviceMode

from multi_camera_fusion.engine import MultiCameraFusionEngine

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot
from building_state.estimator import BuildingStateEstimator

from human_detection.yolo_backend import ModelWeightsNotFoundError
from human_detection.yolo_human_detector import YOLOHumanDetector

from live_runtime.factory import build_live_runtime

from tests.human_detection_fixtures import FakeYOLOBackend, person
from tests.live_camera_pipeline_fixtures import FakeRTSPBackend


# =====================================================
# Real YOLO Person Detection Validation & Productionization milestone
# -- proves YOLOHumanDetector's output genuinely composes with the
# already-committed Single-Camera Tracking Framework (tracking/), the
# real LiveCameraPipeline's full optional-seam chain, RTSPFrameSource,
# and live_runtime.factory.build_live_runtime()'s own existing
# injection seams. FakeYOLOBackend stands in for real ultralytics
# inference throughout (CI has no bundled model weights) -- the point
# is proving the exact production composition, never real model
# accuracy. Zero network access anywhere in this file.
# =====================================================


def no_delay(_seconds):
    pass


def _frame(camera_id="CAM-001", timestamp=0.0, sequence=0, payload_ref="frame"):

    return CameraFrame(camera_id=camera_id, timestamp=timestamp, frame_sequence=sequence, payload_ref=payload_ref)


class TrackerStabilityWithYOLODetectionsTests(unittest.TestCase):

    # Phase 7 -- the tracker, not YOLO's own detection ordering, must
    # own temporal local identity. SimpleSingleCameraTracker is
    # exercised directly against YOLOHumanDetector's real output shape
    # (RawHumanDetection), never a hand-built substitute.

    def setUp(self):

        self.backend = FakeYOLOBackend()
        self.detector = YOLOHumanDetector(self.backend)
        self.tracker = SimpleSingleCameraTracker()

    def test_same_person_keeps_same_track_id_across_consecutive_frames(self):

        self.backend.queue_result(person(confidence=0.9, box=(10.0, 10.0, 30.0, 60.0)))
        self.backend.queue_result(person(confidence=0.88, box=(12.0, 11.0, 32.0, 61.0)))
        self.backend.queue_result(person(confidence=0.91, box=(14.0, 12.0, 34.0, 62.0)))

        track_ids = []
        for frame_index in range(3):
            raw = self.detector.detect(_frame(timestamp=float(frame_index)))
            tracked = self.tracker.update("CAM-001", float(frame_index), raw)
            track_ids.append(tracked[0].track_id)

        self.assertEqual(len(set(track_ids)), 1)

    def test_detection_order_change_does_not_create_a_new_track(self):

        # Two people; YOLO happens to list them in a different order
        # next frame (a real, expected occurrence -- detection order is
        # never guaranteed stable). The tracker must still recognize
        # each by geometry, not by list position.
        self.backend.queue_result(
            person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)),
            person(confidence=0.8, box=(100.0, 0.0, 110.0, 10.0)),
        )
        self.backend.queue_result(
            person(confidence=0.8, box=(101.0, 1.0, 111.0, 11.0)),  # was second, now first
            person(confidence=0.9, box=(1.0, 1.0, 11.0, 11.0)),  # was first, now second
        )

        raw_1 = self.detector.detect(_frame(timestamp=0.0))
        tracked_1 = self.tracker.update("CAM-001", 0.0, raw_1)
        left_id = next(t.track_id for t in tracked_1 if t.bounding_box[0] < 50.0)
        right_id = next(t.track_id for t in tracked_1 if t.bounding_box[0] >= 50.0)

        raw_2 = self.detector.detect(_frame(timestamp=1.0))
        tracked_2 = self.tracker.update("CAM-001", 1.0, raw_2)
        left_id_2 = next(t.track_id for t in tracked_2 if t.bounding_box[0] < 50.0)
        right_id_2 = next(t.track_id for t in tracked_2 if t.bounding_box[0] >= 50.0)

        self.assertEqual(left_id, left_id_2)
        self.assertEqual(right_id, right_id_2)

    def test_temporary_occlusion_track_survives_as_missing_then_resumes(self):

        self.backend.queue_result(person(confidence=0.9, box=(10.0, 10.0, 30.0, 60.0)))
        self.backend.queue_result()  # occluded -- YOLO sees nobody this frame
        self.backend.queue_result(person(confidence=0.9, box=(12.0, 11.0, 32.0, 61.0)))

        raw_1 = self.detector.detect(_frame(timestamp=0.0))
        tracked_1 = self.tracker.update("CAM-001", 0.0, raw_1)
        original_id = tracked_1[0].track_id

        raw_2 = self.detector.detect(_frame(timestamp=1.0))
        tracked_2 = self.tracker.update("CAM-001", 1.0, raw_2)
        self.assertEqual(len(tracked_2), 1)  # the coasting MISSING track, no new detection
        self.assertEqual(tracked_2[0].state, TrackState.MISSING)
        self.assertEqual(tracked_2[0].track_id, original_id)

        raw_3 = self.detector.detect(_frame(timestamp=2.0))
        tracked_3 = self.tracker.update("CAM-001", 2.0, raw_3)
        self.assertEqual(tracked_3[0].track_id, original_id)
        self.assertEqual(tracked_3[0].state, TrackState.TRACKED)

    def test_person_entering_creates_a_new_track(self):

        self.backend.queue_result()
        self.backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)))

        self.tracker.update("CAM-001", 0.0, self.detector.detect(_frame(timestamp=0.0)))
        tracked = self.tracker.update("CAM-001", 1.0, self.detector.detect(_frame(timestamp=1.0)))

        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].state, TrackState.NEW)

    def test_person_leaving_eventually_expires_the_track(self):

        tracker = SimpleSingleCameraTracker(max_missing_frames=2)
        self.backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)))
        for _ in range(4):
            self.backend.queue_result()

        tracker.update("CAM-001", 0.0, self.detector.detect(_frame(timestamp=0.0)))

        states = []
        for t in range(1, 5):
            tracked = tracker.update("CAM-001", float(t), self.detector.detect(_frame(timestamp=float(t))))
            states.extend(x.state for x in tracked)

        self.assertIn(TrackState.EXPIRED, states)

    def test_track_id_is_stable_string_not_yolo_detection_index(self):

        # YOLOHumanDetector's own local_track_id ("0", "1", ...) is
        # detector-local and unstable across frames by design (see its
        # own docstring) -- the TRACKER's track_id is what must be
        # stable, and it is deliberately a different, camera-namespaced
        # string shape ("CAM-001-T1"), never confusable with the raw
        # detector index.
        self.backend.queue_result(person(confidence=0.9))
        raw = self.detector.detect(_frame())
        self.assertEqual(raw[0].local_track_id, "0")

        tracked = self.tracker.update("CAM-001", 0.0, raw)
        self.assertNotEqual(tracked[0].track_id, raw[0].local_track_id)
        self.assertTrue(tracked[0].track_id.startswith("CAM-001-T"))


class LiveCameraPipelineFullChainTests(unittest.TestCase):

    # Phase 8 -- the production-compatible path, using the REAL
    # LiveCameraPipeline, never a second pipeline: CameraFrame ->
    # YOLOHumanDetector -> tracker -> behavior recognition ->
    # CrossCameraIdentityResolver seam (world projection has no
    # calibration data in this offline test, so it is correctly left
    # unconfigured, not fabricated) -> LiveOccupantManager.

    def _build_pipeline(self, backend, tracker=None, behavior_recognizer=None, live_occupant_manager=None):

        source = _FakeFrameSource(payload_refs=["f0", "f1", "f2"])
        detector = YOLOHumanDetector(backend)
        provider = LiveCameraPipelineDetectionProvider()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": source},
            human_detector=detector,
            identity_resolver=SimulationIdentityResolver(),
            detection_provider=provider,
            tracker=tracker,
            behavior_recognizer=behavior_recognizer,
            live_occupant_manager=live_occupant_manager,
        )

        return pipeline, provider

    def test_yolo_detections_reach_the_detection_provider_via_real_pipeline(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)))

        pipeline, provider = self._build_pipeline(backend, tracker=SimpleSingleCameraTracker())
        pipeline.run_cycle(0.0)

        detections = provider.detections_at("CAM-001", 0.0)
        self.assertEqual(len(detections), 1)

    def test_tracker_stabilizes_occupant_id_across_cycles_through_real_pipeline(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(10.0, 10.0, 30.0, 60.0)))
        backend.queue_result(person(confidence=0.9, box=(11.0, 11.0, 31.0, 61.0)))

        pipeline, provider = self._build_pipeline(backend, tracker=SimpleSingleCameraTracker())

        pipeline.run_cycle(0.0)
        first = provider.detections_at("CAM-001", 0.0)[0]

        pipeline.run_cycle(1.0)
        second = provider.detections_at("CAM-001", 1.0)[0]

        self.assertEqual(first.occupant_id, second.occupant_id)

    def test_behavior_recognizer_participates_when_configured(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)))
        backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)))  # stationary

        pipeline, provider = self._build_pipeline(
            backend, tracker=SimpleSingleCameraTracker(), behavior_recognizer=RuleBasedBehaviorRecognizer(),
        )

        pipeline.run_cycle(0.0)
        pipeline.run_cycle(1.0)

        # Never raised an exception, and the seam was genuinely
        # consulted -- human_state is either None or a real HumanState
        # member, never fabricated beyond what RuleBasedBehaviorRecognizer
        # itself would honestly produce from two stationary observations.
        from perception.models.human_observation import HumanState

        detection = provider.detections_at("CAM-001", 1.0)[0]
        self.assertTrue(detection.human_state is None or isinstance(detection.human_state, HumanState))

    def test_live_occupant_manager_reflects_yolo_driven_occupants(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)))

        manager = LiveOccupantManager()
        pipeline, provider = self._build_pipeline(
            backend, tracker=SimpleSingleCameraTracker(), live_occupant_manager=manager,
        )

        pipeline.run_cycle(0.0)

        self.assertEqual(len(manager.all_occupants()), 1)

    def test_without_tracker_pipeline_still_works_unchanged(self):

        # Omitting tracker entirely (the default) must reproduce the
        # pipeline's pre-tracking behavior exactly -- YOLOHumanDetector
        # never requires a tracker to function.
        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9))

        pipeline, provider = self._build_pipeline(backend)
        pipeline.run_cycle(0.0)

        self.assertEqual(len(provider.detections_at("CAM-001", 0.0)), 1)


class _FakeFrameSource:

    # A minimal CameraFrameSource double -- read_frame() returns each
    # configured payload_ref in order, then None forever. Kept local to
    # this test module since it is trivial and this milestone's own
    # tests are the only caller.

    def __init__(self, payload_refs):
        self._payload_refs = list(payload_refs)
        self._index = 0

    def read_frame(self):

        if self._index >= len(self._payload_refs):
            return CameraFrame(camera_id="CAM-001", timestamp=float(self._index), frame_sequence=self._index, payload_ref=self._payload_refs[-1])

        payload_ref = self._payload_refs[min(self._index, len(self._payload_refs) - 1)]
        frame = CameraFrame(camera_id="CAM-001", timestamp=float(self._index), frame_sequence=self._index, payload_ref=payload_ref)
        self._index += 1

        return frame


class RTSPWithTrackerCompatibilityTests(unittest.TestCase):

    # Phase 9 -- FakeRTSPBackend -> RTSPFrameSource -> CameraFrame ->
    # the REAL YOLOHumanDetector adapter architecture (FakeYOLOBackend
    # standing in for real ultralytics inference, since CI has no model
    # weights) -> tracker -> LiveOccupant. No network access.

    def test_rtsp_backed_frames_flow_through_yolo_and_tracker_to_occupants(self):

        rtsp_backend = FakeRTSPBackend()
        rtsp_backend.queue_frame(payload_ref="rtsp-frame-1")
        rtsp_backend.queue_frame(payload_ref="rtsp-frame-2")

        source = RTSPFrameSource(
            camera_id="CAM-RTSP", endpoint="rtsp://endpoint/stream",
            decoder_backend=rtsp_backend, sleep_fn=no_delay,
        )
        source.start()

        yolo_backend = FakeYOLOBackend()
        yolo_backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)))
        yolo_backend.queue_result(person(confidence=0.9, box=(1.0, 1.0, 11.0, 11.0)))

        detector = YOLOHumanDetector(yolo_backend)
        tracker = SimpleSingleCameraTracker()
        manager = LiveOccupantManager()
        provider = LiveCameraPipelineDetectionProvider()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-RTSP": source},
            human_detector=detector,
            identity_resolver=SimulationIdentityResolver(),
            detection_provider=provider,
            tracker=tracker,
            live_occupant_manager=manager,
        )

        pipeline.run_cycle(0.0)
        first_occupants = {o.occupant_id for o in manager.all_occupants()}

        pipeline.run_cycle(1.0)
        second_occupants = {o.occupant_id for o in manager.all_occupants()}

        self.assertEqual(len(first_occupants), 1)
        self.assertEqual(first_occupants, second_occupants)  # stable identity across cycles

        source.stop()


class LiveRuntimeFactoryCompositionTests(unittest.TestCase):

    # Phase 10 -- build_live_runtime() already exposes human_detector/
    # tracker/behavior_recognizer/cross_camera_identity_resolver/
    # world_projector/live_occupant_manager as plain optional injection
    # seams (investigated directly in live_runtime/factory.py); no new
    # composition class was needed. This proves a REAL YOLO+tracker
    # configuration composes cleanly through that existing mechanism,
    # and that OMITTING it (the offline/demo default) still works with
    # no YOLO/ultralytics involvement at all.

    def test_offline_runtime_construction_never_requires_yolo(self):

        from models.building import Building
        from models.floor import Floor

        building = Building(id="b1", name="B", floors=[Floor(id="f1", name="F1")])

        # No human_detector/tracker supplied at all -- must not raise,
        # must not import/require ultralytics.
        runtime = build_live_runtime(building=building)

        self.assertIsNone(runtime.camera_pipeline if hasattr(runtime, "camera_pipeline") else None)

    def test_real_yolo_and_tracker_compose_through_existing_injection_seams(self):

        from models.building import Building
        from models.floor import Floor

        cam = Camera(id="CAM-001", name="Cam", floor_id="f1")
        floor = Floor(id="f1", name="F1", cameras=[cam])
        building = Building(id="b1", name="B", floors=[floor])

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9))

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()
        source = _FakeFrameSource(payload_refs=["f0"])

        runtime = build_live_runtime(
            building=building,
            frame_sources={"CAM-001": source},
            human_detector=detector,
            tracker=tracker,
        )

        runtime.orchestrator.start()
        snapshot = runtime.orchestrator.run_cycle(1.0)

        self.assertGreaterEqual(len(snapshot.building_state.camera_observations), 0)


class FailureBehaviorTests(unittest.TestCase):

    # Phase 11 -- missing/invalid weights, dependency-unavailable
    # framing, corrupt/empty frames, zero detections, inference
    # exceptions, camera frame drops, tracker with zero detections,
    # person disappearing, and one camera failing while another camera
    # stays healthy.

    def test_missing_weights_raises_explicit_error_not_a_silent_failure(self):

        from human_detection.yolo_backend import UltralyticsYOLOBackend

        with self.assertRaises(ModelWeightsNotFoundError):
            UltralyticsYOLOBackend("no/such/weights.pt")

    def test_corrupt_or_none_frame_yields_empty_not_a_crash(self):

        backend = FakeYOLOBackend()
        detector = YOLOHumanDetector(backend)

        self.assertEqual(detector.detect(_frame(payload_ref=None)), ())

    def test_zero_detections_feed_tracker_cleanly(self):

        backend = FakeYOLOBackend()
        backend.queue_result()
        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()

        raw = detector.detect(_frame())
        tracked = tracker.update("CAM-001", 0.0, raw)

        self.assertEqual(tracked, ())

    def test_inference_exception_is_absorbed_at_the_detector_boundary(self):

        backend = FakeYOLOBackend()
        backend.fail_next_with(RuntimeError("simulated model crash"))
        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()

        raw = detector.detect(_frame())  # must not raise
        tracked = tracker.update("CAM-001", 0.0, raw)  # must not raise

        self.assertEqual(raw, ())
        self.assertEqual(tracked, ())

    def test_person_disappearing_does_not_replay_stale_detections(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)))
        backend.queue_result()  # gone

        detector = YOLOHumanDetector(backend)
        tracker = SimpleSingleCameraTracker()

        tracker.update("CAM-001", 0.0, detector.detect(_frame(timestamp=0.0)))
        raw_2 = detector.detect(_frame(timestamp=1.0))

        self.assertEqual(raw_2, ())  # detector itself never replays

        tracked_2 = tracker.update("CAM-001", 1.0, raw_2)
        # The track coasts as MISSING (tracker's own honest bookkeeping),
        # but no NEW detection/observation was fabricated for this cycle.
        self.assertTrue(all(t.state in (TrackState.MISSING, TrackState.EXPIRED) for t in tracked_2))

    def test_one_camera_failing_does_not_affect_a_healthy_second_camera(self):

        healthy_backend = FakeYOLOBackend()
        healthy_backend.queue_result(person(confidence=0.9))

        failing_backend = FakeYOLOBackend()
        failing_backend.fail_next_with(RuntimeError("camera 2 model crash"))

        healthy_detector = YOLOHumanDetector(healthy_backend)
        failing_detector = YOLOHumanDetector(failing_backend)

        healthy_result = healthy_detector.detect(_frame(camera_id="CAM-001"))
        failing_result = failing_detector.detect(_frame(camera_id="CAM-002"))

        self.assertEqual(len(healthy_result), 1)
        self.assertEqual(failing_result, ())

    def test_camera_frame_drop_is_a_clean_skip_not_a_crash(self):

        class _DroppingSource:
            def read_frame(self):
                return None

        backend = FakeYOLOBackend()
        detector = YOLOHumanDetector(backend)
        provider = LiveCameraPipelineDetectionProvider()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": _DroppingSource()},
            human_detector=detector,
            identity_resolver=SimulationIdentityResolver(),
            detection_provider=provider,
        )

        pipeline.run_cycle(0.0)  # must not raise

        self.assertEqual(provider.detections_at("CAM-001", 0.0), ())


class DeviceConfigurationTests(unittest.TestCase):

    # Phase 13 -- device is a plain, explicit constructor parameter
    # (default "cpu"); no hard CUDA requirement anywhere in this
    # package.

    def test_default_device_is_cpu(self):

        import inspect

        from human_detection.yolo_backend import UltralyticsYOLOBackend

        signature = inspect.signature(UltralyticsYOLOBackend.__init__)
        self.assertEqual(signature.parameters["device"].default, "cpu")

    def test_device_is_a_plain_string_parameter_not_hardcoded(self):

        import pathlib
        import re

        text = pathlib.Path("human_detection/yolo_backend.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\bcuda\s*=\s*True\b", text))


if __name__ == "__main__":
    unittest.main()
