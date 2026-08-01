import unittest

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot

from perception.models.human_observation import HumanClassification, HumanState

from models.engineering_asset import DeviceMode

from camera_manager.manager import CameraManager

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrame, CameraFrameSource
from live_camera_pipeline.human_detector import HumanDetector, RawHumanDetection
from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline

from multi_camera_fusion.engine import MultiCameraFusionEngine

from building_state.estimator import BuildingStateEstimator


class FakeFrameSource(CameraFrameSource):

    # A test-only stand-in -- deliberately lives in the test module,
    # not in live_camera_pipeline/, so the production package stays
    # pure interface (Phase 16). Yields one queued frame per
    # read_frame() call, then None -- exactly the shape a real
    # RTSPFrameSource would have once it exists.

    def __init__(self, camera_id):

        self.camera_id = camera_id
        self._queue = []
        self._running = False

    def queue_frame(self, timestamp, frame_sequence, payload_ref=None):

        self._queue.append(
            CameraFrame(
                camera_id=self.camera_id,
                timestamp=timestamp,
                frame_sequence=frame_sequence,
                payload_ref=payload_ref,
            )
        )

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    @property
    def is_running(self):
        return self._running

    def read_frame(self):
        return self._queue.pop(0) if self._queue else None


class FakeHumanDetector(HumanDetector):

    # A test-only stand-in for a real vision model -- maps a
    # CameraFrame's payload_ref (here, just a plain dict the test
    # controls) straight to RawHumanDetection(s), with no image
    # processing of any kind.

    def detect(self, frame):

        if frame.payload_ref is None:
            return ()

        return tuple(
            RawHumanDetection(
                camera_id=frame.camera_id,
                local_track_id=entry["local_track_id"],
                timestamp=frame.timestamp,
                confidence=0.9,
                classification_evidence=HumanClassification.ADULT,
                state_evidence=HumanState.WALKING,
                floor_id="floor-1",
                zone_id=entry.get("zone_id", "zone-1"),
            )
            for entry in frame.payload_ref
        )


def estimate_building_state(fusion_result, timestamp):

    return BuildingStateEstimator().estimate(
        timestamp,
        hazard_snapshot=HazardSnapshot(),
        occupancy_snapshot=OccupancySnapshot(),
        fusion_result=fusion_result,
    )


class LiveCameraPipelineHandoverTests(unittest.TestCase):

    # Phase 13's full fake/in-memory live test -- CameraFrame all the
    # way through to BuildingState occupancy, zero real camera, zero
    # Simulation ground truth involved (a fresh MappingIdentityResolver
    # is the only identity source, exercised explicitly rather than
    # inherited from Simulation).

    def setUp(self):

        self.cam_a = FakeFrameSource("CAM-A")
        self.cam_b = FakeFrameSource("CAM-B")

        self.resolver = MappingIdentityResolver({
            ("CAM-A", "17"): "GLOBAL-001",
            ("CAM-B", "9"): "GLOBAL-001",
        })

        self.detection_provider = LiveCameraPipelineDetectionProvider()

        self.pipeline = LiveCameraPipeline(
            frame_sources={"CAM-A": self.cam_a, "CAM-B": self.cam_b},
            human_detector=FakeHumanDetector(),
            identity_resolver=self.resolver,
            detection_provider=self.detection_provider,
        )

        self.camera_manager = CameraManager()
        self.camera_manager.register_detection_provider(DeviceMode.LIVE, self.detection_provider)

        self.fusion_engine = MultiCameraFusionEngine()

    # =====================================================

    def _tick(self, time):

        self.pipeline.run_cycle(time)

        detections = (
            self.detection_provider.detections_at("CAM-A", time)
            + self.detection_provider.detections_at("CAM-B", time)
        )

        fusion_result = self.fusion_engine.fuse(detections, time)

        return estimate_building_state(fusion_result, time)

    # =====================================================

    def test_two_cameras_one_person_resolved_to_one_global_id_gives_occupancy_one(self):

        self.cam_a.queue_frame(1.0, 1, payload_ref=[{"local_track_id": "17"}])
        self.cam_b.queue_frame(1.0, 1, payload_ref=[{"local_track_id": "9"}])

        state = self._tick(1.0)

        self.assertEqual(len(state.occupant_tracks), 1)
        self.assertIn("GLOBAL-001", state.occupant_tracks)
        self.assertEqual(
            set(state.occupant_tracks["GLOBAL-001"].source_camera_ids), {"CAM-A", "CAM-B"},
        )

    def test_two_different_people_with_no_mapping_gives_occupancy_two(self):

        self.cam_a.queue_frame(1.0, 1, payload_ref=[{"local_track_id": "1"}])
        self.cam_b.queue_frame(1.0, 1, payload_ref=[{"local_track_id": "2"}])

        state = self._tick(1.0)

        self.assertEqual(len(state.occupant_tracks), 2)

    def test_camera_handover_stays_one_persistent_track(self):

        # t=0: only CAM-A sees the person.
        self.cam_a.queue_frame(0.0, 1, payload_ref=[{"local_track_id": "17"}])
        self.cam_b.queue_frame(0.0, 1, payload_ref=[])

        state_t0 = self._tick(0.0)
        self.assertEqual(len(state_t0.occupant_tracks), 1)
        self.assertIn("GLOBAL-001", state_t0.occupant_tracks)

        # t=1: overlap -- both cameras see the same person.
        self.cam_a.queue_frame(1.0, 2, payload_ref=[{"local_track_id": "17"}])
        self.cam_b.queue_frame(1.0, 2, payload_ref=[{"local_track_id": "9"}])

        state_t1 = self._tick(1.0)
        self.assertEqual(len(state_t1.occupant_tracks), 1)
        self.assertEqual(
            set(state_t1.occupant_tracks["GLOBAL-001"].source_camera_ids), {"CAM-A", "CAM-B"},
        )

        # t=2: only CAM-B sees the person now.
        self.cam_a.queue_frame(2.0, 3, payload_ref=[])
        self.cam_b.queue_frame(2.0, 3, payload_ref=[{"local_track_id": "9"}])

        state_t2 = self._tick(2.0)
        self.assertEqual(len(state_t2.occupant_tracks), 1)
        self.assertIn("GLOBAL-001", state_t2.occupant_tracks)

        # The fusion engine's own history recorded the handover.
        history = self.fusion_engine.track_history("GLOBAL-001")
        self.assertEqual(history.previous_camera_id, "CAM-A")
        self.assertEqual(history.current_camera_id, "CAM-B")

    def test_no_frames_produces_no_tracks(self):

        state = self._tick(1.0)
        self.assertEqual(len(state.occupant_tracks), 0)

    def test_stale_detection_does_not_survive_a_frame_with_no_detections(self):

        self.cam_a.queue_frame(1.0, 1, payload_ref=[{"local_track_id": "17"}])
        self.cam_b.queue_frame(1.0, 1, payload_ref=[])
        state_t1 = self._tick(1.0)
        self.assertEqual(len(state_t1.occupant_tracks), 1)

        # Next cycle CAM-A sees nobody -- its buffered detection must
        # clear, not keep reporting the person forever.
        self.cam_a.queue_frame(2.0, 2, payload_ref=[])
        self.cam_b.queue_frame(2.0, 2, payload_ref=[])
        state_t2 = self._tick(2.0)
        self.assertEqual(len(state_t2.occupant_tracks), 0)


class LiveCameraPipelineDependencyDirectionTests(unittest.TestCase):

    # Phase 16: the future frame/detection/identity-resolution seam
    # must never reach into CameraManager, MultiCameraFusionEngine, or
    # BuildingState itself -- a caller wires those, this package never
    # does. Same regex-scan convention every other package boundary in
    # this codebase already uses (see
    # tests.test_camera_manager.CameraManagerPackageDependencyDirectionTests).

    def test_live_camera_pipeline_never_imports_downstream_packages(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "live_camera_pipeline"

        forbidden = r"^\s*(from|import)\s+(camera_manager|multi_camera_fusion|building_state)\b"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"live_camera_pipeline/{path.name} imports a downstream package directly -- "
                f"it must only produce Detections; a caller wires CameraManager/"
                f"MultiCameraFusionEngine/BuildingState itself",
            )


class LiveCameraPipelineLatestFrameCacheTests(unittest.TestCase):

    # Live Camera Viewer milestone -- proves the single-slot
    # latest_frame() cache added to run_cycle() above without touching
    # any existing detection/tracking/fusion behavior: read_frame() is
    # still called exactly once per camera per cycle (no second
    # consumer, no second decode), a genuine new frame overwrites the
    # slot, and a transient miss (queue exhausted) leaves the previous
    # frame in place rather than clearing it.

    def setUp(self):

        self.cam_a = FakeFrameSource("CAM-A")
        self.pipeline = LiveCameraPipeline(
            frame_sources={"CAM-A": self.cam_a},
            human_detector=FakeHumanDetector(),
            identity_resolver=MappingIdentityResolver({}),
            detection_provider=LiveCameraPipelineDetectionProvider(),
        )

    def test_latest_frame_is_none_before_any_cycle(self):

        self.assertIsNone(self.pipeline.latest_frame("CAM-A"))

    def test_latest_frame_is_none_for_an_unknown_camera_id(self):

        self.cam_a.queue_frame(1.0, 1, payload_ref=[])
        self.pipeline.run_cycle(1.0)

        self.assertIsNone(self.pipeline.latest_frame("CAM-B"))

    def test_latest_frame_reflects_the_most_recently_read_frame(self):

        self.cam_a.queue_frame(1.0, 1, payload_ref=[])
        self.pipeline.run_cycle(1.0)

        frame = self.pipeline.latest_frame("CAM-A")
        self.assertIsNotNone(frame)
        self.assertEqual(frame.timestamp, 1.0)
        self.assertEqual(frame.frame_sequence, 1)

        self.cam_a.queue_frame(2.0, 2, payload_ref=[])
        self.pipeline.run_cycle(2.0)

        self.assertEqual(self.pipeline.latest_frame("CAM-A").frame_sequence, 2)

    def test_a_transient_miss_leaves_the_previous_frame_cached(self):

        self.cam_a.queue_frame(1.0, 1, payload_ref=[])
        self.pipeline.run_cycle(1.0)

        # Nothing queued this cycle -- read_frame() returns None.
        self.pipeline.run_cycle(2.0)

        frame = self.pipeline.latest_frame("CAM-A")
        self.assertIsNotNone(frame)
        self.assertEqual(frame.frame_sequence, 1)

    def test_run_cycle_still_calls_read_frame_exactly_once_per_camera(self):

        # Regression guard: adding the cache must never introduce a
        # second read of the same frame source.

        call_count = {"count": 0}
        original_read_frame = self.cam_a.read_frame

        def counting_read_frame():
            call_count["count"] += 1
            return original_read_frame()

        self.cam_a.read_frame = counting_read_frame
        self.cam_a.queue_frame(1.0, 1, payload_ref=[])

        self.pipeline.run_cycle(1.0)

        self.assertEqual(call_count["count"], 1)


if __name__ == "__main__":
    unittest.main()
