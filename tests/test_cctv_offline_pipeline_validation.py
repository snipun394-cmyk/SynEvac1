import tempfile
import unittest
from pathlib import Path

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot

from models.camera import Camera
from models.engineering_asset import ConnectionInfo, DeviceMode

from camera_manager.connection_status import CameraConnectionState
from camera_manager.manager import CameraManager

from credential_store.local_file_store import LocalFileCredentialStore

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrame
from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline
from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from multi_camera_fusion.engine import MultiCameraFusionEngine

from building_state.estimator import BuildingStateEstimator

from tests.live_camera_pipeline_fixtures import MockHumanDetector


# CCTV Pipeline End-to-End Offline Validation milestone.
#
# Proves the full data path -- Digital Twin Camera Asset -> Camera
# Manager -> Camera Mode -> Camera Source Adapter -> Frame Packet ->
# Detection Provider -> Human Detections -> Multi-Camera Fusion ->
# Building State -- works end-to-end using only offline/deterministic
# sources, so that when real CCTV access arrives the only remaining
# work is a real CameraFrameSource/HumanDetector implementation behind
# these exact same seams. No cv2, no RTSP, no real network connection,
# no YOLO anywhere in this file.


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


class MockDetectorCameraIdTests(unittest.TestCase):

    # Phase 9 item 5, standalone: MockHumanDetector -- the deterministic
    # stand-in for a real vision model -- must preserve camera_id on
    # every RawHumanDetection it produces, independent of the rest of
    # the pipeline.

    def test_detect_preserves_camera_id_from_the_frame(self):

        frame = CameraFrame(
            camera_id="CAM-999",
            timestamp=5.0,
            frame_sequence=0,
            payload_ref=[{"local_track_id": "1"}, {"local_track_id": "2"}],
        )

        detections = MockHumanDetector().detect(frame)

        self.assertEqual(len(detections), 2)

        for raw_detection in detections:
            self.assertEqual(raw_detection.camera_id, "CAM-999")

    def test_detect_with_no_payload_returns_no_detections(self):

        frame = CameraFrame(camera_id="CAM-999", timestamp=5.0, frame_sequence=0)

        self.assertEqual(MockHumanDetector().detect(frame), ())


class CameraIdentityContinuityTests(unittest.TestCase):

    # Phase 3: camera_id must survive, unchanged, through every stage:
    # Camera Asset -> Camera Manager -> Offline Source -> Frame Packet
    # -> Detection Provider -> Detection.camera_id -> Multi-Camera
    # Fusion provenance -> Building State.

    def setUp(self):

        self.camera = Camera(id="CAM-001", name="Lobby Cam", floor_id="floor-1")

        self.manager = CameraManager()
        self.manager.register_camera(self.camera)

        self.source = ReplayFrameSource(
            camera_id="CAM-001",
            frames=[(0.0, [{"local_track_id": "17"}])],
        )
        self.source.start()

        self.resolver = MappingIdentityResolver({("CAM-001", "17"): "OCC-1"})
        self.detection_provider = LiveCameraPipelineDetectionProvider()

        self.pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": self.source},
            human_detector=MockHumanDetector(),
            identity_resolver=self.resolver,
            detection_provider=self.detection_provider,
        )

        self.manager.register_detection_provider(DeviceMode.REPLAY, self.detection_provider)
        self.manager.set_camera_mode("CAM-001", DeviceMode.REPLAY)

        self.fusion_engine = MultiCameraFusionEngine()

    def test_camera_id_survives_the_entire_chain(self):

        detections, fusion_result, building_state = _run_one_cycle(
            self.pipeline, self.detection_provider, ["CAM-001"], self.fusion_engine, time=0.0,
        )

        # Detection Provider -> Detection.camera_id
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].camera_id, "CAM-001")

        # Multi-Camera Fusion provenance
        track = fusion_result.track("OCC-1")
        self.assertIsNotNone(track)
        self.assertIn("CAM-001", track.source_camera_ids)

        # Building State
        self.assertIn("OCC-1", building_state.occupant_tracks)
        self.assertIn("CAM-001", building_state.occupant_tracks["OCC-1"].source_camera_ids)

        # Camera Manager routing still resolves the same Camera Asset.
        self.assertIs(self.manager.get_camera("CAM-001"), self.camera)

    def test_camera_asset_identity_unchanged_when_source_changes(self):

        original_id = self.camera.id

        new_source = ReplayFrameSource(
            camera_id="CAM-001", frames=[(1.0, [{"local_track_id": "17"}])],
        )
        new_source.start()

        new_pipeline = LiveCameraPipeline(
            frame_sources={"CAM-001": new_source},
            human_detector=MockHumanDetector(),
            identity_resolver=self.resolver,
            detection_provider=self.detection_provider,
        )

        new_pipeline.run_cycle(1.0)

        camera_after = self.manager.get_camera("CAM-001")

        self.assertIs(camera_after, self.camera)
        self.assertEqual(camera_after.id, original_id)

    def test_camera_asset_identity_unchanged_when_endpoint_changes(self):

        original_id = self.camera.id

        self.camera.connection = ConnectionInfo(
            rtsp_address="rtsp://10.0.0.5/stream1", ip_address="10.0.0.5", username="operator",
        )
        self.camera.connection.rtsp_address = "rtsp://10.0.0.99/stream2"
        self.camera.connection.ip_address = "10.0.0.99"

        camera_after = self.manager.get_camera("CAM-001")

        self.assertIs(camera_after, self.camera)
        self.assertEqual(camera_after.id, original_id)
        self.assertEqual(camera_after.floor_id, "floor-1")


class MultiCameraDeduplicationTests(unittest.TestCase):

    # Phase 5/6: two Camera Assets, three physical people (one seen
    # only by CAM-001, one seen only by CAM-002, one seen by both), four
    # raw detections total. Must fuse to exactly three unique occupants,
    # and Building State must expose exactly three occupant tracks --
    # never four.

    def setUp(self):

        self.cam_1 = Camera(id="CAM-001", name="North Corridor", floor_id="floor-1")
        self.cam_2 = Camera(id="CAM-002", name="South Corridor", floor_id="floor-1")

        self.manager = CameraManager()
        self.manager.register_camera(self.cam_1)
        self.manager.register_camera(self.cam_2)

        self.source_1 = ReplayFrameSource(
            camera_id="CAM-001",
            frames=[(0.0, [
                {"local_track_id": "1"},  # only CAM-001
                {"local_track_id": "9"},  # shared with CAM-002
            ])],
        )
        self.source_2 = ReplayFrameSource(
            camera_id="CAM-002",
            frames=[(0.0, [
                {"local_track_id": "2"},  # only CAM-002
                {"local_track_id": "5"},  # shared with CAM-001
            ])],
        )
        self.source_1.start()
        self.source_2.start()

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

        self.manager.register_detection_provider(DeviceMode.REPLAY, self.detection_provider)
        self.manager.set_camera_mode("CAM-001", DeviceMode.REPLAY)
        self.manager.set_camera_mode("CAM-002", DeviceMode.REPLAY)

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

        _, fusion_result, building_state = _run_one_cycle(
            self.pipeline, self.detection_provider, ["CAM-001", "CAM-002"], self.fusion_engine, time=0.0,
        )

        shared_track = building_state.occupant_track("OCC-SHARED")

        self.assertIsNotNone(shared_track)
        self.assertEqual(set(shared_track.source_camera_ids), {"CAM-001", "CAM-002"})

    def test_camera_only_occupants_have_single_camera_provenance(self):

        _, fusion_result, building_state = _run_one_cycle(
            self.pipeline, self.detection_provider, ["CAM-001", "CAM-002"], self.fusion_engine, time=0.0,
        )

        self.assertEqual(
            building_state.occupant_track("OCC-ONLY-1").source_camera_ids, ("CAM-001",),
        )
        self.assertEqual(
            building_state.occupant_track("OCC-ONLY-2").source_camera_ids, ("CAM-002",),
        )


class BuildingStateNoDoubleCountingTests(unittest.TestCase):

    # Phase 6, restated as its own explicit proof: raw detections (4) !=
    # unique fused occupants (3) == Building State occupant tracks (3).

    def test_raw_four_unique_three_building_state_three(self):

        cam_1 = Camera(id="CAM-A", name="A", floor_id="floor-1")
        cam_2 = Camera(id="CAM-B", name="B", floor_id="floor-1")

        manager = CameraManager()
        manager.register_camera(cam_1)
        manager.register_camera(cam_2)

        source_a = ReplayFrameSource(
            camera_id="CAM-A",
            frames=[(0.0, [{"local_track_id": "a1"}, {"local_track_id": "shared"}])],
        )
        source_b = ReplayFrameSource(
            camera_id="CAM-B",
            frames=[(0.0, [{"local_track_id": "b1"}, {"local_track_id": "shared"}])],
        )
        source_a.start()
        source_b.start()

        resolver = MappingIdentityResolver({
            ("CAM-A", "a1"): "P1",
            ("CAM-B", "b1"): "P2",
            ("CAM-A", "shared"): "P3",
            ("CAM-B", "shared"): "P3",
        })

        detection_provider = LiveCameraPipelineDetectionProvider()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-A": source_a, "CAM-B": source_b},
            human_detector=MockHumanDetector(),
            identity_resolver=resolver,
            detection_provider=detection_provider,
        )

        manager.register_detection_provider(DeviceMode.REPLAY, detection_provider)
        manager.set_camera_mode("CAM-A", DeviceMode.REPLAY)
        manager.set_camera_mode("CAM-B", DeviceMode.REPLAY)

        fusion_engine = MultiCameraFusionEngine()

        detections, fusion_result, building_state = _run_one_cycle(
            pipeline, detection_provider, ["CAM-A", "CAM-B"], fusion_engine, time=0.0,
        )

        self.assertEqual(len(detections), 4, "raw detections")
        self.assertEqual(len(fusion_result.tracks), 3, "unique fused occupants")
        self.assertEqual(len(building_state.occupant_tracks), 3, "Building State occupant tracks")


class ModeIndependenceTests(unittest.TestCase):

    # Phase 7: Simulation / Replay / Live must be interchangeable
    # upstream-only -- Camera Manager routing, Detection shape, Fusion,
    # and Building State must not know or care which mode produced a
    # given Detection. Only the registered DetectionProvider (and, for
    # a real deployment, the CameraFrameSource behind it) differs.

    def setUp(self):

        self.camera = Camera(id="CAM-MODE", name="Test Cam", floor_id="floor-1")

        self.manager = CameraManager()
        self.manager.register_camera(self.camera)

        self.resolver = MappingIdentityResolver({("CAM-MODE", "1"): "OCC-X"})

    def _provider_for(self, timestamp):

        source = ReplayFrameSource(
            camera_id="CAM-MODE", frames=[(timestamp, [{"local_track_id": "1"}])],
        )
        source.start()

        provider = LiveCameraPipelineDetectionProvider()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-MODE": source},
            human_detector=MockHumanDetector(),
            identity_resolver=self.resolver,
            detection_provider=provider,
        )

        pipeline.run_cycle(timestamp)

        return provider

    def test_simulation_replay_and_live_produce_identically_shaped_detections(self):

        results = {}

        for mode in DeviceMode.ALL:

            provider = self._provider_for(timestamp=1.0)

            self.manager.register_detection_provider(mode, provider)
            self.manager.set_camera_mode("CAM-MODE", mode)

            detections = self.manager.detections_for_camera("CAM-MODE", time=1.0)

            self.assertEqual(len(detections), 1)
            results[mode] = detections[0]

        # Every mode produced the same camera_id / occupant_id / shape --
        # downstream code cannot tell which mode it came from.
        reference = results[DeviceMode.SIMULATION]

        for mode, detection in results.items():

            self.assertEqual(detection.camera_id, reference.camera_id)
            self.assertEqual(detection.occupant_id, reference.occupant_id)
            self.assertEqual(detection.classification, reference.classification)

    def test_fusion_and_building_state_identical_regardless_of_mode(self):

        fusion_engine = MultiCameraFusionEngine()

        states = {}

        for mode in DeviceMode.ALL:

            provider = self._provider_for(timestamp=2.0)

            self.manager.register_detection_provider(mode, provider)
            self.manager.set_camera_mode("CAM-MODE", mode)

            detections = self.manager.detections_for_camera("CAM-MODE", time=2.0)
            fusion_result = fusion_engine.fuse(detections, time=2.0)

            states[mode] = BuildingStateEstimator().estimate(
                2.0,
                hazard_snapshot=HazardSnapshot(),
                occupancy_snapshot=OccupancySnapshot(),
                fusion_result=fusion_result,
            )

        for mode, state in states.items():

            self.assertEqual(len(state.occupant_tracks), 1)
            self.assertIn("OCC-X", state.occupant_tracks)
            self.assertEqual(
                state.occupant_tracks["OCC-X"].source_camera_ids, ("CAM-MODE",),
            )

    def test_configuring_live_mode_performs_no_automatic_connection(self):

        # Setting mode=LIVE and populating real-world connection info is
        # only ever configuration -- it must never, by itself, trigger a
        # network attempt or change runtime connection status.

        self.camera.connection = ConnectionInfo(
            rtsp_address="rtsp://10.0.0.5:554/stream1",
            ip_address="10.0.0.5",
            username="operator",
        )
        self.manager.set_camera_mode("CAM-MODE", DeviceMode.LIVE)

        self.assertEqual(
            self.manager.connection_status("CAM-MODE"), CameraConnectionState.CONFIGURED,
        )

        # No provider registered for LIVE yet -- this must be "no
        # detections available," never an error and never an implicit
        # connection attempt.
        self.assertEqual(self.manager.detections_for_camera("CAM-MODE", time=0.0), ())


class LiveCredentialSafetyTests(unittest.TestCase):

    # Phase 9 items 11/13/14: Live mode credential handling must fail
    # gracefully when nothing has been configured yet, and must never
    # leak a real password into any string representation.

    def test_missing_live_credentials_resolve_to_none_not_an_exception(self):

        with tempfile.TemporaryDirectory() as tmp_dir:

            store = LocalFileCredentialStore(path=Path(tmp_dir) / "credentials.json")

            camera = Camera(id="CAM-LIVE", name="Unconfigured Live Cam", floor_id="floor-1")
            camera.mode = DeviceMode.LIVE

            reference = camera.connection.credential_ref or camera.id

            # No password was ever saved for this camera -- resolving it
            # must be an honest None, never a KeyError/crash, and must
            # not create the credential file as a side effect of reading.
            resolved = store.get_credential(reference)

            self.assertIsNone(resolved)
            self.assertFalse(store.has_credential(reference))

    def test_password_never_appears_in_camera_repr(self):

        camera = Camera(id="CAM-LIVE", name="Cam", floor_id="floor-1")
        camera.connection = ConnectionInfo(
            rtsp_address="rtsp://10.0.0.5/stream",
            ip_address="10.0.0.5",
            username="operator",
            password="super-secret-value",
        )

        text = repr(camera)

        self.assertNotIn("super-secret-value", text)
        self.assertIn("<redacted>", text)

    def test_password_never_appears_in_camera_to_dict(self):

        camera = Camera(id="CAM-LIVE", name="Cam", floor_id="floor-1")
        camera.connection = ConnectionInfo(
            rtsp_address="rtsp://10.0.0.5/stream",
            ip_address="10.0.0.5",
            username="operator",
            password="super-secret-value",
        )

        data = camera.to_dict()

        self.assertNotIn("super-secret-value", repr(data))
        self.assertNotIn("password", data["connection"])


if __name__ == "__main__":
    unittest.main()
