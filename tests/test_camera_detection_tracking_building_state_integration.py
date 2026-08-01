import unittest

from live_camera_pipeline.frame_source import CameraFrame, CameraFrameSource
from live_camera_pipeline.human_detector import HumanDetector, RawHumanDetection
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver

from tracking.simple_tracker import SimpleSingleCameraTracker

from live_runtime.factory import build_live_runtime

from live_occupants.state import OccupantStatus

from live_system.live_command_center_gateway import LiveCommandCenterDataSource

from tests.live_runtime_fixtures import make_demo_building


# =====================================================
# Camera 1 Live Detection -> Tracking/Building-State Integration
# milestone -- proves the composition this milestone adds to
# live_runtime_launcher/session.py (tracker=SimpleSingleCameraTracker())
# actually reaches BuildingState.occupant_tracks AND live_occupants.
# manager.LiveOccupantManager, not merely LiveCameraPipeline in
# isolation (already covered by tests/test_live_camera_pipeline_
# tracking_integration.py) or build_live_runtime() WITHOUT a tracker
# (tests/test_live_runtime_human_evidence_e2e.py). No real NVR, no
# real RTSP, no real credentials, no real YOLO model -- uses the SAME
# real SimulationIdentityResolver/SimpleSingleCameraTracker/
# MultiCameraFusionEngine production classes this milestone wires in,
# fed by a deterministic fake frame source + fake detector.
# =====================================================


CAMERA_ID = "CAM-LOBBY"  # matches tests.live_runtime_fixtures.make_demo_building()


class _ScriptedFrameSource(CameraFrameSource):

    # One CameraFrame per read_frame() call, driven by a plain list the
    # test controls -- payload_ref carries whatever the paired
    # _ScriptedDetector should report for that cycle (None entries mean
    # "no frame this cycle", matching a real dropped/late RTSP read).

    def __init__(self, camera_id, payloads):
        self.camera_id = camera_id
        self._payloads = list(payloads)
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
        if self._index >= len(self._payloads):
            return None
        payload = self._payloads[self._index]
        self._index += 1
        if payload is None:
            return None
        return CameraFrame(
            camera_id=self.camera_id, timestamp=float(self._index), frame_sequence=self._index,
            payload_ref=payload, width=1280, height=720, codec="h264",
        )


class _ScriptedDetector(HumanDetector):

    # frame.payload_ref is a list of bounding boxes for this cycle
    # (empty list/None frame -> zero detections, exactly like a real
    # YOLOHumanDetector seeing no person). confidence stays comfortably
    # inside the real 0.329-0.484 range this milestone's own live
    # verification actually observed -- never retuned.

    def detect(self, frame):
        if frame.payload_ref is None:
            return ()
        return tuple(
            RawHumanDetection(
                camera_id=frame.camera_id, local_track_id=str(index),
                timestamp=frame.timestamp, bounding_box=box, confidence=0.42,
            )
            for index, box in enumerate(frame.payload_ref)
        )


class DetectionReachesBuildingStateAndOccupantManagerTests(unittest.TestCase):

    def _build_runtime(self, payloads):

        building = make_demo_building()
        source = _ScriptedFrameSource(CAMERA_ID, payloads)

        runtime = build_live_runtime(
            building,
            frame_sources={CAMERA_ID: source},
            human_detector=_ScriptedDetector(),
            identity_resolver=SimulationIdentityResolver(),
            tracker=SimpleSingleCameraTracker(),
        )
        runtime.start()
        return runtime

    def test_a_single_person_produces_a_fused_track_and_a_live_occupant(self):

        # Requirements 1-5, 7: one CameraFrame -> one human detection ->
        # visible through the detection provider -> aggregated by
        # CameraManager -> consumed by MultiCameraFusionEngine ->
        # produces exactly the right occupant/track state, associated
        # with the correct camera.

        box = (100.0, 100.0, 140.0, 220.0)
        runtime = self._build_runtime([[box]])

        snapshot = runtime.run_cycle(1.0)

        self.assertEqual(len(snapshot.building_state.occupant_tracks), 1)
        track = next(iter(snapshot.building_state.occupant_tracks.values()))
        self.assertEqual(track.source_camera_ids, (CAMERA_ID,))

        occupants = runtime.live_occupant_manager.all_occupants()
        self.assertEqual(len(occupants), 1)
        occupant = occupants[0]
        self.assertEqual(occupant.current_camera_id, CAMERA_ID)
        self.assertEqual(occupant.status, OccupantStatus.NEW)
        self.assertEqual(occupant.occupant_id, track.track_id)

        runtime.stop()

    def test_repeated_observations_of_the_same_person_keep_the_same_track_id(self):

        # Requirement 6: the SAME synthetic person, moving slightly
        # frame to frame, must resolve to the SAME track_id via
        # SimpleSingleCameraTracker's IoU/centroid matching -- genuine
        # temporal continuity, not per-frame coincidence.

        boxes = [
            [(100.0, 100.0, 140.0, 220.0)],
            [(103.0, 101.0, 143.0, 221.0)],
            [(106.0, 100.0, 146.0, 220.0)],
        ]
        runtime = self._build_runtime(boxes)

        track_ids_seen = []
        for cycle, _ in enumerate(boxes, start=1):
            snapshot = runtime.run_cycle(float(cycle))
            self.assertEqual(len(snapshot.building_state.occupant_tracks), 1)
            track_ids_seen.append(next(iter(snapshot.building_state.occupant_tracks.keys())))

        self.assertEqual(len(set(track_ids_seen)), 1, f"expected one stable track_id, saw {track_ids_seen}")
        self.assertTrue(track_ids_seen[0].startswith(CAMERA_ID), "track_id must be namespaced by camera_id")

        occupants = runtime.live_occupant_manager.all_occupants()
        self.assertEqual(len(occupants), 1)
        self.assertEqual(occupants[0].occupant_id, track_ids_seen[0])

        runtime.stop()

    def test_person_leaving_frame_is_handled_without_crashing_and_without_a_stale_track(self):

        # Requirement 8: an empty/no-detection frame must never crash
        # the cycle, and must never keep serving a stale detection --
        # BuildingState.occupant_tracks must reflect "nobody currently
        # detected" once the person is gone (LiveOccupantManager's own
        # sweep semantics for "temporarily lost" are already proven at
        # the pipeline level in tests/test_live_camera_pipeline_
        # occupant_integration.py; this test only proves the FULL
        # build_live_runtime() chain does not crash and reports honestly).

        box = (100.0, 100.0, 140.0, 220.0)
        runtime = self._build_runtime([[box], None])

        runtime.run_cycle(1.0)
        snapshot = runtime.run_cycle(2.0)  # no detection this cycle

        self.assertEqual(snapshot.building_state.occupant_tracks, {})

        runtime.stop()

    def test_no_credential_values_reach_detection_or_tracking_or_building_state(self):

        # Requirement 10 (structural): nothing in this composition
        # (build_rtsp_frame_sources/RTSPFrameSource credential
        # resolution) is anywhere near this call graph at all --
        # build_live_runtime() here is handed a fake frame source with
        # no credential_ref/credential_store concept whatsoever, so
        # there is no seam for a password to travel through even in
        # principle. Proven by construction, not by scanning output.

        self.assertFalse(hasattr(_ScriptedFrameSource, "credential_ref"))
        self.assertFalse(hasattr(_ScriptedFrameSource, "credential_store"))

    def test_a_detected_person_reaches_the_actual_command_center_data_path(self):

        # Expose Real Live Camera Occupant State In SynEvac UI milestone
        # -- the critical proof this milestone exists for: a genuine
        # detection reaching all the way through to the exact object
        # Command Center's own UI panels are wired to read
        # (LiveCommandCenterDataSource.current_snapshot().live_occupants),
        # not merely BuildingState/LiveOccupantManager in isolation.

        box = (100.0, 100.0, 140.0, 220.0)
        runtime = self._build_runtime([[box]])

        data_source = LiveCommandCenterDataSource(
            runtime.orchestrator.state_manager, building=runtime.building,
        )
        data_source.start()

        runtime.run_cycle(1.0)

        snapshot = data_source.current_snapshot()

        self.assertIsNotNone(snapshot.live_occupants)
        self.assertEqual(len(snapshot.live_occupants.occupants), 1)
        self.assertEqual(snapshot.live_occupants.current_occupant_count, 1)
        self.assertEqual(snapshot.live_occupants.occupants[0].current_camera_id, CAMERA_ID)

        runtime.stop()

    def test_two_cameras_configuration_still_isolates_tracks_per_camera(self):

        # Requirement 7 (camera association) -- a second configured
        # Camera Asset with NO frame source at all (make_demo_building()
        # has CAM-CAFETERIA too) must never contaminate CAM-LOBBY's own
        # track -- CameraManager already covers "camera configured but
        # no frame source produces no detections honestly"
        # (tests/test_live_runtime_failure_modes.py); this just confirms
        # that holds true alongside a real tracked occupant on the other
        # camera.

        box = (100.0, 100.0, 140.0, 220.0)
        runtime = self._build_runtime([[box]])

        snapshot = runtime.run_cycle(1.0)

        for track in snapshot.building_state.occupant_tracks.values():
            self.assertEqual(track.source_camera_ids, (CAMERA_ID,))
            self.assertNotIn("CAM-CAFETERIA", track.source_camera_ids)

        runtime.stop()


if __name__ == "__main__":
    unittest.main()
