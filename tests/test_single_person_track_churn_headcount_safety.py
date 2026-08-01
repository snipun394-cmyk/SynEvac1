import unittest

from live_camera_pipeline.frame_source import CameraFrame, CameraFrameSource
from live_camera_pipeline.human_detector import HumanDetector, RawHumanDetection
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver

from tracking.simple_tracker import SimpleSingleCameraTracker, DEFAULT_MAX_MISSING_FRAMES

from live_runtime.factory import build_live_runtime

from live_occupants.state import OccupantStatus

from tests.live_runtime_fixtures import make_demo_building


# =====================================================
# Single-Person Track Lifecycle / Headcount Correctness Audit --
# reproduces, deterministically, the EXACT scenario a real Camera 1
# session actually produced (one physical person, three separate
# SimpleSingleCameraTracker track lifecycles T1/T2/T3 because
# reappearance happened after the tracker's own frame-count-based
# expiration). Proves whether historical track churn ever inflates
# CURRENT physical occupancy anywhere downstream.
#
# Two independent clocks matter here, and this file deliberately keeps
# them decoupled exactly as production does:
#   - SimpleSingleCameraTracker expires a track after
#     max_missing_frames CONSECUTIVE missed CYCLES (frame-count based).
#   - LiveOccupantManager expires an occupant after expire_after_seconds
#     of WALL-CLOCK time since last_seen (see live_occupants/lifecycle.py).
# Every cycle here advances the simulated clock by only 0.1s, so across
# the whole ~2s scenario LiveOccupantManager's own 30s default expiry
# is NEVER reached -- exactly reproducing the real run, where old
# tracks churned (tracker-level) long before any occupant could ever
# wall-clock EXPIRE (occupant-level). This is what makes T1 show up as
# TEMPORARILY_LOST (not removed) at the same time T2/T3 exist.
# =====================================================


CAMERA_ID = "CAM-LOBBY"  # matches tests.live_runtime_fixtures.make_demo_building()

BOX = (100.0, 100.0, 140.0, 220.0)          # one physical person's position
BOX_2 = (400.0, 100.0, 440.0, 220.0)         # a second, clearly-separate physical person


class _ScriptedFrameSource(CameraFrameSource):

    # Same minimal fake as tests/test_camera_detection_tracking_
    # building_state_integration.py -- one CameraFrame per read_frame()
    # call, payload_ref carries this cycle's list of bounding boxes
    # (empty/None -> no detection, exactly like YOLO seeing no person).

    def __init__(self, camera_id):
        self.camera_id = camera_id
        self._pending = None
        self._sequence = 0
        self._running = False

    def push(self, boxes):
        self._pending = boxes

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    @property
    def is_running(self):
        return self._running

    def read_frame(self):
        boxes = self._pending
        self._pending = None
        if boxes is None:
            # No frame at all this cycle (e.g. RTSP hasn't buffered a
            # new one yet) -- distinct from "a frame arrived containing
            # zero people" ([]), which must still reach the tracker so
            # its own missing-frame aging actually advances.
            return None
        self._sequence += 1
        return CameraFrame(
            camera_id=self.camera_id, timestamp=float(self._sequence), frame_sequence=self._sequence,
            payload_ref=boxes, width=1280, height=720, codec="h264",
        )


class _ScriptedDetector(HumanDetector):

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


class SinglePersonTrackChurnTests(unittest.TestCase):

    def setUp(self):

        self.building = make_demo_building()
        self.source = _ScriptedFrameSource(CAMERA_ID)

        self.runtime = build_live_runtime(
            self.building,
            frame_sources={CAMERA_ID: self.source},
            human_detector=_ScriptedDetector(),
            identity_resolver=SimulationIdentityResolver(),
            tracker=SimpleSingleCameraTracker(),
        )
        self.runtime.start()

        self.t = 0.0

    def tearDown(self):
        self.runtime.stop()

    def _cycle(self, boxes):

        # `boxes=None` still pushes a REAL frame with zero detections
        # (a person genuinely absent from an otherwise-flowing camera
        # feed) -- this is what actually drives SimpleSingleCameraTracker's
        # own missing-frame aging (LiveCameraPipeline.run_cycle() skips
        # the tracker entirely for a cycle with NO frame at all, which
        # is a different, rarer condition this test is not simulating).

        self.t += 0.1
        self.source.push(boxes if boxes is not None else [])

        return self.runtime.run_cycle(self.t)

    def _canonical_count(self):
        return self.runtime.live_occupant_manager.canonical_occupancy(self.t).total_observed_count

    def _active_ids(self):
        return {o.occupant_id for o in self.runtime.live_occupant_manager.active_occupants()}

    def _all_ids(self):
        return {o.occupant_id for o in self.runtime.live_occupant_manager.all_occupants()}

    # =====================================================

    def test_full_A_to_F_timeline_never_overcounts_one_physical_person(self):

        # A. Person appears for several sequential frames -> T1.
        self._cycle([BOX])
        self._cycle([BOX])

        active_ids = self._active_ids()
        self.assertEqual(len(active_ids), 1)
        t1_id = next(iter(active_ids))
        self.assertTrue(t1_id.startswith(CAMERA_ID))
        self.assertEqual(self._canonical_count(), 1)

        # B. Person disappears briefly -- LESS than the tracker's own
        # expiration threshold (DEFAULT_MAX_MISSING_FRAMES=5, so 3
        # missed cycles must not expire it).
        for _ in range(3):
            self._cycle(None)

        # Still only T1 known, temporarily lost -- still exactly one
        # physical person's worth of state, and it is honestly NOT
        # counted as currently active while unseen.
        self.assertEqual(self._all_ids(), {t1_id})
        self.assertEqual(self._active_ids(), set())
        self.assertEqual(self._canonical_count(), 0)

        # C. Person reappears BEFORE expiration -- the tracker's own
        # documented behavior: the SAME track_id resumes (still in its
        # internal store), never a new one.
        snapshot = self._cycle([BOX])
        self.assertEqual(set(snapshot.building_state.occupant_tracks.keys()), {t1_id})
        self.assertEqual(self._active_ids(), {t1_id})
        self.assertEqual(self._canonical_count(), 1)

        # D. Person disappears LONGER than the tracker's expiration
        # threshold -- 6 consecutive misses (frames_missing becomes 6,
        # which is > DEFAULT_MAX_MISSING_FRAMES=5) actually expires and
        # deletes T1 at the TRACKER level.
        self.assertEqual(DEFAULT_MAX_MISSING_FRAMES, 5, "test assumes the tracker's own documented default")
        for _ in range(6):
            self._cycle(None)

        # Wall-clock elapsed so far is under 1.2s -- nowhere near
        # LiveOccupantManager's own 30s default expire_after_seconds --
        # so T1 must still exist, honestly TEMPORARILY_LOST, never
        # silently deleted just because the TRACKER forgot about it.
        t1_occupant = self.runtime.live_occupant_manager.get(t1_id)
        self.assertIsNotNone(t1_occupant)
        self.assertEqual(t1_occupant.status, OccupantStatus.TEMPORARILY_LOST)
        self.assertEqual(self._active_ids(), set())
        self.assertEqual(self._canonical_count(), 0)

        # E. Person reappears -- the tracker has no memory of T1
        # anymore (deleted), so a NEW track_id (T2) is legitimately
        # created. This is NOT a bug by itself.
        snapshot = self._cycle([BOX])
        active_ids = self._active_ids()
        self.assertEqual(len(active_ids), 1)
        t2_id = next(iter(active_ids))
        self.assertNotEqual(t2_id, t1_id, "reappearance after real tracker expiry legitimately mints a new id")

        # THE CRITICAL ASSERTION: T1 (historical, TEMPORARILY_LOST) and
        # T2 (current) now BOTH exist in the manager's own store, but
        # canonical/active occupancy must count exactly the one
        # physical person actually present right now -- never 2.
        self.assertEqual(self._all_ids(), {t1_id, t2_id})
        self.assertEqual(self._canonical_count(), 1)
        self.assertEqual(set(snapshot.building_state.occupant_tracks.keys()), {t2_id})

        # F. Repeat once more to produce T3, proving this holds for
        # arbitrarily many historical track lifecycles, not just two.
        for _ in range(6):
            self._cycle(None)

        snapshot = self._cycle([BOX])
        active_ids = self._active_ids()
        self.assertEqual(len(active_ids), 1)
        t3_id = next(iter(active_ids))
        self.assertNotIn(t3_id, {t1_id, t2_id})

        self.assertEqual(self._all_ids(), {t1_id, t2_id, t3_id})
        self.assertEqual(self._canonical_count(), 1, "3 historical track lifecycles for ONE physical person must never read as 3")
        self.assertEqual(set(snapshot.building_state.occupant_tracks.keys()), {t3_id})

        # BuildingState.occupant_tracks is independently self-cleaning
        # too -- it is rebuilt fresh from ONLY this cycle's fused
        # detections (MultiCameraFusionEngine never accumulates stale
        # occupant_ids across calls), so it never shows T1/T2 alongside
        # T3 either.
        self.assertNotIn(t1_id, snapshot.building_state.occupant_tracks)
        self.assertNotIn(t2_id, snapshot.building_state.occupant_tracks)

    def test_two_simultaneous_distinct_people_are_still_counted_as_two(self):

        # Requirement 11 -- the critical guard against a naive "fix"
        # that would collapse genuinely distinct people. Two widely
        # separated bounding boxes, continuously visible together.

        for _ in range(4):
            snapshot = self._cycle([BOX, BOX_2])

        self.assertEqual(self._canonical_count(), 2)
        self.assertEqual(len(self._active_ids()), 2)
        self.assertEqual(len(snapshot.building_state.occupant_tracks), 2)

        # Confirm they were never merged into the same track/occupant.
        track_ids = set(snapshot.building_state.occupant_tracks.keys())
        self.assertEqual(len(track_ids), 2)


if __name__ == "__main__":
    unittest.main()
