import unittest

from live_camera_pipeline.human_detector import RawHumanDetection

from tracking.simple_tracker import SimpleSingleCameraTracker
from tracking.track_state import TrackState


# =====================================================
# Single-Camera Tracking Framework milestone, Phase 9 -- deterministic,
# offline unit tests directly against SimpleSingleCameraTracker. No
# randomness anywhere in this file: every bounding box, confidence, and
# timestamp is hand-chosen so results are 100% reproducible.
# =====================================================


def det(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0), camera_id="CAM-1", timestamp=0.0):

    return RawHumanDetection(
        camera_id=camera_id, local_track_id="unused", timestamp=timestamp,
        bounding_box=box, confidence=confidence,
    )


class OnePersonTests(unittest.TestCase):

    def test_1_one_person_creates_a_new_track(self):

        tracker = SimpleSingleCameraTracker()

        result = tracker.update("CAM-1", 0.0, [det()])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].state, TrackState.NEW)
        self.assertEqual(result[0].age, 1)
        self.assertEqual(result[0].frames_seen, 1)
        self.assertEqual(result[0].frames_missing, 0)

    def test_1_same_person_next_frame_keeps_the_same_track_id(self):

        tracker = SimpleSingleCameraTracker()

        first = tracker.update("CAM-1", 0.0, [det(box=(0.0, 0.0, 10.0, 10.0))])
        second = tracker.update("CAM-1", 1.0, [det(box=(1.0, 1.0, 11.0, 11.0), timestamp=1.0)])

        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(second[0].state, TrackState.TRACKED)
        self.assertEqual(second[0].age, 2)
        self.assertEqual(second[0].frames_seen, 2)


class MultiplePeopleTests(unittest.TestCase):

    def test_2_multiple_people_get_distinct_stable_ids(self):

        tracker = SimpleSingleCameraTracker()

        boxes = [(0.0, 0.0, 10.0, 10.0), (50.0, 0.0, 60.0, 10.0), (100.0, 0.0, 110.0, 10.0)]

        first = tracker.update("CAM-1", 0.0, [det(box=b) for b in boxes])
        second = tracker.update("CAM-1", 1.0, [det(box=b, timestamp=1.0) for b in boxes])

        self.assertEqual(len(first), 3)
        self.assertEqual({t.track_id for t in first}, {t.track_id for t in second})

        first_ids = [t.track_id for t in first]
        second_ids = [t.track_id for t in second]
        self.assertEqual(first_ids, second_ids)  # same order, same boxes -> same pairing


class CrossingPathsTests(unittest.TestCase):

    def test_3_two_people_passing_near_each_other_keep_distinct_ids_while_separated(self):

        tracker = SimpleSingleCameraTracker(iou_threshold=0.3, max_centroid_distance=15.0)

        left = det(box=(0.0, 0.0, 10.0, 10.0))
        right = det(box=(100.0, 0.0, 110.0, 10.0))

        first = tracker.update("CAM-1", 0.0, [left, right])

        # Each moves 5px toward the other -- still far apart relative to
        # max_centroid_distance, unambiguous continuation.
        left_2 = det(box=(5.0, 0.0, 15.0, 10.0), timestamp=1.0)
        right_2 = det(box=(95.0, 0.0, 105.0, 10.0), timestamp=1.0)

        second = tracker.update("CAM-1", 1.0, [left_2, right_2])

        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(first[1].track_id, second[1].track_id)
        self.assertNotEqual(second[0].track_id, second[1].track_id)


class EnteringAndLeavingFrameTests(unittest.TestCase):

    def test_4_person_entering_frame_mid_sequence_gets_a_fresh_track(self):

        tracker = SimpleSingleCameraTracker()

        tracker.update("CAM-1", 0.0, [det(box=(0.0, 0.0, 10.0, 10.0))])

        second = tracker.update(
            "CAM-1", 1.0,
            [det(box=(0.0, 0.0, 10.0, 10.0), timestamp=1.0), det(box=(200.0, 0.0, 210.0, 10.0), timestamp=1.0)],
        )

        self.assertEqual(len(second), 2)
        self.assertEqual(second[1].state, TrackState.NEW)
        self.assertEqual(second[1].frames_seen, 1)

    def test_5_person_leaving_frame_becomes_missing_then_expires(self):

        tracker = SimpleSingleCameraTracker(max_missing_frames=2)

        tracker.update("CAM-1", 0.0, [det()])
        track_id = tracker.update("CAM-1", 0.0, [det()])[0].track_id

        missing_1 = tracker.update("CAM-1", 1.0, [])
        self.assertEqual(len(missing_1), 1)
        self.assertEqual(missing_1[0].state, TrackState.MISSING)
        self.assertEqual(missing_1[0].frames_missing, 1)

        missing_2 = tracker.update("CAM-1", 2.0, [])
        self.assertEqual(missing_2[0].state, TrackState.MISSING)
        self.assertEqual(missing_2[0].frames_missing, 2)

        expired = tracker.update("CAM-1", 3.0, [])
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].state, TrackState.EXPIRED)
        self.assertEqual(expired[0].track_id, track_id)

        gone = tracker.update("CAM-1", 4.0, [])
        self.assertEqual(gone, ())  # never reported again


class OcclusionAndMissedDetectionTests(unittest.TestCase):

    def test_6_temporary_occlusion_resumes_the_same_track_id(self):

        tracker = SimpleSingleCameraTracker(max_missing_frames=3)

        first = tracker.update("CAM-1", 0.0, [det(box=(0.0, 0.0, 10.0, 10.0))])
        track_id = first[0].track_id

        tracker.update("CAM-1", 1.0, [])  # occluded
        tracker.update("CAM-1", 2.0, [])  # still occluded

        reappeared = tracker.update("CAM-1", 3.0, [det(box=(0.0, 0.0, 10.0, 10.0), timestamp=3.0)])

        self.assertEqual(reappeared[0].track_id, track_id)
        self.assertEqual(reappeared[0].state, TrackState.TRACKED)
        self.assertEqual(reappeared[0].frames_missing, 0)

    def test_7_single_missed_detection_does_not_break_continuity(self):

        tracker = SimpleSingleCameraTracker(max_missing_frames=5)

        first = tracker.update("CAM-1", 0.0, [det()])
        track_id = first[0].track_id

        tracker.update("CAM-1", 1.0, [])  # one missed cycle

        third = tracker.update("CAM-1", 2.0, [det(timestamp=2.0)])

        self.assertEqual(third[0].track_id, track_id)


class DuplicateDetectionTests(unittest.TestCase):

    def test_8_duplicate_detections_in_one_frame_handled_deterministically(self):

        tracker = SimpleSingleCameraTracker()

        box = (0.0, 0.0, 10.0, 10.0)
        result = tracker.update("CAM-1", 0.0, [det(box=box), det(box=box)])

        self.assertEqual(len(result), 2)
        # Deterministic, non-crashing -- both get a track id, never the
        # same instance's tracks dict corrupted (no exception, no
        # missing entries).
        self.assertTrue(all(t.track_id for t in result))

        # Re-running the exact same input is byte-for-byte reproducible.
        tracker_2 = SimpleSingleCameraTracker()
        result_2 = tracker_2.update("CAM-1", 0.0, [det(box=box), det(box=box)])
        self.assertEqual([t.track_id for t in result], [t.track_id for t in result_2])


class ConfidenceTests(unittest.TestCase):

    def test_9_confidence_changes_do_not_break_track_continuity(self):

        tracker = SimpleSingleCameraTracker()

        first = tracker.update("CAM-1", 0.0, [det(confidence=0.95)])
        second = tracker.update("CAM-1", 1.0, [det(confidence=0.55, timestamp=1.0)])

        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertAlmostEqual(second[0].confidence, 0.55)


class TrackExpirationTests(unittest.TestCase):

    def test_10_track_expires_exactly_once_and_never_returns(self):

        tracker = SimpleSingleCameraTracker(max_missing_frames=1)

        tracker.update("CAM-1", 0.0, [det()])

        missing = tracker.update("CAM-1", 1.0, [])
        self.assertEqual(missing[0].state, TrackState.MISSING)  # 1 miss <= max_missing_frames, still coasting

        expired = tracker.update("CAM-1", 2.0, [])
        self.assertEqual(expired[0].state, TrackState.EXPIRED)  # 2nd consecutive miss > max_missing_frames

        after = tracker.update("CAM-1", 3.0, [])
        self.assertEqual(after, ())


class IDStabilityTests(unittest.TestCase):

    def test_11_id_stays_stable_across_many_consecutive_cycles(self):

        tracker = SimpleSingleCameraTracker()

        track_id = None

        for i in range(20):

            box = (float(i), 0.0, float(i) + 10.0, 10.0)
            result = tracker.update("CAM-1", float(i), [det(box=box, timestamp=float(i))])

            if track_id is None:
                track_id = result[0].track_id
            else:
                self.assertEqual(result[0].track_id, track_id)

            self.assertEqual(result[0].age, i + 1)
            self.assertEqual(result[0].frames_seen, i + 1)


class TimestampAndFrameOrderingTests(unittest.TestCase):

    def test_12_last_timestamp_reflects_the_most_recent_match(self):

        tracker = SimpleSingleCameraTracker(max_missing_frames=5)

        tracker.update("CAM-1", 10.0, [det(timestamp=10.0)])
        matched = tracker.update("CAM-1", 20.0, [det(timestamp=20.0)])

        self.assertEqual(matched[0].last_timestamp, 20.0)

        missing = tracker.update("CAM-1", 30.0, [])
        # last_timestamp does not advance while merely coasting/missing --
        # it stays the timestamp of the last real match.
        self.assertEqual(missing[0].last_timestamp, 20.0)

    def test_13_frame_sequence_is_irrelevant_to_the_tracker_only_call_order_matters(self):

        # SingleCameraTracker never reads CameraFrame.frame_sequence at
        # all (it is not part of its input) -- only the order update()
        # is called in, and the timestamp explicitly passed each call,
        # matter.
        tracker = SimpleSingleCameraTracker()

        first = tracker.update("CAM-1", 5.0, [det(timestamp=5.0)])
        second = tracker.update("CAM-1", 6.0, [det(timestamp=6.0)])

        self.assertEqual(first[0].track_id, second[0].track_id)


class RepeatedFramesTests(unittest.TestCase):

    def test_14_repeated_identical_frames_increment_age_and_frames_seen(self):

        tracker = SimpleSingleCameraTracker()

        track_id = None

        for i in range(5):

            result = tracker.update("CAM-1", float(i), [det(timestamp=float(i))])

            if track_id is None:
                track_id = result[0].track_id

            self.assertEqual(result[0].track_id, track_id)
            self.assertEqual(result[0].age, i + 1)
            self.assertEqual(result[0].frames_seen, i + 1)
            self.assertEqual(result[0].frames_missing, 0)


class EmptyFrameTests(unittest.TestCase):

    def test_15_empty_frame_with_no_prior_tracks_returns_nothing(self):

        tracker = SimpleSingleCameraTracker()

        self.assertEqual(tracker.update("CAM-1", 0.0, []), ())

    def test_15_empty_frame_after_a_track_exists_reports_it_missing(self):

        tracker = SimpleSingleCameraTracker(max_missing_frames=5)

        tracker.update("CAM-1", 0.0, [det()])
        result = tracker.update("CAM-1", 1.0, [])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].state, TrackState.MISSING)


class TwoCamerasTests(unittest.TestCase):

    def test_16_two_cameras_tracked_simultaneously_by_one_tracker_instance(self):

        tracker = SimpleSingleCameraTracker()

        cam_a = tracker.update("CAM-A", 0.0, [det(camera_id="CAM-A")])
        cam_b = tracker.update("CAM-B", 0.0, [det(camera_id="CAM-B")])

        self.assertEqual(len(cam_a), 1)
        self.assertEqual(len(cam_b), 1)
        self.assertEqual(cam_a[0].camera_id, "CAM-A")
        self.assertEqual(cam_b[0].camera_id, "CAM-B")

    def test_17_identical_bounding_boxes_on_different_cameras_never_share_a_track(self):

        tracker = SimpleSingleCameraTracker()

        box = (0.0, 0.0, 10.0, 10.0)

        cam_a_first = tracker.update("CAM-A", 0.0, [det(box=box, camera_id="CAM-A")])
        cam_b_first = tracker.update("CAM-B", 0.0, [det(box=box, camera_id="CAM-B")])

        self.assertNotEqual(cam_a_first[0].track_id, cam_b_first[0].track_id)

        cam_a_second = tracker.update("CAM-A", 1.0, [det(box=box, camera_id="CAM-A", timestamp=1.0)])
        cam_b_second = tracker.update("CAM-B", 1.0, [det(box=box, camera_id="CAM-B", timestamp=1.0)])

        # Each camera's track continues under its own isolated pool.
        self.assertEqual(cam_a_first[0].track_id, cam_a_second[0].track_id)
        self.assertEqual(cam_b_first[0].track_id, cam_b_second[0].track_id)
        self.assertNotEqual(cam_a_second[0].track_id, cam_b_second[0].track_id)


class ParameterEdgeCaseTests(unittest.TestCase):

    def test_18_max_missing_frames_zero_expires_on_the_first_miss(self):

        tracker = SimpleSingleCameraTracker(max_missing_frames=0)

        tracker.update("CAM-1", 0.0, [det()])

        expired = tracker.update("CAM-1", 1.0, [])
        self.assertEqual(expired[0].state, TrackState.EXPIRED)

    def test_18_minimum_confidence_prevents_low_confidence_detections_from_persisting(self):

        tracker = SimpleSingleCameraTracker(minimum_confidence=0.5)

        low = tracker.update("CAM-1", 0.0, [det(confidence=0.1)])
        self.assertEqual(low[0].state, TrackState.NEW)

        # The low-confidence detection never persisted -- nothing to
        # match against, so a real detection one cycle later at the
        # same location still starts a BRAND NEW track, not a
        # continuation of the disposable one.
        second = tracker.update("CAM-1", 1.0, [det(confidence=0.9, timestamp=1.0)])
        self.assertNotEqual(second[0].track_id, low[0].track_id)

    def test_18_minimum_confidence_does_not_drop_the_output_entry(self):

        tracker = SimpleSingleCameraTracker(minimum_confidence=0.9)

        result = tracker.update("CAM-1", 0.0, [det(confidence=0.1), det(confidence=0.95, box=(50.0, 0.0, 60.0, 10.0))])

        self.assertEqual(len(result), 2)  # both get an entry, positionally

    def test_18_iou_threshold_boundary_just_below_required_overlap_creates_a_new_track(self):

        # Two boxes with deliberately small overlap (IoU well under a
        # strict 0.9 threshold, and far enough apart that the default
        # centroid fallback also does not qualify) -- must be treated
        # as a new person, not silently merged into the existing track.
        tracker = SimpleSingleCameraTracker(iou_threshold=0.9, max_centroid_distance=1.0)

        tracker.update("CAM-1", 0.0, [det(box=(0.0, 0.0, 10.0, 10.0))])

        second = tracker.update("CAM-1", 1.0, [det(box=(5.0, 0.0, 15.0, 10.0), timestamp=1.0)])

        self.assertEqual(second[0].state, TrackState.NEW)

    def test_18_two_cameras_use_independent_track_id_counters(self):

        tracker = SimpleSingleCameraTracker()

        cam_a = tracker.update("CAM-A", 0.0, [det(camera_id="CAM-A")])
        cam_b = tracker.update("CAM-B", 0.0, [det(camera_id="CAM-B")])

        self.assertEqual(cam_a[0].track_id, "CAM-A-T1")
        self.assertEqual(cam_b[0].track_id, "CAM-B-T1")


if __name__ == "__main__":
    unittest.main()
