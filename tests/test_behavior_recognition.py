import unittest

from tracking.track_state import TrackState
from tracking.tracked_human import TrackedHuman

from behavior_recognition.behavior_history import BehaviorHistory
from behavior_recognition.metrics import compute_metrics
from behavior_recognition.observation import RecognizedBehavior
from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer


# =====================================================
# Human Behavior Recognition Framework milestone, Phase 9 -- deterministic,
# offline unit tests directly against RuleBasedBehaviorRecognizer. No
# randomness anywhere in this file.
# =====================================================


def th(
    track_id="T1", camera_id="CAM-1", box=(0.0, 0.0, 10.0, 20.0), confidence=0.9,
    state=TrackState.TRACKED, age=1, frames_seen=1, frames_missing=0, last_timestamp=0.0,
):

    return TrackedHuman(
        track_id=track_id, camera_id=camera_id, bounding_box=box, confidence=confidence,
        state=state, age=age, frames_seen=frames_seen, frames_missing=frames_missing,
        last_timestamp=last_timestamp,
    )


class StationaryTests(unittest.TestCase):

    def test_1_stationary_person_recognized_after_enough_samples(self):

        recognizer = RuleBasedBehaviorRecognizer()

        for i in range(5):
            result = recognizer.recognize("CAM-1", float(i), [th(box=(0.0, 0.0, 10.0, 20.0))])

        self.assertEqual(result[0].recognized_behavior, RecognizedBehavior.STATIONARY)
        self.assertGreater(result[0].supporting_metrics.velocity, -0.001)
        self.assertLess(result[0].supporting_metrics.velocity, 5.0)


class WalkingTests(unittest.TestCase):

    def test_2_steady_moderate_displacement_recognized_as_walking(self):

        recognizer = RuleBasedBehaviorRecognizer()

        result = None
        for i in range(5):
            x = i * 20.0  # 20 px/s -- between stationary (<=5) and running (>=80) thresholds
            result = recognizer.recognize("CAM-1", float(i), [th(box=(x, 0.0, x + 10.0, 20.0))])

        self.assertEqual(result[0].recognized_behavior, RecognizedBehavior.WALKING)


class RunningTests(unittest.TestCase):

    def test_3_large_displacement_recognized_as_running(self):

        recognizer = RuleBasedBehaviorRecognizer()

        result = None
        for i in range(5):
            x = i * 150.0  # 150 px/s -- well above the running threshold (80)
            result = recognizer.recognize("CAM-1", float(i), [th(box=(x, 0.0, x + 10.0, 20.0))])

        self.assertEqual(result[0].recognized_behavior, RecognizedBehavior.RUNNING)


class DirectionTests(unittest.TestCase):

    def test_4_direction_reflects_the_most_recent_displacement(self):

        recognizer = RuleBasedBehaviorRecognizer()

        recognizer.recognize("CAM-1", 0.0, [th(box=(0.0, 0.0, 10.0, 20.0))])
        moving_right = recognizer.recognize("CAM-1", 1.0, [th(box=(50.0, 0.0, 60.0, 20.0), last_timestamp=1.0)])

        self.assertAlmostEqual(moving_right[0].supporting_metrics.direction, 0.0, places=3)  # +x direction -> 0 radians

        moving_up = recognizer.recognize("CAM-1", 2.0, [th(box=(50.0, -50.0, 60.0, -30.0), last_timestamp=2.0)])

        import math
        self.assertAlmostEqual(moving_up[0].supporting_metrics.direction, -math.pi / 2, places=3)  # -y direction


class ShortAndLongTrackTests(unittest.TestCase):

    def test_5_short_track_with_a_single_sample_is_unknown(self):

        recognizer = RuleBasedBehaviorRecognizer()

        result = recognizer.recognize("CAM-1", 0.0, [th()])

        self.assertEqual(result[0].recognized_behavior, RecognizedBehavior.UNKNOWN)
        self.assertEqual(result[0].confidence, 0.0)
        self.assertIsNone(result[0].supporting_metrics.velocity)

    def test_6_long_track_confidence_saturates(self):

        recognizer = RuleBasedBehaviorRecognizer(confidence_saturation_samples=5)

        result = None
        for i in range(20):
            result = recognizer.recognize("CAM-1", float(i), [th(box=(0.0, 0.0, 10.0, 20.0))])

        self.assertEqual(result[0].confidence, 1.0)


class OcclusionTests(unittest.TestCase):

    def test_7_missing_track_produces_no_observation_and_does_not_pollute_history(self):

        recognizer = RuleBasedBehaviorRecognizer()

        recognizer.recognize("CAM-1", 0.0, [th(box=(0.0, 0.0, 10.0, 20.0))])

        missing_result = recognizer.recognize(
            "CAM-1", 1.0, [th(state=TrackState.MISSING, frames_missing=1, last_timestamp=0.0)],
        )
        self.assertEqual(missing_result, ())

        # History still has only the one real sample -- the missing
        # cycle appended nothing.
        self.assertEqual(len(recognizer.history.recent("CAM-1", "T1")), 1)

        reappeared = recognizer.recognize("CAM-1", 2.0, [th(box=(0.0, 0.0, 10.0, 20.0), last_timestamp=2.0)])
        self.assertEqual(len(reappeared), 1)


class TrackResetTests(unittest.TestCase):

    def test_8_expired_track_clears_its_history(self):

        recognizer = RuleBasedBehaviorRecognizer()

        recognizer.recognize("CAM-1", 0.0, [th(box=(0.0, 0.0, 10.0, 20.0))])
        self.assertEqual(len(recognizer.history), 1)

        expired_result = recognizer.recognize("CAM-1", 1.0, [th(state=TrackState.EXPIRED)])
        self.assertEqual(expired_result, ())
        self.assertEqual(len(recognizer.history), 0)  # no leak

        # A brand new track reusing the same track_id string starts
        # fresh -- UNKNOWN on its first sample, exactly like any other
        # first-ever sighting.
        fresh = recognizer.recognize("CAM-1", 2.0, [th(box=(999.0, 999.0, 1009.0, 1019.0), last_timestamp=2.0)])
        self.assertEqual(fresh[0].recognized_behavior, RecognizedBehavior.UNKNOWN)


class HistoryTrimmingTests(unittest.TestCase):

    def test_9_history_never_grows_past_max_length(self):

        history = BehaviorHistory(max_length=3)

        for i in range(10):
            history.append("CAM-1", "T1", float(i), (float(i), 0.0, float(i) + 10.0, 20.0))

        self.assertEqual(len(history.recent("CAM-1", "T1")), 3)
        # Only the 3 most recent samples survive.
        self.assertEqual(history.recent("CAM-1", "T1")[0][0], 7.0)
        self.assertEqual(history.recent("CAM-1", "T1")[-1][0], 9.0)

    def test_9_clear_removes_a_single_track_without_affecting_others(self):

        history = BehaviorHistory()

        history.append("CAM-1", "T1", 0.0, (0.0, 0.0, 10.0, 20.0))
        history.append("CAM-1", "T2", 0.0, (0.0, 0.0, 10.0, 20.0))

        history.clear("CAM-1", "T1")

        self.assertEqual(history.recent("CAM-1", "T1"), ())
        self.assertEqual(len(history.recent("CAM-1", "T2")), 1)


class ConfidenceHandlingTests(unittest.TestCase):

    def test_10_confidence_scales_with_sample_count(self):

        recognizer = RuleBasedBehaviorRecognizer(confidence_saturation_samples=4)

        confidences = []
        for i in range(4):
            result = recognizer.recognize("CAM-1", float(i), [th(box=(0.0, 0.0, 10.0, 20.0))])
            confidences.append(result[0].confidence)

        # First sample: UNKNOWN, 0.0. Then increasing toward 1.0.
        self.assertEqual(confidences[0], 0.0)
        self.assertTrue(all(confidences[i] <= confidences[i + 1] for i in range(len(confidences) - 1)))
        self.assertEqual(confidences[-1], 1.0)

    def test_10_unknown_always_has_zero_confidence(self):

        recognizer = RuleBasedBehaviorRecognizer()

        result = recognizer.recognize("CAM-1", 0.0, [th()])

        self.assertEqual(result[0].recognized_behavior, RecognizedBehavior.UNKNOWN)
        self.assertEqual(result[0].confidence, 0.0)


class MultipleCamerasTests(unittest.TestCase):

    def test_11_two_cameras_tracked_independently(self):

        recognizer = RuleBasedBehaviorRecognizer()

        recognizer.recognize("CAM-A", 0.0, [th(camera_id="CAM-A", box=(0.0, 0.0, 10.0, 20.0))])
        recognizer.recognize("CAM-B", 0.0, [th(camera_id="CAM-B", box=(500.0, 500.0, 510.0, 520.0))])

        self.assertEqual(len(recognizer.history.recent("CAM-A", "T1")), 1)
        self.assertEqual(len(recognizer.history.recent("CAM-B", "T1")), 1)

        recognizer.recognize("CAM-A", 1.0, [th(camera_id="CAM-A", box=(0.0, 0.0, 10.0, 20.0), last_timestamp=1.0)])

        # CAM-B's history is untouched by a CAM-A-only cycle.
        self.assertEqual(len(recognizer.history.recent("CAM-B", "T1")), 1)


class MultiplePeopleTests(unittest.TestCase):

    def test_12_two_people_on_one_camera_classified_independently(self):

        recognizer = RuleBasedBehaviorRecognizer()

        for i in range(5):
            stationary = th(track_id="T1", box=(0.0, 0.0, 10.0, 20.0), last_timestamp=float(i))
            fast_x = i * 150.0
            running = th(track_id="T2", box=(fast_x, 0.0, fast_x + 10.0, 20.0), last_timestamp=float(i))
            result = recognizer.recognize("CAM-1", float(i), [stationary, running])

        by_track = {obs.track_id: obs for obs in result}
        self.assertEqual(by_track["T1"].recognized_behavior, RecognizedBehavior.STATIONARY)
        self.assertEqual(by_track["T2"].recognized_behavior, RecognizedBehavior.RUNNING)


class BehaviorPersistenceTests(unittest.TestCase):

    def test_13_consistent_walking_pace_persists_across_cycles(self):

        recognizer = RuleBasedBehaviorRecognizer()

        behaviors = []
        for i in range(6):
            x = i * 20.0
            result = recognizer.recognize("CAM-1", float(i), [th(box=(x, 0.0, x + 10.0, 20.0))])
            behaviors.append(result[0].recognized_behavior)

        # First cycle is always UNKNOWN (no velocity yet); every cycle
        # after that stays WALKING.
        self.assertEqual(behaviors[0], RecognizedBehavior.UNKNOWN)
        self.assertTrue(all(b == RecognizedBehavior.WALKING for b in behaviors[1:]))


class EmptyAndNoTrackTests(unittest.TestCase):

    def test_14_empty_frame_returns_nothing(self):

        recognizer = RuleBasedBehaviorRecognizer()

        self.assertEqual(recognizer.recognize("CAM-1", 0.0, []), ())

    def test_15_only_missing_or_expired_tracks_returns_nothing(self):

        recognizer = RuleBasedBehaviorRecognizer()

        result = recognizer.recognize(
            "CAM-1", 0.0,
            [th(track_id="T1", state=TrackState.MISSING), th(track_id="T2", state=TrackState.EXPIRED)],
        )

        self.assertEqual(result, ())


class PossiblyFallenHeuristicTests(unittest.TestCase):

    def test_disabled_by_default_never_reports_possibly_fallen(self):

        recognizer = RuleBasedBehaviorRecognizer()  # default: heuristic disabled

        result = None
        for i in range(5):
            # Wide, low box, perfectly stationary -- would trip the
            # heuristic if enabled.
            result = recognizer.recognize("CAM-1", float(i), [th(box=(0.0, 0.0, 40.0, 10.0))])

        self.assertNotEqual(result[0].recognized_behavior, RecognizedBehavior.POSSIBLY_FALLEN)
        self.assertEqual(result[0].recognized_behavior, RecognizedBehavior.STATIONARY)

    def test_enabled_reports_possibly_fallen_for_a_wide_low_stationary_box(self):

        recognizer = RuleBasedBehaviorRecognizer(
            enable_possibly_fallen_heuristic=True,
            possibly_fallen_min_stationary_duration=2.0,
        )

        result = None
        for i in range(6):
            result = recognizer.recognize("CAM-1", float(i), [th(box=(0.0, 0.0, 40.0, 10.0))])

        self.assertEqual(result[0].recognized_behavior, RecognizedBehavior.POSSIBLY_FALLEN)
        # Confidence is scaled down relative to a plain STATIONARY call.
        self.assertLess(result[0].confidence, 1.0)

    def test_enabled_but_tall_box_never_reports_possibly_fallen(self):

        recognizer = RuleBasedBehaviorRecognizer(enable_possibly_fallen_heuristic=True)

        result = None
        for i in range(6):
            # Tall, narrow box (normal standing aspect ratio), stationary.
            result = recognizer.recognize("CAM-1", float(i), [th(box=(0.0, 0.0, 10.0, 40.0))])

        self.assertEqual(result[0].recognized_behavior, RecognizedBehavior.STATIONARY)


class MetricsPurityTests(unittest.TestCase):

    def test_compute_metrics_is_a_pure_function_of_samples_and_thresholds(self):

        samples = [(0.0, (0.0, 0.0, 10.0, 10.0)), (1.0, (10.0, 0.0, 20.0, 10.0))]

        first = compute_metrics(samples, track_age=2, stationary_velocity_threshold=5.0)
        second = compute_metrics(samples, track_age=2, stationary_velocity_threshold=5.0)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first.velocity, 10.0)
        self.assertEqual(first.track_age, 2)


if __name__ == "__main__":
    unittest.main()
