import unittest

from perception.models.human_observation import HumanClassification, HumanState

from multi_camera_fusion.confidence import (
    HighestConfidenceStrategy,
    MultiCameraConfidenceBoostStrategy,
    WeightedAverageConfidenceStrategy,
)
from multi_camera_fusion.engine import MultiCameraFusionEngine
from multi_camera_fusion.track import FusedTrack, FusionResult, TrackHistory


class FakeDetection:

    # A minimal stand-in for virtual_camera.detection.Detection --
    # MultiCameraFusionEngine never imports that type at all (see
    # engine.py's own docstring and this module's own
    # MultiCameraFusionPackageDependencyDirectionTests below), so
    # nothing here needs to subclass it.

    def __init__(
        self, occupant_id, camera_id, floor_id="floor-1", zone_id="zone-1",
        confidence=1.0, classification=HumanClassification.ADULT,
        human_state=HumanState.WALKING, is_false_positive=False,
    ):

        self.occupant_id = occupant_id
        self.camera_id = camera_id
        self.floor_id = floor_id
        self.zone_id = zone_id
        self.confidence = confidence
        self.classification = classification
        self.human_state = human_state
        self.is_false_positive = is_false_positive


class SingleCameraTrackingTests(unittest.TestCase):

    def test_one_detection_produces_one_fused_track(self):

        engine = MultiCameraFusionEngine()

        detection = FakeDetection("occ-1", "cam-1", classification=HumanClassification.CHILD, human_state=HumanState.STANDING)

        result = engine.fuse([detection], time=0.0)

        self.assertEqual(len(result.tracks), 1)

        track = result.tracks[0]
        self.assertIsInstance(track, FusedTrack)
        self.assertEqual(track.track_id, "occ-1")
        self.assertEqual(track.source_camera_ids, ("cam-1",))
        self.assertEqual(track.floor_id, "floor-1")
        self.assertEqual(track.zone_id, "zone-1")
        self.assertEqual(track.classification, HumanClassification.CHILD)
        self.assertEqual(track.human_state, HumanState.STANDING)
        self.assertEqual(track.confidence, 1.0)
        self.assertEqual(track.timestamp, 0.0)

    def test_no_detections_produces_no_tracks(self):

        engine = MultiCameraFusionEngine()

        result = engine.fuse([], time=0.0)

        self.assertEqual(result.tracks, ())

    def test_two_different_occupants_produce_two_distinct_tracks(self):

        engine = MultiCameraFusionEngine()

        result = engine.fuse(
            [FakeDetection("occ-1", "cam-1"), FakeDetection("occ-2", "cam-1")], time=0.0,
        )

        self.assertEqual({t.track_id for t in result.tracks}, {"occ-1", "occ-2"})


class MultiCameraOverlapTests(unittest.TestCase):

    def test_two_cameras_seeing_the_same_occupant_fuse_into_one_track(self):

        engine = MultiCameraFusionEngine()

        detections = [
            FakeDetection("occ-1", "cam-1", confidence=0.8),
            FakeDetection("occ-1", "cam-2", confidence=0.9),
        ]

        result = engine.fuse(detections, time=0.0)

        self.assertEqual(len(result.tracks), 1)

        track = result.tracks[0]
        self.assertEqual(track.track_id, "occ-1")
        self.assertEqual(track.source_camera_ids, ("cam-1", "cam-2"))

    def test_source_camera_ids_are_sorted_regardless_of_input_order(self):

        engine = MultiCameraFusionEngine()

        detections = [
            FakeDetection("occ-1", "cam-zebra"),
            FakeDetection("occ-1", "cam-alpha"),
        ]

        result = engine.fuse(detections, time=0.0)

        self.assertEqual(result.tracks[0].source_camera_ids, ("cam-alpha", "cam-zebra"))

    def test_fused_fields_come_from_the_highest_confidence_source(self):

        engine = MultiCameraFusionEngine()

        detections = [
            FakeDetection(
                "occ-1", "cam-1", zone_id="zone-far", confidence=0.4,
                classification=HumanClassification.UNKNOWN, human_state=HumanState.STANDING,
            ),
            FakeDetection(
                "occ-1", "cam-2", zone_id="zone-near", confidence=0.95,
                classification=HumanClassification.WHEELCHAIR_USER, human_state=HumanState.WALKING,
            ),
        ]

        result = engine.fuse(detections, time=0.0)
        track = result.tracks[0]

        self.assertEqual(track.zone_id, "zone-near")
        self.assertEqual(track.classification, HumanClassification.WHEELCHAIR_USER)
        self.assertEqual(track.human_state, HumanState.WALKING)


class DuplicateSuppressionTests(unittest.TestCase):

    def test_three_cameras_on_the_same_occupant_still_yield_exactly_one_track(self):

        engine = MultiCameraFusionEngine()

        detections = [
            FakeDetection("occ-1", "cam-1"),
            FakeDetection("occ-1", "cam-2"),
            FakeDetection("occ-1", "cam-3"),
        ]

        result = engine.fuse(detections, time=0.0)

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].source_camera_ids, ("cam-1", "cam-2", "cam-3"))

    def test_false_positives_never_produce_a_track(self):

        engine = MultiCameraFusionEngine()

        detections = [
            FakeDetection("occ-1", "cam-1"),
            FakeDetection("false-positive-abc", "cam-1", is_false_positive=True),
        ]

        result = engine.fuse(detections, time=0.0)

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].track_id, "occ-1")

    def test_only_false_positives_produces_no_tracks_at_all(self):

        engine = MultiCameraFusionEngine()

        result = engine.fuse(
            [FakeDetection("fp-1", "cam-1", is_false_positive=True)], time=0.0,
        )

        self.assertEqual(result.tracks, ())


class CameraHandoverTests(unittest.TestCase):

    def test_handover_continues_the_same_track_not_a_new_one(self):

        engine = MultiCameraFusionEngine()

        result_1 = engine.fuse([FakeDetection("occ-1", "cam-1", zone_id="zone-a")], time=0.0)
        self.assertEqual(len(result_1.tracks), 1)

        # cam-1 no longer sees them; cam-2 now does -- same person.
        result_2 = engine.fuse([FakeDetection("occ-1", "cam-2", zone_id="zone-b")], time=1.0)

        self.assertEqual(len(result_2.tracks), 1)
        self.assertEqual(result_2.tracks[0].track_id, "occ-1")

    def test_previous_and_current_camera_reflect_the_handover(self):

        engine = MultiCameraFusionEngine()

        engine.fuse([FakeDetection("occ-1", "cam-1")], time=0.0)

        history_before = engine.track_history("occ-1")
        self.assertIsNone(history_before.previous_camera_id)
        self.assertEqual(history_before.current_camera_id, "cam-1")

        engine.fuse([FakeDetection("occ-1", "cam-2")], time=1.0)

        history_after = engine.track_history("occ-1")
        self.assertEqual(history_after.previous_camera_id, "cam-1")
        self.assertEqual(history_after.current_camera_id, "cam-2")

    def test_camera_stays_settled_after_the_handover_completes(self):

        engine = MultiCameraFusionEngine()

        engine.fuse([FakeDetection("occ-1", "cam-1")], time=0.0)
        engine.fuse([FakeDetection("occ-1", "cam-2")], time=1.0)
        engine.fuse([FakeDetection("occ-1", "cam-2")], time=2.0)

        history = engine.track_history("occ-1")
        self.assertEqual(history.previous_camera_id, "cam-2")
        self.assertEqual(history.current_camera_id, "cam-2")

    def test_track_history_survives_a_tick_with_no_detection_at_all(self):

        engine = MultiCameraFusionEngine()

        engine.fuse([FakeDetection("occ-1", "cam-1")], time=0.0)

        # Momentarily unseen by any camera (e.g. a blind corridor).
        result = engine.fuse([], time=1.0)
        self.assertEqual(result.tracks, ())

        history = engine.track_history("occ-1")
        self.assertIsNotNone(history)
        self.assertEqual(history.current_camera_id, "cam-1")

        # Reappears -- continues the same track.
        result_2 = engine.fuse([FakeDetection("occ-1", "cam-2")], time=2.0)
        self.assertEqual(result_2.tracks[0].track_id, "occ-1")


class ZoneAndFloorTransitionTests(unittest.TestCase):

    def test_zone_change_is_recorded_as_a_transition(self):

        engine = MultiCameraFusionEngine()

        engine.fuse([FakeDetection("occ-1", "cam-1", zone_id="zone-a")], time=0.0)
        engine.fuse([FakeDetection("occ-1", "cam-1", zone_id="zone-b")], time=1.0)

        history = engine.track_history("occ-1")

        self.assertEqual(len(history.zone_transitions), 1)
        transition = history.zone_transitions[0]
        self.assertEqual(transition.time, 1.0)
        self.assertEqual(transition.from_value, "zone-a")
        self.assertEqual(transition.to_value, "zone-b")
        self.assertEqual(history.current_zone_id, "zone-b")

    def test_staying_in_the_same_zone_records_no_transition(self):

        engine = MultiCameraFusionEngine()

        engine.fuse([FakeDetection("occ-1", "cam-1", zone_id="zone-a")], time=0.0)
        engine.fuse([FakeDetection("occ-1", "cam-1", zone_id="zone-a")], time=1.0)

        self.assertEqual(engine.track_history("occ-1").zone_transitions, ())

    def test_floor_change_is_recorded_independently_of_zone(self):

        engine = MultiCameraFusionEngine()

        engine.fuse([FakeDetection("occ-1", "cam-1", floor_id="floor-1", zone_id="zone-a")], time=0.0)
        engine.fuse([FakeDetection("occ-1", "cam-2", floor_id="floor-2", zone_id="zone-a")], time=1.0)

        history = engine.track_history("occ-1")

        self.assertEqual(len(history.floor_transitions), 1)
        self.assertEqual(history.floor_transitions[0].from_value, "floor-1")
        self.assertEqual(history.floor_transitions[0].to_value, "floor-2")
        # zone_id string is the same across floors here, so no zone
        # transition should be recorded even though the floor changed.
        self.assertEqual(history.zone_transitions, ())

    def test_history_is_bounded_to_max_history_length(self):

        engine = MultiCameraFusionEngine(max_history_length=3)

        for i in range(10):
            engine.fuse([FakeDetection("occ-1", "cam-1", zone_id=f"zone-{i}")], time=float(i))

        history = engine.track_history("occ-1")

        self.assertLessEqual(len(history.zone_transitions), 3)
        # The most recent transitions are kept, oldest dropped first.
        self.assertEqual(history.zone_transitions[-1].to_value, "zone-9")


class ConfidenceFusionTests(unittest.TestCase):

    def test_highest_confidence_strategy(self):

        strategy = HighestConfidenceStrategy()
        self.assertEqual(strategy.fuse((0.6, 0.9, 0.3)), 0.9)

    def test_weighted_average_confidence_strategy(self):

        strategy = WeightedAverageConfidenceStrategy()
        self.assertAlmostEqual(strategy.fuse((0.6, 0.9)), 0.75)

    def test_multi_camera_confidence_boost_strategy(self):

        strategy = MultiCameraConfidenceBoostStrategy()
        # base (highest) = 0.9, one extra camera -> +0.05
        self.assertAlmostEqual(strategy.fuse((0.6, 0.9)), 0.95)

    def test_multi_camera_confidence_boost_is_clamped_at_one(self):

        strategy = MultiCameraConfidenceBoostStrategy()
        self.assertEqual(strategy.fuse((0.99, 0.99, 0.99, 0.99, 0.99)), 1.0)

    def test_multi_camera_confidence_boost_wraps_a_custom_base_strategy(self):

        strategy = MultiCameraConfidenceBoostStrategy(base_strategy=WeightedAverageConfidenceStrategy())
        # base (average) = 0.75, one extra camera -> +0.05
        self.assertAlmostEqual(strategy.fuse((0.6, 0.9)), 0.80)

    def test_engine_uses_the_configured_strategy(self):

        engine = MultiCameraFusionEngine(confidence_strategy=WeightedAverageConfidenceStrategy())

        detections = [
            FakeDetection("occ-1", "cam-1", confidence=0.6),
            FakeDetection("occ-1", "cam-2", confidence=0.8),
        ]

        result = engine.fuse(detections, time=0.0)

        self.assertAlmostEqual(result.tracks[0].confidence, 0.7)

    def test_engine_defaults_to_highest_confidence_strategy(self):

        engine = MultiCameraFusionEngine()

        detections = [
            FakeDetection("occ-1", "cam-1", confidence=0.5),
            FakeDetection("occ-1", "cam-2", confidence=0.95),
        ]

        result = engine.fuse(detections, time=0.0)

        self.assertEqual(result.tracks[0].confidence, 0.95)


class DeterminismTests(unittest.TestCase):

    def _detections_for_tick(self, tick):

        if tick == 0:
            return [FakeDetection("occ-1", "cam-1", zone_id="zone-a", confidence=0.8)]

        if tick == 1:
            return [
                FakeDetection("occ-1", "cam-1", zone_id="zone-a", confidence=0.7),
                FakeDetection("occ-1", "cam-2", zone_id="zone-a", confidence=0.9),
            ]

        return [FakeDetection("occ-1", "cam-2", zone_id="zone-b", confidence=0.85)]

    def test_two_independently_constructed_engines_produce_identical_results(self):

        engine_a = MultiCameraFusionEngine(confidence_strategy=MultiCameraConfidenceBoostStrategy())
        engine_b = MultiCameraFusionEngine(confidence_strategy=MultiCameraConfidenceBoostStrategy())

        for tick in range(3):

            detections = self._detections_for_tick(tick)

            result_a = engine_a.fuse(detections, time=float(tick))
            result_b = engine_b.fuse(detections, time=float(tick))

            self.assertEqual(result_a, result_b)

        self.assertEqual(engine_a.track_history("occ-1"), engine_b.track_history("occ-1"))

    def test_a_single_fuse_call_is_reproducible_on_a_fresh_engine(self):

        # A meaningful determinism check for a *stateful* engine is not
        # "call fuse() twice on the same instance at the same
        # timestamp" -- history legitimately advances every call
        # regardless of whether time moved (previous_camera_id tracks
        # "whatever was current as of the last call", by construction --
        # see MultiCameraFusionEngine._update_history()), so a second
        # call at an unchanged timestamp is not a no-op and should not
        # be expected to look identical to the first. The real
        # guarantee is: two independently-constructed engines, given
        # the exact same single call, produce byte-identical output --
        # which is what this asserts.

        detections = [FakeDetection("occ-1", "cam-1", confidence=0.5)]

        result_a = MultiCameraFusionEngine().fuse(detections, time=5.0)
        result_b = MultiCameraFusionEngine().fuse(detections, time=5.0)

        self.assertEqual(result_a, result_b)


class FusionResultTests(unittest.TestCase):

    def test_track_lookup_by_id(self):

        engine = MultiCameraFusionEngine()

        result = engine.fuse(
            [FakeDetection("occ-1", "cam-1"), FakeDetection("occ-2", "cam-1")], time=0.0,
        )

        self.assertEqual(result.track("occ-1").track_id, "occ-1")
        self.assertEqual(result.track("occ-2").track_id, "occ-2")
        self.assertIsNone(result.track("no-such-track"))


class TrackHistoryLookupTests(unittest.TestCase):

    def test_track_history_is_none_before_any_fuse_call(self):

        engine = MultiCameraFusionEngine()

        self.assertIsNone(engine.track_history("occ-1"))


class MultiCameraFusionPackageDependencyDirectionTests(unittest.TestCase):

    # Same regex-scan-the-source-files convention every other package
    # boundary in this codebase already enforces (see
    # tests.test_camera_manager.CameraManagerPackageDependencyDirectionTests,
    # tests.test_virtual_camera.VirtualCameraPackageDependencyDirectionTests).
    # multi_camera_fusion/ must reach detections only through whatever
    # duck-typed object it is handed -- it does not even need to import
    # virtual_camera/camera_manager/visibility to do that.

    def test_never_imports_simulation_or_camera_internals(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "multi_camera_fusion"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(simulator|ground_truth|behavior|behavior_library|behaviour_profile_resolver|"
            r"simulation_runtime|hazard_evolution|ai_training|rl_training|advisory_system|"
            r"command_center|designer|virtual_camera|visibility|camera_manager|"
            r"cv2|torch|ultralytics|onvif)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"multi_camera_fusion/{path.name} imports a simulation/camera/decision-layer "
                f"module directly -- it must only fuse whatever Detection-shaped objects it is "
                f"handed",
            )


if __name__ == "__main__":
    unittest.main()
