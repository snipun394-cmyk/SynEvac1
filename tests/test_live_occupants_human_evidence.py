import unittest

from behavior_recognition.observation import RecognizedBehavior

from perception.models.human_observation import HumanClassification, HumanState

from human_evidence.reconciliation import HumanEvidenceConfig

from live_system.event_bus import EventBus, EventType

from live_occupants.manager import LiveOccupantManager


# =====================================================
# Live Human State & Assistance Perception Bridge milestone, Phase 23 --
# LiveOccupantManager.update()'s own extended classification/state
# evidence behavior. Covers required test-matrix items 1-11, 19-21,
# 25-30. No randomness anywhere in this file.
# =====================================================


class ClassificationSurvivalTests(unittest.TestCase):

    def test_1_unknown_classification_reaches_live_occupant_as_unknown(self):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
        )

        self.assertEqual(occupant.human_classification, HumanClassification.UNKNOWN)

    def test_2_genuine_adult_evidence_survives(self):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.ADULT, classification_confidence=0.9,
        )

        self.assertEqual(occupant.human_classification, HumanClassification.ADULT)
        self.assertEqual(occupant.human_classification_source, "CAM-A")

    def test_3_genuine_child_evidence_survives(self):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.CHILD, classification_confidence=0.8,
        )

        self.assertEqual(occupant.human_classification, HumanClassification.CHILD)

    def test_4_genuine_elderly_evidence_survives(self):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.ELDERLY, classification_confidence=0.8,
        )

        self.assertEqual(occupant.human_classification, HumanClassification.ELDERLY)

    def test_5_genuine_wheelchair_user_evidence_survives(self):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.WHEELCHAIR_USER, classification_confidence=0.8,
        )

        self.assertEqual(occupant.human_classification, HumanClassification.WHEELCHAIR_USER)


class StateSurvivalTests(unittest.TestCase):

    def _survives(self, state):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            state_evidence=state, state_confidence=0.8,
        )

        self.assertEqual(occupant.human_state, state)

    def test_6_walking_survives(self):
        self._survives(HumanState.WALKING)

    def test_7_running_survives(self):
        self._survives(HumanState.RUNNING)

    def test_8_fallen_survives(self):
        self._survives(HumanState.FALLEN)

    def test_9_crawling_survives(self):
        self._survives(HumanState.CRAWLING)

    def test_10_being_assisted_survives(self):
        self._survives(HumanState.BEING_ASSISTED)

    def test_11_helping_survives(self):
        self._survives(HumanState.HELPING_ANOTHER_OCCUPANT)


class UnknownDoesNotErasePersistenceTests(unittest.TestCase):

    def test_15_unknown_classification_does_not_erase_valid_recent_known_classification_immediately(self):

        manager = LiveOccupantManager()

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.ADULT, classification_confidence=0.8,
        )
        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 1.0,
            classification_evidence=HumanClassification.UNKNOWN,
        )

        self.assertEqual(occupant.human_classification, HumanClassification.ADULT)

    def test_16_genuine_changed_classification_updates_per_reconciliation_rules(self):

        manager = LiveOccupantManager()

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.CHILD, classification_confidence=0.5,
        )
        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 1.0,
            classification_evidence=HumanClassification.ADULT, classification_confidence=0.9,
        )

        # A strictly later timestamp with a genuine, different reading
        # from the SAME source supersedes the earlier one outright
        # (temporal recency -- Phase 8).
        self.assertEqual(occupant.human_classification, HumanClassification.ADULT)


class MultiCameraConflictTests(unittest.TestCase):

    def test_17_conflicting_cameras_produce_deterministic_resolution(self):

        manager = LiveOccupantManager()

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.ADULT, classification_confidence=0.5,
        )
        occupant = manager.update(
            "OCC-1", "CAM-B", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.CHILD, classification_confidence=0.9,
        )

        self.assertEqual(occupant.human_classification, HumanClassification.CHILD)

    def test_18_detection_order_does_not_change_reconciliation_result(self):

        def run(order):

            manager = LiveOccupantManager()
            occupant = None

            for camera_id, classification, confidence in order:
                occupant = manager.update(
                    "OCC-1", camera_id, "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
                    classification_evidence=classification, classification_confidence=confidence,
                )

            return occupant.human_classification

        forward = run([("CAM-B", HumanClassification.CHILD, 0.5), ("CAM-A", HumanClassification.ADULT, 0.5)])
        backward = run([("CAM-A", HumanClassification.ADULT, 0.5), ("CAM-B", HumanClassification.CHILD, 0.5)])

        self.assertEqual(forward, backward)

    def test_19_two_cameras_seeing_same_occupant_still_produce_one_live_occupant(self):

        manager = LiveOccupantManager()

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.ADULT, classification_confidence=0.9,
        )
        manager.update(
            "OCC-1", "CAM-B", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.ADULT, classification_confidence=0.9,
        )

        self.assertEqual(len(manager), 1)


class CrossCameraHandoverTests(unittest.TestCase):

    def test_20_handover_preserves_classification(self):

        manager = LiveOccupantManager()

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.ELDERLY, classification_confidence=0.8,
        )
        # Occupant hands over to CAM-B the next cycle -- no new
        # classification evidence from CAM-B (it may not classify at
        # all), the earlier known value survives.
        occupant = manager.update(
            "OCC-1", "CAM-B", "T1", "zone-2", "floor-1", (5.0, 5.0), None, None, 0.9, 1.0,
        )

        self.assertEqual(occupant.human_classification, HumanClassification.ELDERLY)

    def test_21_handover_preserves_current_state_per_freshness_rules(self):

        manager = LiveOccupantManager()

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            state_evidence=HumanState.RUNNING, state_confidence=0.8,
        )
        occupant = manager.update(
            "OCC-1", "CAM-B", "T1", "zone-2", "floor-1", (5.0, 5.0), None, None, 0.9, 1.0,
        )

        self.assertEqual(occupant.human_state, HumanState.RUNNING)


class StateTransitionTests(unittest.TestCase):

    def test_22_walking_to_running_works(self):

        manager = LiveOccupantManager()

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            state_evidence=HumanState.WALKING, state_confidence=0.8,
        )
        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 1.0,
            state_evidence=HumanState.RUNNING, state_confidence=0.8,
        )

        self.assertEqual(occupant.human_state, HumanState.RUNNING)

    def test_23_running_to_fallen_works(self):

        manager = LiveOccupantManager()

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            state_evidence=HumanState.RUNNING, state_confidence=0.8,
        )
        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 1.0,
            state_evidence=HumanState.FALLEN, state_confidence=0.8,
        )

        self.assertEqual(occupant.human_state, HumanState.FALLEN)

    def test_24_fallen_to_being_assisted_works(self):

        manager = LiveOccupantManager()

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            state_evidence=HumanState.FALLEN, state_confidence=0.8,
        )
        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 1.0,
            state_evidence=HumanState.BEING_ASSISTED, state_confidence=0.8,
        )

        self.assertEqual(occupant.human_state, HumanState.BEING_ASSISTED)


class NoSpamTests(unittest.TestCase):

    def test_25_identical_state_every_cycle_does_not_spam_history(self):

        manager = LiveOccupantManager()

        occupant = None
        for t in range(5):
            occupant = manager.update(
                "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, float(t),
                state_evidence=HumanState.WALKING, state_confidence=0.8,
            )

        self.assertEqual(len(occupant.history.state_changes), 1)

    def test_26_state_change_event_fires_exactly_once_per_transition(self):

        bus = EventBus()
        manager = LiveOccupantManager(event_bus=bus)

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            state_evidence=HumanState.WALKING, state_confidence=0.8,
        )
        for t in range(1, 4):
            manager.update(
                "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, float(t),
                state_evidence=HumanState.RUNNING, state_confidence=0.8,
            )

        events = bus.history_of(EventType.OCCUPANT_STATE_CHANGED)
        self.assertEqual(len(events), 1)  # WALKING->RUNNING once, never re-fired for the two repeats

    def test_27_classification_update_event_fires_only_on_genuine_update(self):

        bus = EventBus()
        manager = LiveOccupantManager(event_bus=bus)

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.ADULT, classification_confidence=0.8,
        )
        for t in range(1, 4):
            manager.update(
                "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, float(t),
                classification_evidence=HumanClassification.ADULT, classification_confidence=0.8,
            )

        events = bus.history_of(EventType.OCCUPANT_CLASSIFICATION_UPDATED)
        self.assertEqual(len(events), 0)  # created UNKNOWN->ADULT fires OCCUPANT_CREATED path, not this event


class StalenessExpiryTests(unittest.TestCase):

    def test_28_stale_evidence_expires_according_to_configuration(self):

        manager = LiveOccupantManager(human_evidence_config=HumanEvidenceConfig(state_staleness_seconds=2.0))

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            state_evidence=HumanState.FALLEN, state_confidence=0.8,
        )
        # No new state evidence -- re-observed with no state_evidence
        # supplied this cycle, well past the staleness window.
        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 5.0,
        )

        self.assertIsNone(occupant.human_state)

    def test_28_staleness_expires_even_while_missing_via_sweep(self):

        manager = LiveOccupantManager(human_evidence_config=HumanEvidenceConfig(state_staleness_seconds=2.0))

        manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            state_evidence=HumanState.FALLEN, state_confidence=0.8,
        )
        manager.sweep_missing(5.0, seen_occupant_ids=set())

        self.assertIsNone(manager.get("OCC-1").human_state)


class YoloUnknownBoundaryTests(unittest.TestCase):

    def test_29_current_yolo_only_pipeline_remains_unknown_classification(self):

        # Mirrors human_detection.yolo_human_detector.YOLOHumanDetector's
        # own honest RawHumanDetection(classification_evidence=
        # HumanClassification.UNKNOWN) output -- proving the manager
        # itself never fabricates a richer classification from nothing.
        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
            classification_evidence=HumanClassification.UNKNOWN,
        )

        self.assertEqual(occupant.human_classification, HumanClassification.UNKNOWN)

    def test_30_missing_human_state_evidence_remains_none(self):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, None, 0.9, 0.0,
        )

        self.assertIsNone(occupant.human_state)


class PossiblyFallenSeparationTests(unittest.TestCase):

    # Phase 6's own hard safety requirement -- RecognizedBehavior.
    # POSSIBLY_FALLEN must NEVER become HumanState.FALLEN anywhere in
    # this manager.

    def test_possibly_fallen_behavior_never_promotes_to_fallen_state(self):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, RecognizedBehavior.POSSIBLY_FALLEN,
            0.9, 0.0,
        )

        self.assertEqual(occupant.behavior, RecognizedBehavior.POSSIBLY_FALLEN)
        self.assertIsNone(occupant.human_state)

    def test_possibly_fallen_and_confirmed_fallen_are_independent_fields(self):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (0.0, 0.0), None, RecognizedBehavior.POSSIBLY_FALLEN,
            0.9, 0.0, state_evidence=HumanState.FALLEN, state_confidence=0.9,
        )

        self.assertEqual(occupant.behavior, RecognizedBehavior.POSSIBLY_FALLEN)
        self.assertEqual(occupant.human_state, HumanState.FALLEN)


if __name__ == "__main__":
    unittest.main()
