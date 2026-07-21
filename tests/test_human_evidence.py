import unittest

from perception.models.human_observation import HumanClassification, HumanState

from human_evidence.reconciliation import (
    HumanEvidenceConfig, apply_classification_staleness, apply_state_staleness,
    reconcile_classification, reconcile_state,
)


# =====================================================
# Live Human State & Assistance Perception Bridge milestone, Phase 7/8/
# 23 -- the reconciliation module is pure, deterministic functions with
# no randomness and no dependency on call order; every test below
# proves that directly (item 15-18, 28 of the milestone's own required
# test matrix).
# =====================================================


class KnownBeatsUnknownTests(unittest.TestCase):

    def test_unknown_reading_never_overwrites_known_classification(self):

        result = reconcile_classification(
            existing_classification=HumanClassification.ADULT, existing_confidence=0.8,
            existing_source="CAM-A", existing_last_observed_at=5.0,
            new_classification=HumanClassification.UNKNOWN, new_confidence=None, new_source="CAM-B",
            timestamp=6.0,
        )

        self.assertEqual(result, (HumanClassification.ADULT, 0.8, "CAM-A", 5.0))

    def test_none_reading_never_overwrites_known_state(self):

        result = reconcile_state(
            existing_state=HumanState.WALKING, existing_confidence=0.9, existing_source="CAM-A",
            existing_last_observed_at=5.0,
            new_state=None, new_confidence=None, new_source="CAM-B", timestamp=6.0,
        )

        self.assertEqual(result, (HumanState.WALKING, 0.9, "CAM-A", 5.0))

    def test_known_classification_adopted_when_nothing_existed_before(self):

        result = reconcile_classification(
            existing_classification=HumanClassification.UNKNOWN, existing_confidence=None,
            existing_source=None, existing_last_observed_at=None,
            new_classification=HumanClassification.CHILD, new_confidence=0.7, new_source="CAM-A",
            timestamp=1.0,
        )

        self.assertEqual(result, (HumanClassification.CHILD, 0.7, "CAM-A", 1.0))

    def test_known_state_adopted_when_nothing_existed_before(self):

        result = reconcile_state(
            existing_state=None, existing_confidence=None, existing_source=None, existing_last_observed_at=None,
            new_state=HumanState.FALLEN, new_confidence=0.85, new_source="CAM-A", timestamp=1.0,
        )

        self.assertEqual(result, (HumanState.FALLEN, 0.85, "CAM-A", 1.0))


class AgreementRefreshTests(unittest.TestCase):

    def test_agreeing_classification_refreshes_recency_and_confidence(self):

        result = reconcile_classification(
            existing_classification=HumanClassification.ADULT, existing_confidence=0.6,
            existing_source="CAM-A", existing_last_observed_at=1.0,
            new_classification=HumanClassification.ADULT, new_confidence=0.95, new_source="CAM-A",
            timestamp=2.0,
        )

        self.assertEqual(result, (HumanClassification.ADULT, 0.95, "CAM-A", 2.0))


class TemporalRecencyTests(unittest.TestCase):

    def test_sequential_state_change_at_a_later_timestamp_always_wins(self):

        # A later, genuinely different reading always supersedes an
        # earlier one, regardless of confidence -- HumanState must
        # update readily (Phase 6/8).
        result = reconcile_state(
            existing_state=HumanState.WALKING, existing_confidence=0.99, existing_source="CAM-A",
            existing_last_observed_at=1.0,
            new_state=HumanState.RUNNING, new_confidence=0.4, new_source="CAM-A", timestamp=2.0,
        )

        self.assertEqual(result, (HumanState.RUNNING, 0.4, "CAM-A", 2.0))

    def test_walking_to_running_to_fallen_to_being_assisted_all_update(self):

        state = (None, None, None, None)

        for t, new_state in enumerate(
            (HumanState.WALKING, HumanState.RUNNING, HumanState.FALLEN, HumanState.BEING_ASSISTED), start=1,
        ):
            state = reconcile_state(
                existing_state=state[0], existing_confidence=state[1], existing_source=state[2],
                existing_last_observed_at=state[3],
                new_state=new_state, new_confidence=0.8, new_source="CAM-A", timestamp=float(t),
            )
            self.assertEqual(state[0], new_state)


class DeterministicConflictResolutionTests(unittest.TestCase):

    # Phase 23 tests 17/18 -- conflicting cameras produce a deterministic
    # resolution, and that resolution does not depend on which camera's
    # reading was applied first.

    def test_same_cycle_conflict_resolved_by_higher_confidence(self):

        result = reconcile_classification(
            existing_classification=HumanClassification.ADULT, existing_confidence=0.5,
            existing_source="CAM-A", existing_last_observed_at=1.0,
            new_classification=HumanClassification.CHILD, new_confidence=0.9, new_source="CAM-B",
            timestamp=1.0,
        )

        self.assertEqual(result[0], HumanClassification.CHILD)

    def test_same_cycle_conflict_tie_breaks_on_lexicographically_smaller_source(self):

        result_a_then_b = reconcile_classification(
            existing_classification=HumanClassification.ADULT, existing_confidence=0.5,
            existing_source="CAM-A", existing_last_observed_at=1.0,
            new_classification=HumanClassification.CHILD, new_confidence=0.5, new_source="CAM-B",
            timestamp=1.0,
        )
        result_b_then_a = reconcile_classification(
            existing_classification=HumanClassification.CHILD, existing_confidence=0.5,
            existing_source="CAM-B", existing_last_observed_at=1.0,
            new_classification=HumanClassification.ADULT, new_confidence=0.5, new_source="CAM-A",
            timestamp=1.0,
        )

        self.assertEqual(result_a_then_b[0], HumanClassification.ADULT)
        self.assertEqual(result_b_then_a[0], HumanClassification.ADULT)
        self.assertEqual(result_a_then_b[0], result_b_then_a[0])

    def test_detection_order_never_changes_the_final_reconciled_state(self):

        readings = [
            ("CAM-B", HumanState.RUNNING, 0.5),
            ("CAM-A", HumanState.WALKING, 0.5),
        ]

        def fold(order):
            state = (None, None, None, None)
            for source, value, confidence in order:
                state = reconcile_state(
                    existing_state=state[0], existing_confidence=state[1], existing_source=state[2],
                    existing_last_observed_at=state[3],
                    new_state=value, new_confidence=confidence, new_source=source, timestamp=1.0,
                )
            return state[0]

        self.assertEqual(fold(readings), fold(list(reversed(readings))))


class StalenessTests(unittest.TestCase):

    def test_classification_reverts_to_unknown_after_configured_staleness(self):

        config = HumanEvidenceConfig(classification_staleness_seconds=10.0)

        result = apply_classification_staleness(
            HumanClassification.ADULT, 0.8, "CAM-A", 0.0, now=11.0, config=config,
        )

        self.assertEqual(result, (HumanClassification.UNKNOWN, None, None, None))

    def test_classification_not_yet_stale_is_preserved(self):

        config = HumanEvidenceConfig(classification_staleness_seconds=10.0)

        result = apply_classification_staleness(
            HumanClassification.ADULT, 0.8, "CAM-A", 0.0, now=9.0, config=config,
        )

        self.assertEqual(result, (HumanClassification.ADULT, 0.8, "CAM-A", 0.0))

    def test_state_reverts_to_none_after_configured_staleness(self):

        config = HumanEvidenceConfig(state_staleness_seconds=5.0)

        result = apply_state_staleness(HumanState.FALLEN, 0.8, "CAM-A", 0.0, now=6.0, config=config)

        self.assertEqual(result, (None, None, None, None))

    def test_state_staleness_window_is_shorter_than_classification_by_default(self):

        config = HumanEvidenceConfig()

        self.assertLess(config.state_staleness_seconds, config.classification_staleness_seconds)


if __name__ == "__main__":
    unittest.main()
