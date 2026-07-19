import unittest

from ai_explainability.comparison import (
    DecisionPolicyComparisonRecord,
    compare_bottleneck_location_to_decision_policy,
    decision_policy_flagged,
    summarize_comparisons,
)

from tests.ai_explainability_fixtures import RealTrainedModelsTestCase


class DecisionPolicyFlaggedTests(unittest.TestCase):

    def test_exit_concerning_statuses_are_flagged(self):

        for status in ("CLOSE", "HIGH_CONGESTION"):

            decision_policy = {"exit_decisions": [{"exit_id": "exit-1", "status": status}]}
            self.assertTrue(decision_policy_flagged(decision_policy, "exit-1", "exit"))

    def test_exit_non_concerning_statuses_are_not_flagged(self):

        for status in ("KEEP_OPEN", "OPEN"):

            decision_policy = {"exit_decisions": [{"exit_id": "exit-1", "status": status}]}
            self.assertFalse(decision_policy_flagged(decision_policy, "exit-1", "exit"))

    def test_stair_concerning_statuses_are_flagged(self):

        for status in ("AVOID", "CONGESTED"):

            decision_policy = {"stair_decisions": [{"stair_id": "stair-1", "status": status}]}
            self.assertTrue(decision_policy_flagged(decision_policy, "stair-1", "stair"))

    def test_stair_use_status_is_not_flagged(self):

        decision_policy = {"stair_decisions": [{"stair_id": "stair-1", "status": "USE"}]}

        self.assertFalse(decision_policy_flagged(decision_policy, "stair-1", "stair"))

    def test_zone_concerning_actions_are_flagged(self):

        for action in ("EVACUATE_IMMEDIATELY", "SHELTER_IN_PLACE"):

            decision_policy = {"zone_decisions": [{"zone_id": "zone-1", "action": action}]}
            self.assertTrue(decision_policy_flagged(decision_policy, "zone-1", "zone"))

    def test_zone_wait_action_is_not_flagged(self):

        decision_policy = {"zone_decisions": [{"zone_id": "zone-1", "action": "WAIT"}]}

        self.assertFalse(decision_policy_flagged(decision_policy, "zone-1", "zone"))

    def test_unknown_location_type_returns_none(self):

        decision_policy = {"exit_decisions": [], "stair_decisions": [], "zone_decisions": []}

        self.assertIsNone(decision_policy_flagged(decision_policy, "door-1", "door"))

    def test_id_not_present_in_the_relevant_decision_list_returns_none(self):

        decision_policy = {"exit_decisions": [{"exit_id": "exit-1", "status": "CLOSE"}]}

        self.assertIsNone(decision_policy_flagged(decision_policy, "exit-999", "exit"))


class CompareBottleneckLocationToDecisionPolicyTests(RealTrainedModelsTestCase):

    def test_returns_one_record_per_scenario(self):

        records = compare_bottleneck_location_to_decision_policy(self.bottleneck_model, self.dataset)

        self.assertEqual(len(records), len(self.dataset))

        for record in records:
            self.assertIsInstance(record, DecisionPolicyComparisonRecord)
            self.assertIn(record.scenario_id, self.dataset.scenario_ids)

    def test_agreement_is_none_when_prediction_is_no_bottleneck(self):

        from ai_training.models.bottleneck_model import NO_BOTTLENECK_LABEL

        records = compare_bottleneck_location_to_decision_policy(self.bottleneck_model, self.dataset)

        for record in records:
            if record.ai_prediction == NO_BOTTLENECK_LABEL:
                self.assertIsNone(record.agreement)
                self.assertIsNone(record.decision_policy_flagged)

    def test_agreement_matches_decision_policy_flagged_when_comparable(self):

        records = compare_bottleneck_location_to_decision_policy(self.bottleneck_model, self.dataset)

        for record in records:
            if record.decision_policy_flagged is not None:
                self.assertEqual(record.agreement, record.decision_policy_flagged)

    def test_confidence_is_a_probability_when_present(self):

        records = compare_bottleneck_location_to_decision_policy(self.bottleneck_model, self.dataset)

        for record in records:
            if record.ai_confidence is not None:
                self.assertGreaterEqual(record.ai_confidence, 0.0)
                self.assertLessEqual(record.ai_confidence, 1.0)

    def test_is_deterministic_across_repeated_calls(self):

        first = compare_bottleneck_location_to_decision_policy(self.bottleneck_model, self.dataset)
        second = compare_bottleneck_location_to_decision_policy(self.bottleneck_model, self.dataset)

        self.assertEqual(first, second)


class SummarizeComparisonsTests(RealTrainedModelsTestCase):

    def test_summary_counts_are_internally_consistent(self):

        records = compare_bottleneck_location_to_decision_policy(self.bottleneck_model, self.dataset)
        summary = summarize_comparisons(records)

        self.assertEqual(summary["total"], len(records))
        self.assertEqual(summary["comparable"], summary["agreements"] + summary["disagreements"])
        self.assertLessEqual(summary["comparable"], summary["total"])

        if summary["comparable"] > 0:
            self.assertAlmostEqual(
                summary["agreement_rate"], summary["agreements"] / summary["comparable"],
            )
        else:
            self.assertIsNone(summary["agreement_rate"])

    def test_summary_of_empty_records_has_none_agreement_rate(self):

        summary = summarize_comparisons([])

        self.assertEqual(summary["total"], 0)
        self.assertIsNone(summary["agreement_rate"])


if __name__ == "__main__":
    unittest.main()
