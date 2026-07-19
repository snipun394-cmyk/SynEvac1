import unittest

import ai_training as at
from ai_training.models.bottleneck_model import NO_BOTTLENECK_LABEL
from ai_inference.loader import load_model
from ai_inference.recommendation import (
    ACTION_ESCALATE,
    ACTION_FOLLOW_DECISION_POLICY,
    ACTION_REVIEW,
    Recommendation,
    build_recommendation,
    decision_policy_flagged,
)

from tests.ai_inference_fixtures import RealSavedModelsTestCase


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

    def test_id_not_present_returns_none(self):

        decision_policy = {"exit_decisions": [{"exit_id": "exit-1", "status": "CLOSE"}]}

        self.assertIsNone(decision_policy_flagged(decision_policy, "exit-999", "exit"))


class BuildRecommendationTests(RealSavedModelsTestCase):

    def setUp(self):

        self.loaded = load_model(self.bottleneck_location_dir)
        self.X_rows, _y, _extra = at.BottleneckModel.build_table(self.dataset, target="location")

    def test_returns_a_recommendation_with_every_field(self):

        decision_policy = self.dataset.decision_policy_rows()[0]
        ground_truth = self.dataset.ground_truth_rows()[0]

        recommendation = build_recommendation(
            self.loaded, self.X_rows[0], decision_policy,
            location_type=ground_truth.get("peak_congestion_location_type"),
        )

        self.assertIsInstance(recommendation, Recommendation)
        self.assertIn(
            recommendation.recommended_action,
            (ACTION_ESCALATE, ACTION_REVIEW, ACTION_FOLLOW_DECISION_POLICY),
        )
        self.assertIsInstance(recommendation.rationale, str)
        self.assertGreater(len(recommendation.rationale), 0)

    def test_agreement_mirrors_decision_policy_flagged(self):

        for row, decision_policy, ground_truth in zip(
            self.X_rows, self.dataset.decision_policy_rows(), self.dataset.ground_truth_rows(),
        ):

            recommendation = build_recommendation(
                self.loaded, row, decision_policy,
                location_type=ground_truth.get("peak_congestion_location_type"),
            )

            self.assertEqual(recommendation.agreement, recommendation.decision_policy_flagged)

    def test_no_bottleneck_prediction_always_defers_to_decision_policy(self):

        for row, decision_policy, ground_truth in zip(
            self.X_rows, self.dataset.decision_policy_rows(), self.dataset.ground_truth_rows(),
        ):

            recommendation = build_recommendation(
                self.loaded, row, decision_policy,
                location_type=ground_truth.get("peak_congestion_location_type"),
            )

            if recommendation.ai_prediction == NO_BOTTLENECK_LABEL:
                self.assertEqual(recommendation.recommended_action, ACTION_FOLLOW_DECISION_POLICY)
                self.assertIsNone(recommendation.agreement)

    def test_flagged_true_produces_escalate(self):

        for row, decision_policy, ground_truth in zip(
            self.X_rows, self.dataset.decision_policy_rows(), self.dataset.ground_truth_rows(),
        ):

            recommendation = build_recommendation(
                self.loaded, row, decision_policy,
                location_type=ground_truth.get("peak_congestion_location_type"),
            )

            if recommendation.decision_policy_flagged is True:
                self.assertEqual(recommendation.recommended_action, ACTION_ESCALATE)

            if recommendation.decision_policy_flagged is False:
                self.assertEqual(recommendation.recommended_action, ACTION_REVIEW)

    def test_ai_confidence_is_a_probability(self):

        decision_policy = self.dataset.decision_policy_rows()[0]

        recommendation = build_recommendation(self.loaded, self.X_rows[0], decision_policy)

        self.assertIsNotNone(recommendation.ai_confidence)
        self.assertGreaterEqual(recommendation.ai_confidence, 0.0)
        self.assertLessEqual(recommendation.ai_confidence, 1.0)

    def test_to_dict_contains_every_field(self):

        decision_policy = self.dataset.decision_policy_rows()[0]

        recommendation = build_recommendation(self.loaded, self.X_rows[0], decision_policy)
        payload = recommendation.to_dict()

        self.assertEqual(
            set(payload.keys()),
            {
                "ai_prediction", "ai_confidence", "decision_policy_flagged",
                "agreement", "recommended_action", "rationale",
            },
        )

    def test_never_replaces_decision_policy_never_mutates_it(self):

        decision_policy = self.dataset.decision_policy_rows()[0]
        original = {
            "exit_decisions": list(decision_policy.get("exit_decisions", ())),
            "stair_decisions": list(decision_policy.get("stair_decisions", ())),
            "zone_decisions": list(decision_policy.get("zone_decisions", ())),
        }

        build_recommendation(self.loaded, self.X_rows[0], decision_policy)

        self.assertEqual(list(decision_policy.get("exit_decisions", ())), original["exit_decisions"])
        self.assertEqual(list(decision_policy.get("stair_decisions", ())), original["stair_decisions"])
        self.assertEqual(list(decision_policy.get("zone_decisions", ())), original["zone_decisions"])


if __name__ == "__main__":
    unittest.main()
