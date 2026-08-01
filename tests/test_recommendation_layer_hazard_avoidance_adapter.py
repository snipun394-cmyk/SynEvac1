import unittest

from evacuation_recommendation.models import RecommendationStatus

from recommendation_layer.adapters import hazard_avoidance_adapter
from recommendation_layer.models import RecommendationPriority, RecommendationSource, RecommendationType, TriggerCondition

from emergency_response.models import ResponseReason

from tests.recommendation_layer_fixtures import (
    make_advisory_report, make_emergency_response_snapshot, make_recommendation_snapshot, make_zone_recommendation,
    make_zone_response_priority,
)


class HazardAvoidanceAdapterTests(unittest.TestCase):

    def test_no_safe_exit_zone_produces_critical_hazard_recommendation(self):

        zone = make_zone_recommendation(zone_id="zone-1", status=RecommendationStatus.NO_SAFE_EXIT_AVAILABLE, recommended_exit_id=None)
        snapshot = make_recommendation_snapshot(zones=[zone])

        candidates = hazard_avoidance_adapter.adapt(snapshot, None, None)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].priority, RecommendationPriority.CRITICAL)
        self.assertEqual(candidates[0].trigger_condition, TriggerCondition.ZONE_NO_SAFE_EXIT)
        self.assertEqual(candidates[0].primary_source, RecommendationSource.EVACUATION_RECOMMENDATION)

    def test_emergency_response_hazard_present_works_with_no_other_inputs(self):

        zone = make_zone_response_priority(zone_id="zone-2", reason_codes=(ResponseReason.HAZARD_PRESENT,), hazard_severity="HIGH")
        snapshot = make_emergency_response_snapshot(zones=[zone])

        candidates = hazard_avoidance_adapter.adapt(None, snapshot, None)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].priority, RecommendationPriority.HIGH)
        self.assertEqual(candidates[0].severity, "HIGH")
        self.assertEqual(candidates[0].primary_source, RecommendationSource.EMERGENCY_RESPONSE)

    def test_advisory_only_critical_zone_works_alone(self):

        report = make_advisory_report(critical_zones=["zone-3"])

        candidates = hazard_avoidance_adapter.adapt(None, None, report)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].affected_zones, ("zone-3",))
        self.assertEqual(candidates[0].primary_source, RecommendationSource.ADVISORY_SYSTEM)

    def test_all_none_produces_no_candidates(self):

        self.assertEqual(hazard_avoidance_adapter.adapt(None, None, None), ())


if __name__ == "__main__":
    unittest.main()
