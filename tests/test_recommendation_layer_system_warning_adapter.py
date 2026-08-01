import unittest

from evacuation_recommendation.models import RecommendationReason, RecommendationStatus

from evacuation_guidance.models import GuidanceInconsistency

from recommendation_layer.adapters import system_warning_adapter
from recommendation_layer.models import RecommendationPriority, TriggerCondition

from tests.recommendation_layer_fixtures import (
    make_guidance_plan, make_guidance_snapshot, make_recommendation_snapshot, make_zone_recommendation,
)


class SystemWarningAdapterTests(unittest.TestCase):

    def test_low_confidence_produces_warning(self):

        zone = make_zone_recommendation(zone_id="zone-1", confidence=0.1)
        snapshot = make_recommendation_snapshot(zones=[zone])

        candidates = system_warning_adapter.adapt(snapshot, None)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].trigger_condition, TriggerCondition.RECOMMENDATION_LOW_CONFIDENCE)
        self.assertEqual(candidates[0].priority, RecommendationPriority.MEDIUM)

    def test_ai_bottleneck_risk_reason_code_produces_low_priority_warning(self):

        zone = make_zone_recommendation(zone_id="zone-1", reason_codes=(RecommendationReason.AI_BOTTLENECK_RISK_ELEVATED,))
        snapshot = make_recommendation_snapshot(zones=[zone])

        candidates = system_warning_adapter.adapt(snapshot, None)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].trigger_condition, TriggerCondition.RECOMMENDATION_AI_BOTTLENECK_RISK)
        self.assertEqual(candidates[0].priority, RecommendationPriority.LOW)

    def test_widespread_no_safe_exit_produces_critical_building_wide_warning(self):

        zones = [
            make_zone_recommendation(zone_id=f"zone-{i}", status=RecommendationStatus.NO_SAFE_EXIT_AVAILABLE, recommended_exit_id=None)
            for i in range(3)
        ]
        snapshot = make_recommendation_snapshot(zones=zones)

        candidates = system_warning_adapter.adapt(snapshot, None)

        widespread = [c for c in candidates if c.trigger_condition == TriggerCondition.BUILDING_NO_SAFE_EXIT_WIDESPREAD]

        self.assertEqual(len(widespread), 1)
        self.assertEqual(widespread[0].priority, RecommendationPriority.CRITICAL)
        self.assertEqual(set(widespread[0].affected_zones), {"zone-0", "zone-1", "zone-2"})

    def test_guidance_inconsistency_produces_warning(self):

        plan = make_guidance_plan(zone_id="zone-1", inconsistencies=(GuidanceInconsistency.NO_SPEAKER_COVERAGE,))
        guidance_snapshot = make_guidance_snapshot(plans=[plan])

        candidates = system_warning_adapter.adapt(None, guidance_snapshot)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].trigger_condition, TriggerCondition.GUIDANCE_INCONSISTENCY)

    def test_all_none_produces_no_candidates(self):

        self.assertEqual(system_warning_adapter.adapt(None, None), ())


if __name__ == "__main__":
    unittest.main()
