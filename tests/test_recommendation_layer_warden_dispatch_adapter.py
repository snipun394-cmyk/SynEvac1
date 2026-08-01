import unittest

from emergency_response.models import ResponsePriorityLevel

from recommendation_layer.adapters import warden_dispatch_adapter
from recommendation_layer.models import RecommendationPriority, RecommendationSource, TriggerCondition

from tests.recommendation_layer_fixtures import (
    make_advisory_report, make_emergency_response_snapshot, make_zone_response_priority,
)


class WardenDispatchAdapterTests(unittest.TestCase):

    def test_critical_priority_level_alone_produces_warden_dispatch(self):

        zone = make_zone_response_priority(zone_id="zone-1", priority_level=ResponsePriorityLevel.CRITICAL)
        snapshot = make_emergency_response_snapshot(zones=[zone])

        candidates = warden_dispatch_adapter.adapt(snapshot, None)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].priority, RecommendationPriority.CRITICAL)
        self.assertEqual(candidates[0].trigger_condition, TriggerCondition.ZONE_RESPONSE_ELEVATED)
        self.assertEqual(candidates[0].primary_source, RecommendationSource.EMERGENCY_RESPONSE)

    def test_confirmed_assistance_produces_separate_assistance_recommendation(self):

        zone = make_zone_response_priority(zone_id="zone-1", confirmed_assistance_count=1)
        snapshot = make_emergency_response_snapshot(zones=[zone])

        candidates = warden_dispatch_adapter.adapt(snapshot, None)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].trigger_condition, TriggerCondition.ZONE_ASSISTANCE_REQUIRED)
        self.assertEqual(candidates[0].priority, RecommendationPriority.HIGH)

    def test_critical_and_assistance_both_present_produce_two_recommendations(self):

        zone = make_zone_response_priority(
            zone_id="zone-1", priority_level=ResponsePriorityLevel.CRITICAL, possible_assistance_count=1,
        )
        snapshot = make_emergency_response_snapshot(zones=[zone])

        candidates = warden_dispatch_adapter.adapt(snapshot, None)

        self.assertEqual(len(candidates), 2)
        trigger_conditions = {c.trigger_condition for c in candidates}
        self.assertEqual(trigger_conditions, {TriggerCondition.ZONE_RESPONSE_ELEVATED, TriggerCondition.ZONE_ASSISTANCE_REQUIRED})

    def test_advisory_source_produces_independent_candidate(self):

        report = make_advisory_report(live_priority_zone_ids=["zone-2"])

        candidates = warden_dispatch_adapter.adapt(None, report)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].primary_source, RecommendationSource.ADVISORY_SYSTEM)
        self.assertEqual(candidates[0].affected_zones, ("zone-2",))

    def test_all_none_produces_no_candidates(self):

        self.assertEqual(warden_dispatch_adapter.adapt(None, None), ())


if __name__ == "__main__":
    unittest.main()
