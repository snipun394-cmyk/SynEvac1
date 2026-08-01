import unittest

from evacuation_recommendation.models import RecommendationReason, RecommendationStatus

from recommendation_layer.adapters import occupant_routing_adapter
from recommendation_layer.models import RecommendationPriority, RecommendationSource, RecommendationType, TriggerCondition

from tests.recommendation_layer_fixtures import make_recommendation_snapshot, make_zone_recommendation


class OccupantRoutingAdapterTests(unittest.TestCase):

    def test_recommended_zone_produces_occupant_routing_recommendation(self):

        zone = make_zone_recommendation(zone_id="zone-1", recommended_exit_id="exit-1")
        snapshot = make_recommendation_snapshot(zones=[zone])

        candidates = occupant_routing_adapter.adapt(snapshot)

        self.assertEqual(len(candidates), 1)

        candidate = candidates[0]

        self.assertEqual(candidate.type, RecommendationType.OCCUPANT_ROUTING)
        self.assertEqual(candidate.trigger_condition, TriggerCondition.ZONE_EXIT_RECOMMENDED)
        self.assertEqual(candidate.affected_zones, ("zone-1",))
        self.assertEqual(candidate.affected_exits, ("exit-1",))
        self.assertEqual(candidate.primary_source, RecommendationSource.EVACUATION_RECOMMENDATION)
        self.assertEqual(candidate.priority, RecommendationPriority.LOW)

    def test_congestion_reason_code_bumps_priority(self):

        zone = make_zone_recommendation(
            zone_id="zone-1", recommended_exit_id="exit-1", reason_codes=(RecommendationReason.HIGH_CONGESTION,),
        )
        snapshot = make_recommendation_snapshot(zones=[zone])

        candidates = occupant_routing_adapter.adapt(snapshot)

        self.assertEqual(candidates[0].priority, RecommendationPriority.MEDIUM)

    def test_no_safe_exit_zone_produces_no_routing_recommendation(self):

        zone = make_zone_recommendation(zone_id="zone-1", status=RecommendationStatus.NO_SAFE_EXIT_AVAILABLE, recommended_exit_id=None)
        snapshot = make_recommendation_snapshot(zones=[zone])

        candidates = occupant_routing_adapter.adapt(snapshot)

        self.assertEqual(candidates, ())

    def test_none_snapshot_produces_no_candidates(self):

        self.assertEqual(occupant_routing_adapter.adapt(None), ())


if __name__ == "__main__":
    unittest.main()
