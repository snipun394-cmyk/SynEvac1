import unittest

from evacuation_recommendation.models import RecommendationReason, RecommendationStatus

from emergency_response.models import ResponsePriorityLevel

from recommendation_layer.layer import RecommendationLayer
from recommendation_layer.models import RecommendationStatus as UnifiedStatus, RecommendationType

from tests.recommendation_layer_fixtures import (
    make_emergency_response_snapshot, make_exit_candidate, make_recommendation_snapshot, make_zone_recommendation,
    make_zone_response_priority,
)


# =====================================================
# The Recommendation Layer milestone -- the 6 scenario demonstrations
# the brief required, each a standalone end-to-end test through the
# real RecommendationLayer (adapters + manager together) against
# fabricated but realistic upstream snapshots.
# =====================================================


class RecommendationLayerScenarioDemonstrationTests(unittest.TestCase):

    def test_congestion_prediction_triggers_congestion_mitigation(self):

        candidate = make_exit_candidate(exit_id="exit-1", reason_codes=(RecommendationReason.QUEUE_PRESENT,))
        zone = make_zone_recommendation(
            zone_id="zone-1", recommended_exit_id="exit-1", candidates=(candidate,),
            reason_codes=(RecommendationReason.QUEUE_PRESENT,),
        )
        snapshot = make_recommendation_snapshot(zones=[zone])

        result = RecommendationLayer().compute(1.0, evacuation_recommendation_snapshot=snapshot)

        self.assertTrue(result.by_type(RecommendationType.CONGESTION_MITIGATION))

    def test_hazard_avoidance_on_unreachable_exit(self):

        zone = make_zone_recommendation(zone_id="zone-1", status=RecommendationStatus.NO_SAFE_EXIT_AVAILABLE, recommended_exit_id=None)
        snapshot = make_recommendation_snapshot(zones=[zone])

        result = RecommendationLayer().compute(1.0, evacuation_recommendation_snapshot=snapshot)

        hazard = result.by_type(RecommendationType.HAZARD_AVOIDANCE)
        self.assertTrue(hazard)
        self.assertEqual(hazard[0].priority, "CRITICAL")

    def test_exit_redistribution_flags_overutilized_exit(self):

        zones = [
            make_zone_recommendation(zone_id=f"zone-{i}", recommended_exit_id="exit-1", alternative_exit_ids=("exit-2",))
            for i in range(4)
        ]
        snapshot = make_recommendation_snapshot(zones=zones)

        result = RecommendationLayer().compute(1.0, evacuation_recommendation_snapshot=snapshot)

        self.assertTrue(result.by_type(RecommendationType.EXIT_UTILIZATION))

    def test_warden_dispatch_on_critical_response_priority(self):

        zone = make_zone_response_priority(zone_id="zone-1", priority_level=ResponsePriorityLevel.CRITICAL)
        snapshot = make_emergency_response_snapshot(zones=[zone])

        result = RecommendationLayer().compute(1.0, emergency_response_snapshot=snapshot)

        warden = result.by_type(RecommendationType.WARDEN_DISPATCH)
        self.assertTrue(warden)
        self.assertEqual(warden[0].priority, "CRITICAL")

    def test_multiple_simultaneous_recommendations_across_all_six_categories(self):

        candidate = make_exit_candidate(exit_id="exit-1", reason_codes=(RecommendationReason.QUEUE_PRESENT, RecommendationReason.AI_BOTTLENECK_RISK_ELEVATED))
        routed_zone = make_zone_recommendation(
            zone_id="zone-1", recommended_exit_id="exit-1", candidates=(candidate,),
            reason_codes=(RecommendationReason.QUEUE_PRESENT, RecommendationReason.AI_BOTTLENECK_RISK_ELEVATED),
        )
        no_safe_exit_zone = make_zone_recommendation(
            zone_id="zone-2", status=RecommendationStatus.NO_SAFE_EXIT_AVAILABLE, recommended_exit_id=None,
        )
        overload_zones = [
            make_zone_recommendation(zone_id=f"zone-load-{i}", recommended_exit_id="exit-9", alternative_exit_ids=("exit-10",))
            for i in range(3)
        ]
        recommendation_snapshot = make_recommendation_snapshot(zones=[routed_zone, no_safe_exit_zone] + overload_zones)

        response_zone = make_zone_response_priority(zone_id="zone-1", priority_level=ResponsePriorityLevel.HIGH)
        response_snapshot = make_emergency_response_snapshot(zones=[response_zone])

        result = RecommendationLayer().compute(
            1.0, evacuation_recommendation_snapshot=recommendation_snapshot, emergency_response_snapshot=response_snapshot,
        )

        present_types = {r.type for r in result.recommendations}

        self.assertEqual(present_types, {
            RecommendationType.OCCUPANT_ROUTING, RecommendationType.HAZARD_AVOIDANCE,
            RecommendationType.CONGESTION_MITIGATION, RecommendationType.EXIT_UTILIZATION,
            RecommendationType.WARDEN_DISPATCH, RecommendationType.SYSTEM_WARNING,
        })

    def test_recommendation_expiration(self):

        zone = make_zone_recommendation(zone_id="zone-1", recommended_exit_id="exit-1")
        snapshot = make_recommendation_snapshot(zones=[zone])

        layer = RecommendationLayer(grace_period_seconds=2.0)

        first = layer.compute(1.0, evacuation_recommendation_snapshot=snapshot)
        self.assertTrue(first.by_type(RecommendationType.OCCUPANT_ROUTING))

        # First missed cycle (2.0) starts the grace clock: expires_at =
        # 2.0 + grace_period_seconds (2.0) = 4.0 -- still ACTIVE here.
        still_active = layer.compute(2.0, evacuation_recommendation_snapshot=None)
        routing = still_active.by_type(RecommendationType.OCCUPANT_ROUTING)
        self.assertEqual(len(routing), 1)
        self.assertEqual(routing[0].status, UnifiedStatus.ACTIVE)

        expired = layer.compute(4.0, evacuation_recommendation_snapshot=None)
        routing = expired.by_type(RecommendationType.OCCUPANT_ROUTING)
        self.assertEqual(len(routing), 1)
        self.assertEqual(routing[0].status, UnifiedStatus.EXPIRED)

        gone = layer.compute(5.0, evacuation_recommendation_snapshot=None)
        self.assertEqual(gone.by_type(RecommendationType.OCCUPANT_ROUTING), ())


if __name__ == "__main__":
    unittest.main()
