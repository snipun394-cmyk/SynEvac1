import unittest

from evacuation_recommendation.models import RecommendationReason

from recommendation_layer.adapters import congestion_mitigation_adapter
from recommendation_layer.models import RecommendationPriority, RecommendationSource

from tests.recommendation_layer_fixtures import (
    make_advisory_report, make_building_recommendation, make_crowd_snapshot, make_exit_candidate,
    make_recommendation_snapshot, make_zone_recommendation,
)


class CongestionMitigationAdapterTests(unittest.TestCase):

    def test_queue_present_reason_code_alone_fires(self):

        candidate = make_exit_candidate(exit_id="exit-1", reason_codes=(RecommendationReason.QUEUE_PRESENT,))
        zone = make_zone_recommendation(
            zone_id="zone-1", recommended_exit_id="exit-1", candidates=(candidate,),
            reason_codes=(RecommendationReason.QUEUE_PRESENT,),
        )
        snapshot = make_recommendation_snapshot(zones=[zone])

        candidates = congestion_mitigation_adapter.adapt(snapshot, None, None)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].priority, RecommendationPriority.HIGH)
        self.assertEqual(candidates[0].primary_source, RecommendationSource.EVACUATION_RECOMMENDATION)

    def test_crowd_intelligence_enriches_without_being_sole_trigger_alone(self):

        crowd_snapshot = make_crowd_snapshot(congested_exits=["exit-9"])

        candidates = congestion_mitigation_adapter.adapt(None, crowd_snapshot, None)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].primary_source, RecommendationSource.CROWD_INTELLIGENCE)
        self.assertEqual(candidates[0].affected_exits, ("exit-9",))

    def test_advisory_congestion_recommendation_is_picked_up(self):

        report = make_advisory_report(building_recommendations=[
            make_building_recommendation(action="Monitor Congestion at Exit exit-5", target_type="exit", target_id="exit-5"),
        ])

        candidates = congestion_mitigation_adapter.adapt(None, None, report)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].primary_source, RecommendationSource.ADVISORY_SYSTEM)
        self.assertEqual(candidates[0].affected_exits, ("exit-5",))

    def test_all_none_produces_no_candidates(self):

        self.assertEqual(congestion_mitigation_adapter.adapt(None, None, None), ())


if __name__ == "__main__":
    unittest.main()
