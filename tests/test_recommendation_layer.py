import unittest
from unittest.mock import patch

from recommendation_layer.layer import RecommendationLayer
from recommendation_layer.models import RecommendationType

from evacuation_recommendation.models import RecommendationReason

from tests.recommendation_layer_fixtures import (
    make_emergency_response_snapshot, make_exit_candidate, make_recommendation_snapshot, make_zone_recommendation,
    make_zone_response_priority,
)
from emergency_response.models import ResponsePriorityLevel


class RecommendationLayerTests(unittest.TestCase):

    def test_compute_produces_multiple_categories_across_all_adapters(self):

        candidate = make_exit_candidate(exit_id="exit-1", reason_codes=(RecommendationReason.QUEUE_PRESENT,))
        zone = make_zone_recommendation(
            zone_id="zone-1", recommended_exit_id="exit-1", candidates=(candidate,),
            reason_codes=(RecommendationReason.QUEUE_PRESENT,),
        )
        recommendation_snapshot = make_recommendation_snapshot(zones=[zone])

        response_zone = make_zone_response_priority(zone_id="zone-1", priority_level=ResponsePriorityLevel.CRITICAL)
        response_snapshot = make_emergency_response_snapshot(zones=[response_zone])

        layer = RecommendationLayer()

        result = layer.compute(
            1.0, evacuation_recommendation_snapshot=recommendation_snapshot, emergency_response_snapshot=response_snapshot,
        )

        types = {r.type for r in result.recommendations}

        self.assertIn(RecommendationType.OCCUPANT_ROUTING, types)
        self.assertIn(RecommendationType.CONGESTION_MITIGATION, types)
        self.assertIn(RecommendationType.WARDEN_DISPATCH, types)

    def test_compute_always_returns_a_recommendation_set_never_none(self):

        layer = RecommendationLayer()

        result = layer.compute(1.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.recommendations, ())

    def test_a_raising_adapter_never_blanks_the_other_five(self):

        zone = make_zone_recommendation(zone_id="zone-1", recommended_exit_id="exit-1")
        recommendation_snapshot = make_recommendation_snapshot(zones=[zone])

        layer = RecommendationLayer()

        with patch(
            "recommendation_layer.layer.hazard_avoidance_adapter.adapt", side_effect=RuntimeError("boom"),
        ):
            result = layer.compute(1.0, evacuation_recommendation_snapshot=recommendation_snapshot)

        types = {r.type for r in result.recommendations}
        self.assertIn(RecommendationType.OCCUPANT_ROUTING, types)

    def test_latest_property_reflects_most_recent_compute(self):

        layer = RecommendationLayer()

        self.assertIsNone(layer.latest)

        result = layer.compute(1.0)

        self.assertIs(layer.latest, result)


if __name__ == "__main__":
    unittest.main()
