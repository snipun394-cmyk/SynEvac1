import unittest

from recommendation_layer.adapters import exit_utilization_adapter
from recommendation_layer.models import RecommendationSource, RecommendationType, TriggerCondition

from tests.recommendation_layer_fixtures import make_recommendation_snapshot, make_zone_recommendation


class ExitUtilizationAdapterTests(unittest.TestCase):

    def test_skewed_routing_flags_overutilized_and_underutilized(self):

        zones = [
            make_zone_recommendation(zone_id=f"zone-{i}", recommended_exit_id="exit-1", alternative_exit_ids=("exit-2",))
            for i in range(4)
        ]
        snapshot = make_recommendation_snapshot(zones=zones)

        candidates = exit_utilization_adapter.adapt(snapshot)

        types = {c.trigger_condition for c in candidates}

        self.assertIn(TriggerCondition.EXIT_OVERUTILIZED, types)
        self.assertIn(TriggerCondition.EXIT_UNDERUTILIZED_ALTERNATIVE, types)

        for candidate in candidates:
            self.assertEqual(candidate.primary_source, RecommendationSource.RECOMMENDATION_LAYER)
            self.assertEqual(candidate.type, RecommendationType.EXIT_UTILIZATION)

    def test_balanced_routing_flags_neither(self):

        zones = [
            make_zone_recommendation(zone_id="zone-1", recommended_exit_id="exit-1", alternative_exit_ids=("exit-2",)),
            make_zone_recommendation(zone_id="zone-2", recommended_exit_id="exit-2", alternative_exit_ids=("exit-1",)),
        ]
        snapshot = make_recommendation_snapshot(zones=zones)

        candidates = exit_utilization_adapter.adapt(snapshot)

        self.assertEqual(candidates, ())

    def test_below_minimum_zone_count_never_flags(self):

        zones = [
            make_zone_recommendation(zone_id=f"zone-{i}", recommended_exit_id="exit-1", alternative_exit_ids=("exit-2",))
            for i in range(2)
        ]
        snapshot = make_recommendation_snapshot(zones=zones)

        candidates = exit_utilization_adapter.adapt(snapshot)

        self.assertEqual(candidates, ())

    def test_none_snapshot_produces_no_candidates(self):

        self.assertEqual(exit_utilization_adapter.adapt(None), ())


if __name__ == "__main__":
    unittest.main()
