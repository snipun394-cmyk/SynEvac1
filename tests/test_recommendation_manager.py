import unittest

from recommendation_layer.manager import RecommendationManager
from recommendation_layer.models import Recommendation, RecommendationPriority, RecommendationStatus, RecommendationType, TriggerCondition


def make_candidate(zone_id="zone-1", primary_source="evacuation_recommendation", priority=RecommendationPriority.MEDIUM, **kwargs):

    return Recommendation(
        type=RecommendationType.OCCUPANT_ROUTING, priority=priority, trigger_condition=TriggerCondition.ZONE_EXIT_RECOMMENDED,
        affected_zones=(zone_id,), primary_source=primary_source, **kwargs,
    )


class RecommendationManagerTests(unittest.TestCase):

    def test_create_mints_a_fresh_id(self):

        manager = RecommendationManager()

        result = manager.ingest([make_candidate()], time=1.0)

        self.assertEqual(len(result.recommendations), 1)
        self.assertTrue(result.recommendations[0].recommendation_id)
        self.assertEqual(result.recommendations[0].status, RecommendationStatus.ACTIVE)

    def test_id_stable_across_reappearing_cycles(self):

        manager = RecommendationManager()

        first = manager.ingest([make_candidate()], time=1.0).recommendations[0]
        second = manager.ingest([make_candidate()], time=2.0).recommendations[0]
        third = manager.ingest([make_candidate()], time=3.0).recommendations[0]

        self.assertEqual(first.recommendation_id, second.recommendation_id)
        self.assertEqual(second.recommendation_id, third.recommendation_id)
        self.assertEqual(third.updated_at, 3.0)
        self.assertEqual(third.created_at, 1.0)

    def test_expire_after_grace_period_elapses(self):

        manager = RecommendationManager(grace_period_seconds=5.0)

        manager.ingest([make_candidate()], time=1.0)

        during_grace = manager.ingest([], time=3.0).recommendations
        self.assertEqual(len(during_grace), 1)
        self.assertEqual(during_grace[0].status, RecommendationStatus.ACTIVE)
        # The grace clock starts from the FIRST cycle it's noticed
        # missing (3.0), not from when it was last confirmed present.
        self.assertEqual(during_grace[0].expires_at, 8.0)

        after_grace = manager.ingest([], time=8.0).recommendations
        self.assertEqual(len(after_grace), 1)
        self.assertEqual(after_grace[0].status, RecommendationStatus.EXPIRED)

        dropped = manager.ingest([], time=9.0).recommendations
        self.assertEqual(dropped, ())

    def test_recovery_before_grace_period_clears_expiry_same_id(self):

        manager = RecommendationManager(grace_period_seconds=5.0)

        first = manager.ingest([make_candidate()], time=1.0).recommendations[0]
        manager.ingest([], time=2.0)
        recovered = manager.ingest([make_candidate()], time=3.0).recommendations[0]

        self.assertEqual(recovered.recommendation_id, first.recommendation_id)
        self.assertIsNone(recovered.expires_at)
        self.assertEqual(recovered.status, RecommendationStatus.ACTIVE)

    def test_same_cycle_merge_sets_supporting_sources_and_evidence_origin(self):

        manager = RecommendationManager()

        winner = make_candidate(
            primary_source="evacuation_recommendation", supporting_evidence={"a": 1},
        )
        loser = make_candidate(
            primary_source="crowd_intelligence", supporting_evidence={"b": 2},
        )

        result = manager.ingest([winner, loser], time=1.0)

        self.assertEqual(len(result.recommendations), 1)

        merged = result.recommendations[0]
        self.assertEqual(merged.primary_source, "evacuation_recommendation")
        self.assertEqual(merged.supporting_sources, ("crowd_intelligence",))
        self.assertEqual(dict(merged.supporting_evidence), {"a": 1, "b": 2})
        self.assertEqual(merged.evidence_origin["a"], "evacuation_recommendation")
        self.assertEqual(merged.evidence_origin["b"], "crowd_intelligence")

    def test_ranking_sorts_by_priority_descending(self):

        manager = RecommendationManager()

        low = make_candidate(zone_id="zone-low", priority=RecommendationPriority.LOW)
        critical = make_candidate(zone_id="zone-critical", priority=RecommendationPriority.CRITICAL)

        result = manager.ingest([low, critical], time=1.0)

        self.assertEqual(result.recommendations[0].priority, RecommendationPriority.CRITICAL)
        self.assertEqual(result.recommendations[1].priority, RecommendationPriority.LOW)


if __name__ == "__main__":
    unittest.main()
