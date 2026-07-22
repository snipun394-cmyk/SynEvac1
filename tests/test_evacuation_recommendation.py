import unittest

from behavior_recognition.observation import RecognizedBehavior

from hazard.severity import HazardSeverity

from crowd_intelligence.models import IntensityLevel

from emergency_response.models import ResponsePriorityLevel

from evacuation_recommendation.models import (
    RecommendationConfig, RecommendationReason, RecommendationStatus, RecommendationWeights,
)

from tests.evacuation_recommendation_fixtures import (
    FakeAIPredictionSnapshot, make_building_state, make_crowd_snapshot, make_emergency_response_snapshot,
    make_engine, make_evacuation_progress_snapshot, make_exit_flow, make_exit_metrics, make_zone_priority,
)


# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone, Phase 13 --
# deterministic engine-level unit coverage. No randomness anywhere in
# this file.
# =====================================================


class SafeExitCandidateTests(unittest.TestCase):

    def test_1_unsafe_exit_excluded_from_ranking(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, make_building_state({"z1": HazardSeverity.HIGH}))

        result = snapshot.zone("z1")
        self.assertEqual(result.status, RecommendationStatus.NO_SAFE_EXIT_AVAILABLE)
        self.assertNotIn("EXIT-1", result.ranked_exit_ids)

    def test_2_no_safe_exit_available_when_zone_excluded(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, make_building_state({"z1": HazardSeverity.HIGH}))
        result = snapshot.zone("z1")

        self.assertIsNone(result.recommended_exit_id)
        self.assertEqual(result.reason_codes, (RecommendationReason.NO_SAFE_EXIT_REACHABLE,))

    def test_3_unoccupied_zone_produces_no_recommendation(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, make_building_state())

        self.assertIsNone(snapshot.zone("z2"))
        self.assertIsNone(snapshot.zone("z4"))


class RankingTests(unittest.TestCase):

    def test_4_shortest_safe_exit_preferred(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, make_building_state())
        result = snapshot.zone("z2")

        self.assertEqual(result.status, RecommendationStatus.RECOMMENDED)
        self.assertEqual(result.recommended_exit_id, "EXIT-1")
        self.assertEqual(result.ranked_exit_ids, ("EXIT-1", "EXIT-2"))
        self.assertIn(RecommendationReason.SHORTEST_SAFE_ROUTE, result.reason_codes)

    def test_5_congested_safe_exit_loses_rank(self):

        # Distance dominates by default (Phase 4's own "shortest safe
        # exit preferred" weighting) -- proving congestion genuinely
        # participates in ranking (Phase 4's own "Weights must be
        # configurable" requirement) uses a config that weighs it
        # heavily enough to matter, isolating the mechanism itself
        # rather than depending on real-world-plausible default weights
        # happening to overturn a 3x distance gap.
        weights = RecommendationWeights(route_distance_weight=0.10, congestion_weight=0.60, queue_weight=0.20)
        engine, manager = make_engine(weights=weights)

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        crowd = make_crowd_snapshot({
            "EXIT-1": make_exit_metrics("EXIT-1", congestion_level=IntensityLevel.CRITICAL, queue_candidate_count=10),
            "EXIT-2": make_exit_metrics("EXIT-2", congestion_level=IntensityLevel.LOW, queue_candidate_count=0),
        })

        uncongested = engine.compute(0.0, make_building_state()).zone("z2")
        self.assertEqual(uncongested.recommended_exit_id, "EXIT-1")

        congested = engine.compute(0.0, make_building_state(), crowd_snapshot=crowd).zone("z2")

        # EXIT-1 is still the shorter route, but EXIT-2's overwhelming
        # congestion/queue advantage is enough to overturn it once
        # congestion is genuinely weighted.
        self.assertEqual(congested.recommended_exit_id, "EXIT-2")

    def test_6_throughput_influences_rank(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        progress = make_evacuation_progress_snapshot({
            "EXIT-1": make_exit_flow("EXIT-1", recent_flow_per_minute=0.0),
            "EXIT-2": make_exit_flow("EXIT-2", recent_flow_per_minute=40.0),
        })

        distance_only = engine.compute(0.0, make_building_state()).zone("z2")
        with_throughput = engine.compute(0.0, make_building_state(), evacuation_progress_snapshot=progress).zone("z2")

        exit_1_candidate = next(c for c in with_throughput.candidates if c.exit_id == "EXIT-1")
        exit_2_candidate = next(c for c in with_throughput.candidates if c.exit_id == "EXIT-2")

        self.assertGreater(exit_2_candidate.throughput_per_minute, exit_1_candidate.throughput_per_minute)
        # Throughput alone (small weight) doesn't overturn a much shorter
        # route -- distance_only's own top pick is preserved.
        self.assertEqual(distance_only.recommended_exit_id, with_throughput.recommended_exit_id)

    def test_7_trajectory_supports_ranking(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        from trajectory_intelligence.engine import TrajectoryIntelligenceEngine

        trajectory_engine = TrajectoryIntelligenceEngine(engine.building, engine.graph, manager)
        trajectory_snapshot = trajectory_engine.compute(0.0, make_building_state())

        snapshot = engine.compute(0.0, make_building_state(), trajectory_snapshot=trajectory_snapshot)
        result = snapshot.zone("z2")

        exit_1_candidate = next(c for c in result.candidates if c.exit_id == "EXIT-1")
        self.assertIn(exit_1_candidate.trajectory_support, ("PROGRESSING", None))

    def test_8_crowd_supports_ranking(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        crowd = make_crowd_snapshot({"EXIT-1": make_exit_metrics("EXIT-1", congestion_level=IntensityLevel.LOW)})
        snapshot = engine.compute(0.0, make_building_state(), crowd_snapshot=crowd)
        result = snapshot.zone("z2")

        exit_1_candidate = next(c for c in result.candidates if c.exit_id == "EXIT-1")
        self.assertEqual(exit_1_candidate.congestion_level, "LOW")
        self.assertIn(RecommendationReason.LOW_CONGESTION, exit_1_candidate.reason_codes)

    def test_9_emergency_response_supports_ranking(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        response = make_emergency_response_snapshot({"z1": make_zone_priority("z1", ResponsePriorityLevel.CRITICAL)})
        snapshot = engine.compute(0.0, make_building_state(), emergency_response_snapshot=response)
        result = snapshot.zone("z2")

        exit_1_candidate = next(c for c in result.candidates if c.exit_id == "EXIT-1")
        self.assertTrue(exit_1_candidate.emergency_response_elevated)
        self.assertIn(RecommendationReason.EMERGENCY_RESPONSE_ZONE_ELEVATED, exit_1_candidate.reason_codes)

    def test_10_ai_only_supports_never_changes_relative_order(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        without_ai = engine.compute(0.0, make_building_state()).zone("z2")
        with_low_ai = engine.compute(0.0, make_building_state(), ai_prediction_snapshot=FakeAIPredictionSnapshot(0.05)).zone("z2")
        with_high_ai = engine.compute(0.0, make_building_state(), ai_prediction_snapshot=FakeAIPredictionSnapshot(0.95)).zone("z2")

        self.assertEqual(without_ai.ranked_exit_ids, with_low_ai.ranked_exit_ids)
        self.assertEqual(without_ai.ranked_exit_ids, with_high_ai.ranked_exit_ids)

    def test_11_multiple_zones_ranked_independently(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        manager.update("OCC-2", "CAM-2", "T2", "z4", "f1", (41.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        snapshot = engine.compute(0.0, make_building_state())

        self.assertEqual(snapshot.zone("z1").recommended_exit_id, "EXIT-1")
        self.assertEqual(snapshot.zone("z4").recommended_exit_id, "EXIT-2")


class ConfidenceTests(unittest.TestCase):

    def test_12_coverage_changes_confidence(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        from crowd_intelligence.models import CrowdIntelligenceSnapshot, ZoneCrowdMetrics

        good_coverage = CrowdIntelligenceSnapshot(zone_metrics={"z2": ZoneCrowdMetrics(zone_id="z2", position_coverage_fraction=1.0)})
        poor_coverage = CrowdIntelligenceSnapshot(zone_metrics={"z2": ZoneCrowdMetrics(zone_id="z2", position_coverage_fraction=0.1)})

        high_confidence = engine.compute(0.0, make_building_state(), crowd_snapshot=good_coverage).zone("z2").confidence
        low_confidence = engine.compute(0.0, make_building_state(), crowd_snapshot=poor_coverage).zone("z2").confidence

        self.assertGreater(high_confidence, low_confidence)

    def test_13_no_evidence_at_all_still_produces_a_recommendation_with_neutral_confidence(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        result = engine.compute(0.0, building_state=None).zone("z2")

        self.assertEqual(result.status, RecommendationStatus.RECOMMENDED)
        self.assertIsNotNone(result.confidence)


class ExitRecoveryTests(unittest.TestCase):

    def test_14_hazard_change_reroutes_recommendation(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        before = engine.compute(0.0, make_building_state()).zone("z2")
        self.assertEqual(before.recommended_exit_id, "EXIT-1")

        during = engine.compute(1.0, make_building_state({"z1": HazardSeverity.HIGH})).zone("z2")
        self.assertEqual(during.recommended_exit_id, "EXIT-2")

    def test_15_exit_recovers_once_hazard_clears(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        engine.compute(0.0, make_building_state({"z1": HazardSeverity.HIGH}))
        recovered = engine.compute(1.0, make_building_state()).zone("z2")

        self.assertEqual(recovered.recommended_exit_id, "EXIT-1")


class ExplanationTests(unittest.TestCase):

    def test_16_recommendation_explains_itself(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        result = engine.compute(0.0, make_building_state()).zone("z2")

        self.assertIn("EXIT-1", result.explanation)
        self.assertNotEqual(result.explanation, "")

    def test_17_no_safe_exit_explanation_is_honest(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        result = engine.compute(0.0, make_building_state({"z1": HazardSeverity.HIGH})).zone("z1")

        self.assertIn("no safe exit", result.explanation.lower())


class DeterminismTests(unittest.TestCase):

    def test_18_deterministic_regardless_of_occupant_order(self):

        engine_a, manager_a = make_engine()
        engine_b, manager_b = make_engine()

        for occupant_id in ("A", "B", "C"):
            manager_a.update(occupant_id, "CAM-1", occupant_id, "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        for occupant_id in ("C", "B", "A"):
            manager_b.update(occupant_id, "CAM-1", occupant_id, "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        snapshot_a = engine_a.compute(0.0, make_building_state())
        snapshot_b = engine_b.compute(0.0, make_building_state())

        self.assertEqual(snapshot_a.zone("z2").to_dict(), snapshot_b.zone("z2").to_dict())


if __name__ == "__main__":
    unittest.main()
