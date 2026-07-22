import unittest

from behavior_recognition.observation import RecognizedBehavior

from hazard.severity import HazardSeverity

from trajectory_intelligence.models import (
    AnomalyFlag, MovementStatus, RouteDistanceTrend, RouteProgressStatus, TrajectoryConfig,
)

from tests.trajectory_intelligence_fixtures import make_building_state, make_engine


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone, Phase 27 -- deterministic engine-level unit
# coverage. No randomness anywhere in this file. Every scenario drives
# LiveOccupantManager.update() directly (the same "no full camera
# pipeline needed for engine-level coverage" convention
# tests/test_emergency_response.py already established) and asserts on
# TrajectoryIntelligenceEngine.compute()'s own TrajectoryIntelligenceSnapshot.
# =====================================================


class MovementFactsTests(unittest.TestCase):

    def test_1_position_history_produces_distance_speed_and_direction(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (0.0, 0.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (3.0, 4.0), None, RecognizedBehavior.WALKING, 0.9, 2.0)

        snapshot = engine.compute(2.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertTrue(result.position_available)
        self.assertEqual(result.position_sample_count, 2)
        self.assertAlmostEqual(result.distance_travelled, 5.0)
        self.assertAlmostEqual(result.net_displacement, 5.0)
        self.assertAlmostEqual(result.current_speed, 2.5)
        self.assertEqual(result.movement_status, MovementStatus.MOVING)

    def test_2_no_world_position_is_honest_not_fabricated(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", None, None, RecognizedBehavior.WALKING, 0.9, 0.0)

        snapshot = engine.compute(0.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertFalse(result.position_available)
        self.assertEqual(result.position_sample_count, 0)
        self.assertIsNone(result.current_position)


class RouteProgressTests(unittest.TestCase):

    def test_3_progressing_toward_exit_after_zone_transition(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        engine.compute(0.0, make_building_state())

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 4.0)
        engine.compute(4.0, make_building_state())

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 8.0)
        snapshot = engine.compute(8.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertEqual(result.route_progress_status, RouteProgressStatus.PROGRESSING_TOWARD_EXIT)
        self.assertEqual(result.route_distance_trend, RouteDistanceTrend.DECREASING)
        self.assertEqual(result.nearest_safe_exit_id, "EXIT-1")

    def test_4_moving_away_from_exit_after_zone_transition(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        engine.compute(0.0, make_building_state())

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 3.0)
        engine.compute(3.0, make_building_state())

        manager.update("OCC-1", "CAM-1", "T1", "z4", "f1", (41.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 7.0)
        snapshot = engine.compute(7.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertEqual(result.route_progress_status, RouteProgressStatus.MOVING_AWAY_FROM_EXIT)

    def test_5_first_cycle_is_route_uncertain_not_fabricated(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertEqual(result.route_progress_status, RouteProgressStatus.ROUTE_UNCERTAIN)

    def test_6_no_zone_identity_prevents_graph_route_claim(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", None, "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertEqual(result.route_progress_status, RouteProgressStatus.ROUTE_UNCERTAIN)

    def test_7_graph_distance_not_euclidean_shortcut(self):

        # z2 (Hall) and z4 (Annex) are geometrically adjacent (only 10m
        # of empty space) but only connected through the Navigation
        # Graph's own Door -- route_distance_m must reflect the much
        # longer walked path (Door + Exit), never the short straight
        # line, proving no through-wall Euclidean shortcut is ever used.

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z4", "f1", (41.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertGreater(result.route_distance_m, 40.0)

    def test_8_unsafe_zone_excluded_from_safe_routing(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, make_building_state({"z1": HazardSeverity.HIGH}))
        result = snapshot.occupant("OCC-1")

        self.assertEqual(result.nearest_safe_exit_id, "EXIT-2")

    def test_9_no_safe_route_when_own_zone_excluded(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, make_building_state({"z1": HazardSeverity.HIGH}))
        result = snapshot.occupant("OCC-1")

        self.assertEqual(result.route_progress_status, RouteProgressStatus.NO_SAFE_ROUTE)
        self.assertIn(AnomalyFlag.NO_SAFE_ROUTE, result.anomaly_flags)

    def test_10_structurally_blocked_exit_excluded(self):

        building_state = make_building_state()
        engine, manager = make_engine()

        # Structural blockage (Edge.traversable), independent of hazard.
        exit_one = engine.graph.find_node("z1").reference
        for door_or_exit in engine.building.ordered_floors()[0].exits:
            if door_or_exit.id == "EXIT-1":
                door_or_exit.is_blocked = True

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, building_state)
        result = snapshot.occupant("OCC-1")

        self.assertNotEqual(result.nearest_safe_exit_id, "EXIT-1")

    def test_11_no_hazard_information_is_uncertainty_not_fabrication(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, building_state=None)
        result = snapshot.occupant("OCC-1")

        self.assertNotEqual(result.route_progress_status, RouteProgressStatus.NO_SAFE_ROUTE)


class MovingAwayPersistenceTests(unittest.TestCase):

    def test_12_one_noisy_backward_sample_does_not_trigger_anomaly(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        engine.compute(0.0, make_building_state())

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 3.0)
        engine.compute(3.0, make_building_state())

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 6.0)
        snapshot = engine.compute(6.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertNotIn(AnomalyFlag.MOVING_AWAY_FROM_SAFE_EXIT, result.anomaly_flags)

    def test_13_persistent_moving_away_triggers_anomaly(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        engine.compute(0.0, make_building_state())

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 3.0)
        engine.compute(3.0, make_building_state())

        manager.update("OCC-1", "CAM-1", "T1", "z4", "f1", (41.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 7.0)
        snapshot = engine.compute(7.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertIn(AnomalyFlag.MOVING_AWAY_FROM_SAFE_EXIT, result.anomaly_flags)


class ReversalTests(unittest.TestCase):

    def test_14_zone_a_b_a_b_triggers_reversal(self):

        engine, manager = make_engine()

        for index, (zone_id, position) in enumerate([
            ("z1", (1.0, 1.0)), ("z2", (21.0, 1.0)), ("z1", (1.0, 1.0)), ("z2", (21.0, 1.0)),
        ]):
            manager.update("OCC-1", "CAM-1", "T1", zone_id, "f1", position, None, RecognizedBehavior.WALKING, 0.9, float(index))
            snapshot = engine.compute(float(index), make_building_state())

        result = snapshot.occupant("OCC-1")
        self.assertIn(AnomalyFlag.REPEATED_ROUTE_REVERSAL, result.anomaly_flags)

    def test_15_single_reversal_does_not_trigger(self):

        engine, manager = make_engine()

        for index, (zone_id, position) in enumerate([
            ("z1", (1.0, 1.0)), ("z2", (21.0, 1.0)), ("z1", (1.0, 1.0)),
        ]):
            manager.update("OCC-1", "CAM-1", "T1", zone_id, "f1", position, None, RecognizedBehavior.WALKING, 0.9, float(index))
            snapshot = engine.compute(float(index), make_building_state())

        result = snapshot.occupant("OCC-1")
        self.assertNotIn(AnomalyFlag.REPEATED_ROUTE_REVERSAL, result.anomaly_flags)


class MovementStalledTests(unittest.TestCase):

    def test_16_no_route_improvement_for_configured_duration_stalls(self):

        config = TrajectoryConfig(movement_stall_duration_seconds=10.0)
        engine, manager = make_engine(config=config)

        for t in (0.0, 4.0, 8.0, 12.0):
            manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.STATIONARY, 0.9, t)
            snapshot = engine.compute(t, make_building_state())

        result = snapshot.occupant("OCC-1")
        self.assertIn(AnomalyFlag.MOVEMENT_STALLED, result.anomaly_flags)

    def test_17_stationary_behavior_alone_is_not_stall_without_duration(self):

        config = TrajectoryConfig(movement_stall_duration_seconds=10.0)
        engine, manager = make_engine(config=config)

        for t in (0.0, 2.0):
            manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.STATIONARY, 0.9, t)
            snapshot = engine.compute(t, make_building_state())

        result = snapshot.occupant("OCC-1")
        self.assertNotIn(AnomalyFlag.MOVEMENT_STALLED, result.anomaly_flags)


class HazardousZoneTests(unittest.TestCase):

    def test_18_entering_hazardous_zone_detected(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        engine.compute(0.0, make_building_state())

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 2.0)
        snapshot = engine.compute(2.0, make_building_state({"z2": HazardSeverity.HIGH}))
        result = snapshot.occupant("OCC-1")

        self.assertIn(AnomalyFlag.ENTERED_HAZARDOUS_ZONE, result.anomaly_flags)
        self.assertNotIn(AnomalyFlag.REMAINS_IN_HAZARDOUS_ZONE, result.anomaly_flags)

    def test_19_remaining_in_hazardous_zone_detected_next_cycle(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        engine.compute(0.0, make_building_state({"z2": HazardSeverity.HIGH}))

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 2.0)
        snapshot = engine.compute(2.0, make_building_state({"z2": HazardSeverity.HIGH}))
        result = snapshot.occupant("OCC-1")

        self.assertIn(AnomalyFlag.REMAINS_IN_HAZARDOUS_ZONE, result.anomaly_flags)

    def test_20_no_hazard_evidence_no_fabricated_hazardous_flag(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-1", "T1", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(0.0, building_state=None)
        result = snapshot.occupant("OCC-1")

        self.assertNotIn(AnomalyFlag.ENTERED_HAZARDOUS_ZONE, result.anomaly_flags)
        self.assertNotIn(AnomalyFlag.REMAINS_IN_HAZARDOUS_ZONE, result.anomaly_flags)


class AgainstFlowTests(unittest.TestCase):

    def _walking_occupant(self, manager, occupant_id, zone_id, position, t):

        manager.update(occupant_id, "CAM-1", occupant_id, zone_id, "f1", position, None, RecognizedBehavior.WALKING, 0.9, t)

    def test_21_against_flow_detected_with_sufficient_evidence(self):

        config = TrajectoryConfig(against_flow_min_occupants=3, against_flow_min_coverage_fraction=0.5)
        engine, manager = make_engine(config=config)

        for occupant_id in ("A", "B", "C"):
            self._walking_occupant(manager, occupant_id, "z2", (20.0, 1.0), 0.0)
            self._walking_occupant(manager, occupant_id, "z1", (10.0, 1.0), 2.0)

        self._walking_occupant(manager, "D", "z1", (10.0, 1.0), 0.0)
        self._walking_occupant(manager, "D", "z2", (20.0, 1.0), 2.0)

        snapshot = engine.compute(2.0, make_building_state())

        self.assertIn(AnomalyFlag.AGAINST_DOMINANT_FLOW, snapshot.occupant("D").anomaly_flags)
        self.assertNotIn(AnomalyFlag.AGAINST_DOMINANT_FLOW, snapshot.occupant("A").anomaly_flags)

    def test_22_against_flow_unknown_with_too_few_occupants(self):

        config = TrajectoryConfig(against_flow_min_occupants=10)
        engine, manager = make_engine(config=config)

        for occupant_id in ("A", "B"):
            self._walking_occupant(manager, occupant_id, "z2", (20.0, 1.0), 0.0)
            self._walking_occupant(manager, occupant_id, "z1", (10.0, 1.0), 2.0)

        snapshot = engine.compute(2.0, make_building_state())

        self.assertEqual(snapshot.dominant_flow_direction_by_floor, {})
        for occupant_id in ("A", "B"):
            self.assertNotIn(AnomalyFlag.AGAINST_DOMINANT_FLOW, snapshot.occupant(occupant_id).anomaly_flags)

    def test_23_against_flow_unknown_with_poor_coverage(self):

        config = TrajectoryConfig(against_flow_min_occupants=2, against_flow_min_coverage_fraction=0.9)
        engine, manager = make_engine(config=config)

        for occupant_id in ("A", "B"):
            self._walking_occupant(manager, occupant_id, "z2", (20.0, 1.0), 0.0)
            self._walking_occupant(manager, occupant_id, "z1", (10.0, 1.0), 2.0)

        # Many stationary occupants dilute coverage below the required fraction.
        for occupant_id in ("S1", "S2", "S3", "S4", "S5", "S6"):
            manager.update(occupant_id, "CAM-1", occupant_id, "z1", "f1", (5.0, 5.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 2.0)

        snapshot = engine.compute(2.0, make_building_state())

        self.assertEqual(snapshot.dominant_flow_direction_by_floor, {})


class GroupAnomalyTests(unittest.TestCase):

    def test_24_shared_route_deviation_detected_for_multiple_occupants(self):

        engine, manager = make_engine()

        for zone_id, position, t in (("z1", (1.0, 1.0), 0.0), ("z2", (21.0, 1.0), 3.0), ("z4", (41.0, 1.0), 7.0)):

            for occupant_id in ("A", "B"):
                manager.update(occupant_id, "CAM-1", occupant_id, zone_id, "f1", position, None, RecognizedBehavior.WALKING, 0.9, t)

            snapshot = engine.compute(t, make_building_state())

        self.assertIn(AnomalyFlag.SHARED_ROUTE_DEVIATION, snapshot.occupant("A").anomaly_flags)
        self.assertIn(AnomalyFlag.SHARED_ROUTE_DEVIATION, snapshot.occupant("B").anomaly_flags)
        self.assertTrue(any(record.anomaly_type == "SHARED_ROUTE_DEVIATION" for record in snapshot.group_anomalies))

    def test_25_single_occupant_does_not_trigger_group_anomaly(self):

        engine, manager = make_engine()

        manager.update("A", "CAM-1", "A", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        engine.compute(0.0, make_building_state())

        manager.update("A", "CAM-1", "A", "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 3.0)
        engine.compute(3.0, make_building_state())

        manager.update("A", "CAM-1", "A", "z4", "f1", (41.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 7.0)
        snapshot = engine.compute(7.0, make_building_state())

        self.assertNotIn(AnomalyFlag.SHARED_ROUTE_DEVIATION, snapshot.occupant("A").anomaly_flags)
        self.assertEqual(snapshot.group_anomalies, ())


class FloorTransitionTests(unittest.TestCase):

    def test_26_valid_stair_floor_transition_handled(self):

        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-2", "T1", "z3", "f2", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        engine.compute(0.0, make_building_state())

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 5.0)
        snapshot = engine.compute(5.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertNotEqual(result.route_progress_status, RouteProgressStatus.ROUTE_UNCERTAIN)

    def test_27_floor_change_without_known_route_is_uncertain(self):

        # z3 (f2) and z4 (f1) have no Stair connecting them directly --
        # an "instant" floor jump between them must be treated as
        # ungrounded, never a valid transition.
        engine, manager = make_engine()

        manager.update("OCC-1", "CAM-2", "T1", "z3", "f2", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        engine.compute(0.0, make_building_state())

        manager.update("OCC-1", "CAM-1", "T1", "z4", "f1", (41.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 5.0)
        snapshot = engine.compute(5.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertEqual(result.route_progress_status, RouteProgressStatus.ROUTE_UNCERTAIN)


class StalenessTests(unittest.TestCase):

    def test_28_long_gap_marks_trajectory_stale(self):

        config = TrajectoryConfig(staleness_seconds=5.0)
        engine, manager = make_engine(config=config)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(20.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertTrue(result.stale)

    def test_29_short_gap_preserves_continuity(self):

        config = TrajectoryConfig(staleness_seconds=5.0)
        engine, manager = make_engine(config=config)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot = engine.compute(1.0, make_building_state())
        result = snapshot.occupant("OCC-1")

        self.assertFalse(result.stale)


class ClassificationIsolationTests(unittest.TestCase):

    def test_30_human_classification_has_no_trajectory_effect(self):

        from perception.models.human_observation import HumanClassification

        engine, manager = make_engine()

        manager.update(
            "OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0,
            classification_evidence=HumanClassification.CHILD, classification_confidence=0.9,
        )
        snapshot_with = engine.compute(0.0, make_building_state())

        engine2, manager2 = make_engine()
        manager2.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)
        snapshot_without = engine2.compute(0.0, make_building_state())

        self.assertEqual(
            snapshot_with.occupant("OCC-1").route_progress_status,
            snapshot_without.occupant("OCC-1").route_progress_status,
        )
        self.assertEqual(snapshot_with.occupant("OCC-1").anomaly_flags, snapshot_without.occupant("OCC-1").anomaly_flags)


class DeterminismTests(unittest.TestCase):

    def test_31_result_independent_of_occupant_iteration_order(self):

        engine_a, manager_a = make_engine()
        engine_b, manager_b = make_engine()

        for occupant_id in ("A", "B", "C"):
            manager_a.update(occupant_id, "CAM-1", occupant_id, "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        for occupant_id in ("C", "B", "A"):
            manager_b.update(occupant_id, "CAM-1", occupant_id, "z2", "f1", (21.0, 1.0), None, RecognizedBehavior.WALKING, 0.9, 0.0)

        snapshot_a = engine_a.compute(0.0, make_building_state())
        snapshot_b = engine_b.compute(0.0, make_building_state())

        for occupant_id in ("A", "B", "C"):
            self.assertEqual(
                snapshot_a.occupant(occupant_id).to_dict(), snapshot_b.occupant(occupant_id).to_dict(),
            )


if __name__ == "__main__":
    unittest.main()
