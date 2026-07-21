import unittest

from behavior_recognition.observation import RecognizedBehavior

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from crowd_intelligence.congestion import CongestionThresholds, compute_congestion_level
from crowd_intelligence.density import compute_zone_density
from crowd_intelligence.engine import CrowdIntelligenceEngine
from crowd_intelligence.models import DensityThresholds, IntensityLevel, TrendDirection
from crowd_intelligence.queue import QueueMetrics
from crowd_intelligence.trends import TrendConfig, TrendTracker


# =====================================================
# Live Occupancy, Crowd Density & Congestion Intelligence milestone,
# Phase 15 -- deterministic unit coverage. No randomness anywhere in
# this file: every occupant/geometry input is hand-constructed.
# =====================================================


def make_single_zone_building(zone_kwargs=None, door=None, exit_obj=None, stair=None):

    zone_kwargs = zone_kwargs or {}

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z1", name="Zone 1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1", **zone_kwargs)],
        doors=[door] if door is not None else [],
        exits=[exit_obj] if exit_obj is not None else [],
        stairs=[stair] if stair is not None else [],
    )

    return Building(id="b1", name="Building 1", floors=[floor])


def add_occupant(manager, occupant_id, zone_id, floor_id, position, behavior, time, velocity=0.0):

    return manager.update(
        occupant_id, "CAM-1", occupant_id, zone_id, floor_id, position, velocity, behavior, 0.9, time,
    )


class DensityCalculationTests(unittest.TestCase):

    def test_single_occupant_density(self):

        self.assertAlmostEqual(compute_zone_density(1, 10.0), 0.1)

    def test_multiple_occupants_density(self):

        self.assertAlmostEqual(compute_zone_density(5, 10.0), 0.5)

    def test_zero_area_produces_no_density(self):

        self.assertIsNone(compute_zone_density(3, 0.0))

    def test_none_area_produces_no_density(self):

        self.assertIsNone(compute_zone_density(3, None))


class DensityClassificationTests(unittest.TestCase):

    def setUp(self):
        self.thresholds = DensityThresholds()

    def test_low(self):
        self.assertEqual(self.thresholds.classify(0.5), IntensityLevel.LOW)

    def test_moderate(self):
        self.assertEqual(self.thresholds.classify(1.0), IntensityLevel.MODERATE)

    def test_high(self):
        self.assertEqual(self.thresholds.classify(2.0), IntensityLevel.HIGH)

    def test_very_high(self):
        self.assertEqual(self.thresholds.classify(3.0), IntensityLevel.VERY_HIGH)

    def test_critical(self):
        self.assertEqual(self.thresholds.classify(4.0), IntensityLevel.CRITICAL)

    def test_none_density_has_no_classification(self):
        self.assertIsNone(self.thresholds.classify(None))

    def test_thresholds_are_configurable(self):

        custom = DensityThresholds(moderate_at=0.1, high_at=0.2, very_high_at=0.3, critical_at=0.4)
        self.assertEqual(custom.classify(0.25), IntensityLevel.HIGH)


class EngineZoneTests(unittest.TestCase):

    def test_empty_zone_reports_zero_occupants_not_none(self):

        building = make_single_zone_building()
        manager = LiveOccupantManager()
        engine = CrowdIntelligenceEngine(building, manager)

        snapshot = engine.compute(0.0)
        zone = snapshot.zone("z1")

        self.assertEqual(zone.occupant_count, 0)
        self.assertEqual(zone.density_people_per_m2, 0.0)
        self.assertEqual(zone.density_classification, IntensityLevel.LOW)

    def test_single_occupant_in_zone(self):

        building = make_single_zone_building()
        manager = LiveOccupantManager()
        add_occupant(manager, "OCC-1", "z1", "f1", (5.0, 5.0), RecognizedBehavior.STATIONARY, 0.0)

        engine = CrowdIntelligenceEngine(building, manager)
        zone = engine.compute(0.0).zone("z1")

        self.assertEqual(zone.occupant_count, 1)
        self.assertEqual(zone.stationary_count, 1)
        self.assertAlmostEqual(zone.density_people_per_m2, 0.01)

    def test_multiple_occupants_in_zone(self):

        building = make_single_zone_building()
        manager = LiveOccupantManager()
        for i in range(4):
            add_occupant(manager, f"OCC-{i}", "z1", "f1", (float(i), 5.0), RecognizedBehavior.STATIONARY, 0.0)

        engine = CrowdIntelligenceEngine(building, manager)
        zone = engine.compute(0.0).zone("z1")

        self.assertEqual(zone.occupant_count, 4)

    def test_multi_zone_building_keeps_zones_independent(self):

        floor = Floor(
            id="f1", name="Floor 1",
            zones=[
                Zone(id="z1", name="Zone 1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
                Zone(id="z2", name="Zone 2", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            ],
        )
        building = Building(id="b1", name="Building 1", floors=[floor])

        manager = LiveOccupantManager()
        add_occupant(manager, "OCC-1", "z1", "f1", (5.0, 5.0), RecognizedBehavior.STATIONARY, 0.0)
        add_occupant(manager, "OCC-2", "z2", "f1", (25.0, 5.0), RecognizedBehavior.STATIONARY, 0.0)
        add_occupant(manager, "OCC-3", "z2", "f1", (26.0, 5.0), RecognizedBehavior.STATIONARY, 0.0)

        engine = CrowdIntelligenceEngine(building, manager)
        snapshot = engine.compute(0.0)

        self.assertEqual(snapshot.zone("z1").occupant_count, 1)
        self.assertEqual(snapshot.zone("z2").occupant_count, 2)

    def test_moving_occupants_counted(self):

        building = make_single_zone_building()
        manager = LiveOccupantManager()
        add_occupant(manager, "OCC-1", "z1", "f1", (5.0, 5.0), RecognizedBehavior.WALKING, 0.0, velocity=1.0)

        zone = CrowdIntelligenceEngine(building, manager).compute(0.0).zone("z1")

        self.assertEqual(zone.moving_count, 1)
        self.assertEqual(zone.stationary_count, 0)

    def test_stationary_occupants_counted(self):

        building = make_single_zone_building()
        manager = LiveOccupantManager()
        add_occupant(manager, "OCC-1", "z1", "f1", (5.0, 5.0), RecognizedBehavior.STATIONARY, 0.0)

        zone = CrowdIntelligenceEngine(building, manager).compute(0.0).zone("z1")

        self.assertEqual(zone.stationary_count, 1)
        self.assertEqual(zone.moving_count, 0)

    def test_running_occupants_counted_as_both_moving_and_running(self):

        building = make_single_zone_building()
        manager = LiveOccupantManager()
        add_occupant(manager, "OCC-1", "z1", "f1", (5.0, 5.0), RecognizedBehavior.RUNNING, 0.0, velocity=3.0)

        zone = CrowdIntelligenceEngine(building, manager).compute(0.0).zone("z1")

        self.assertEqual(zone.running_count, 1)
        self.assertEqual(zone.moving_count, 1)

    def test_occupant_transition_between_zones_is_reflected_next_cycle(self):

        floor = Floor(
            id="f1", name="Floor 1",
            zones=[
                Zone(id="z1", name="Zone 1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
                Zone(id="z2", name="Zone 2", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            ],
        )
        building = Building(id="b1", name="Building 1", floors=[floor])
        manager = LiveOccupantManager()

        add_occupant(manager, "OCC-1", "z1", "f1", (5.0, 5.0), RecognizedBehavior.WALKING, 0.0, velocity=1.0)
        engine = CrowdIntelligenceEngine(building, manager)
        self.assertEqual(engine.compute(0.0).zone("z1").occupant_count, 1)
        self.assertEqual(engine.compute(0.0).zone("z2").occupant_count, 0)

        add_occupant(manager, "OCC-1", "z2", "f1", (25.0, 5.0), RecognizedBehavior.WALKING, 1.0, velocity=1.0)
        self.assertEqual(engine.compute(1.0).zone("z1").occupant_count, 0)
        self.assertEqual(engine.compute(1.0).zone("z2").occupant_count, 1)

    def test_temporarily_lost_occupant_reported_separately_never_double_counted(self):

        building = make_single_zone_building()
        manager = LiveOccupantManager(expire_after_seconds=1000.0)

        add_occupant(manager, "OCC-1", "z1", "f1", (5.0, 5.0), RecognizedBehavior.STATIONARY, 0.0)
        manager.sweep_missing(1.0, seen_occupant_ids=set())  # not seen this cycle -> TEMPORARILY_LOST

        zone = CrowdIntelligenceEngine(building, manager).compute(1.0).zone("z1")

        self.assertEqual(zone.occupant_count, 0)  # not ACTIVE -- never counted here
        self.assertEqual(zone.temporarily_lost_count, 1)  # but honestly reported separately

    def test_expired_occupant_is_not_reported_anywhere(self):

        building = make_single_zone_building()
        manager = LiveOccupantManager(expire_after_seconds=1.0)

        add_occupant(manager, "OCC-1", "z1", "f1", (5.0, 5.0), RecognizedBehavior.STATIONARY, 0.0)
        manager.sweep_missing(10.0, seen_occupant_ids=set())  # far beyond expire_after_seconds

        zone = CrowdIntelligenceEngine(building, manager).compute(10.0).zone("z1")

        self.assertEqual(zone.occupant_count, 0)
        self.assertEqual(zone.temporarily_lost_count, 0)


class AssetApproachTests(unittest.TestCase):

    def test_door_approach_detected(self):

        door = Door(id="d1", floor_id="f1", start_point=(10.0, 4.0), end_point=(10.0, 6.0), width=1.0)
        building = make_single_zone_building(door=door)
        manager = LiveOccupantManager()

        add_occupant(manager, "OCC-1", "z1", "f1", (9.0, 5.0), RecognizedBehavior.WALKING, 0.0, velocity=1.0)
        engine = CrowdIntelligenceEngine(building, manager)
        engine.compute(0.0)

        add_occupant(manager, "OCC-1", "z1", "f1", (9.5, 5.0), RecognizedBehavior.WALKING, 1.0, velocity=1.0)
        door_metrics = engine.compute(1.0).door("d1")

        self.assertTrue(door_metrics.position_available)
        self.assertEqual(door_metrics.approaching_count, 1)

    def test_exit_approach_detected(self):

        exit_obj = Exit(id="e1", floor_id="f1", start_point=(10.0, 4.0), end_point=(10.0, 6.0), width=1.2, capacity=10)
        building = make_single_zone_building(exit_obj=exit_obj)
        manager = LiveOccupantManager()

        add_occupant(manager, "OCC-1", "z1", "f1", (9.0, 5.0), RecognizedBehavior.WALKING, 0.0, velocity=1.0)
        engine = CrowdIntelligenceEngine(building, manager)
        engine.compute(0.0)

        add_occupant(manager, "OCC-1", "z1", "f1", (9.5, 5.0), RecognizedBehavior.WALKING, 1.0, velocity=1.0)
        exit_metrics = engine.compute(1.0).exit("e1")

        self.assertEqual(exit_metrics.approaching_count, 1)

    def test_stair_approach_detected(self):

        stair = Staircase(
            id="s1", from_position=(9.0, 5.0), to_position=(9.0, 5.0),
            from_floor_id="f1", to_floor_id="f2", width=1.5,
        )
        building = make_single_zone_building(stair=stair)
        manager = LiveOccupantManager()

        add_occupant(manager, "OCC-1", "z1", "f1", (5.0, 5.0), RecognizedBehavior.WALKING, 0.0, velocity=1.0)
        engine = CrowdIntelligenceEngine(building, manager)
        engine.compute(0.0)

        add_occupant(manager, "OCC-1", "z1", "f1", (7.0, 5.0), RecognizedBehavior.WALKING, 1.0, velocity=1.0)
        stair_metrics = engine.compute(1.0).stair("s1")

        self.assertTrue(stair_metrics.position_available)
        self.assertEqual(stair_metrics.approaching_count, 1)

    def test_multi_floor_stair_scenario_tracks_both_sides_independently(self):

        stair = Staircase(
            id="s1", from_position=(1.0, 1.0), to_position=(2.0, 2.0),
            from_floor_id="f1", to_floor_id="f2", width=1.5,
        )
        floor1 = Floor(id="f1", name="Floor 1", zones=[Zone(id="z1", name="Z1", x=0, y=0, width=10, height=10, floor_id="f1")], stairs=[stair])
        floor2 = Floor(id="f2", name="Floor 2", zones=[Zone(id="z2", name="Z2", x=0, y=0, width=10, height=10, floor_id="f2")])
        building = Building(id="b1", name="B", floors=[floor1, floor2])

        manager = LiveOccupantManager()
        add_occupant(manager, "OCC-F1", "z1", "f1", (1.0, 1.0), RecognizedBehavior.STATIONARY, 0.0)
        add_occupant(manager, "OCC-F2", "z2", "f2", (2.0, 2.0), RecognizedBehavior.STATIONARY, 0.0)

        stair_metrics = CrowdIntelligenceEngine(building, manager).compute(0.0).stair("s1")

        # Both floor-1 and floor-2 occupants are candidates for the SAME
        # stair (its own two independent sides), never conflated into
        # the wrong floor's coordinate space.
        self.assertEqual(stair_metrics.queue_candidate_count, 2)

    def test_wide_asset_has_higher_capacity_than_narrow_asset(self):

        narrow_door = Door(id="d-narrow", floor_id="f1", start_point=(10.0, 4.0), end_point=(10.0, 5.0), width=0.9)
        wide_door = Door(id="d-wide", floor_id="f1", start_point=(10.0, 4.0), end_point=(10.0, 5.0), width=3.0)

        floor = Floor(
            id="f1", name="Floor 1",
            zones=[Zone(id="z1", name="Z1", x=0, y=0, width=10, height=10, floor_id="f1")],
            doors=[narrow_door, wide_door],
        )
        building = Building(id="b1", name="B", floors=[floor])
        manager = LiveOccupantManager()

        snapshot = CrowdIntelligenceEngine(building, manager).compute(0.0)

        self.assertGreater(snapshot.door("d-wide").simulation_style_capacity, snapshot.door("d-narrow").simulation_style_capacity)

    def test_asset_with_missing_geometry_still_reports_a_capacity_floor(self):

        # Door.width always has a default (0.90) -- there is no "missing
        # width" case for Door/Exit in this codebase's own models, so
        # this proves the floor behavior for the smallest configured
        # width instead of a genuinely absent one.
        door = Door(id="d1", floor_id="f1", start_point=(10.0, 4.0), end_point=(10.0, 5.0), width=0.01)
        building = make_single_zone_building(door=door)
        manager = LiveOccupantManager()

        door_metrics = CrowdIntelligenceEngine(building, manager).compute(0.0).door("d1")

        self.assertGreaterEqual(door_metrics.simulation_style_capacity, 1)


class QueueFormationTests(unittest.TestCase):

    def test_queue_forms_as_occupants_accumulate_and_stop(self):

        exit_obj = Exit(id="e1", floor_id="f1", start_point=(10.0, 4.0), end_point=(10.0, 6.0), width=1.2, capacity=10)
        building = make_single_zone_building(exit_obj=exit_obj)
        manager = LiveOccupantManager()

        engine = CrowdIntelligenceEngine(building, manager)

        before = engine.compute(0.0).exit("e1")
        self.assertEqual(before.queue_candidate_count, 0)

        for i in range(3):
            add_occupant(manager, f"OCC-{i}", "z1", "f1", (9.0, 5.0 + i * 0.1), RecognizedBehavior.STATIONARY, 1.0)

        after = engine.compute(1.0).exit("e1")
        self.assertEqual(after.queue_candidate_count, 3)
        self.assertEqual(after.estimated_queue_length, 3)

    def test_queue_disappears_once_occupants_leave(self):

        exit_obj = Exit(id="e1", floor_id="f1", start_point=(10.0, 4.0), end_point=(10.0, 6.0), width=1.2, capacity=10)
        building = make_single_zone_building(exit_obj=exit_obj)
        manager = LiveOccupantManager()
        engine = CrowdIntelligenceEngine(building, manager)

        for i in range(3):
            add_occupant(manager, f"OCC-{i}", "z1", "f1", (9.0, 5.0 + i * 0.1), RecognizedBehavior.STATIONARY, 0.0)

        self.assertEqual(engine.compute(0.0).exit("e1").queue_candidate_count, 3)

        manager.sweep_missing(1.0, seen_occupant_ids=set())  # everyone leaves/goes missing

        after = engine.compute(1.0).exit("e1")
        self.assertEqual(after.queue_candidate_count, 0)
        self.assertEqual(after.estimated_queue_length, 0)

    def test_no_queue_fabricated_from_zone_occupancy_alone(self):

        # Occupants exist in the zone but nowhere near the exit -- never
        # counted as a queue just because the zone itself is occupied.
        exit_obj = Exit(id="e1", floor_id="f1", start_point=(10.0, 4.0), end_point=(10.0, 6.0), width=1.2, capacity=10)
        building = make_single_zone_building(exit_obj=exit_obj)
        manager = LiveOccupantManager()

        for i in range(3):
            add_occupant(manager, f"OCC-{i}", "z1", "f1", (0.5, 0.5 + i * 0.1), RecognizedBehavior.STATIONARY, 0.0)

        exit_metrics = CrowdIntelligenceEngine(building, manager).compute(0.0).exit("e1")

        self.assertEqual(exit_metrics.queue_candidate_count, 0)


class CongestionTests(unittest.TestCase):

    def test_congestion_increases_with_demand(self):

        thresholds = CongestionThresholds()

        low_demand = compute_congestion_level(QueueMetrics(approaching_count=0, queue_candidate_count=1), 10, thresholds)
        high_demand = compute_congestion_level(QueueMetrics(approaching_count=5, queue_candidate_count=10), 10, thresholds)

        self.assertLess(low_demand.value, high_demand.value)

    def test_congestion_none_without_known_capacity(self):

        self.assertIsNone(compute_congestion_level(QueueMetrics(queue_candidate_count=3), None, CongestionThresholds()))

    def test_congestion_low_when_no_demand(self):

        self.assertEqual(compute_congestion_level(QueueMetrics(), 10, CongestionThresholds()), IntensityLevel.LOW)

    def test_congestion_rises_over_time_as_queue_grows(self):

        exit_obj = Exit(id="e1", floor_id="f1", start_point=(10.0, 4.0), end_point=(10.0, 6.0), width=1.2, capacity=2)
        building = make_single_zone_building(exit_obj=exit_obj)
        manager = LiveOccupantManager()
        engine = CrowdIntelligenceEngine(building, manager)

        add_occupant(manager, "OCC-0", "z1", "f1", (9.0, 5.0), RecognizedBehavior.STATIONARY, 0.0)
        level_1 = engine.compute(0.0).exit("e1").congestion_level

        for i in range(1, 5):
            add_occupant(manager, f"OCC-{i}", "z1", "f1", (9.0, 5.0 + i * 0.1), RecognizedBehavior.STATIONARY, float(i))

        level_2 = engine.compute(4.0).exit("e1").congestion_level

        self.assertLess(level_1.value, level_2.value)

    def test_congestion_falls_as_queue_clears(self):

        exit_obj = Exit(id="e1", floor_id="f1", start_point=(10.0, 4.0), end_point=(10.0, 6.0), width=1.2, capacity=2)
        building = make_single_zone_building(exit_obj=exit_obj)
        manager = LiveOccupantManager()
        engine = CrowdIntelligenceEngine(building, manager)

        for i in range(5):
            add_occupant(manager, f"OCC-{i}", "z1", "f1", (9.0, 5.0 + i * 0.1), RecognizedBehavior.STATIONARY, 0.0)

        crowded = engine.compute(0.0).exit("e1").congestion_level

        manager.sweep_missing(1.0, seen_occupant_ids=set())
        cleared = engine.compute(1.0).exit("e1").congestion_level

        self.assertGreater(crowded.value, cleared.value)


class TrendTests(unittest.TestCase):

    def test_unknown_with_insufficient_history(self):

        tracker = TrendTracker()
        self.assertEqual(tracker.observe("k", 0.0, 1.0), TrendDirection.UNKNOWN)

    def test_rising_trend(self):

        tracker = TrendTracker(TrendConfig(trend_window_seconds=100.0))
        tracker.observe("k", 0.0, 1.0)
        self.assertEqual(tracker.observe("k", 1.0, 10.0), TrendDirection.RISING)

    def test_falling_trend(self):

        tracker = TrendTracker(TrendConfig(trend_window_seconds=100.0))
        tracker.observe("k", 0.0, 10.0)
        self.assertEqual(tracker.observe("k", 1.0, 1.0), TrendDirection.FALLING)

    def test_stable_trend_within_tolerance(self):

        tracker = TrendTracker(TrendConfig(trend_window_seconds=100.0, stable_absolute_tolerance=0.5))
        tracker.observe("k", 0.0, 1.0)
        self.assertEqual(tracker.observe("k", 1.0, 1.2), TrendDirection.STABLE)

    def test_none_value_never_recorded_or_treated_as_zero(self):

        tracker = TrendTracker(TrendConfig(trend_window_seconds=100.0))
        tracker.observe("k", 0.0, 10.0)
        self.assertEqual(tracker.observe("k", 1.0, None), TrendDirection.UNKNOWN)
        # The None observation was never recorded -- a real reading right
        # after it still compares against the last REAL value (10.0).
        self.assertEqual(tracker.observe("k", 2.0, 10.5), TrendDirection.STABLE)

    def test_bounded_history_never_grows_unbounded(self):

        tracker = TrendTracker(TrendConfig(max_history_length=3, trend_window_seconds=1000.0))

        for i in range(50):
            tracker.observe("k", float(i), float(i))

        self.assertEqual(len(tracker._history["k"]), 3)

    def test_trend_window_excludes_samples_older_than_configured_window(self):

        tracker = TrendTracker(TrendConfig(max_history_length=100, trend_window_seconds=5.0))

        tracker.observe("k", 0.0, 1.0)     # outside the window once we reach t=100
        result = tracker.observe("k", 100.0, 1.0)

        # No sample within the last 5 seconds except the current one --
        # honestly UNKNOWN, never comparing against a 100-second-stale baseline.
        self.assertEqual(result, TrendDirection.UNKNOWN)


class DeterministicOrderingTests(unittest.TestCase):

    def test_building_summary_ordering_is_deterministic_across_runs(self):

        floor = Floor(
            id="f1", name="Floor 1",
            zones=[
                Zone(id="z1", name="Z1", x=0, y=0, width=10, height=10, floor_id="f1"),
                Zone(id="z2", name="Z2", x=20, y=0, width=10, height=10, floor_id="f1"),
                Zone(id="z3", name="Z3", x=40, y=0, width=10, height=10, floor_id="f1"),
            ],
        )
        building = Building(id="b1", name="B", floors=[floor])
        manager = LiveOccupantManager()

        add_occupant(manager, "OCC-1", "z2", "f1", (25.0, 5.0), RecognizedBehavior.STATIONARY, 0.0)

        results = [
            CrowdIntelligenceEngine(building, manager).compute(0.0).building_summary.zones_above_configured_density_threshold
            for _ in range(5)
        ]

        self.assertTrue(all(result == results[0] for result in results))


if __name__ == "__main__":
    unittest.main()
