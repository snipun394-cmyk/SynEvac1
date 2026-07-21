import unittest

from behavior_recognition.observation import RecognizedBehavior

from models.building import Building
from models.door import Door
from models.floor import Floor
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from crowd_intelligence.engine import CrowdIntelligenceEngine


# =====================================================
# Live Occupancy, Crowd Density & Congestion Intelligence milestone,
# Phases 13/14 -- mixed camera calibration coverage. A "known occupant"
# (a resolved global identity, current_zone_id set) is NOT the same
# thing as a "known precise position" (world_position set) -- this
# package must distinguish the two, never silently assigning an
# arbitrary coordinate, and must report coverage honestly rather than
# a falsely-precise number.
# =====================================================


def make_building(door=None):

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1")],
        doors=[door] if door is not None else [],
    )

    return Building(id="b1", name="B", floors=[floor])


class ZoneOccupancyWithoutWorldPositionTests(unittest.TestCase):

    def test_zone_occupancy_usable_when_zone_known_but_position_unavailable(self):

        # Simulates an uncalibrated camera: the pipeline still resolved
        # this occupant's current_zone_id (a fixed camera's own known
        # zone assignment), but camera_calibration.projection.
        # WorldProjector had no calibration for that camera, so
        # world_position stays honestly None (see camera_calibration.
        # projection.WorldProjector.project()'s own "no calibration ->
        # every field None" contract).
        building = make_building()
        manager = LiveOccupantManager()
        manager.update("OCC-1", "CAM-UNCALIBRATED", "T1", "z1", "f1", None, None, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        zone = CrowdIntelligenceEngine(building, manager).compute(0.0).zone("z1")

        # Occupant count/density are ZONE-IDENTITY-based, not position-
        # based -- still honestly computed.
        self.assertEqual(zone.occupant_count, 1)
        self.assertAlmostEqual(zone.density_people_per_m2, 0.01)

        # But position coverage honestly reports zero -- never fabricated.
        self.assertEqual(zone.position_coverage_count, 0)
        self.assertEqual(zone.position_coverage_fraction, 0.0)

    def test_partial_calibration_coverage_reports_a_fraction_not_a_guess(self):

        building = make_building()
        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-A", "T1", "z1", "f1", (5.0, 5.0), 0.1, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.update("OCC-2", "CAM-B", "T2", "z1", "f1", None, None, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        zone = CrowdIntelligenceEngine(building, manager).compute(0.0).zone("z1")

        self.assertEqual(zone.occupant_count, 2)
        self.assertEqual(zone.position_coverage_count, 1)
        self.assertAlmostEqual(zone.position_coverage_fraction, 0.5)

    def test_zero_calibration_coverage_reports_zero_fraction_not_none(self):

        building = make_building()
        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-A", "T1", "z1", "f1", None, None, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.update("OCC-2", "CAM-B", "T2", "z1", "f1", None, None, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        zone = CrowdIntelligenceEngine(building, manager).compute(0.0).zone("z1")

        self.assertEqual(zone.position_coverage_count, 0)
        self.assertEqual(zone.position_coverage_fraction, 0.0)  # a real, computed 0.0 -- known denominator, known numerator

    def test_position_coverage_fraction_is_none_when_zone_has_no_occupants_at_all(self):

        building = make_building()
        manager = LiveOccupantManager()

        zone = CrowdIntelligenceEngine(building, manager).compute(0.0).zone("z1")

        # No honest denominator to compute a fraction of -- never a
        # fabricated 0.0/1.0.
        self.assertIsNone(zone.position_coverage_fraction)


class BuildingSummaryCoverageTests(unittest.TestCase):

    def test_building_summary_reports_calibrated_vs_total_honestly(self):

        building = make_building()
        manager = LiveOccupantManager()

        manager.update("OCC-1", "CAM-A", "T1", "z1", "f1", (5.0, 5.0), 0.1, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.update("OCC-2", "CAM-B", "T2", "z1", "f1", None, None, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        manager.update("OCC-3", "CAM-B", "T3", "z1", "f1", None, None, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        summary = CrowdIntelligenceEngine(building, manager).compute(0.0).building_summary

        self.assertEqual(summary.total_observed_occupants, 3)
        self.assertEqual(summary.calibrated_occupant_count, 1)
        self.assertAlmostEqual(summary.position_coverage_fraction, 1 / 3)


class AssetMetricsPositionUnavailableTests(unittest.TestCase):

    def test_asset_metrics_report_unavailable_when_a_known_occupant_on_its_floor_has_no_position(self):

        door = Door(id="d1", floor_id="f1", start_point=(5.0, 0.0), end_point=(5.0, 1.0))
        building = make_building(door=door)
        manager = LiveOccupantManager()

        # A known occupant, on the SAME floor as this door, but with no
        # usable position -- Phase 13's own core distinguishing case.
        manager.update("OCC-1", "CAM-UNCALIBRATED", "T1", "z1", "f1", None, None, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        door_metrics = CrowdIntelligenceEngine(building, manager).compute(0.0).door("d1")

        self.assertFalse(door_metrics.position_available)
        self.assertIsNone(door_metrics.estimated_queue_length)
        self.assertIsNone(door_metrics.mean_approach_speed)
        self.assertEqual(door_metrics.approaching_count, 0)

    def test_asset_metrics_available_and_honestly_zero_when_nobody_present_at_all(self):

        door = Door(id="d1", floor_id="f1", start_point=(5.0, 0.0), end_point=(5.0, 1.0))
        building = make_building(door=door)
        manager = LiveOccupantManager()

        door_metrics = CrowdIntelligenceEngine(building, manager).compute(0.0).door("d1")

        # Vacuously available (nobody at all to have missing coverage
        # for) -- a genuine, position-confirmed zero, distinct from the
        # "unavailable" case above.
        self.assertTrue(door_metrics.position_available)
        self.assertEqual(door_metrics.estimated_queue_length, 0)

    def test_asset_metrics_available_when_the_only_occupant_missing_position_is_on_a_different_floor(self):

        door = Door(id="d1", floor_id="f1", start_point=(5.0, 0.0), end_point=(5.0, 1.0))
        floor2 = Floor(id="f2", name="Floor 2", zones=[Zone(id="z2", name="Z2", x=0, y=0, width=10, height=10, floor_id="f2")])
        building = make_building(door=door)
        building.floors.append(floor2)

        manager = LiveOccupantManager()
        manager.update("OCC-1", "CAM-OTHER", "T1", "z2", "f2", None, None, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        door_metrics = CrowdIntelligenceEngine(building, manager).compute(0.0).door("d1")

        # An occupant with no position exists SOMEWHERE in the building,
        # but not on this door's own floor -- must not falsely mark this
        # unrelated asset's own coverage as reduced.
        self.assertTrue(door_metrics.position_available)


if __name__ == "__main__":
    unittest.main()
