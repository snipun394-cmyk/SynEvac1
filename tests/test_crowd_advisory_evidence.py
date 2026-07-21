import unittest

from behavior_recognition.observation import RecognizedBehavior

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from crowd_intelligence.engine import CrowdIntelligenceEngine

from advisory_system.crowd_evidence import UNAVAILABLE_CROWD_DECISION_EVIDENCE

from live_system.live_advisory_gateway import crowd_decision_evidence_from_snapshot


# =====================================================
# Live Crowd Intelligence -> Operational Advisory Integration milestone,
# Phase 2/3 -- the adapter itself: CrowdIntelligenceSnapshot ->
# CrowdDecisionEvidence. Must preserve ids/levels/trends/coverage
# honestly, never invent missing data, and never balloon into a copy of
# the entire snapshot.
# =====================================================


def make_building():

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1", max_occupancy=1)],
        exits=[Exit(id="e1", floor_id="f1", start_point=(9.0, 4.0), end_point=(9.0, 6.0), width=1.2, capacity=1)],
    )

    return Building(id="b1", name="B", floors=[floor])


class UnavailableEvidenceTests(unittest.TestCase):

    def test_none_snapshot_produces_the_canonical_unavailable_instance(self):

        self.assertEqual(crowd_decision_evidence_from_snapshot(None), UNAVAILABLE_CROWD_DECISION_EVIDENCE)
        self.assertFalse(crowd_decision_evidence_from_snapshot(None).available)

    def test_unavailable_evidence_never_reports_a_fabricated_low_congestion_or_zero_density(self):

        evidence = crowd_decision_evidence_from_snapshot(None)

        self.assertEqual(evidence.congested_exit_ids, ())
        self.assertIsNone(evidence.highest_density_zone_id)
        self.assertIsNone(evidence.position_coverage_fraction)


class RealSnapshotAdapterTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.manager = LiveOccupantManager()

    def _engine(self):
        return CrowdIntelligenceEngine(self.building, self.manager)

    def test_empty_building_produces_available_but_no_flagged_evidence(self):

        snapshot = self._engine().compute(0.0)
        evidence = crowd_decision_evidence_from_snapshot(snapshot)

        self.assertTrue(evidence.available)
        self.assertEqual(evidence.congested_exit_ids, ())
        # zone z1 is trivially "highest density" (the only zone, at a
        # real, honestly-computed 0.0 density -- never fabricated), but
        # is correctly NOT above the configured density threshold, and
        # no asset is flagged at all.
        self.assertEqual(evidence.zones_above_density_threshold, ())
        self.assertEqual(evidence.asset_details, {})

    def test_congested_exit_id_and_level_preserved(self):

        # capacity=1, so 1 stationary occupant near the exit already
        # exceeds demand/capacity -- deterministic, no randomness.
        self.manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (8.9, 5.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        snapshot = self._engine().compute(0.0)
        evidence = crowd_decision_evidence_from_snapshot(snapshot)

        self.assertIn("e1", evidence.congested_exit_ids)
        self.assertIn("e1", evidence.asset_details)
        self.assertEqual(evidence.asset_details["e1"].asset_type, "Exit")
        self.assertIsNotNone(evidence.asset_details["e1"].congestion_level)
        self.assertEqual(evidence.most_congested_asset_id, "e1")
        self.assertEqual(evidence.most_congested_asset_type, "Exit")
        self.assertIsNotNone(evidence.most_congested_level)

    def test_zone_density_and_max_occupancy_threshold_preserved(self):

        self.manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        self.manager.update("OCC-2", "CAM-1", "T2", "z1", "f1", (2.0, 2.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        snapshot = self._engine().compute(0.0)
        evidence = crowd_decision_evidence_from_snapshot(snapshot)

        # zone.max_occupancy == 1, occupant_count == 2 -> flagged.
        self.assertIn("z1", evidence.zones_above_density_threshold)
        self.assertEqual(evidence.highest_density_zone_id, "z1")
        self.assertIn("z1", evidence.zone_details)
        self.assertIsNotNone(evidence.zone_details["z1"].density_people_per_m2)

    def test_position_coverage_fraction_preserved_honestly(self):

        self.manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        self.manager.update("OCC-2", "CAM-1", "T2", "z1", "f1", None, None, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        snapshot = self._engine().compute(0.0)
        evidence = crowd_decision_evidence_from_snapshot(snapshot)

        self.assertAlmostEqual(evidence.position_coverage_fraction, 0.5)

    def test_position_unavailable_asset_ids_reflects_zero_coverage(self):

        # No occupants at all on floor f1 -> exit e1's own metrics are
        # vacuously position_available=True (nobody to have missing
        # coverage for) -- confirm the honest OPPOSITE case: an occupant
        # present on that floor with no position at all.
        self.manager.update("OCC-1", "CAM-UNCALIBRATED", "T1", "z1", "f1", None, None, RecognizedBehavior.STATIONARY, 0.9, 0.0)

        snapshot = self._engine().compute(0.0)
        evidence = crowd_decision_evidence_from_snapshot(snapshot)

        self.assertIn("e1", evidence.position_unavailable_asset_ids)
        self.assertNotIn("e1", evidence.congested_exit_ids)  # never fabricated as congested OR clear

    def test_asset_details_never_include_unflagged_assets(self):

        # A single, non-congested, non-queueing, non-trending exit ->
        # no entry at all (Phase 2's own "do not duplicate the entire
        # snapshot unnecessarily").
        snapshot = self._engine().compute(0.0)
        evidence = crowd_decision_evidence_from_snapshot(snapshot)

        self.assertNotIn("e1", evidence.asset_details)

    def test_to_dict_round_trips(self):

        self.manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (8.9, 5.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        snapshot = self._engine().compute(0.0)
        evidence = crowd_decision_evidence_from_snapshot(snapshot)

        data = evidence.to_dict()
        self.assertEqual(data["available"], True)
        self.assertIsInstance(data["congested_exit_ids"], list)
        self.assertIsInstance(data["asset_details"], dict)


if __name__ == "__main__":
    unittest.main()
