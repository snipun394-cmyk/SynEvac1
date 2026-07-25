import unittest
from types import SimpleNamespace

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from crowd_intelligence.models import AssetApproachMetrics, CrowdIntelligenceSnapshot, IntensityLevel

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.live_extractor import extract_live_candidate_features


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0)],
        exits=[Exit(id="exit-1", zone_id="zone-1", capacity=2)],
    )

    return Building(name="Live Extractor Building", id="building-1", floors=[floor])


def make_occupancy_facts(total_observed_count=0, occupant_ids_by_zone=None):

    return SimpleNamespace(
        total_observed_count=total_observed_count,
        occupant_ids_by_zone=occupant_ids_by_zone or {},
    )


class LiveExtractorTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.candidate = enumerate_candidates(self.building)[0]
        self.edge = edges_by_candidate_id(self.building)[self.candidate.candidate_id]

    def test_reads_queue_and_approach_directly_from_crowd_intelligence_snapshot(self):

        metrics = AssetApproachMetrics(
            asset_id="exit-1", asset_type="Exit", position_available=True,
            approaching_count=3, queue_candidate_count=2, estimated_queue_length=2,
            simulation_style_capacity=2, congestion_level=IntensityLevel.HIGH,
        )
        crowd_snapshot = CrowdIntelligenceSnapshot(exit_metrics={"exit-1": metrics})

        occupancy_facts = make_occupancy_facts(total_observed_count=5, occupant_ids_by_zone={"zone-1": ("occ-1", "occ-2")})

        features = extract_live_candidate_features(
            self.candidate, self.edge, building=self.building,
            crowd_snapshot=crowd_snapshot, occupancy_facts=occupancy_facts,
        )

        self.assertEqual(features["candidate_queue_length"], 2)
        self.assertEqual(features["candidate_approaching_count"], 3)
        self.assertEqual(features["candidate_congestion_level"], "HIGH")
        self.assertEqual(features["candidate_capacity"], 2)
        self.assertEqual(features["candidate_adjacent_zone_occupancy"], 2)
        self.assertEqual(features["total_active_occupant_count"], 5)
        self.assertEqual(features["candidate_type"], "Exit")

    def test_missing_position_evidence_is_honestly_none_not_a_fabricated_zero(self):

        metrics = AssetApproachMetrics(
            asset_id="exit-1", asset_type="Exit", position_available=False,
            simulation_style_capacity=2,
        )
        crowd_snapshot = CrowdIntelligenceSnapshot(exit_metrics={"exit-1": metrics})

        features = extract_live_candidate_features(
            self.candidate, self.edge, building=self.building,
            crowd_snapshot=crowd_snapshot, occupancy_facts=None,
        )

        self.assertIsNone(features["candidate_queue_length"])
        self.assertIsNone(features["candidate_approaching_count"])
        self.assertIsNone(features["candidate_congestion_level"])
        self.assertIsNone(features["total_active_occupant_count"])
        # Capacity is structural -- still honestly known even with no
        # position evidence for THIS cycle's occupants.
        self.assertEqual(features["candidate_capacity"], 2)

    def test_missing_asset_entirely_falls_back_to_structural_capacity(self):

        crowd_snapshot = CrowdIntelligenceSnapshot()  # no exit_metrics entry for exit-1 at all

        features = extract_live_candidate_features(
            self.candidate, self.edge, building=self.building,
            crowd_snapshot=crowd_snapshot, occupancy_facts=None,
        )

        self.assertIsNone(features["candidate_queue_length"])
        self.assertEqual(features["candidate_capacity"], 2)

    def test_no_crowd_snapshot_at_all_still_produces_structural_fields(self):

        features = extract_live_candidate_features(
            self.candidate, self.edge, building=self.building,
            crowd_snapshot=None, occupancy_facts=None,
        )

        self.assertEqual(features["candidate_type"], "Exit")
        self.assertEqual(features["candidate_capacity"], 2)
        self.assertTrue(features["candidate_traversable"])
        self.assertIsNone(features["candidate_queue_length"])


if __name__ == "__main__":
    unittest.main()
