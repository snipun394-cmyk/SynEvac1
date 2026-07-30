import unittest

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.live_extractor_v2_1 import (
    build_stair_flow_snapshot_for_prediction, extract_live_experimental_candidate_features,
)
from predictive_dataset.simulation_extractor_v2_1 import build_alternative_route_counts


# =====================================================
# Stair Predictive-Feature Live Parity milestone, Phase 12 -- proves
# every offline/no-camera/legacy-project code path is completely
# unaffected by this milestone. No new required parameter, no new
# crash surface, no fabricated flow when calibration/observation is
# absent.
# =====================================================


def make_building_with_stair_and_door():

    floor_1 = Floor(name="Floor 1", id="floor-1", display_order=0, height=3.0)
    floor_2 = Floor(name="Floor 2", id="floor-2", display_order=1, height=3.0)

    stair = Staircase(
        id="stair-1", name="S1", from_floor_id="floor-1", to_floor_id="floor-2",
        from_zone_id="zone-1", to_zone_id="zone-2", width=1.5,
        # No from_observable_region/to_observable_region authored at all
        # -- a legacy .syn project, or one where an operator simply has
        # not authored Stair perception geometry yet.
    )
    floor_1.stairs.append(stair)

    floor_1.zones.append(Zone(id="zone-1", name="Zone 1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="floor-1"))
    floor_2.zones.append(Zone(id="zone-2", name="Zone 2", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="floor-2"))
    floor_1.exits.append(Exit(id="exit-1", zone_id="zone-1", capacity=2))

    building = Building(name="Building", id="building-1", floors=[floor_1, floor_2])
    return building, floor_1, floor_2, stair


class DesignerOnlyAndOfflineTests(unittest.TestCase):

    def setUp(self):

        self.building, self.floor_1, self.floor_2, self.stair = make_building_with_stair_and_door()
        self.edges = edges_by_candidate_id(self.building)
        self.candidates = {c.candidate_id: c for c in enumerate_candidates(self.building)}
        self.alt_counts = build_alternative_route_counts(tuple(self.candidates.values()))

    def _features(self, candidate_id, **kwargs):

        candidate = self.candidates[candidate_id]
        edge = self.edges[candidate_id]

        return extract_live_experimental_candidate_features(
            candidate, edge, kwargs.pop("time", 100.0),
            building=self.building, crowd_snapshot=kwargs.pop("crowd_snapshot", None),
            occupancy_facts=kwargs.pop("occupancy_facts", None),
            alternative_route_counts=self.alt_counts,
            evacuation_snapshot=kwargs.pop("evacuation_snapshot", None),
            occupants=kwargs.pop("occupants", None),
            **kwargs,
        )

    def test_designer_only_stair_candidate_extraction_omitting_new_parameter_entirely(self):

        # A caller written before this milestone -- never even aware
        # `stair_flow_snapshot` exists -- must keep working unchanged.
        features = self._features("stair-1")

        self.assertIsNone(features["candidate_recent_flow_rate"])  # no occupants -> honest None
        self.assertEqual(features["candidate_type"], "Stair")

    def test_exit_candidate_entirely_unaffected(self):

        exit_features = self._features("exit-1")
        self.assertIsNone(exit_features["candidate_recent_flow_rate"])

    def test_no_camera_live_runtime_stair_flow_snapshot_is_none(self):

        features = self._features("stair-1", stair_flow_snapshot=None)
        self.assertIsNone(features["candidate_recent_flow_rate"])

    def test_legacy_stair_with_no_observable_region_never_fabricates_flow(self):

        stair_flow_snapshot = build_stair_flow_snapshot_for_prediction(
            [self.stair], [], self.building, 100.0,
        )
        metrics = stair_flow_snapshot.for_stair("stair-1")

        self.assertIsNone(metrics.entries)
        self.assertIsNone(metrics.exits)

        features = self._features("stair-1", stair_flow_snapshot=stair_flow_snapshot)
        self.assertIsNone(features["candidate_recent_flow_rate"])

    def test_missing_calibration_never_fabricates_flow(self):

        # No ObservableAssetSnapshot at all -- equivalent to "no
        # calibrated camera has ever reported on this stair."
        stair_flow_snapshot = build_stair_flow_snapshot_for_prediction(
            [self.stair], [], self.building, 100.0, observable_assets=None,
        )

        self.assertEqual(stair_flow_snapshot.for_stair("stair-1").status.name, "UNKNOWN")
        self.assertIsNone(stair_flow_snapshot.for_stair("stair-1").exits)

    def test_stair_without_observable_region_remains_a_valid_candidate(self):

        # Candidate enumeration/feature extraction never raises just
        # because Stair perception geometry was never authored.
        self.assertIn("stair-1", self.candidates)
        features = self._features("stair-1")
        self.assertEqual(features["candidate_type"], "Stair")
        self.assertIn("candidate_capacity", features)


if __name__ == "__main__":
    unittest.main()
