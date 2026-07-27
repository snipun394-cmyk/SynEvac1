import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from navigation.edge import Edge

from crowd_intelligence.models import AssetApproachMetrics, CrowdIntelligenceSnapshot, TrendDirection
from evacuation_progress.models import EvacuationProgressSnapshot, ExitFlow
from live_occupants.history import OccupantHistory, ZoneTransitionRecord
from live_occupants.occupant import LiveOccupant
from live_occupants.state import OccupantStatus

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.live_extractor_v2_1 import extract_live_experimental_candidate_features
from predictive_dataset.simulation_extractor_v2_1 import build_alternative_route_counts


# =====================================================
# Localized Predictive Model V2.2 milestone, Phase 5 -- live extractor
# tests for the 3 experimental fields. Synthetic fixtures only, matching
# tests/test_predictive_dataset_extractors.py's own style.
# =====================================================


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Quiet Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[
            Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2"),
        ],
        exits=[
            Exit(id="exit-1", zone_id="zone-1", capacity=2),
        ],
    )

    return Building(name="Building", id="building-1", floors=[floor])


def _make_occupant(occupant_id, transitions):

    history = OccupantHistory(zone_transitions=tuple(transitions))

    return LiveOccupant(
        occupant_id=occupant_id,
        current_camera_id=None,
        current_track_id=None,
        current_zone_id=None,
        current_floor_id=None,
        world_position=None,
        world_velocity=None,
        behavior=None,
        confidence=1.0,
        first_seen=0.0,
        last_seen=0.0,
        status=OccupantStatus.ACTIVE,
        history=history,
    )


class LiveExperimentalFeaturesTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.edges = edges_by_candidate_id(self.building)
        self.candidates = {c.candidate_id: c for c in enumerate_candidates(self.building)}
        self.alt_counts = build_alternative_route_counts(tuple(self.candidates.values()))

    def _features(self, candidate_id, time, **kwargs):

        candidate = self.candidates[candidate_id]
        edge = self.edges[candidate_id]

        return extract_live_experimental_candidate_features(
            candidate, edge, time,
            building=self.building, crowd_snapshot=kwargs.get("crowd_snapshot"),
            occupancy_facts=kwargs.get("occupancy_facts"),
            alternative_route_counts=self.alt_counts,
            evacuation_snapshot=kwargs.get("evacuation_snapshot"),
            occupants=kwargs.get("occupants"),
        )

    # --- candidate_congestion_trend ---

    def test_trend_reads_directly_from_crowd_snapshot(self):

        snapshot = CrowdIntelligenceSnapshot(
            door_metrics={"door-1": AssetApproachMetrics(asset_id="door-1", asset_type="Door", trend=TrendDirection.RISING)},
        )

        features = self._features("door-1", time=100.0, crowd_snapshot=snapshot)

        self.assertEqual(features["candidate_congestion_trend"], "RISING")

    def test_trend_is_none_when_no_crowd_snapshot_available(self):

        features = self._features("door-1", time=100.0, crowd_snapshot=None)

        self.assertIsNone(features["candidate_congestion_trend"])

    def test_trend_unknown_is_a_real_value_not_none(self):

        snapshot = CrowdIntelligenceSnapshot(
            door_metrics={"door-1": AssetApproachMetrics(asset_id="door-1", asset_type="Door", trend=TrendDirection.UNKNOWN)},
        )

        features = self._features("door-1", time=100.0, crowd_snapshot=snapshot)

        self.assertEqual(features["candidate_congestion_trend"], "UNKNOWN")

    # --- candidate_alternative_route_count ---

    def test_alternative_route_count_matches_structural_computation(self):

        features = self._features("exit-1", time=100.0)

        self.assertEqual(features["candidate_alternative_route_count"], self.alt_counts["exit-1"])

    # --- candidate_recent_flow_rate: Exit ---

    def test_exit_flow_rate_reads_from_evacuation_snapshot(self):

        snapshot = EvacuationProgressSnapshot(exits={"exit-1": ExitFlow(exit_id="exit-1", recent_flow_per_minute=7.0)})

        features = self._features("exit-1", time=100.0, evacuation_snapshot=snapshot)

        self.assertEqual(features["candidate_recent_flow_rate"], 7.0)

    def test_exit_flow_rate_is_none_without_evacuation_snapshot(self):

        features = self._features("exit-1", time=100.0, evacuation_snapshot=None)

        self.assertIsNone(features["candidate_recent_flow_rate"])

    # --- candidate_recent_flow_rate: Door/Stair ---

    def test_door_flow_rate_counts_matching_zone_transitions_in_window(self):

        occupants = (
            _make_occupant("occ-1", [ZoneTransitionRecord(timestamp=90.0, from_zone_id="zone-1", to_zone_id="zone-2")]),
            _make_occupant("occ-2", [ZoneTransitionRecord(timestamp=95.0, from_zone_id="zone-2", to_zone_id="zone-1")]),
        )

        features = self._features("door-1", time=100.0, occupants=occupants)

        self.assertEqual(features["candidate_recent_flow_rate"], 2)

    def test_door_flow_rate_excludes_transitions_outside_window(self):

        occupants = (
            _make_occupant("occ-1", [ZoneTransitionRecord(timestamp=10.0, from_zone_id="zone-1", to_zone_id="zone-2")]),
        )

        features = self._features("door-1", time=100.0, occupants=occupants)  # window is (40, 100]

        self.assertEqual(features["candidate_recent_flow_rate"], 0)

    def test_door_flow_rate_excludes_unrelated_zone_transitions(self):

        occupants = (
            _make_occupant("occ-1", [ZoneTransitionRecord(timestamp=95.0, from_zone_id="zone-1", to_zone_id="zone-3")]),
        )

        features = self._features("door-1", time=100.0, occupants=occupants)

        self.assertEqual(features["candidate_recent_flow_rate"], 0)

    def test_door_flow_rate_is_none_without_occupants(self):

        features = self._features("door-1", time=100.0, occupants=None)

        self.assertIsNone(features["candidate_recent_flow_rate"])

    def test_exit_type_never_uses_door_stair_mechanism(self):
        """Exit candidates must go through the evacuation-progress path
        even if `occupants` happens to be supplied -- the two mechanisms
        must never silently mix."""

        occupants = (
            _make_occupant("occ-1", [ZoneTransitionRecord(timestamp=95.0, from_zone_id="zone-1", to_zone_id=None)]),
        )
        snapshot = EvacuationProgressSnapshot(exits={"exit-1": ExitFlow(exit_id="exit-1", recent_flow_per_minute=3.0)})

        features = self._features("exit-1", time=100.0, occupants=occupants, evacuation_snapshot=snapshot)

        self.assertEqual(features["candidate_recent_flow_rate"], 3.0)


if __name__ == "__main__":
    unittest.main()
