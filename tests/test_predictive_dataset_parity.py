import unittest
from types import SimpleNamespace

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from navigation.node import Node

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from pathfinding.route import Route

from crowd_intelligence.models import AssetApproachMetrics, CrowdIntelligenceSnapshot, IntensityLevel

from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline, OccupantTimelineStep
from simulator.occupant import OccupantState

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.live_extractor import extract_live_candidate_features
from predictive_dataset.simulation_extractor import extract_simulation_candidate_features


# =====================================================
# Phase 13 -- controlled matched-state test. A DELIBERATELY constructed
# pair of (simulation state, live state) representing "the same"
# observed situation -- 2 occupants queued, 1 approaching, on the same
# candidate, same Building -- fed through both extractors. Structural
# fields (type/capacity/walking_distance/traversable) and the shared
# congestion-classification code path must match EXACTLY. Queue/
# approaching counts are sourced by genuinely different mechanisms on
# each side (Phase 6/7's own disclosed divergence, see schema.py) --
# this test constructs matching COUNTS by hand rather than asserting
# the two mechanisms always agree (they do not, and are not expected
# to). The missing-evidence case is tested separately: live honestly
# reports None where simulation still has ground truth, never forced
# into fake equality.
# =====================================================


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0)],
        exits=[Exit(id="exit-1", zone_id="zone-1", capacity=2)],
    )

    return Building(name="Parity Building", id="building-1", floors=[floor])


class MatchedStateParityTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.candidate = enumerate_candidates(self.building)[0]
        self.edge = edges_by_candidate_id(self.building)[self.candidate.candidate_id]

    def test_structural_fields_match_exactly(self):

        sim_features = extract_simulation_candidate_features(
            self.candidate, self.edge, time=0.0,
            building=self.building, movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            occupancy_snapshot=OccupancySnapshot(),
        )
        live_features = extract_live_candidate_features(
            self.candidate, self.edge, building=self.building, crowd_snapshot=None, occupancy_facts=None,
        )

        for field in ("candidate_type", "candidate_capacity", "candidate_walking_distance", "candidate_traversable"):
            self.assertEqual(sim_features[field], live_features[field], field)

    def test_congestion_classification_matches_exactly_given_the_same_queue_and_approach_counts(self):

        # -- Simulation side: hand-build 2 occupants queued for the edge --
        node = Node(id="zone-1", name="Lobby", floor_id="floor-1", node_type=Node.ZONE)
        outside = Node(id=Node.OUTSIDE_NODE_ID, name="Outside", floor_id="", node_type=Node.OUTSIDE)

        def queued_timeline(occupant_id, join_time, start_time, end_time):
            step = OccupantTimelineStep(
                index=0, from_node=node, to_node=outside, edge=self.edge,
                queue_wait_time=start_time - join_time, start_time=start_time, end_time=end_time,
            )
            route = Route(nodes=[], edges=[self.edge], total_cost=0.0, total_distance=0.0)
            return OccupantTimeline(
                occupant_id=occupant_id, route=route, steps=[step],
                state=OccupantState.ARRIVED, depart_time=join_time, arrival_time=end_time,
            )

        movement_result = MultiAgentSimulationResult(
            occupants={
                "occ-1": queued_timeline("occ-1", join_time=0.0, start_time=10.0, end_time=20.0),
                "occ-2": queued_timeline("occ-2", join_time=1.0, start_time=12.0, end_time=22.0),
            },
            total_evacuation_time=22.0,
        )

        sim_features = extract_simulation_candidate_features(
            self.candidate, self.edge, time=5.0,
            building=self.building, movement_result=movement_result, occupancy_snapshot=OccupancySnapshot(),
        )

        self.assertEqual(sim_features["candidate_queue_length"], 2)
        self.assertEqual(sim_features["candidate_approaching_count"], 0)

        # -- Live side: hand-build the SAME counts (2 queued, 0 approaching) --
        metrics = AssetApproachMetrics(
            asset_id="exit-1", asset_type="Exit", position_available=True,
            queue_candidate_count=2, approaching_count=0,
            simulation_style_capacity=sim_features["candidate_capacity"],
            congestion_level=IntensityLevel.HIGH,  # compute_congestion_level's own classification for demand=2/cap=2 (ratio 1.0 -> high_at)
        )
        crowd_snapshot = CrowdIntelligenceSnapshot(exit_metrics={"exit-1": metrics})

        live_features = extract_live_candidate_features(
            self.candidate, self.edge, building=self.building, crowd_snapshot=crowd_snapshot, occupancy_facts=None,
        )

        self.assertEqual(live_features["candidate_queue_length"], 2)
        self.assertEqual(live_features["candidate_approaching_count"], 0)

        # The classification ITSELF -- same shared code path both sides
        # -- agrees exactly given the same counts and the same capacity.
        self.assertEqual(sim_features["candidate_congestion_level"], live_features["candidate_congestion_level"])

    def test_missing_live_position_evidence_is_never_forced_into_fake_equality(self):

        movement_result = MultiAgentSimulationResult(occupants={}, total_evacuation_time=None)

        sim_features = extract_simulation_candidate_features(
            self.candidate, self.edge, time=0.0,
            building=self.building, movement_result=movement_result, occupancy_snapshot=OccupancySnapshot(),
        )

        metrics = AssetApproachMetrics(asset_id="exit-1", asset_type="Exit", position_available=False, simulation_style_capacity=2)
        crowd_snapshot = CrowdIntelligenceSnapshot(exit_metrics={"exit-1": metrics})

        live_features = extract_live_candidate_features(
            self.candidate, self.edge, building=self.building, crowd_snapshot=crowd_snapshot, occupancy_facts=None,
        )

        # Simulation has exact ground truth (0 queued) -- live honestly
        # has NO evidence (None) -- these must NOT be coerced to match.
        self.assertEqual(sim_features["candidate_queue_length"], 0)
        self.assertIsNone(live_features["candidate_queue_length"])
        self.assertNotEqual(sim_features["candidate_queue_length"], live_features["candidate_queue_length"])


if __name__ == "__main__":
    unittest.main()
