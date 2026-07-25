import unittest

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from navigation.edge import Edge
from navigation.node import Node

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from pathfinding.route import Route

from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline, OccupantTimelineStep
from simulator.occupant import OccupantState

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.simulation_extractor import extract_simulation_candidate_features
from predictive_dataset.target_generator import generate_candidate_label


# =====================================================
# Phase 11 fixture -- ONE simulation timestep, ONE global building
# state, TWO exits (E1 busy/queued, E2 empty) with genuinely different
# candidate-LOCAL conditions -- the fundamental capability
# CANONICAL_LIVE_SCHEMA's whole-building model lacks (docs/architecture/
# ai_operational_role.md §4/§9).
# =====================================================


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Quiet Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        exits=[
            Exit(id="exit-1", zone_id="zone-1", capacity=2),
            Exit(id="exit-2", zone_id="zone-2", capacity=2),
        ],
    )

    return Building(name="Two Exit Building", id="building-1", floors=[floor])


def _zone_node(zone):
    return Node(id=zone.id, name=zone.name, floor_id="floor-1", node_type=Node.ZONE)


def _outside_node():
    return Node(id=Node.OUTSIDE_NODE_ID, name="Outside", floor_id="", node_type=Node.OUTSIDE)


def _make_exit_edge(exit_obj, zone):
    return Edge(
        id=exit_obj.id, edge_type=Edge.EXIT, from_node=zone.id, to_node=Node.OUTSIDE_NODE_ID,
        walking_distance=8.0, reference=exit_obj,
    )


def _make_timeline(occupant_id, route_edge, steps, depart_time, arrival_time):

    route = Route(nodes=[], edges=[route_edge], total_cost=0.0, total_distance=0.0)

    return OccupantTimeline(
        occupant_id=occupant_id, route=route, steps=steps,
        state=OccupantState.ARRIVED if arrival_time is not None else OccupantState.TRAVERSING,
        depart_time=depart_time, arrival_time=arrival_time,
    )


class BusyVsQuietExitFixture(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.zone_a, self.zone_b = self.building.floors[0].zones
        self.exit_1, self.exit_2 = self.building.floors[0].exits

        self.edges = edges_by_candidate_id(self.building)
        self.candidates = {c.candidate_id: c for c in enumerate_candidates(self.building)}

        exit1_edge = self.edges["exit-1"]

        # occ-1: on exit-1's edge for [0, 12] -- present at t=5.
        occ1_step = OccupantTimelineStep(
            index=0, from_node=_zone_node(self.zone_a), to_node=_outside_node(), edge=exit1_edge,
            queue_wait_time=0.0, start_time=0.0, end_time=12.0,
        )
        occ1 = _make_timeline("occ-1", exit1_edge, [occ1_step], depart_time=0.0, arrival_time=12.0)

        # occ-2: joins exit-1's queue at t=1 (start_time 6 - queue_wait_time 5) --
        # still queued at t=5 (join_time=1 <= 5 < start_time=6), then admitted
        # onto the edge at t=6 and overlaps occ-1 there until occ-1 leaves at 12
        # (concurrent count 2 on exit-1's edge during [6, 12]).
        occ2_step = OccupantTimelineStep(
            index=0, from_node=_zone_node(self.zone_a), to_node=_outside_node(), edge=exit1_edge,
            queue_wait_time=5.0, start_time=6.0, end_time=20.0,
        )
        occ2 = _make_timeline("occ-2", exit1_edge, [occ2_step], depart_time=1.0, arrival_time=20.0)

        # occ-3: route ends at exit-1, already departed (t=0), not yet arrived --
        # "approaching" exit-1 at t=5 even though it has taken no step yet.
        occ3 = _make_timeline("occ-3", exit1_edge, [], depart_time=0.0, arrival_time=None)

        self.movement_result = MultiAgentSimulationResult(
            occupants={"occ-1": occ1, "occ-2": occ2, "occ-3": occ3},
            total_evacuation_time=20.0,
        )

        self.occupancy_snapshot = OccupancySnapshot(observations={
            self.zone_a.id: OccupancyObservation(occupant_count=4),
            self.zone_b.id: OccupancyObservation(occupant_count=0),
        })

    def _features(self, candidate_id, time=5.0):

        candidate = self.candidates[candidate_id]
        edge = self.edges[candidate_id]

        return extract_simulation_candidate_features(
            candidate, edge, time,
            building=self.building, movement_result=self.movement_result,
            occupancy_snapshot=self.occupancy_snapshot,
        )


class CurrentFeatureDifferentiationTests(BusyVsQuietExitFixture):

    def test_two_exits_same_global_state_produce_different_feature_rows(self):

        features_e1 = self._features("exit-1")
        features_e2 = self._features("exit-2")

        self.assertNotEqual(features_e1, features_e2)

    def test_busy_exit_shows_queueing_and_approach_demand(self):

        features = self._features("exit-1")

        self.assertEqual(features["candidate_queue_length"], 1)  # occ-2
        self.assertEqual(features["candidate_approaching_count"], 1)  # occ-3
        self.assertEqual(features["candidate_adjacent_zone_occupancy"], 4)

    def test_quiet_exit_shows_no_demand(self):

        features = self._features("exit-2")

        self.assertEqual(features["candidate_queue_length"], 0)
        self.assertEqual(features["candidate_approaching_count"], 0)
        self.assertEqual(features["candidate_adjacent_zone_occupancy"], 0)

    def test_congestion_level_differs(self):

        features_e1 = self._features("exit-1")
        features_e2 = self._features("exit-2")

        self.assertNotEqual(features_e1["candidate_congestion_level"], features_e2["candidate_congestion_level"])
        self.assertEqual(features_e2["candidate_congestion_level"], "LOW")

    def test_structural_fields_are_shared_and_still_correct_per_candidate(self):

        features_e1 = self._features("exit-1")
        features_e2 = self._features("exit-2")

        self.assertEqual(features_e1["candidate_type"], Edge.EXIT)
        self.assertEqual(features_e2["candidate_type"], Edge.EXIT)
        self.assertEqual(features_e1["candidate_capacity"], 2)
        self.assertTrue(features_e1["candidate_traversable"])


class FutureTargetDifferentiationTests(BusyVsQuietExitFixture):

    def test_busy_exit_becomes_congested_within_horizon_positive(self):

        label = generate_candidate_label("exit-1", self.movement_result, time=5.0, horizon=30.0)

        self.assertFalse(label.currently_congested)
        self.assertTrue(label.target)

    def test_quiet_exit_never_congests_within_horizon_negative(self):

        label = generate_candidate_label("exit-2", self.movement_result, time=5.0, horizon=30.0)

        self.assertFalse(label.currently_congested)
        self.assertFalse(label.target)
        self.assertFalse(label.had_any_activity_in_window)

    def test_e1_positive_e2_negative_same_scenario_same_observation_time(self):

        label_e1 = generate_candidate_label("exit-1", self.movement_result, time=5.0, horizon=30.0)
        label_e2 = generate_candidate_label("exit-2", self.movement_result, time=5.0, horizon=30.0)

        self.assertEqual(label_e1.target, True)
        self.assertEqual(label_e2.target, False)


class DeterministicExtractionTests(BusyVsQuietExitFixture):

    def test_repeated_extraction_is_identical(self):

        first = self._features("exit-1")
        second = self._features("exit-1")

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
