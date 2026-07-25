import unittest

from models.building import Building
from models.door import Door
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

        # occ-3: departed at t=0, hasn't yet started queueing for exit-1's
        # edge (join_time=10.0 > observation time 5.0) -- "approaching"
        # exit-1 at t=5, not yet counted in candidate_queue_length.
        occ3_step = OccupantTimelineStep(
            index=0, from_node=_zone_node(self.zone_a), to_node=_outside_node(), edge=exit1_edge,
            queue_wait_time=0.0, start_time=10.0, end_time=15.0,
        )
        occ3 = _make_timeline("occ-3", exit1_edge, [occ3_step], depart_time=0.0, arrival_time=15.0)

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


# =====================================================
# Large-Scale Predictive Dataset Campaign & Validation milestone --
# regression coverage for the bug that campaign's own Phase 5 feature-
# distribution check caught (candidate_traversable reporting constant
# True across 2.5M real campaign rows, because the CALLING campaign
# script passed the pristine template Building instead of the scenario-
# initialized copy -- see simulation_extractor.py's own updated
# docstring). This proves the LIBRARY function itself has always
# correctly reported a blocked Exit's traversability when given the
# right Edge -- the bug was in caller wiring, not here, but this is the
# regression guard for the underlying mechanism regardless.
# =====================================================


class BlockedCandidateTraversabilityTests(unittest.TestCase):

    def test_blocked_exit_reports_not_traversable(self):

        building = Building(
            name="Blocked Exit Building", id="building-2",
            floors=[Floor(
                name="Ground", id="floor-1",
                zones=[Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0)],
                exits=[Exit(id="exit-blocked", zone_id="zone-1", capacity=2, is_blocked=True)],
            )],
        )

        candidate = enumerate_candidates(building)[0]
        edge = edges_by_candidate_id(building)[candidate.candidate_id]

        features = extract_simulation_candidate_features(
            candidate, edge, time=0.0,
            building=building, movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            occupancy_snapshot=OccupancySnapshot(),
        )

        self.assertFalse(features["candidate_traversable"])

    def test_open_exit_reports_traversable(self):

        building = Building(
            name="Open Exit Building", id="building-3",
            floors=[Floor(
                name="Ground", id="floor-1",
                zones=[Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0)],
                exits=[Exit(id="exit-open", zone_id="zone-1", capacity=2, is_blocked=False)],
            )],
        )

        candidate = enumerate_candidates(building)[0]
        edge = edges_by_candidate_id(building)[candidate.candidate_id]

        features = extract_simulation_candidate_features(
            candidate, edge, time=0.0,
            building=building, movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=None),
            occupancy_snapshot=OccupancySnapshot(),
        )

        self.assertTrue(features["candidate_traversable"])


# =====================================================
# Large-Scale Predictive Dataset Campaign & Validation milestone --
# regression coverage for TWO bugs that campaign's own Phase 4/5 checks
# caught over a real 2.5M-row run (see simulation_extractor.py's own
# "BUG HISTORY" comment for the full story): v1's route.edges[-1] made
# candidate_approaching_count structurally zero for Door/Stair (a
# complete Route always ends at an Exit); v2's "immediate next hop"
# fix was ALSO structurally zero for every candidate, because this
# simulator's occupant movement model has no observable gap between
# "departed/finished an edge" and "joined the next edge's queue".
# Proves the current (v3) "any not-yet-reached edge in my remaining
# route" definition avoids both failure modes: a Door -- a genuinely
# INTERMEDIATE hop -- registers approach demand, and multiple future
# hops can register simultaneously.
# =====================================================


class MultiHopApproachingCountTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(
            name="Two Hop Building", id="building-4",
            floors=[Floor(
                name="Ground", id="floor-1",
                zones=[
                    Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
                    Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
                ],
                doors=[Door(id="door-1", zone_a_id="zone-1", zone_b_id="zone-2")],
                exits=[Exit(id="exit-1", zone_id="zone-2", capacity=2)],
            )],
        )

        self.edges = edges_by_candidate_id(self.building)
        self.candidates = {c.candidate_id: c for c in enumerate_candidates(self.building)}

        zone_1 = Node(id="zone-1", name="Lobby", floor_id="floor-1", node_type=Node.ZONE)
        zone_2 = Node(id="zone-2", name="Office", floor_id="floor-1", node_type=Node.ZONE)
        outside = Node(id=Node.OUTSIDE_NODE_ID, name="Outside", floor_id="", node_type=Node.OUTSIDE)

        door_edge = self.edges["door-1"]
        exit_edge = self.edges["exit-1"]

        # Occupant's route is [door-1, exit-1] -- door-1 is an
        # INTERMEDIATE hop, never the route's final edge. At t=0 they
        # have departed but not yet started queueing for door-1
        # (join_time=5.0 > 0.0).
        door_step = OccupantTimelineStep(
            index=0, from_node=zone_1, to_node=zone_2, edge=door_edge,
            queue_wait_time=0.0, start_time=5.0, end_time=8.0,
        )
        exit_step = OccupantTimelineStep(
            index=1, from_node=zone_2, to_node=outside, edge=exit_edge,
            queue_wait_time=0.0, start_time=8.0, end_time=16.0,
        )
        route = Route(nodes=[zone_1, zone_2, outside], edges=[door_edge, exit_edge], total_cost=0.0, total_distance=0.0)

        occupant = OccupantTimeline(
            occupant_id="occ-1", route=route, steps=[door_step, exit_step],
            state=OccupantState.ARRIVED, depart_time=0.0, arrival_time=16.0,
        )

        self.movement_result = MultiAgentSimulationResult(occupants={"occ-1": occupant}, total_evacuation_time=16.0)

    def test_intermediate_door_registers_approach_demand_before_it_is_reached(self):

        features = extract_simulation_candidate_features(
            self.candidates["door-1"], self.edges["door-1"], time=0.0,
            building=self.building, movement_result=self.movement_result,
            occupancy_snapshot=OccupancySnapshot(),
        )

        self.assertEqual(features["candidate_approaching_count"], 1)

    def test_exit_also_registers_approach_demand_as_a_later_not_yet_reached_hop(self):

        # v3's whole-remaining-route definition: exit-1 is a LATER hop
        # than door-1 in occ-1's route, and it hasn't been reached
        # either (occ-1 hasn't even started queueing for door-1 yet) --
        # so it legitimately registers demand too, simultaneously with
        # door-1. This is the deliberate difference from a stricter
        # "only the immediate next hop" definition, which this campaign
        # proved was always zero in practice (see simulation_extractor.
        # py's own BUG HISTORY comment).
        features = extract_simulation_candidate_features(
            self.candidates["exit-1"], self.edges["exit-1"], time=0.0,
            building=self.building, movement_result=self.movement_result,
            occupancy_snapshot=OccupancySnapshot(),
        )

        self.assertEqual(features["candidate_approaching_count"], 1)

    def test_approach_demand_leaves_the_door_once_it_is_cleared(self):

        # At t=8, the occupant has finished door-1 (end_time=8.0) and
        # gone straight onto exit-1 with zero queue wait (start_time=8.0,
        # queue_wait_time=0.0 -- no [join_time, start_time) queueing
        # interval at all) -- no longer "approaching" door-1 (already
        # past it), and not counted as "approaching" exit-1 either
        # (they're already ON it, which candidate_queue_length /
        # candidate_approaching_count deliberately do not double-report
        # as demand -- see this module's own docstring).
        door_features = extract_simulation_candidate_features(
            self.candidates["door-1"], self.edges["door-1"], time=8.0,
            building=self.building, movement_result=self.movement_result,
            occupancy_snapshot=OccupancySnapshot(),
        )
        exit_features = extract_simulation_candidate_features(
            self.candidates["exit-1"], self.edges["exit-1"], time=8.0,
            building=self.building, movement_result=self.movement_result,
            occupancy_snapshot=OccupancySnapshot(),
        )

        self.assertEqual(door_features["candidate_approaching_count"], 0)
        self.assertEqual(door_features["candidate_queue_length"], 0)
        self.assertEqual(exit_features["candidate_approaching_count"], 0)
        self.assertEqual(exit_features["candidate_queue_length"], 0)


if __name__ == "__main__":
    unittest.main()
