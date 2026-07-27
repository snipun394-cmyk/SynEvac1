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
from predictive_dataset.simulation_extractor_v2_1 import (
    EXPERIMENTAL_FEATURE_NAMES,
    build_alternative_route_counts,
    extract_experimental_candidate_features,
)


# =====================================================
# Localized Predictive Model V2.1 milestone, Phase 15 -- tests for the
# 3 experimental fields. Mirrors tests/test_predictive_dataset_
# extractors.py's own fixture style (synthetic MultiAgentSimulationResult,
# no full simulation run needed).
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
            Exit(id="exit-2", zone_id="zone-2", capacity=2),
        ],
    )

    return Building(name="Two Exit Building", id="building-1", floors=[floor])


def _zone_node(zone):
    return Node(id=zone.id, name=zone.name, floor_id="floor-1", node_type=Node.ZONE)


def _outside_node():
    return Node(id=Node.OUTSIDE_NODE_ID, name="Outside", floor_id="", node_type=Node.OUTSIDE)


def _make_timeline(occupant_id, route_edge, steps, depart_time, arrival_time):

    route = Route(nodes=[], edges=[route_edge], total_cost=0.0, total_distance=0.0)

    return OccupantTimeline(
        occupant_id=occupant_id, route=route, steps=steps,
        state=OccupantState.ARRIVED if arrival_time is not None else OccupantState.TRAVERSING,
        depart_time=depart_time, arrival_time=arrival_time,
    )


class AlternativeRouteCountTests(unittest.TestCase):

    def setUp(self):
        self.building = make_building()
        self.candidates = enumerate_candidates(self.building)

    def test_exit_1_sees_door_1_as_its_only_alternative(self):
        """exit-1 touches zone-1 only; door-1 also touches zone-1 (and
        zone-2); exit-2 touches zone-2 only -- so exit-1's alternatives
        are just door-1, not exit-2 (no shared zone)."""

        counts = build_alternative_route_counts(self.candidates)

        self.assertEqual(counts["exit-1"], 1)  # door-1
        self.assertEqual(counts["exit-2"], 1)  # door-1
        self.assertEqual(counts["door-1"], 2)  # exit-1 and exit-2

    def test_isolated_candidate_has_zero_alternatives(self):

        floor = Floor(
            name="Isolated", id="floor-iso",
            zones=[Zone(id="zone-iso", name="Room", x=0.0, y=0.0, width=5.0, height=5.0)],
            exits=[Exit(id="exit-iso", zone_id="zone-iso", capacity=2)],
        )
        building = Building(name="Isolated Building", id="building-iso", floors=[floor])
        candidates = enumerate_candidates(building)

        counts = build_alternative_route_counts(candidates)

        self.assertEqual(counts["exit-iso"], 0)

    def test_every_candidate_gets_a_count(self):

        counts = build_alternative_route_counts(self.candidates)

        for candidate in self.candidates:
            self.assertIn(candidate.candidate_id, counts)
            self.assertIsInstance(counts[candidate.candidate_id], int)


class RecentFlowRateAndTrendTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.zone_a = self.building.floors[0].zones[0]
        self.edges = edges_by_candidate_id(self.building)
        self.candidates = {c.candidate_id: c for c in enumerate_candidates(self.building)}

        exit1_edge = self.edges["exit-1"]

        # occ-1 completes crossing exit-1 at t=50 (within a 60s window of t=100)
        occ1_step = OccupantTimelineStep(
            index=0, from_node=_zone_node(self.zone_a), to_node=_outside_node(), edge=exit1_edge,
            queue_wait_time=0.0, start_time=45.0, end_time=50.0,
        )
        occ1 = _make_timeline("occ-1", exit1_edge, [occ1_step], depart_time=45.0, arrival_time=50.0)

        # occ-2 completes crossing exit-1 at t=20 (OUTSIDE a 60s window of t=100 -- too old)
        occ2_step = OccupantTimelineStep(
            index=0, from_node=_zone_node(self.zone_a), to_node=_outside_node(), edge=exit1_edge,
            queue_wait_time=0.0, start_time=15.0, end_time=20.0,
        )
        occ2 = _make_timeline("occ-2", exit1_edge, [occ2_step], depart_time=15.0, arrival_time=20.0)

        self.movement_result = MultiAgentSimulationResult(
            occupants={"occ-1": occ1, "occ-2": occ2},
            total_evacuation_time=100.0,
        )

        self.occupancy_snapshot = OccupancySnapshot(observations={
            self.zone_a.id: OccupancyObservation(occupant_count=2),
        })

    def _features(self, candidate_id, time, alt_counts=None):

        candidate = self.candidates[candidate_id]
        edge = self.edges[candidate_id]

        return extract_experimental_candidate_features(
            candidate, edge, time,
            building=self.building, movement_result=self.movement_result,
            occupancy_snapshot=self.occupancy_snapshot,
            alternative_route_counts=alt_counts or build_alternative_route_counts(tuple(self.candidates.values())),
        )

    def test_recent_flow_rate_counts_only_completions_within_window(self):

        features = self._features("exit-1", time=100.0)

        # occ-1 (end_time=50) is within (100-60, 100] = (40, 100]; occ-2
        # (end_time=20) is not.
        self.assertEqual(features["candidate_recent_flow_rate"], 1)

    def test_recent_flow_rate_excludes_future_completions(self):

        features = self._features("exit-1", time=10.0)

        # neither occupant has completed crossing by t=10
        self.assertEqual(features["candidate_recent_flow_rate"], 0)

    def test_congestion_trend_is_unknown_before_the_trend_window_elapses(self):

        features = self._features("exit-1", time=5.0)  # < 30s

        self.assertEqual(features["candidate_congestion_trend"], "UNKNOWN")

    def test_congestion_trend_is_stable_when_nothing_changes(self):

        # no queueing/approaching activity at all -- demand proxy is 0
        # both now and 30s ago -> STABLE
        features = self._features("exit-1", time=60.0)

        self.assertEqual(features["candidate_congestion_trend"], "STABLE")

    def test_extract_experimental_candidate_features_includes_base_and_new_fields(self):

        features = self._features("exit-1", time=100.0)

        for name in EXPERIMENTAL_FEATURE_NAMES:
            self.assertIn(name, features)

        # base V2 fields are still present, unchanged
        self.assertIn("candidate_queue_length", features)
        self.assertIn("candidate_capacity", features)


if __name__ == "__main__":
    unittest.main()
