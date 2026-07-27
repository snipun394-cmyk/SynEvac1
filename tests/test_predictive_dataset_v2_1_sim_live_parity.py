import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from navigation.node import Node

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from pathfinding.route import Route

from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline, OccupantTimelineStep
from simulator.occupant import OccupantState

from live_occupants.history import OccupantHistory, ZoneTransitionRecord
from live_occupants.occupant import LiveOccupant
from live_occupants.state import OccupantStatus

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.live_extractor_v2_1 import extract_live_experimental_candidate_features
from predictive_dataset.simulation_extractor_v2_1 import (
    build_alternative_route_counts,
    extract_experimental_candidate_features,
)


# =====================================================
# Localized Predictive Model V2.2 milestone, Phase 5 -- proves semantic
# parity between the simulation and live extractors for the 3 V2.1
# experimental fields, per this milestone's own instruction: "exact
# numeric equality is required where the underlying evidence is
# identical [...] where epistemic differences prevent equality, document
# the difference and prove missing evidence remains None rather than
# fabricated."
#
#   candidate_alternative_route_count -- ALWAYS exact (both call the
#     literal same function).
#   candidate_recent_flow_rate (Door/Stair) -- exact when the underlying
#     crossing evidence is identical (same timestamps), because both
#     mechanisms answer the identical question ("how many occupants
#     crossed this edge's two zones in the last 60s") even though one
#     reads OccupantTimelineStep and the other reads ZoneTransitionRecord.
#   candidate_congestion_trend / candidate_recent_flow_rate (Exit) --
#     NOT asserted numerically equal (genuinely different computation
#     bases, disclosed in both modules' own docstrings) -- this test
#     instead proves the "missing evidence stays None" half of the rule.
# =====================================================


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-1", capacity=2)],
    )

    return Building(name="Building", id="building-1", floors=[floor])


def _zone_node(zone_id):
    return Node(id=zone_id, name=zone_id, floor_id="floor-1", node_type=Node.ZONE)


def _outside_node():
    return Node(id=Node.OUTSIDE_NODE_ID, name="Outside", floor_id="", node_type=Node.OUTSIDE)


class DoorStairFlowRateParityTests(unittest.TestCase):
    """Identical underlying crossing evidence, fed to both extractors via
    their own native representations -- must produce the identical count."""

    def setUp(self):

        self.building = make_building()
        self.edges = edges_by_candidate_id(self.building)
        self.candidates = {c.candidate_id: c for c in enumerate_candidates(self.building)}
        self.alt_counts = build_alternative_route_counts(tuple(self.candidates.values()))

    def _sim_flow_rate(self, crossing_end_times, observation_time):

        door_edge = self.edges["door-1"]
        steps = [
            OccupantTimelineStep(
                index=0, from_node=_zone_node("zone-1"), to_node=_zone_node("zone-2"), edge=door_edge,
                queue_wait_time=0.0, start_time=end_time - 1.0, end_time=end_time,
            )
            for end_time in crossing_end_times
        ]
        occupants = {
            f"occ-{i}": OccupantTimeline(
                occupant_id=f"occ-{i}", route=Route(nodes=[], edges=[door_edge], total_cost=0.0, total_distance=0.0),
                steps=[step], state=OccupantState.ARRIVED, depart_time=step.start_time, arrival_time=step.end_time,
            )
            for i, step in enumerate(steps)
        }
        movement_result = MultiAgentSimulationResult(occupants=occupants, total_evacuation_time=observation_time)
        occupancy_snapshot = OccupancySnapshot(observations={"zone-1": OccupancyObservation(occupant_count=0)})

        features = extract_experimental_candidate_features(
            self.candidates["door-1"], door_edge, observation_time,
            building=self.building, movement_result=movement_result, occupancy_snapshot=occupancy_snapshot,
            alternative_route_counts=self.alt_counts,
        )
        return features["candidate_recent_flow_rate"]

    def _live_flow_rate(self, crossing_end_times, observation_time):

        occupants = tuple(
            LiveOccupant(
                occupant_id=f"occ-{i}", current_camera_id=None, current_track_id=None,
                current_zone_id=None, current_floor_id=None, world_position=None, world_velocity=None,
                behavior=None, confidence=1.0, first_seen=0.0, last_seen=0.0, status=OccupantStatus.ACTIVE,
                history=OccupantHistory(zone_transitions=(
                    ZoneTransitionRecord(timestamp=end_time, from_zone_id="zone-1", to_zone_id="zone-2"),
                )),
            )
            for i, end_time in enumerate(crossing_end_times)
        )

        door_edge = self.edges["door-1"]
        features = extract_live_experimental_candidate_features(
            self.candidates["door-1"], door_edge, observation_time,
            building=self.building, crowd_snapshot=None, occupancy_facts=None,
            alternative_route_counts=self.alt_counts, evacuation_snapshot=None, occupants=occupants,
        )
        return features["candidate_recent_flow_rate"]

    def test_identical_crossing_evidence_produces_identical_flow_rate(self):

        crossing_end_times = [45.0, 50.0, 70.0]  # all within (40, 100]
        observation_time = 100.0

        sim_rate = self._sim_flow_rate(crossing_end_times, observation_time)
        live_rate = self._live_flow_rate(crossing_end_times, observation_time)

        self.assertEqual(sim_rate, live_rate)
        self.assertEqual(sim_rate, 3)

    def test_identical_evidence_outside_window_produces_identical_zero(self):

        crossing_end_times = [10.0, 20.0]  # outside (40, 100]
        observation_time = 100.0

        sim_rate = self._sim_flow_rate(crossing_end_times, observation_time)
        live_rate = self._live_flow_rate(crossing_end_times, observation_time)

        self.assertEqual(sim_rate, live_rate)
        self.assertEqual(sim_rate, 0)

    def test_alternative_route_count_is_always_exact(self):
        """Both extractors call the literal same function -- this is a
        structural guarantee, not a coincidence of matching test data."""

        door_edge = self.edges["door-1"]
        candidate = self.candidates["door-1"]

        sim_features = extract_experimental_candidate_features(
            candidate, door_edge, 100.0,
            building=self.building,
            movement_result=MultiAgentSimulationResult(occupants={}, total_evacuation_time=100.0),
            occupancy_snapshot=OccupancySnapshot(observations={"zone-1": OccupancyObservation(occupant_count=0)}),
            alternative_route_counts=self.alt_counts,
        )
        live_features = extract_live_experimental_candidate_features(
            candidate, door_edge, 100.0,
            building=self.building, crowd_snapshot=None, occupancy_facts=None,
            alternative_route_counts=self.alt_counts, evacuation_snapshot=None, occupants=None,
        )

        self.assertEqual(
            sim_features["candidate_alternative_route_count"],
            live_features["candidate_alternative_route_count"],
        )


class MissingEvidenceStaysNoneTests(unittest.TestCase):
    """The epistemic-difference case: candidate_congestion_trend and
    Exit's candidate_recent_flow_rate use genuinely different live
    computation bases than simulation (disclosed in both modules'
    docstrings) -- exact numeric equality is not claimed for those, but
    missing live evidence must always surface as None, never a
    fabricated 0/UNKNOWN-as-if-computed."""

    def setUp(self):
        self.building = make_building()
        self.edges = edges_by_candidate_id(self.building)
        self.candidates = {c.candidate_id: c for c in enumerate_candidates(self.building)}
        self.alt_counts = build_alternative_route_counts(tuple(self.candidates.values()))

    def test_trend_is_none_not_fabricated_when_crowd_intelligence_unavailable(self):

        features = extract_live_experimental_candidate_features(
            self.candidates["door-1"], self.edges["door-1"], 100.0,
            building=self.building, crowd_snapshot=None, occupancy_facts=None,
            alternative_route_counts=self.alt_counts, evacuation_snapshot=None, occupants=None,
        )

        self.assertIsNone(features["candidate_congestion_trend"])

    def test_exit_flow_rate_is_none_not_fabricated_when_evacuation_progress_unavailable(self):

        features = extract_live_experimental_candidate_features(
            self.candidates["exit-1"], self.edges["exit-1"], 100.0,
            building=self.building, crowd_snapshot=None, occupancy_facts=None,
            alternative_route_counts=self.alt_counts, evacuation_snapshot=None, occupants=None,
        )

        self.assertIsNone(features["candidate_recent_flow_rate"])

    def test_door_stair_flow_rate_is_none_not_fabricated_when_occupants_unavailable(self):

        features = extract_live_experimental_candidate_features(
            self.candidates["door-1"], self.edges["door-1"], 100.0,
            building=self.building, crowd_snapshot=None, occupancy_facts=None,
            alternative_route_counts=self.alt_counts, evacuation_snapshot=None, occupants=None,
        )

        self.assertIsNone(features["candidate_recent_flow_rate"])


if __name__ == "__main__":
    unittest.main()
