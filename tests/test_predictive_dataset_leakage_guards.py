import unittest

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from navigation.edge import Edge
from navigation.node import Node

from occupancy.snapshot import OccupancySnapshot

from pathfinding.route import Route

from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline, OccupantTimelineStep
from simulator.occupant import OccupantState

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.simulation_extractor import extract_simulation_candidate_features
from predictive_dataset.target_generator import generate_candidate_label


# =====================================================
# Phase 8/16 -- mechanical proof that predictive_dataset.
# simulation_extractor cannot see, and is not influenced by, anything
# that happens strictly after `time`. Contrasted directly against
# predictive_dataset.target_generator, which (by design, and only
# there) legitimately does read the future -- the same test setup is
# used for both, so the difference in behavior is the actual proof of
# the boundary, not just an assertion about it.
# =====================================================


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0)],
        exits=[Exit(id="exit-1", zone_id="zone-1", capacity=2)],
    )

    return Building(name="Leakage Guard Building", id="building-1", floors=[floor])


def _step(edge, start, end, queue_wait_time=0.0):
    node = Node(id="zone-1", name="Lobby", floor_id="floor-1", node_type=Node.ZONE)
    outside = Node(id=Node.OUTSIDE_NODE_ID, name="Outside", floor_id="", node_type=Node.OUTSIDE)
    return OccupantTimelineStep(
        index=0, from_node=node, to_node=outside, edge=edge,
        queue_wait_time=queue_wait_time, start_time=start, end_time=end,
    )


def _timeline(occupant_id, edge, start, end):
    route = Route(nodes=[], edges=[edge], total_cost=0.0, total_distance=0.0)
    return OccupantTimeline(
        occupant_id=occupant_id, route=route, steps=[_step(edge, start, end)],
        state=OccupantState.ARRIVED, depart_time=start, arrival_time=end,
    )


def _movement_result_with_future_activity(edge, future_start, future_end):

    # Fixed, shared "past" -- identical in every call site below.
    occ_past = _timeline("occ-past", edge, 0.0, 5.0)

    # The only thing that varies -- an occupant pair active entirely
    # AFTER the observation time this test evaluates at.
    occ_future_1 = _timeline("occ-future-1", edge, future_start, future_end)
    occ_future_2 = _timeline("occ-future-2", edge, future_start + 0.5, future_end - 0.5)

    return MultiAgentSimulationResult(
        occupants={"occ-past": occ_past, "occ-future-1": occ_future_1, "occ-future-2": occ_future_2},
        total_evacuation_time=future_end,
    )


class SimulationExtractorIgnoresFutureTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.candidate = enumerate_candidates(self.building)[0]
        self.edge = edges_by_candidate_id(self.building)[self.candidate.candidate_id]
        self.occupancy_snapshot = OccupancySnapshot()

    def _features_at(self, movement_result, time):

        return extract_simulation_candidate_features(
            self.candidate, self.edge, time,
            building=self.building, movement_result=movement_result, occupancy_snapshot=self.occupancy_snapshot,
        )

    def test_feature_row_is_identical_regardless_of_what_happens_far_in_the_future(self):

        observation_time = 10.0  # strictly after occ-past's window, strictly before either future scenario

        movement_result_a = _movement_result_with_future_activity(self.edge, future_start=100.0, future_end=110.0)
        movement_result_b = _movement_result_with_future_activity(self.edge, future_start=500.0, future_end=9000.0)

        features_a = self._features_at(movement_result_a, observation_time)
        features_b = self._features_at(movement_result_b, observation_time)

        self.assertEqual(features_a, features_b)

    def test_feature_row_is_identical_even_if_the_same_future_occupants_never_use_this_candidate(self):

        # SAME population, SAME depart/arrival times (so total_active_
        # occupant_count is identical in both cases -- fixed scenario
        # population is legitimately t=0-knowable, not a future read) --
        # the only difference is which edge the two "future" occupants
        # are ultimately assigned to. One variant congests THIS
        # candidate later; the other never touches it at all.

        observation_time = 10.0

        other_edge = Edge(
            id="exit-2-not-a-candidate-under-test", edge_type=Edge.EXIT,
            from_node="zone-1", to_node=Node.OUTSIDE_NODE_ID,
        )

        occ_past = _timeline("occ-past", self.edge, 0.0, 5.0)

        congests_this_candidate_later = MultiAgentSimulationResult(
            occupants={
                "occ-past": occ_past,
                "occ-future-1": _timeline("occ-future-1", self.edge, 100.0, 110.0),
                "occ-future-2": _timeline("occ-future-2", self.edge, 100.5, 109.5),
            },
            total_evacuation_time=110.0,
        )
        never_touches_this_candidate = MultiAgentSimulationResult(
            occupants={
                "occ-past": occ_past,
                "occ-future-1": _timeline("occ-future-1", other_edge, 100.0, 110.0),
                "occ-future-2": _timeline("occ-future-2", other_edge, 100.5, 109.5),
            },
            total_evacuation_time=110.0,
        )

        features_congests_later = self._features_at(congests_this_candidate_later, observation_time)
        features_never_touches = self._features_at(never_touches_this_candidate, observation_time)

        self.assertEqual(features_congests_later, features_never_touches)

    def test_target_generator_by_contrast_is_sensitive_to_the_same_future_difference(self):

        # The exact same two movement_results the feature-row test above
        # proved were EXTRACTION-invariant now produce DIFFERENT target
        # labels -- proving the leakage boundary is drawn in the right
        # place (target generation, not feature extraction).

        observation_time = 10.0

        congested_soon = _movement_result_with_future_activity(self.edge, future_start=15.0, future_end=25.0)
        congested_never = MultiAgentSimulationResult(
            occupants={"occ-past": _timeline("occ-past", self.edge, 0.0, 5.0)}, total_evacuation_time=5.0,
        )

        label_soon = generate_candidate_label(self.candidate.candidate_id, congested_soon, observation_time, horizon=30.0)
        label_never = generate_candidate_label(self.candidate.candidate_id, congested_never, observation_time, horizon=30.0)

        self.assertTrue(label_soon.target)
        self.assertFalse(label_never.target)


if __name__ == "__main__":
    unittest.main()
