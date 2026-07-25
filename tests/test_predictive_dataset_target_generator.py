import unittest

from navigation.edge import Edge
from navigation.node import Node

from pathfinding.route import Route

from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline, OccupantTimelineStep
from simulator.occupant import OccupantState

from predictive_dataset.target_generator import generate_candidate_label


CANDIDATE_ID = "exit-1"


def _edge():
    return Edge(id=CANDIDATE_ID, edge_type=Edge.EXIT, from_node="zone-1", to_node=Node.OUTSIDE_NODE_ID)


def _step(edge, start, end, queue_wait_time=0.0, index=0):
    node = Node(id="zone-1", name="Lobby", floor_id="floor-1", node_type=Node.ZONE)
    outside = Node(id=Node.OUTSIDE_NODE_ID, name="Outside", floor_id="", node_type=Node.OUTSIDE)
    return OccupantTimelineStep(
        index=index, from_node=node, to_node=outside, edge=edge,
        queue_wait_time=queue_wait_time, start_time=start, end_time=end,
    )


def _timeline(occupant_id, steps, depart_time, arrival_time):
    edge = steps[0].edge if steps else _edge()
    route = Route(nodes=[], edges=[edge], total_cost=0.0, total_distance=0.0)
    return OccupantTimeline(
        occupant_id=occupant_id, route=route, steps=steps,
        state=OccupantState.ARRIVED if arrival_time is not None else OccupantState.TRAVERSING,
        depart_time=depart_time, arrival_time=arrival_time,
    )


def _movement_result(occupants):
    return MultiAgentSimulationResult(occupants=occupants, total_evacuation_time=None)


# =====================================================
# Phase 10's five required cases, verbatim.
# =====================================================


class CaseATests(unittest.TestCase):
    # A. Clear now -> congested within 30s -> POSITIVE.

    def test_clear_now_congested_within_horizon_is_positive(self):

        edge = _edge()

        # Nobody on the edge at t=0. Two occupants overlap on it during
        # [20, 40] -- inside the (0, 30] window.
        occ1 = _timeline("occ-1", [_step(edge, 20.0, 40.0)], depart_time=20.0, arrival_time=40.0)
        occ2 = _timeline("occ-2", [_step(edge, 22.0, 35.0)], depart_time=22.0, arrival_time=35.0)

        movement_result = _movement_result({"occ-1": occ1, "occ-2": occ2})

        label = generate_candidate_label(CANDIDATE_ID, movement_result, time=0.0, horizon=30.0)

        self.assertFalse(label.currently_congested)
        self.assertTrue(label.target)


class CaseBTests(unittest.TestCase):
    # B. Clear now -> remains clear -> NEGATIVE.

    def test_clear_now_remains_clear_is_negative(self):

        edge = _edge()

        occ1 = _timeline("occ-1", [_step(edge, 5.0, 10.0)], depart_time=5.0, arrival_time=10.0)

        movement_result = _movement_result({"occ-1": occ1})

        label = generate_candidate_label(CANDIDATE_ID, movement_result, time=0.0, horizon=30.0)

        self.assertFalse(label.currently_congested)
        self.assertFalse(label.target)


class CaseCTests(unittest.TestCase):
    # C. Already congested now -> handled per Phase 5 policy: currently_congested=True, target=None (not applicable).

    def test_already_congested_now_is_not_applicable(self):

        edge = _edge()

        occ1 = _timeline("occ-1", [_step(edge, 0.0, 20.0)], depart_time=0.0, arrival_time=20.0)
        occ2 = _timeline("occ-2", [_step(edge, 0.0, 20.0)], depart_time=0.0, arrival_time=20.0)

        movement_result = _movement_result({"occ-1": occ1, "occ-2": occ2})

        label = generate_candidate_label(CANDIDATE_ID, movement_result, time=5.0, horizon=30.0)

        self.assertTrue(label.currently_congested)
        self.assertIsNone(label.target)


class CaseDTests(unittest.TestCase):
    # D. Congestion begins after the prediction horizon -> NEGATIVE.

    def test_congestion_after_horizon_is_negative(self):

        edge = _edge()

        # Congestion starts at t=40 -- outside a 30s horizon from t=0.
        occ1 = _timeline("occ-1", [_step(edge, 40.0, 60.0)], depart_time=40.0, arrival_time=60.0)
        occ2 = _timeline("occ-2", [_step(edge, 42.0, 55.0)], depart_time=42.0, arrival_time=55.0)

        movement_result = _movement_result({"occ-1": occ1, "occ-2": occ2})

        label = generate_candidate_label(CANDIDATE_ID, movement_result, time=0.0, horizon=30.0)

        self.assertFalse(label.currently_congested)
        self.assertFalse(label.target)


class CaseETests(unittest.TestCase):
    # E. Candidate becomes unavailable/unsafe before congestion -- this
    # package's chosen treatment: an unavailable/unused candidate simply
    # has zero recorded edge activity, which flows through the ordinary
    # "never reached the threshold" negative path with no special-casing
    # required. had_any_activity_in_window makes the distinction visible
    # for analysis rather than silently collapsing it into an ordinary
    # negative.

    def test_no_activity_at_all_is_negative_and_flagged_as_no_activity(self):

        movement_result = _movement_result({})

        label = generate_candidate_label(CANDIDATE_ID, movement_result, time=0.0, horizon=30.0)

        self.assertFalse(label.currently_congested)
        self.assertFalse(label.target)
        self.assertFalse(label.had_any_activity_in_window)

    def test_light_activity_that_never_congests_is_negative_but_flagged_as_active(self):

        edge = _edge()
        occ1 = _timeline("occ-1", [_step(edge, 5.0, 10.0)], depart_time=5.0, arrival_time=10.0)

        movement_result = _movement_result({"occ-1": occ1})

        label = generate_candidate_label(CANDIDATE_ID, movement_result, time=0.0, horizon=30.0)

        self.assertFalse(label.target)
        self.assertTrue(label.had_any_activity_in_window)


# =====================================================
# Horizon boundary behavior.
# =====================================================


class HorizonBoundaryTests(unittest.TestCase):

    def test_congestion_exactly_at_horizon_boundary_is_included(self):

        edge = _edge()

        occ1 = _timeline("occ-1", [_step(edge, 25.0, 35.0)], depart_time=25.0, arrival_time=35.0)
        occ2 = _timeline("occ-2", [_step(edge, 28.0, 30.0)], depart_time=28.0, arrival_time=30.0)

        movement_result = _movement_result({"occ-1": occ1, "occ-2": occ2})

        # Congestion window [28, 30] -- 30.0 is exactly time + horizon.
        label = generate_candidate_label(CANDIDATE_ID, movement_result, time=0.0, horizon=30.0)

        self.assertTrue(label.target)

    def test_congestion_one_instant_after_horizon_boundary_is_excluded(self):

        edge = _edge()

        occ1 = _timeline("occ-1", [_step(edge, 30.1, 40.0)], depart_time=30.1, arrival_time=40.0)
        occ2 = _timeline("occ-2", [_step(edge, 30.1, 35.0)], depart_time=30.1, arrival_time=35.0)

        movement_result = _movement_result({"occ-1": occ1, "occ-2": occ2})

        label = generate_candidate_label(CANDIDATE_ID, movement_result, time=0.0, horizon=30.0)

        self.assertFalse(label.target)

    def test_different_horizons_can_disagree_on_the_same_observation(self):

        edge = _edge()

        occ1 = _timeline("occ-1", [_step(edge, 25.0, 40.0)], depart_time=25.0, arrival_time=40.0)
        occ2 = _timeline("occ-2", [_step(edge, 28.0, 32.0)], depart_time=28.0, arrival_time=32.0)

        movement_result = _movement_result({"occ-1": occ1, "occ-2": occ2})

        short_horizon_label = generate_candidate_label(CANDIDATE_ID, movement_result, time=0.0, horizon=20.0)
        long_horizon_label = generate_candidate_label(CANDIDATE_ID, movement_result, time=0.0, horizon=30.0)

        self.assertFalse(short_horizon_label.target)
        self.assertTrue(long_horizon_label.target)


if __name__ == "__main__":
    unittest.main()
