import unittest

from navigation.edge import Edge
from navigation.node import Node

from pathfinding.route import Route

from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline, OccupantTimelineStep
from simulator.occupant import OccupantState

from predictive_dataset.target_semantics_analysis import (
    is_persistently_congested,
    occupancy_episodes,
    occupancy_intervals,
    persistent_onset_within_window,
    qualifying_onset_times,
    queue_episodes,
    queue_intervals,
)


# =====================================================
# Predictive Congestion Target V2 milestone, Phase 25 -- unit tests for
# the new queue/occupancy interval and onset primitives added to
# predictive_dataset/target_semantics_analysis.py this milestone.
# =====================================================


def _edge(edge_id="edge-1"):
    return Edge(id=edge_id, edge_type=Edge.DOOR, from_node="zone-a", to_node="zone-b", walking_distance=5.0)


def _zone_node(name):
    return Node(id=name, name=name, floor_id="floor-1", node_type=Node.ZONE)


def _step(edge, start, end, queue_wait_time=0.0):
    return OccupantTimelineStep(
        index=0, from_node=_zone_node("zone-a"), to_node=_zone_node("zone-b"), edge=edge,
        queue_wait_time=queue_wait_time, start_time=start, end_time=end,
    )


def _movement_result(*specs):
    occupants = {}
    for occ_id, edge, step in specs:
        route = Route(nodes=[], edges=[edge], total_cost=0.0, total_distance=0.0)
        occupants[occ_id] = OccupantTimeline(
            occupant_id=occ_id, route=route, steps=[step],
            state=OccupantState.ARRIVED, depart_time=step.start_time - step.queue_wait_time, arrival_time=step.end_time,
        )
    return MultiAgentSimulationResult(occupants=occupants, total_evacuation_time=100.0)


class QueueIntervalsTests(unittest.TestCase):

    def test_zero_wait_produces_no_interval(self):

        edge = _edge()
        mr = _movement_result(("occ-1", edge, _step(edge, 0.0, 10.0, queue_wait_time=0.0)))

        self.assertEqual(queue_intervals(mr, edge.id), [])

    def test_real_wait_produces_correct_interval(self):

        edge = _edge()
        # queue_wait_time=5 on a step starting at 10 -> join_time=5, interval (5, 10)
        mr = _movement_result(("occ-1", edge, _step(edge, 10.0, 20.0, queue_wait_time=5.0)))

        self.assertEqual(queue_intervals(mr, edge.id), [(5.0, 10.0)])

    def test_ignores_other_edges(self):

        edge = _edge("edge-1")
        other = _edge("edge-2")
        mr = _movement_result(("occ-1", other, _step(other, 10.0, 20.0, queue_wait_time=5.0)))

        self.assertEqual(queue_intervals(mr, edge.id), [])


class OccupancyIntervalsTests(unittest.TestCase):

    def test_returns_start_end_for_every_step_on_edge(self):

        edge = _edge()
        mr = _movement_result(
            ("occ-1", edge, _step(edge, 0.0, 10.0)),
            ("occ-2", edge, _step(edge, 5.0, 15.0)),
        )

        self.assertEqual(sorted(occupancy_intervals(mr, edge.id)), [(0.0, 10.0), (5.0, 15.0)])


class QueueEpisodesTests(unittest.TestCase):

    def test_single_waiting_occupant_meets_threshold_one(self):

        edge = _edge()
        mr = _movement_result(("occ-1", edge, _step(edge, 10.0, 20.0, queue_wait_time=5.0)))

        episodes = queue_episodes(mr, edge.id, threshold=1)
        self.assertEqual(episodes, [(5.0, 10.0, 5.0)])

    def test_single_waiting_occupant_does_not_meet_threshold_two(self):

        edge = _edge()
        mr = _movement_result(("occ-1", edge, _step(edge, 10.0, 20.0, queue_wait_time=5.0)))

        episodes = queue_episodes(mr, edge.id, threshold=2)
        self.assertEqual(episodes, [])


class OccupancyEpisodesTests(unittest.TestCase):

    def test_two_concurrent_occupants_meet_default_threshold(self):

        edge = _edge()
        mr = _movement_result(
            ("occ-1", edge, _step(edge, 0.0, 20.0)),
            ("occ-2", edge, _step(edge, 5.0, 15.0)),
        )

        episodes = occupancy_episodes(mr, edge.id)  # default threshold=2
        self.assertEqual(episodes, [(5.0, 15.0, 10.0)])

    def test_single_occupant_never_meets_default_threshold(self):

        edge = _edge()
        mr = _movement_result(("occ-1", edge, _step(edge, 0.0, 20.0)))

        self.assertEqual(occupancy_episodes(mr, edge.id), [])


class QualifyingOnsetTimesTests(unittest.TestCase):

    def test_filters_out_episodes_shorter_than_min_duration(self):

        episodes = [(0.0, 2.0, 2.0), (10.0, 20.0, 10.0)]

        onsets = qualifying_onset_times(episodes, min_duration=5.0)
        self.assertEqual(onsets, [(15.0, 20.0)])

    def test_empty_episodes_produce_empty_onsets(self):
        self.assertEqual(qualifying_onset_times([], min_duration=3.0), [])

    def test_zero_min_duration_onset_equals_episode_start(self):

        episodes = [(10.0, 20.0, 10.0)]
        onsets = qualifying_onset_times(episodes, min_duration=0.0)
        self.assertEqual(onsets, [(10.0, 20.0)])


class PersistenceQueryTests(unittest.TestCase):

    def setUp(self):
        self.onsets = [(15.0, 20.0)]  # qualifies starting at t=15, episode ends at t=20

    def test_not_congested_before_onset(self):
        self.assertFalse(is_persistently_congested(self.onsets, 14.9))

    def test_congested_at_onset(self):
        self.assertTrue(is_persistently_congested(self.onsets, 15.0))

    def test_congested_just_before_end(self):
        self.assertTrue(is_persistently_congested(self.onsets, 19.9))

    def test_not_congested_at_or_after_end(self):
        self.assertFalse(is_persistently_congested(self.onsets, 20.0))

    def test_onset_within_window_true(self):
        self.assertTrue(persistent_onset_within_window(self.onsets, 10.0, 10.0))  # (10,20], onset=15

    def test_onset_within_window_false_when_before_window(self):
        self.assertFalse(persistent_onset_within_window(self.onsets, 16.0, 10.0))  # onset=15 <= time=16

    def test_onset_within_window_false_when_after_window(self):
        self.assertFalse(persistent_onset_within_window(self.onsets, 0.0, 10.0))  # (0,10], onset=15 not in range


if __name__ == "__main__":
    unittest.main()
