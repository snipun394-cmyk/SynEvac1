import unittest

from navigation.edge import Edge
from navigation.node import Node

from pathfinding.route import Route

from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline, OccupantTimelineStep
from simulator.occupant import OccupantState

from predictive_dataset.target_semantics_analysis import (
    counterfactual_positive,
    episode_durations_and_gaps,
    summarize_durations,
)


# =====================================================
# Localized Predictive Model V2.2 milestone, Phase 20 -- tests for the
# target-semantics analysis machinery (Phase 7/8). Synthetic fixtures,
# same style as tests/test_predictive_dataset_extractors.py.
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


def _timeline(occupant_id, edge, step):
    route = Route(nodes=[], edges=[edge], total_cost=0.0, total_distance=0.0)
    return OccupantTimeline(
        occupant_id=occupant_id, route=route, steps=[step],
        state=OccupantState.ARRIVED, depart_time=step.start_time, arrival_time=step.end_time,
    )


class EpisodeDurationsAndGapsTests(unittest.TestCase):

    def test_zero_duration_episode_from_exact_fifo_handoff(self):
        """occ-1 on [0, 10], occ-2 on [10, 20] -- an exact handoff, no
        real overlap -- must register a momentary threshold-crossing
        episode of exactly 0.0 duration (matches target_generator's own
        inclusive-bounds convention) and a 0.0 adjacent gap."""

        edge = _edge()
        occ1 = _timeline("occ-1", edge, _step(edge, 0.0, 10.0))
        occ2 = _timeline("occ-2", edge, _step(edge, 10.0, 20.0))
        movement_result = MultiAgentSimulationResult(occupants={"occ-1": occ1, "occ-2": occ2}, total_evacuation_time=20.0)

        durations, gaps = episode_durations_and_gaps(movement_result, edge.id)

        self.assertEqual(durations, [0.0])
        self.assertEqual(gaps, [0.0])

    def test_genuine_sustained_overlap_has_real_duration(self):
        """occ-1 on [0, 30], occ-2 on [10, 20] -- a real 10s overlap."""

        edge = _edge()
        occ1 = _timeline("occ-1", edge, _step(edge, 0.0, 30.0))
        occ2 = _timeline("occ-2", edge, _step(edge, 10.0, 20.0))
        movement_result = MultiAgentSimulationResult(occupants={"occ-1": occ1, "occ-2": occ2}, total_evacuation_time=30.0)

        durations, gaps = episode_durations_and_gaps(movement_result, edge.id)

        self.assertEqual(durations, [10.0])
        # occ-2 is nested inside occ-1's step -- a NEGATIVE gap (real
        # overlap), not a zero/positive one; distinguishes genuine
        # overlap from a FIFO handoff (gap==0) or real idle time (gap>0).
        self.assertEqual(gaps, [-20.0])

    def test_positive_gap_when_edge_is_idle_between_crossings(self):

        edge = _edge()
        occ1 = _timeline("occ-1", edge, _step(edge, 0.0, 5.0))
        occ2 = _timeline("occ-2", edge, _step(edge, 15.0, 20.0))
        movement_result = MultiAgentSimulationResult(occupants={"occ-1": occ1, "occ-2": occ2}, total_evacuation_time=20.0)

        durations, gaps = episode_durations_and_gaps(movement_result, edge.id)

        self.assertEqual(durations, [])
        self.assertEqual(gaps, [10.0])

    def test_no_steps_on_edge_returns_empty(self):

        edge = _edge()
        other_edge = _edge("edge-2")
        occ1 = _timeline("occ-1", other_edge, _step(other_edge, 0.0, 5.0))
        movement_result = MultiAgentSimulationResult(occupants={"occ-1": occ1}, total_evacuation_time=5.0)

        durations, gaps = episode_durations_and_gaps(movement_result, edge.id)

        self.assertEqual(durations, [])
        self.assertEqual(gaps, [])


class CounterfactualPositiveTests(unittest.TestCase):

    def test_min_duration_zero_matches_any_threshold_crossing(self):
        """A zero-duration handoff at t=15 (within (10, 30]) must count
        as positive under min_duration=0.0 -- exactly production
        target_generator semantics."""

        edge = _edge()
        occ1 = _timeline("occ-1", edge, _step(edge, 5.0, 15.0))
        occ2 = _timeline("occ-2", edge, _step(edge, 15.0, 25.0))
        movement_result = MultiAgentSimulationResult(occupants={"occ-1": occ1, "occ-2": occ2}, total_evacuation_time=25.0)

        self.assertTrue(counterfactual_positive(movement_result, edge.id, 10.0, 20.0, min_duration=0.0))

    def test_min_duration_one_second_rejects_zero_duration_handoff(self):

        edge = _edge()
        occ1 = _timeline("occ-1", edge, _step(edge, 5.0, 15.0))
        occ2 = _timeline("occ-2", edge, _step(edge, 15.0, 25.0))
        movement_result = MultiAgentSimulationResult(occupants={"occ-1": occ1, "occ-2": occ2}, total_evacuation_time=25.0)

        self.assertFalse(counterfactual_positive(movement_result, edge.id, 10.0, 20.0, min_duration=1.0))

    def test_min_duration_is_satisfied_by_genuine_sustained_overlap(self):

        edge = _edge()
        occ1 = _timeline("occ-1", edge, _step(edge, 5.0, 30.0))
        occ2 = _timeline("occ-2", edge, _step(edge, 12.0, 22.0))  # 10s overlap with occ-1
        movement_result = MultiAgentSimulationResult(occupants={"occ-1": occ1, "occ-2": occ2}, total_evacuation_time=30.0)

        self.assertTrue(counterfactual_positive(movement_result, edge.id, 10.0, 20.0, min_duration=5.0))
        self.assertFalse(counterfactual_positive(movement_result, edge.id, 10.0, 20.0, min_duration=15.0))

    def test_never_looks_past_the_horizon_window(self):
        """A real, sustained overlap starting well AFTER (time+horizon]
        must never count -- future information outside the window is
        inaccessible, same discipline as target_generator itself."""

        edge = _edge()
        occ1 = _timeline("occ-1", edge, _step(edge, 100.0, 130.0))
        occ2 = _timeline("occ-2", edge, _step(edge, 105.0, 125.0))
        movement_result = MultiAgentSimulationResult(occupants={"occ-1": occ1, "occ-2": occ2}, total_evacuation_time=130.0)

        self.assertFalse(counterfactual_positive(movement_result, edge.id, 10.0, 20.0, min_duration=0.0))

    def test_no_evidence_at_all_is_negative(self):

        edge = _edge()
        movement_result = MultiAgentSimulationResult(occupants={}, total_evacuation_time=10.0)

        self.assertFalse(counterfactual_positive(movement_result, edge.id, 0.0, 20.0, min_duration=0.0))


class SummarizeDurationsTests(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(summarize_durations([]), {"n_episodes": 0})

    def test_all_zero_durations(self):

        summary = summarize_durations([0.0, 0.0, 0.0])

        self.assertEqual(summary["n_episodes"], 3)
        self.assertEqual(summary["zero_duration_fraction"], 1.0)
        self.assertEqual(summary["mean"], 0.0)

    def test_mixed_durations(self):

        summary = summarize_durations([0.0, 10.0, 20.0])

        self.assertEqual(summary["n_episodes"], 3)
        self.assertAlmostEqual(summary["mean"], 10.0)
        self.assertEqual(summary["zero_duration_count"], 1)


if __name__ == "__main__":
    unittest.main()
