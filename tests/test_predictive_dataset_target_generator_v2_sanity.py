import unittest

from navigation.edge import Edge
from navigation.node import Node

from pathfinding.route import Route

from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline, OccupantTimelineStep
from simulator.occupant import OccupantState

from predictive_dataset.target_generator_v2 import (
    MIN_PERSISTENCE_SECONDS,
    compute_qualifying_onsets,
    generate_candidate_label_v2,
)


# =====================================================
# Predictive Congestion Target V2 milestone, Phase 5/9/25 -- the 8
# controlled physical/operational sanity scenarios this milestone's own
# Phase 5 requires, PLUS the "already congested -> None" and "onset
# within horizon" unit checks. Synthetic OccupantTimelineStep fixtures,
# same style as tests/test_predictive_dataset_extractors.py -- these
# test the TARGET FORMULA's logic directly, independent of whether the
# real simulator's capacity model would ever produce this exact
# evidence for a given candidate type (e.g. Scenario 5 hand-builds a
# queue on an Exit-typed edge specifically to prove the formula
# generalizes, even though Exit's real capacity=50 never queues in
# practice -- documented separately in docs/architecture/
# predictive_congestion_target_v2.md).
# =====================================================


def _edge(edge_id="edge-1", edge_type=Edge.DOOR):
    return Edge(id=edge_id, edge_type=edge_type, from_node="zone-a", to_node="zone-b", walking_distance=5.0)


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
        state=OccupantState.ARRIVED, depart_time=step.start_time - step.queue_wait_time, arrival_time=step.end_time,
    )


def _movement_result(*occupant_specs, total_evacuation_time=100.0):
    """occupant_specs: sequence of (occupant_id, edge, step)."""

    occupants = {occ_id: _timeline(occ_id, edge, step) for occ_id, edge, step in occupant_specs}
    return MultiAgentSimulationResult(occupants=occupants, total_evacuation_time=total_evacuation_time)


class ControlledSanityScenarioTests(unittest.TestCase):

    def test_scenario_1_single_occupant_no_congestion(self):
        """One occupant passes through a Door -- expected: NO congestion."""

        edge = _edge()
        movement_result = _movement_result(("occ-1", edge, _step(edge, 0.0, 10.0)))

        label = generate_candidate_label_v2(edge.id, movement_result, 0.0, 20.0)

        self.assertFalse(label.currently_congested)
        self.assertFalse(label.target)

    def test_scenario_2_sequential_no_meaningful_waiting(self):
        """Two occupants pass sequentially, negligible wait (well under
        the persistence bar) -- expected: NO congestion."""

        edge = _edge()
        # occ-2 waits only 0.5s before being admitted -- far short of MIN_PERSISTENCE_SECONDS
        movement_result = _movement_result(
            ("occ-1", edge, _step(edge, 0.0, 10.0)),
            ("occ-2", edge, _step(edge, 10.0, 20.0, queue_wait_time=0.5)),
        )

        label = generate_candidate_label_v2(edge.id, movement_result, 5.0, 20.0)

        self.assertFalse(label.currently_congested)
        self.assertFalse(label.target)

    def test_scenario_3_accumulation_at_narrow_door_is_congestion(self):
        """Several occupants accumulate at a narrow Door (capacity 1) --
        a real, sustained queue -- expected: congestion."""

        edge = _edge()
        # occ-1 occupies [0,10]; occ-2,3,4,5 queue behind it, admitted in
        # sequence, each waiting a meaningful, growing amount of time.
        specs = [("occ-1", edge, _step(edge, 0.0, 10.0))]
        join_time = 1.0  # occ-2 starts waiting at t=1
        admit_time = 10.0
        for i in range(2, 6):
            wait = admit_time - join_time
            specs.append((f"occ-{i}", edge, _step(edge, admit_time, admit_time + 10.0, queue_wait_time=wait)))
            join_time = admit_time
            admit_time += 10.0

        movement_result = _movement_result(*specs)

        # observe early, while the queue is building (t=3): a qualifying
        # onset should occur within the next 20s.
        label = generate_candidate_label_v2(edge.id, movement_result, 3.0, 20.0)
        self.assertTrue(label.target)

    def test_scenario_4_exit_demand_that_clears_efficiently_is_not_congestion(self):
        """Occupants approach an Exit but clear efficiently (no sustained
        overlap) -- expected: NOT congestion."""

        exit_edge = _edge("exit-1", Edge.EXIT)
        # each occupant's crossing barely overlaps the next, never for
        # longer than the persistence bar.
        movement_result = _movement_result(
            ("occ-1", exit_edge, _step(exit_edge, 0.0, 5.0)),
            ("occ-2", exit_edge, _step(exit_edge, 4.5, 9.5)),
            ("occ-3", exit_edge, _step(exit_edge, 9.0, 14.0)),
        )

        label = generate_candidate_label_v2(exit_edge.id, movement_result, 0.0, 20.0)

        self.assertFalse(label.currently_congested)
        self.assertFalse(label.target)

    def test_scenario_5_exit_persistent_overlap_is_congestion(self):
        """Exit demand exceeds clearing rate and occupants genuinely,
        sustainedly overlap on the edge -- expected: congestion (via the
        OCCUPANCY mechanism, not queueing -- Exit structurally never
        queues under this project's capacity model)."""

        exit_edge = _edge("exit-1", Edge.EXIT)
        # occ-1 and occ-2 overlap on the exit edge for a full 15s -- a
        # genuine, sustained (>=3s) concurrent-occupancy episode.
        movement_result = _movement_result(
            ("occ-1", exit_edge, _step(exit_edge, 0.0, 20.0)),
            ("occ-2", exit_edge, _step(exit_edge, 5.0, 20.0)),
        )

        label = generate_candidate_label_v2(exit_edge.id, movement_result, 0.0, 20.0)
        self.assertTrue(label.target)

    def test_scenario_6_stair_sustained_demand_is_congestion(self):
        """Stair receives sustained demand and occupants accumulate in a
        real queue -- expected: congestion."""

        stair_edge = _edge("stair-1", Edge.STAIR)
        specs = [("occ-1", stair_edge, _step(stair_edge, 0.0, 15.0))]
        specs.append(("occ-2", stair_edge, _step(stair_edge, 15.0, 30.0, queue_wait_time=14.0)))  # waited since t=1

        movement_result = _movement_result(*specs)

        label = generate_candidate_label_v2(stair_edge.id, movement_result, 2.0, 20.0)
        self.assertTrue(label.target)

    def test_scenario_7_timestamp_handoff_alone_is_never_congestion(self):
        """The exact bug Target V1 had: occupant A leaves exactly when
        occupant B enters (zero-duration FIFO handoff, zero queue wait)
        -- expected: NO congestion, purely from the boundary touch."""

        edge = _edge()
        movement_result = _movement_result(
            ("occ-1", edge, _step(edge, 0.0, 10.0)),
            ("occ-2", edge, _step(edge, 10.0, 20.0, queue_wait_time=0.0)),
        )

        label = generate_candidate_label_v2(edge.id, movement_result, 5.0, 20.0)

        self.assertFalse(label.currently_congested)
        self.assertFalse(label.target)

    def test_scenario_8_congestion_is_computed_independently_per_candidate(self):
        """Each candidate's label depends only on its OWN edge's steps --
        proves congestion on one candidate never leaks into or is
        contaminated by another (the same independence a real obstacle-
        induced reroute to a different candidate would rely on)."""

        busy_edge = _edge("door-busy")
        quiet_edge = _edge("door-quiet")

        specs = [("occ-1", busy_edge, _step(busy_edge, 0.0, 10.0))]
        join_time = 1.0
        admit_time = 10.0
        for i in range(2, 5):
            wait = admit_time - join_time
            specs.append((f"occ-{i}", busy_edge, _step(busy_edge, admit_time, admit_time + 10.0, queue_wait_time=wait)))
            join_time = admit_time
            admit_time += 10.0
        specs.append(("occ-quiet", quiet_edge, _step(quiet_edge, 0.0, 5.0)))

        movement_result = _movement_result(*specs)

        busy_label = generate_candidate_label_v2(busy_edge.id, movement_result, 3.0, 20.0)
        quiet_label = generate_candidate_label_v2(quiet_edge.id, movement_result, 3.0, 20.0)

        self.assertTrue(busy_label.target)
        self.assertFalse(quiet_label.target)


class AlreadyCongestedExclusionTests(unittest.TestCase):

    def test_currently_congested_candidate_gets_target_none(self):

        edge = _edge()
        specs = [("occ-1", edge, _step(edge, 0.0, 20.0))]
        join_time = 1.0
        admit_time = 20.0
        for i in range(2, 4):
            wait = admit_time - join_time
            specs.append((f"occ-{i}", edge, _step(edge, admit_time, admit_time + 20.0, queue_wait_time=wait)))
            join_time = admit_time
            admit_time += 20.0

        movement_result = _movement_result(*specs)

        # by t=10, occ-2 has already been queued since t=1 -- 9s of
        # elapsed wait, well past the 3.0s persistence bar -- already
        # congested.
        label = generate_candidate_label_v2(edge.id, movement_result, 10.0, 20.0)

        self.assertTrue(label.currently_congested)
        self.assertIsNone(label.target)


class OnsetTimingTests(unittest.TestCase):

    def test_onset_exactly_at_persistence_bar(self):
        """A queue that starts at t=1 and is admitted at t=4 (exactly a
        3.0s wait, MIN_PERSISTENCE_SECONDS) should have its onset at
        t=1+3.0=4.0."""

        edge = _edge()
        movement_result = _movement_result(
            ("occ-1", edge, _step(edge, 0.0, 50.0)),
            ("occ-2", edge, _step(edge, 50.0, 60.0, queue_wait_time=MIN_PERSISTENCE_SECONDS)),
        )

        onsets = compute_qualifying_onsets(movement_result, edge.id)
        self.assertEqual(len(onsets), 1)
        self.assertAlmostEqual(onsets[0][0], 50.0 - MIN_PERSISTENCE_SECONDS + MIN_PERSISTENCE_SECONDS)

    def test_onset_outside_horizon_window_is_negative(self):

        edge = _edge()
        specs = [("occ-1", edge, _step(edge, 0.0, 100.0))]
        movement_result = _movement_result(
            ("occ-1", edge, _step(edge, 0.0, 100.0)),
            ("occ-2", edge, _step(edge, 100.0, 110.0, queue_wait_time=50.0)),  # onset far in the future
        )

        # observing at t=10 with a 20s horizon -- the queue onset (at
        # t=100-50+3=53) is well outside (10, 30].
        label = generate_candidate_label_v2(edge.id, movement_result, 10.0, 20.0)
        self.assertFalse(label.target)


if __name__ == "__main__":
    unittest.main()
