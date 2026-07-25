from typing import Any, Dict

from navigation.node import Node

from predictive_dataset.candidate import CandidateIdentity
from predictive_dataset.congestion import candidate_capacity, candidate_congestion_level


# =====================================================
# Per-Candidate Predictive AI Data Foundation milestone, Phase 8 --
# THE FEATURE EXTRACTOR. Every value this module produces is a
# function of simulation state AT OR BEFORE `time` only -- see the
# per-field comments below for exactly which already-realized interval
# data each one reads, and predictive_dataset/target_generator.py's own
# module docstring for why THAT module, not this one, is the only place
# allowed to inspect what happens after `time`.
#
# `movement_result` (simulator.multi_agent_result.MultiAgentSimulationResult)
# holds every OccupantTimelineStep for the WHOLE completed run, future
# ones included -- this module never rejects that on principle (the
# object it's handed already contains the whole run, as
# dataset_builder/timeline.py's own identical `_current_*` helpers
# already establish is fine, see docs/architecture/
# localized_predictive_ai_dataset.md's own leakage-boundary section).
# What matters, and what tests/test_predictive_dataset_leakage_guards.py
# proves mechanically, is that every read here is gated by a comparison
# against `time` that only ever asks "does this already-recorded
# interval CONTAIN time" or "did this already-fixed plan target this
# candidate" -- never "what happens strictly after time".
# =====================================================


def extract_simulation_candidate_features(
    candidate: CandidateIdentity,
    edge,
    time: float,
    *,
    building,
    movement_result,
    occupancy_snapshot,
) -> Dict[str, Any]:

    queue_length = _current_queue_length(movement_result, candidate.candidate_id, time)
    approaching_count = _current_approaching_count(movement_result, candidate.candidate_id, time)
    capacity = candidate_capacity(candidate.candidate_type, edge, building)

    return {
        "total_active_occupant_count": _total_active_occupant_count(movement_result, time),
        "candidate_type": candidate.candidate_type,
        "candidate_capacity": capacity,
        "candidate_walking_distance": edge.walking_distance,
        "candidate_traversable": edge.traversable,
        "candidate_adjacent_zone_occupancy": _adjacent_zone_occupancy(occupancy_snapshot, edge),
        "candidate_queue_length": queue_length,
        "candidate_approaching_count": approaching_count,
        "candidate_congestion_level": candidate_congestion_level(queue_length, approaching_count, capacity),
    }


# =====================================================
# Global context -- "how many occupants are still evacuating, whole
# building, right now" -- the SAME arrival_time <= time test dataset_
# builder.timeline.extract_timeline_rows() already uses for
# people_evacuated/people_remaining, reused here rather than
# reimplemented a second way.
# =====================================================


def _total_active_occupant_count(movement_result, time: float) -> int:

    total = len(movement_result.occupants)

    arrived_by_time = sum(
        1
        for timeline in movement_result.occupants.values()
        if timeline.arrival_time is not None and timeline.arrival_time <= time
    )

    return total - arrived_by_time


# =====================================================


def _adjacent_zone_occupancy(occupancy_snapshot, edge):

    if edge.from_node == Node.OUTSIDE_NODE_ID:
        return None

    return occupancy_snapshot.observation_at(edge.from_node).occupant_count


# =====================================================
# "Currently queued FOR THIS CANDIDATE" -- exactly dataset_builder.
# timeline._current_queue_length()'s own join_time <= time < start_time
# window, filtered to one edge id instead of aggregated across every
# edge. join_time/start_time are both already-realized facts about an
# admission that has, by construction of a completed run, already
# happened -- asking "does [join_time, start_time) contain time" is a
# pure interval-membership test, the same one HazardSnapshot/
# OccupancySnapshot's own snapshot_at(time) queries already rely on.
# =====================================================


def _current_queue_length(movement_result, candidate_id: str, time: float) -> int:

    count = 0

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:

            if step.edge.id != candidate_id:
                continue

            join_time = step.start_time - step.queue_wait_time

            if join_time <= time < step.start_time:
                count += 1

    return count


# =====================================================
# "Currently approaching THIS CANDIDATE" -- occupants whose Route (a
# plan FIXED at scenario/behaviour-resolution time, before any movement
# is simulated -- simulator/occupant.py's own "no dynamic rerouting of
# an occupant already in flight" contract) ends at this candidate's
# edge, who have already departed but not yet arrived as of `time`, and
# who are not ALREADY queued or on the candidate's own edge right now
# (that overlap is what candidate_queue_length already reports --
# double-counting the same occupant under both fields would make them
# redundant rather than two distinct demand signals). Route.edges[-1]
# is knowable at t=0 already (it is a plan, not an outcome) -- reading
# it at any `time` is therefore not a future read, it is the same fact
# available for the whole run. depart_time/arrival_time are the same
# already-realized OccupantTimeline fields _total_active_occupant_
# count() above already uses.
# =====================================================


def _current_approaching_count(movement_result, candidate_id: str, time: float) -> int:

    count = 0

    for timeline in movement_result.occupants.values():

        if timeline.route is None or not timeline.route.edges:
            continue

        if timeline.route.edges[-1].id != candidate_id:
            continue

        if timeline.depart_time > time:
            continue

        if timeline.arrival_time is not None and timeline.arrival_time <= time:
            continue

        if _is_currently_queued_or_on_edge(timeline, candidate_id, time):
            continue

        count += 1

    return count


def _is_currently_queued_or_on_edge(timeline, candidate_id: str, time: float) -> bool:

    for step in timeline.steps:

        if step.edge.id != candidate_id:
            continue

        join_time = step.start_time - step.queue_wait_time

        if join_time <= time <= step.end_time:
            return True

    return False
