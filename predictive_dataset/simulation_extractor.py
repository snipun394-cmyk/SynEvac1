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
    # CALLER CONTRACT (found the hard way -- Large-Scale Predictive
    # Dataset Campaign & Validation milestone's own Phase 5 feature-
    # distribution check caught candidate_traversable reporting a
    # constant True across an entire 2.5M-row campaign): `building` and
    # `edge` must come from the SAME scenario-initialized Building
    # scenario_runner.run() returns as `context.building` -- NOT the
    # shared template Building a caller may be reusing across many
    # scenarios. Only context.building has this scenario's actual
    # resolved door/exit blocked/locked state applied (scenario_runner.
    # building_initializer.build_initialized_building()); a pristine
    # template's Door/Exit objects never receive it, so Edge.traversable
    # (read straight off Edge.reference, see navigation/edge.py) would
    # silently always report the template's own default state instead
    # of this scenario's. See scripts/run_predictive_dataset_campaign_v1.py's
    # own comment at its TimelineRun(...) construction for the concrete
    # fix.
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
# "Currently approaching THIS CANDIDATE" -- occupants whose REMAINING
# (not-yet-reached) route includes this candidate: any edge later than
# the one they are currently queued for/on, who have already departed
# but not yet arrived as of `time`. NOT restricted to the immediate
# next hop, and NOT restricted to the route's final edge -- both were
# tried and both broke empirically (see BUG HISTORY below); this is a
# genuine "how much future demand pressure is headed toward this
# candidate" signal, matching this field's own documented semantic
# ("long-range demand signal known from route assignment at scenario
# start" -- see predictive_dataset/schema.py's own field description).
# Occupants already queued for or on the candidate's own edge right now
# are excluded (that overlap is what candidate_queue_length already
# reports -- double-counting the same occupant under both fields would
# make them redundant rather than two distinct demand signals).
#
# BUG HISTORY (Large-Scale Predictive Dataset Campaign & Validation
# milestone, Phase 4/5/8, all caught over a real 2.5M-row campaign,
# never anticipated by hand-built unit fixtures):
#
#   v1 ("route.edges[-1]", the route's FINAL edge): every complete
#   evacuation Route necessarily ends at an Exit (the only way to reach
#   Node.OUTSIDE_NODE_ID) -- STRUCTURALLY, PERMANENTLY zero for every
#   Door and Stair candidate. Caught by predictive_dataset/diversity.py's
#   candidate_utilization report (stair-1: 0% active across 501,696
#   rows).
#
#   v2 ("the immediate next un-started step"): generalizes correctly to
#   Door/Stair in principle, but this simulator's actual occupant
#   movement model (simulator/coordinator.py) transitions INSTANTLY
#   from departure (or finishing one edge) straight into joining the
#   next edge's queue -- verified directly against real simulation
#   output: every recorded OccupantTimelineStep has join_time ==
#   depart_time (first step) or join_time == the previous step's
#   end_time (later steps), with NO exceptions found. There is no
#   "walking toward but not yet queued" gap for this definition to ever
#   observe -- STRUCTURALLY, PERMANENTLY zero for every candidate type.
#   Caught by predictive_dataset/feature_statistics.py's own constant-
#   value detection (candidate_approaching_count: is_constant=True,
#   value 0.0, across all 2,508,480 rows).
#
# v3 (this one) has no such degenerate window: it only requires "this
# candidate appears somewhere in my remaining plan", which is routinely
# true the instant an occupant departs.
# =====================================================


def _current_approaching_count(movement_result, candidate_id: str, time: float) -> int:

    count = 0

    for timeline in movement_result.occupants.values():

        if timeline.route is None or not timeline.route.edges:
            continue

        if timeline.depart_time > time:
            continue

        if timeline.arrival_time is not None and timeline.arrival_time <= time:
            continue

        if candidate_id in _not_yet_reached_edge_ids(timeline, time):
            count += 1

    return count


def _not_yet_reached_edge_ids(timeline, time: float):

    # "Reached" = this occupant has, as of `time`, already joined the
    # queue for (or is already past) route.edges[0..reached_count-1] --
    # timeline.steps is ordered by OccupantTimelineStep.index, the same
    # order as route.edges, so walking it forward and stopping at the
    # first not-yet-joined step (time < join_time) gives exactly that
    # boundary. Everything at or after `reached_count` -- including any
    # step never even recorded yet for an occupant who got stuck
    # earlier in their route -- is "not yet reached".

    reached_count = 0

    for step in timeline.steps:

        join_time = step.start_time - step.queue_wait_time

        if time < join_time:
            break

        reached_count = step.index + 1

    return {edge.id for edge in timeline.route.edges[reached_count:]}
