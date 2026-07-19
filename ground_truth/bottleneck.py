from typing import Any, Dict, List, Optional

from navigation.edge import Edge

from simulator.capacity import derive_stair_capacity
from simulator.occupant import OccupantState


# =====================================================
# Congestion -- worst single object per type, plus the single overall
# peak location (across zones/doors/exits/stairs) and how long that
# location stayed congested. Everything here reads
# MultiAgentSimulationResult.peak_edge_occupancy/peak_node_occupancy
# (already-computed peaks) or OccupantTimelineStep.start_time/end_time
# (already-realized step intervals) -- nothing is resimulated.
# =====================================================


def compute_congestion(movement_result, zones, doors, exits, stairs) -> Dict[str, Any]:

    worst_exit = _worst_by_peak_occupancy(exits, movement_result.peak_edge_occupancy)
    worst_stair = _worst_by_peak_occupancy(stairs, movement_result.peak_edge_occupancy)
    worst_door = _worst_by_peak_occupancy(doors, movement_result.peak_edge_occupancy)

    candidates = []

    for zone in zones:
        value = movement_result.peak_node_occupancy.get(zone.id, 0)
        if value > 0:
            candidates.append((value, "zone", zone.id))

    for door in doors:
        value = movement_result.peak_edge_occupancy.get(door.id, 0)
        if value > 0:
            candidates.append((value, "door", door.id))

    for exit_obj in exits:
        value = movement_result.peak_edge_occupancy.get(exit_obj.id, 0)
        if value > 0:
            candidates.append((value, "exit", exit_obj.id))

    for stair in stairs:
        value = movement_result.peak_edge_occupancy.get(stair.id, 0)
        if value > 0:
            candidates.append((value, "stair", stair.id))

    peak_location_id = None
    peak_location_type = None
    peak_value = None
    duration = None

    if candidates:

        # max() on tuples ties on the first differing element -- with
        # value first, the earliest-appended (zone, then door, then
        # exit, then stair, each in Building enumeration order)
        # candidate wins any tie, making this deterministic.
        peak_value, peak_location_type, peak_location_id = max(candidates, key=lambda c: c[0])

        if peak_location_type != "zone":
            duration = _congestion_duration_for_edge(movement_result, peak_location_id)

    return {
        "worst_exit": worst_exit,
        "worst_stair": worst_stair,
        "worst_door": worst_door,
        "peak_congestion_location_id": peak_location_id,
        "peak_congestion_location_type": peak_location_type,
        "peak_congestion_value": peak_value,
        "congestion_duration": duration,
    }


# =====================================================


def _worst_by_peak_occupancy(objects, peak_edge_occupancy) -> Optional[str]:

    best_id = None
    best_value = 0

    for obj in objects:

        value = peak_edge_occupancy.get(obj.id, 0)

        if value > best_value:
            best_value = value
            best_id = obj.id

    return best_id


# =====================================================


def _congestion_duration_for_edge(movement_result, edge_id, threshold: int = 2) -> float:

    # "Congested" is defined here as 2+ occupants concurrently on the
    # same edge (a disclosed, fixed policy threshold, not a fabricated
    # data value) -- computed with a standard interval sweep over each
    # occupant-step's already-recorded [start_time, end_time] on this
    # edge.

    events = []

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            if step.edge.id == edge_id:
                events.append((step.start_time, 1))
                events.append((step.end_time, -1))

    if not events:
        return 0.0

    events.sort()

    duration = 0.0
    count = 0
    previous_time = None

    for time, delta in events:

        if previous_time is not None and count >= threshold:
            duration += time - previous_time

        count += delta
        previous_time = time

    return duration


# =====================================================
# Engineering findings -- discrete, ruleset-based conclusions about
# which doors/exits/stairs need attention.
# =====================================================


def compute_engineering_findings(movement_result, doors, exits, stairs) -> Dict[str, Any]:

    doors_bottleneck = [
        door.id for door in doors if _has_any_queueing(movement_result, door.id)
    ]

    exit_usage = _exit_usage_counts(movement_result, exits)

    exits_underutilized = [
        exit_obj.id for exit_obj in exits if exit_usage.get(exit_obj.id, 0) == 0
    ]

    exits_exceeding_capacity = [
        exit_obj.id
        for exit_obj in exits
        if exit_obj.capacity
        and movement_result.peak_edge_occupancy.get(exit_obj.id, 0) > exit_obj.capacity
    ]

    # Staircase has no explicit capacity field of its own (models/
    # staircase.py has width but no capacity, and navigation/
    # edge.py::Edge.capacity reads straight off that field so it's
    # None for a Stair edge too) -- but simulator.capacity's
    # derive_stair_capacity() is the same width-plus-vertical-travel
    # formula StairCapacityModel applies during simulation, so it is
    # reused here rather than inventing a second number.
    stairs_exceeding_capacity: List[str] = [
        stair.id
        for stair in stairs
        if movement_result.peak_edge_occupancy.get(stair.id, 0)
        > derive_stair_capacity(stair.width, _edge_walking_distance(movement_result, stair.id))
    ]

    return {
        "doors_that_became_bottlenecks": doors_bottleneck,
        "exits_underutilized": exits_underutilized,
        "exits_exceeding_capacity": exits_exceeding_capacity,
        "stairs_exceeding_capacity": stairs_exceeding_capacity,
    }


# =====================================================


def _has_any_queueing(movement_result, edge_id) -> bool:

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            if step.edge.id == edge_id and step.queue_wait_time > 0:
                return True

    return False


# =====================================================


def _edge_walking_distance(movement_result, edge_id) -> float:

    # Edge.walking_distance for a Stair is already derived from the
    # Building's floor heights (see navigation/graph_builder.py) --
    # reading it back off an already-recorded step's edge reuses that
    # value directly rather than needing a Building passed in here.

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            if step.edge.id == edge_id:
                return step.edge.walking_distance or 0.0

    return 0.0


# =====================================================


def _exit_usage_counts(movement_result, exits) -> Dict[str, int]:

    exit_ids = {exit_obj.id for exit_obj in exits}
    counts: Dict[str, int] = {}

    for timeline in movement_result.occupants.values():

        if (
            timeline.state != OccupantState.ARRIVED
            or timeline.route is None
            or not timeline.route.edges
        ):
            continue

        last_edge = timeline.route.edges[-1]

        if last_edge.edge_type == Edge.EXIT and last_edge.id in exit_ids:
            counts[last_edge.id] = counts.get(last_edge.id, 0) + 1

    return counts
