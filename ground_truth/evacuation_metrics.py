from typing import Any, Dict, List, Optional

from navigation.edge import Edge

from simulator.occupant import OccupantState


# =====================================================
# Evacuation summary -- one row's worth of top-level outcome numbers,
# read or aggregated directly from MultiAgentSimulationResult. No
# rerun, no estimate: an occupant either has state ARRIVED in the
# already-completed result or it doesn't.
# =====================================================


def compute_evacuation_metrics(scenario, movement_result) -> Dict[str, Any]:

    # Production Pipeline Completion -- civilian-scoped. total_occupants
    # is, and always was, len(scenario.occupants) alone (never
    # Scenario.firefighters); movement_result.occupants, however, now
    # legitimately carries firefighter entries too once a caller
    # registers them (behaviour_profile_resolver.register_population/
    # register_population_dynamic -- see those modules' own docstrings).
    # people_evacuated/people_trapped is only an honest partition of
    # total_occupants when both sides of that arithmetic are scoped to
    # the exact same population, so every lookup below is restricted to
    # civilian occupant_ids -- a firefighter's own outcome is never
    # counted as an evacuated/trapped *occupant* here (it has no
    # meaning against a total that never included them), the same
    # "count exactly the population the total already represents"
    # discipline ground_truth.profile_outcomes.compute_profile_outcomes()
    # already establishes by iterating scenario.occupants first.

    civilian_ids = {occupant.occupant_id for occupant in scenario.occupants}

    total_occupants = len(scenario.occupants)
    timelines = {
        occupant_id: timeline
        for occupant_id, timeline in movement_result.occupants.items()
        if occupant_id in civilian_ids
    }

    arrived = [
        timeline for timeline in timelines.values() if timeline.state == OccupantState.ARRIVED
    ]

    unreachable_occupants = sum(
        1 for occupant_id in movement_result.unreachable_occupant_ids if occupant_id in civilian_ids
    )
    reachable_occupants = total_occupants - unreachable_occupants
    people_evacuated = len(arrived)
    people_trapped = total_occupants - people_evacuated

    return {
        "total_evacuation_time": movement_result.total_evacuation_time,
        "building_cleared": people_trapped == 0,
        "reachable_occupants": reachable_occupants,
        "unreachable_occupants": unreachable_occupants,
        "people_trapped": people_trapped,
        "people_evacuated": people_evacuated,
    }


# =====================================================
# Per-zone route statistics -- "preferred" exit/stair reads each
# zone-origin occupant's own already-fixed Route (decided once at
# registration, never replanned -- see simulator/occupant.py), not
# just those who happened to finish, since the assigned route is
# already the engineering-relevant "where were they supposed to go"
# signal regardless of whether hazard/congestion later stopped them.
# =====================================================


def compute_zone_route_stats(scenario, movement_result, zones) -> List[Dict[str, Any]]:

    timelines = movement_result.occupants
    rows = []

    for zone in zones:

        origin_occupants = [
            occupant for occupant in scenario.occupants if occupant.zone_id == zone.id
        ]

        origin_timelines = [
            timelines[occupant.occupant_id]
            for occupant in origin_occupants
            if occupant.occupant_id in timelines
        ]

        exit_ids = [
            timeline.route.edges[-1].id
            for timeline in origin_timelines
            if timeline.route is not None
            and timeline.route.edges
            and timeline.route.edges[-1].edge_type == Edge.EXIT
        ]

        stair_ids = [
            edge.id
            for timeline in origin_timelines
            if timeline.route is not None
            for edge in timeline.route.edges
            if edge.edge_type == Edge.STAIR
        ]

        distances = [
            timeline.route.total_distance
            for timeline in origin_timelines
            if timeline.route is not None and timeline.route.total_distance is not None
        ]

        travel_times = [
            timeline.arrival_time - timeline.depart_time
            for timeline in origin_timelines
            if timeline.state == OccupantState.ARRIVED and timeline.arrival_time is not None
        ]

        rows.append(
            {
                "zone_id": zone.id,
                "preferred_exit": _mode(exit_ids),
                "preferred_stair": _mode(stair_ids),
                "average_travel_distance": _mean(distances),
                "average_travel_time": _mean(travel_times),
            }
        )

    return rows


# =====================================================


def _mean(values) -> Optional[float]:

    values = [value for value in values if value is not None]

    if not values:
        return None

    return sum(values) / len(values)


def _mode(values):

    # Most frequent non-None value, ties broken by first occurrence --
    # deterministic for a given input order.

    values = [value for value in values if value is not None]

    if not values:
        return None

    counts: Dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    best_count = max(counts.values())

    for value in values:
        if counts[value] == best_count:
            return value

    return None
