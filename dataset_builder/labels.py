from typing import Any, Dict, List, Optional

from navigation.edge import Edge
from navigation.node import Node

from simulator.occupant import OccupantState

from behaviour_profile_resolver.category import OccupantCategory, occupant_category

from ground_truth import decision_events, human_behavior, rescue_outcomes

from dataset_builder.schema import ordered_exits, ordered_stairs, ordered_zones
from dataset_builder.statistics import mean, mode


# =====================================================
# Dataset 2: Simulation Outcomes (one row per scenario).
#
# Every value is read or aggregated from MultiAgentSimulationResult
# (the completed movement result already produced by the Simulation
# Runtime) plus Scenario/Building -- nothing here reruns or replays a
# simulation.
# =====================================================


def extract_simulation_outcome(run) -> Dict[str, Any]:

    scenario = run.scenario
    building = run.building
    movement_result = run.movement_result

    # Production Pipeline Completion -- civilian-scoped, mirroring
    # ground_truth.evacuation_metrics.compute_evacuation_metrics()'s own
    # identical fix and its own docstring's full reasoning: total_
    # occupants is len(scenario.occupants) alone, so every count that
    # must honestly partition it (people_evacuated/trapped, reachable/
    # unreachable) is restricted to civilian occupant_ids, never
    # including a firefighter entry movement_result.occupants may now
    # carry.
    civilian_ids = {occupant.occupant_id for occupant in scenario.occupants}

    total_occupants = len(scenario.occupants)
    timelines = {
        occupant_id: timeline
        for occupant_id, timeline in movement_result.occupants.items()
        if occupant_id in civilian_ids
    }

    arrived = [
        timeline
        for timeline in timelines.values()
        if timeline.state == OccupantState.ARRIVED
    ]

    unreachable_occupants = sum(
        1 for occupant_id in movement_result.unreachable_occupant_ids if occupant_id in civilian_ids
    )
    reachable_occupants = total_occupants - unreachable_occupants
    people_evacuated = len(arrived)
    people_trapped = total_occupants - people_evacuated

    return {
        "scenario_id": scenario.metadata.scenario_id,
        "total_evacuation_time": movement_result.total_evacuation_time,
        "average_evacuation_time": mean(timeline.arrival_time for timeline in arrived),
        "last_occupant_exit_time": movement_result.total_evacuation_time,
        "people_evacuated": people_evacuated,
        "people_trapped": people_trapped,
        "reachable_occupants": reachable_occupants,
        "unreachable_occupants": unreachable_occupants,
        "building_cleared": people_trapped == 0,
        "simulation_finished": run.simulation_finished,
        "maximum_queue_length": _maximum_queue_length(movement_result),
        "maximum_density": _maximum_density(building, movement_result),
        "maximum_congestion": _maximum_congestion(movement_result),
        "most_congested_exit": _most_congested_id(
            ordered_exits(building), movement_result.peak_edge_occupancy,
        ),
        "most_congested_stair": _most_congested_id(
            ordered_stairs(building), movement_result.peak_edge_occupancy,
        ),
        **_profile_category_counts(scenario),
        **_human_behavior_counts(run),
        "Firefighter_Count": len(scenario.firefighters),
        "Firefighter_Rescues": _firefighter_rescue_count(scenario, movement_result),
        **_decision_event_counts(run),
    }


# =====================================================
# Human Population & Assisted Evacuation Modeling Phase 6 -- new
# Dataset 2 columns. Profile category counts read straight off
# Scenario.occupants (behaviour_profile_resolver.category is the one
# place allowed to interpret behaviour_profile_id, same as every other
# consumer of it); Fallen/Possible_Injury/Helping/Assisted counts are
# ground_truth.human_behavior's own already-established derivation,
# reused rather than reimplemented here (see that module's own
# docstring for why this must be one shared computation, not two
# independently-evolving copies).
# =====================================================

_PROFILE_CATEGORY_COLUMNS = {
    OccupantCategory.ADULT: "Adult_Count",
    OccupantCategory.CHILD: "Child_Count",
    OccupantCategory.ELDERLY: "Elderly_Count",
    OccupantCategory.WHEELCHAIR_USER: "Wheelchair_Count",
    OccupantCategory.VISITOR: "Visitor_Count",
}


def _profile_category_counts(scenario) -> Dict[str, int]:

    counts = {column: 0 for column in _PROFILE_CATEGORY_COLUMNS.values()}

    for occupant in scenario.occupants:

        column = _PROFILE_CATEGORY_COLUMNS.get(occupant_category(occupant.behaviour_profile_id))

        if column is not None:
            counts[column] += 1

    return counts


# =====================================================


def _human_behavior_counts(run) -> Dict[str, int]:

    tick_results = getattr(run, "tick_results", ())

    records = human_behavior.compute_occupant_behavior(run.scenario, run.movement_result, tick_results)
    summary = human_behavior.summarize_behavior_counts(records)

    return {
        "Helping_Group_Count": summary["helping_group_count"],
        "Assisted_Occupant_Count": summary["assisted_occupant_count"],
        "Fallen_Count": summary["fallen_count"],
        "Possible_Injury_Count": summary["possible_injury_count"],
    }


# =====================================================


def _firefighter_rescue_count(scenario, movement_result) -> int:

    outcomes = rescue_outcomes.compute_rescue_outcomes(scenario, movement_result)

    return rescue_outcomes.summarize_rescue_outcomes(outcomes)["firefighter_rescue_count"]


# =====================================================
# Human Decision Engine Phase 6 -- new Dataset 2 columns. run.
# decision_events is the same optional, empty-by-default field
# SimulationRun.tick_results already establishes (see that dataclass's
# own docstring); every count is honestly 0 when it was never supplied.
# =====================================================


def _decision_event_counts(run) -> Dict[str, int]:

    events = getattr(run, "decision_events", ())
    summary = decision_events.summarize_decision_events(events)

    return {
        "Help_Decision_Count": summary["help_decision_count"],
        "Help_Rejected_Count": summary["help_rejected_count"],
        "Firefighter_Task_Count": summary["firefighter_task_count"],
        "Group_Formed_Count": summary["group_formed_count"],
        "Group_Dissolved_Count": summary["group_dissolved_count"],
        "Rescue_Initiated_Count": summary["rescue_initiated_count"],
        "Rescue_Completed_Count": summary["rescue_completed_count"],
    }


# =====================================================
# Dataset 3: Zone-Level Results (one row per zone per scenario).
# =====================================================


def extract_zone_results(run) -> List[Dict[str, Any]]:

    scenario = run.scenario
    building = run.building
    movement_result = run.movement_result

    timelines = movement_result.occupants
    occupants_by_id = {occupant.occupant_id: occupant for occupant in scenario.occupants}

    rows = []

    for zone in ordered_zones(building):

        origin_occupants = [
            occupant for occupant in scenario.occupants if occupant.zone_id == zone.id
        ]

        origin_timelines = [
            timelines[occupant.occupant_id]
            for occupant in origin_occupants
            if occupant.occupant_id in timelines
        ]

        evacuated_timelines = [
            timeline
            for timeline in origin_timelines
            if timeline.state == OccupantState.ARRIVED
        ]

        final_occupants = sum(
            1
            for timeline in timelines.values()
            if _current_location_zone_id(
                occupants_by_id.get(timeline.occupant_id), timeline,
            )
            == zone.id
        )

        exit_ids_used = [
            timeline.route.edges[-1].id
            for timeline in evacuated_timelines
            if timeline.route is not None
            and timeline.route.edges
            and timeline.route.edges[-1].edge_type == Edge.EXIT
        ]

        route_lengths = [
            timeline.route.total_distance
            for timeline in origin_timelines
            if timeline.route is not None and timeline.route.total_distance is not None
        ]

        rows.append(
            {
                "scenario_id": scenario.metadata.scenario_id,
                "zone_id": zone.id,
                "initial_occupants": len(origin_occupants),
                "final_occupants": final_occupants,
                "evacuated": len(evacuated_timelines),
                "trapped": len(origin_occupants) - len(evacuated_timelines),
                "average_delay": mean(
                    timeline.depart_time for timeline in origin_timelines
                ),
                "average_travel_time": mean(
                    timeline.arrival_time - timeline.depart_time
                    for timeline in evacuated_timelines
                    if timeline.arrival_time is not None
                ),
                "average_waiting_time": mean(
                    timeline.total_queue_wait_time for timeline in origin_timelines
                ),
                "exit_used": mode(exit_ids_used),
                "route_length": mean(route_lengths),
            }
        )

    return rows


# =====================================================


def _current_location_zone_id(occupant, timeline) -> Optional[str]:

    # Where an occupant physically ended up when the simulation ended,
    # derived only from data the movement result already carries -- an
    # ARRIVED occupant has left the building entirely (no zone); an
    # occupant with realized steps is wherever their last completed
    # step left them; an occupant who never departed (UNREACHABLE, or
    # STATIONARY with no steps) is still at their scenario-assigned
    # starting zone.

    if timeline.state == OccupantState.ARRIVED:
        return None

    if timeline.steps:

        last_node = timeline.steps[-1].to_node

        if last_node.node_type == Node.ZONE:
            return last_node.id

        return None

    return occupant.zone_id if occupant is not None else None


# =====================================================


def _maximum_queue_length(movement_result) -> int:

    # No per-tick queue-depth history is recorded anywhere upstream --
    # the closest available signal is how many occupant-steps, across
    # the whole run, had to wait at all for a given edge. The largest
    # such count, across every edge, is what is reported here.

    queue_counts: Dict[str, int] = {}

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            if step.queue_wait_time > 0:
                queue_counts[step.edge.id] = queue_counts.get(step.edge.id, 0) + 1

    if not queue_counts:
        return 0

    return max(queue_counts.values())


# =====================================================


def _maximum_density(building, movement_result) -> Optional[float]:

    densities = [
        movement_result.peak_node_occupancy.get(zone.id, 0) / zone.area
        for zone in ordered_zones(building)
        if zone.area
    ]

    if not densities:
        return None

    return max(densities)


# =====================================================


def _maximum_congestion(movement_result) -> Optional[int]:

    values = list(movement_result.peak_edge_occupancy.values()) + list(
        movement_result.peak_node_occupancy.values()
    )

    if not values:
        return None

    return max(values)


# =====================================================


def _most_congested_id(objects, peak_edge_occupancy) -> Optional[str]:

    best_id = None
    best_value = 0

    for obj in objects:

        value = peak_edge_occupancy.get(obj.id, 0)

        if value > best_value:
            best_value = value
            best_id = obj.id

    return best_id
