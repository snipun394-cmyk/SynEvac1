from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from models.building import Building
from scenario.engineering_state import DeviceAvailability, DoorState, StairAvailability
from scenario.scenario import Scenario
from simulation_runtime.result import TickResult
from simulator.multi_agent_result import MultiAgentSimulationResult

from dataset_builder.schema import (
    camera_state_columns,
    detector_state_columns,
    door_state_columns,
    exit_state_columns,
    ordered_cameras,
    ordered_detectors,
    ordered_doors,
    ordered_exits,
    ordered_stairs,
    ordered_zones,
    stair_state_columns,
    zone_occupancy_columns,
)
from dataset_builder.timeline_schema import (
    zone_smoke_columns,
    zone_temperature_columns,
    zone_visibility_columns,
)


@dataclass(frozen=True)
class TimelineRun:

    # Dataset Builder V1's SimulationRun plus the one additional piece
    # a per-timestep dataset needs: the ordered TickResults the same
    # completed SimulationRuntime run already produced. Deliberately a
    # separate class rather than a field added to
    # dataset_builder.builder.SimulationRun -- V1's API is frozen.

    scenario: Scenario
    building: Building
    movement_result: MultiAgentSimulationResult
    tick_results: Tuple[TickResult, ...]


# =====================================================
# Dataset Builder V2: Timeline (one row per simulation timestep).
#
# Every column is read directly off Scenario/Building/
# MultiAgentSimulationResult/TickResult, or is a plain arithmetic
# rollup over them (a running max, a per-tick headcount, ...) -- this
# module never reruns, replans, or estimates anything a completed
# SimulationRuntime did not already decide. In particular, per-tick
# engineering state (Door/Exit/Stair/Detector/Camera) is reconstructed
# by replaying TickResult.fired_events forward from the same baseline
# V1 uses (Scenario override, else Building baseline) -- fired_events
# is itself already-recorded output of the completed run (see
# ScenarioEventExecutor.advance_to()), not new simulated behaviour.
# =====================================================


def extract_timeline_rows(run: TimelineRun) -> List[Dict[str, Any]]:

    scenario = run.scenario
    building = run.building
    movement_result = run.movement_result

    total_occupants = len(scenario.occupants)
    trapped_occupants = len(movement_result.unreachable_occupant_ids)

    zones = ordered_zones(building)
    doors = ordered_doors(building)
    exits = ordered_exits(building)
    stairs = ordered_stairs(building)
    detectors = ordered_detectors(building)
    cameras = ordered_cameras(building)

    zone_occupancy_cols = zone_occupancy_columns(building)
    zone_smoke_cols = zone_smoke_columns(building)
    zone_temperature_cols = zone_temperature_columns(building)
    zone_visibility_cols = zone_visibility_columns(building)

    door_cols = door_state_columns(building)
    exit_cols = exit_state_columns(building)
    stair_cols = stair_state_columns(building)
    detector_cols = detector_state_columns(building)
    camera_cols = camera_state_columns(building)

    engineering_state = _initial_engineering_state(building, scenario)

    ignition_zone = scenario.fire.ignition_zone_id if scenario.fire is not None else None

    rows: List[Dict[str, Any]] = []

    running_max_congestion = 0
    previous_time = 0.0
    previous_evacuated = 0

    for tick in run.tick_results:

        for event in tick.fired_events:
            _apply_event(engineering_state, event)

        hazard_snapshot = tick.hazard_snapshot
        occupancy_snapshot = tick.occupancy_snapshot

        row: Dict[str, Any] = {}

        row["scenario_id"] = scenario.metadata.scenario_id
        row["simulation_time"] = tick.time

        row["ignition_zone"] = ignition_zone

        # One node_state() read per zone this tick, shared across every
        # column below -- node_state() is a pure function of
        # (hazard_snapshot, zone.id), so re-reading it separately for
        # hazard_score/smoke_level/temperature/visibility (as this loop
        # used to) recomputed the identical lookup four times per zone.
        zone_hazard_states = {zone.id: hazard_snapshot.node_state(zone.id) for zone in zones}

        row["current_fire_zones"] = ";".join(
            zone_id for zone_id, state in zone_hazard_states.items() if state.hazard_score > 0
        )
        row["current_hazard_score"] = (
            max(state.hazard_score for state in zone_hazard_states.values())
            if zone_hazard_states else None
        )

        for column, zone in zip(zone_occupancy_cols, zones):
            row[column] = occupancy_snapshot.observation_at(zone.id).occupant_count

        for column, zone in zip(zone_smoke_cols, zones):
            row[column] = zone_hazard_states[zone.id].smoke_level

        for column, zone in zip(zone_temperature_cols, zones):
            row[column] = zone_hazard_states[zone.id].temperature

        for column, zone in zip(zone_visibility_cols, zones):
            row[column] = zone_hazard_states[zone.id].visibility

        for column, door in zip(door_cols, doors):
            row[column] = engineering_state["door"][door.id].name

        for column, exit_obj in zip(exit_cols, exits):
            row[column] = "OPEN" if engineering_state["exit"][exit_obj.id] else "CLOSED"

        for column, stair in zip(stair_cols, stairs):
            row[column] = engineering_state["stair"][stair.id].name

        for column, detector in zip(detector_cols, detectors):
            row[column] = engineering_state["detector"][detector.id].name

        for column, camera in zip(camera_cols, cameras):
            row[column] = engineering_state["camera"][camera.id].name

        evacuated_so_far = sum(
            1
            for timeline in movement_result.occupants.values()
            if timeline.arrival_time is not None and timeline.arrival_time <= tick.time
        )

        row["people_evacuated"] = evacuated_so_far
        row["people_remaining"] = total_occupants - evacuated_so_far
        row["people_trapped"] = trapped_occupants

        current_congestion = _current_congestion(movement_result, occupancy_snapshot, tick.time)
        running_max_congestion = max(running_max_congestion, current_congestion or 0)

        row["current_congestion"] = current_congestion
        row["maximum_congestion_so_far"] = running_max_congestion

        elapsed = tick.time - previous_time
        row["current_evacuation_rate"] = (
            (evacuated_so_far - previous_evacuated) / elapsed if elapsed > 0 else 0.0
        )

        row["current_queue_length"] = _current_queue_length(movement_result, tick.time)
        row["current_route_utilization"] = _current_route_utilization(
            movement_result, tick.time,
        )
        row["current_bottleneck"] = _current_bottleneck(movement_result, tick.time)

        # No non-invented derivation of "hazard frontier" exists from
        # the data available here (it would require Building adjacency
        # reasoning this package has no other need for) -- left blank
        # rather than guessed at, per this package's own "never
        # fabricate" contract.
        row["current_hazard_frontier"] = None

        rows.append(row)

        previous_time = tick.time
        previous_evacuated = evacuated_so_far

    return rows


# =====================================================
# Engineering-state replay.
#
# Starts from exactly the same baseline Dataset Builder V1 uses
# (Scenario override for an object id, else that object's own Building
# baseline field), then applies every fired ScenarioEvent in the order
# TickResult.fired_events already delivers them (chronological, since
# ScenarioEventExecutor.advance_to() only ever fires each scheduled
# event once, in schedule order). event.parameters' shape for each
# target_type mirrors scenario_event_executor/handlers.py exactly
# (read, not reimplemented: this module never decides *whether* an
# event fires, only records what already did).
# =====================================================


def _initial_engineering_state(building: Building, scenario: Scenario) -> Dict[str, Dict[str, Any]]:

    door_state = {}
    for door in ordered_doors(building):

        if door.locked:
            door_state[door.id] = DoorState.LOCKED
        elif not door.active:
            door_state[door.id] = DoorState.CLOSED
        else:
            door_state[door.id] = DoorState.OPEN

    for state in scenario.door_states:
        door_state[state.door_id] = state.state

    exit_open = {
        exit_obj.id: not exit_obj.is_blocked for exit_obj in ordered_exits(building)
    }
    for state in scenario.exit_states:
        exit_open[state.exit_id] = state.is_open

    stair_availability = {
        stair.id: StairAvailability.AVAILABLE for stair in ordered_stairs(building)
    }
    for state in scenario.stair_states:
        stair_availability[state.stair_id] = state.availability

    detector_availability = {
        detector.id: (
            DeviceAvailability.AVAILABLE if detector.active else DeviceAvailability.FAILED
        )
        for detector in ordered_detectors(building)
    }
    for state in scenario.detector_states:
        detector_availability[state.detector_id] = state.availability

    camera_availability = {
        camera.id: (
            DeviceAvailability.AVAILABLE if camera.active else DeviceAvailability.FAILED
        )
        for camera in ordered_cameras(building)
    }
    for state in scenario.camera_states:
        camera_availability[state.camera_id] = state.availability

    return {
        "door": door_state,
        "exit": exit_open,
        "stair": stair_availability,
        "detector": detector_availability,
        "camera": camera_availability,
    }


# =====================================================


def _apply_event(engineering_state: Dict[str, Dict[str, Any]], event) -> None:

    if event.target_type == "door":
        engineering_state["door"][event.target_id] = DoorState[event.parameters["state"]]

    elif event.target_type == "exit":
        engineering_state["exit"][event.target_id] = bool(event.parameters["is_open"])

    elif event.target_type == "stair":
        engineering_state["stair"][event.target_id] = StairAvailability[
            event.parameters["availability"]
        ]

    elif event.target_type == "detector":
        engineering_state["detector"][event.target_id] = DeviceAvailability[
            event.parameters["availability"]
        ]

    elif event.target_type == "camera":
        engineering_state["camera"][event.target_id] = DeviceAvailability[
            event.parameters["availability"]
        ]

    # Obstacle events are silently ignored -- the required Timeline
    # column set (per the spec this module implements) has no
    # Obstacle_N_State column, unlike Dataset Builder V1's Scenario
    # Features.


# =====================================================
# Per-tick congestion/queueing/utilization -- all reconstructed from
# OccupantTimelineStep.start_time/end_time/queue_wait_time, the only
# place a "who is where, and since when" signal exists at sub-run
# granularity.
# =====================================================


def _current_congestion(movement_result, occupancy_snapshot, time: float) -> Optional[int]:

    node_values = [
        observation.occupant_count
        for observation in occupancy_snapshot.observations.values()
        if observation.occupant_count is not None
    ]

    edge_counts: Dict[str, int] = {}
    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            if step.start_time <= time <= step.end_time:
                edge_counts[step.edge.id] = edge_counts.get(step.edge.id, 0) + 1

    values = node_values + list(edge_counts.values())

    if not values:
        return None

    return max(values)


# =====================================================


def _current_queue_length(movement_result, time: float) -> int:

    # An occupant is queued for [start_time - queue_wait_time,
    # start_time) -- the interval between joining an edge's queue and
    # actually being admitted onto it (see simulator/coordinator.py's
    # _admit_onto_edge(), the only place queue_wait_time is computed).

    count = 0

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:

            join_time = step.start_time - step.queue_wait_time

            if join_time <= time < step.start_time:
                count += 1

    return count


# =====================================================


def _current_route_utilization(movement_result, time: float) -> Optional[float]:

    # Only edges with a modeled capacity (Exit.capacity -- Door/
    # Staircase have none, see navigation/edge.py::Edge.capacity)
    # produce a meaningful ratio; edges without one are excluded
    # rather than assigned a guessed capacity.

    edge_counts: Dict[str, int] = {}
    edge_capacity: Dict[str, Any] = {}

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            if step.start_time <= time <= step.end_time:
                edge_counts[step.edge.id] = edge_counts.get(step.edge.id, 0) + 1
                edge_capacity[step.edge.id] = step.edge.capacity

    utilizations = [
        edge_counts[edge_id] / edge_capacity[edge_id]
        for edge_id in edge_counts
        if edge_capacity.get(edge_id)
    ]

    if not utilizations:
        return None

    return max(utilizations)


# =====================================================


def _current_bottleneck(movement_result, time: float) -> Optional[str]:

    edge_counts: Dict[str, int] = {}

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            if step.start_time <= time <= step.end_time:
                edge_counts[step.edge.id] = edge_counts.get(step.edge.id, 0) + 1

    if not edge_counts:
        return None

    return max(edge_counts, key=edge_counts.get)
