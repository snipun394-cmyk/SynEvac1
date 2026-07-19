from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

from dataset_builder.schema import (
    ordered_cameras,
    ordered_detectors,
    ordered_doors,
    ordered_exits,
    ordered_stairs,
)

from navigation.edge import Edge

from scenario_runner.context import SimulationContext

from simulator.occupant import OccupantState

from behaviour_profile_resolver.category import occupant_category

from simulation_interactive.movement_stepper import MovementStepper


# =====================================================
# Honesty discipline: every field here is derived purely from state
# an interactive run has already decided by `time` -- current occupant
# state/position, the current (possibly already-mutated) engineering
# Building fields, the current hazard snapshot, and live coordinator
# congestion/queue bookkeeping. Nothing here re-simulates, replans, or
# estimates, mirroring dataset_builder/timeline.py's own "never
# fabricate" contract (its current_hazard_frontier is left None rather
# than guessed; this module follows the same principle by simply not
# including a field it cannot honestly derive).
# =====================================================


@dataclass(frozen=True)
class OccupantSnapshot:

    occupant_id: str
    state: str
    current_node_id: Optional[str]
    depart_time: float
    arrival_time: Optional[float]

    # Human Population & Assisted Evacuation Modeling Phase 8 --
    # additive. profile_category is behaviour_profile_resolver.
    # category.OccupantCategory's own .name string (e.g. "CHILD",
    # "WHEELCHAIR_USER", "FIREFIGHTER" for a firefighter -- see
    # build_snapshot() below), resolved once, here, from Scenario data
    # already in context -- the one, honest way rl_training.
    # observation_space can read a profile-aware aggregate without
    # itself importing behaviour_profile_resolver (a forbidden import
    # there, see tests.test_rl_training.IndependenceTests). Every
    # dynamic-behavior state (Fallen/Possible Injury/...) is
    # deliberately NOT exposed here -- ground_truth.human_behavior's
    # own derivation requires a *completed* MultiAgentSimulationResult
    # plus its full tick_results history, neither of which a still-
    # running, step-driven InteractiveSimulation has; exposing a
    # partial or reimplemented version of that derivation here would
    # risk silently diverging from the one true implementation.
    profile_category: str = "UNKNOWN"


@dataclass(frozen=True)
class SimulationSnapshot:

    time: float
    occupants: Tuple[OccupantSnapshot, ...]
    hazard_snapshot: object
    door_states: Mapping[str, str]
    exit_states: Mapping[str, str]
    stair_states: Mapping[str, str]
    detector_states: Mapping[str, str]
    camera_states: Mapping[str, str]
    edge_congestion: Mapping[str, int]
    edge_queue_lengths: Mapping[str, int]
    node_occupancy: Mapping[str, int]
    people_evacuated: int
    people_remaining: int


# =====================================================


def _door_state_name(door) -> str:

    if door.locked:
        return "LOCKED"

    if door.normally_open:
        return "OPEN"

    return "CLOSED"


def _exit_state_name(exit_obj) -> str:

    return "CLOSED" if exit_obj.is_blocked else "OPEN"


def _stair_state_name(stair, graph) -> str:

    is_included = any(
        edge.edge_type == Edge.STAIR and edge.id == stair.id for edge in graph.edges
    )

    return "AVAILABLE" if is_included else "CLOSED"


def _device_state_name(device) -> str:

    return "AVAILABLE" if device.active else "FAILED"


# =====================================================


def build_snapshot(
    context: SimulationContext, stepper: MovementStepper, hazard_snapshot, time: float,
) -> SimulationSnapshot:

    result = stepper.snapshot_result()

    # occupant_id -> behaviour_profile_id, from Scenario data already
    # in context (civilians and firefighters both -- see
    # OccupantSnapshot.profile_category's own docstring for why this is
    # resolved here rather than by rl_training itself).
    profile_id_by_occupant_id = {
        occupant.occupant_id: occupant.behaviour_profile_id for occupant in context.occupants
    }
    profile_id_by_occupant_id.update(
        {firefighter.firefighter_id: firefighter.behaviour_profile_id
         for firefighter in context.firefighters}
    )

    occupants = tuple(
        OccupantSnapshot(
            occupant_id=occupant_id,
            state=timeline.state.name,
            current_node_id=stepper.occupant_current_node_id(occupant_id),
            depart_time=timeline.depart_time,
            arrival_time=timeline.arrival_time,
            profile_category=occupant_category(
                profile_id_by_occupant_id.get(occupant_id, ""),
            ).name,
        )
        for occupant_id, timeline in result.occupants.items()
    )

    people_evacuated = sum(1 for o in occupants if o.state == OccupantState.ARRIVED.name)
    people_remaining = len(occupants) - people_evacuated

    door_states = {
        door.id: _door_state_name(door) for door in ordered_doors(context.building)
    }
    exit_states = {
        exit_obj.id: _exit_state_name(exit_obj) for exit_obj in ordered_exits(context.building)
    }
    stair_states = {
        stair.id: _stair_state_name(stair, context.graph)
        for stair in ordered_stairs(context.building)
    }
    detector_states = {
        detector.id: _device_state_name(detector)
        for detector in ordered_detectors(context.building)
    }
    camera_states = {
        camera.id: _device_state_name(camera) for camera in ordered_cameras(context.building)
    }

    edge_congestion = {
        edge.id: stepper.current_edge_occupancy(edge.id) for edge in context.graph.edges
    }
    edge_queue_lengths = {
        edge.id: stepper.current_queue_length(edge.id) for edge in context.graph.edges
    }
    node_occupancy = {
        node_id: stepper.current_node_occupancy(node_id) for node_id in context.graph.nodes
    }

    return SimulationSnapshot(
        time=time,
        occupants=occupants,
        hazard_snapshot=hazard_snapshot,
        door_states=door_states,
        exit_states=exit_states,
        stair_states=stair_states,
        detector_states=detector_states,
        camera_states=camera_states,
        edge_congestion=edge_congestion,
        edge_queue_lengths=edge_queue_lengths,
        node_occupancy=node_occupancy,
        people_evacuated=people_evacuated,
        people_remaining=people_remaining,
    )
