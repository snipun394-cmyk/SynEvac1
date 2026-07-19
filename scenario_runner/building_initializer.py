from models.building import Building

from scenario import DeviceAvailability, DoorState, PresenceState


# Building initialization -- architecture doc
# docs/architecture/scenario_runner.md §5. Edge.traversable
# (navigation/edge.py) reads Door/Exit state live off the Building's
# own engineering objects, with no pluggable override -- so applying
# a Scenario's resolved engineering state means writing it onto a
# Building's own fields, before NavigationGraphGenerator.build() ever
# runs (navigation_initializer.py). This module never touches the
# caller's original Building: build_initialized_building() always
# returns a fresh deep copy, mutated in place only on that copy.
#
# Stair is deliberately absent from this module -- models/staircase.py
# has no availability/blocked field at all, so a Scenario's
# stair_states cannot be applied as Building state the way the other
# five categories can. That is handled entirely in
# navigation_initializer.py, as a graph-level edge exclusion after the
# NavigationGraph is built, per §5's own documented exception.
#
# find_*()/apply_*_state() below are public -- architecture doc
# docs/architecture/scenario_event_execution.md §6's own compliance
# finding: the Scenario Event Executor needs to apply the exact same
# per-object field mapping this module already established for initial
# (t=0) state, one object at a time instead of once for every object.
# Exposing the single-object primitives here (rather than duplicating
# the same five trivial field assignments a second time in
# scenario_event_executor/) is a pure, additive factoring of this
# module's own existing logic -- _apply_*_states() below now call
# these same public functions internally, so build_initialized_building
# ()'s own behavior is completely unchanged.


def build_initialized_building(scenario, building: Building) -> Building:

    building_copy = Building.from_dict(building.to_dict())

    _apply_door_states(building_copy, scenario.door_states)
    _apply_exit_states(building_copy, scenario.exit_states)
    _apply_obstacle_states(building_copy, scenario.obstacle_states)
    _apply_camera_states(building_copy, scenario.camera_states)
    _apply_detector_states(building_copy, scenario.detector_states)

    return building_copy


# =====================================================
# Per-category id -> engineering object lookups. Public: the Event
# Executor needs to resolve one object at a time by id, the same way
# every other consumer of a Building resolves an id.
# =====================================================


def find_door(building, door_id):
    return _doors_by_id(building).get(door_id)


def find_exit(building, exit_id):
    return _exits_by_id(building).get(exit_id)


def find_obstacle(building, obstacle_id):
    return _obstacles_by_id(building).get(obstacle_id)


def find_camera(building, camera_id):
    return _cameras_by_id(building).get(camera_id)


def find_detector(building, detector_id):
    return _detectors_by_id(building).get(detector_id)


def _doors_by_id(building):
    return {door.id: door for floor in building.floors for door in floor.doors}


def _exits_by_id(building):
    return {exit_obj.id: exit_obj for floor in building.floors for exit_obj in floor.exits}


def _obstacles_by_id(building):
    return {obs.id: obs for floor in building.floors for obs in floor.obstacles}


def _cameras_by_id(building):
    return {cam.id: cam for floor in building.floors for cam in floor.cameras}


def _detectors_by_id(building):
    return {det.id: det for floor in building.floors for det in floor.detectors}


# =====================================================
# Single-object state setters -- the one place each category's
# resolved-value-to-field mapping is defined. Both this module's own
# _apply_*_states() (below) and scenario_event_executor's handlers
# call these, never restating the mapping themselves.
# =====================================================


def apply_door_state(door, state: DoorState) -> None:

    door.locked = state == DoorState.LOCKED
    door.normally_open = state == DoorState.OPEN


def apply_exit_state(exit_obj, is_open: bool) -> None:

    exit_obj.is_blocked = not is_open


def apply_obstacle_state(obstacle, presence: PresenceState) -> None:

    obstacle.active = presence == PresenceState.ACTIVE


def apply_camera_state(camera, availability: DeviceAvailability) -> None:

    camera.active = availability == DeviceAvailability.AVAILABLE


def apply_detector_state(detector, availability: DeviceAvailability) -> None:

    detector.active = availability == DeviceAvailability.AVAILABLE


# =====================================================
# Applying resolved Scenario state onto the copy. An id present in a
# Scenario's states but absent from the Building copy is silently
# skipped -- not this package's job to validate that a Scenario's
# referenced ids exist (§1: "not a validator"; that already happened,
# upstream, in scenario_validator's Building Validation).
# =====================================================


def _apply_door_states(building_copy, door_states):

    for state in door_states:

        door = find_door(building_copy, state.door_id)

        if door is None:
            continue

        apply_door_state(door, state.state)


def _apply_exit_states(building_copy, exit_states):

    for state in exit_states:

        exit_obj = find_exit(building_copy, state.exit_id)

        if exit_obj is None:
            continue

        apply_exit_state(exit_obj, state.is_open)


def _apply_obstacle_states(building_copy, obstacle_states):

    for state in obstacle_states:

        obstacle = find_obstacle(building_copy, state.obstacle_id)

        if obstacle is None:
            continue

        apply_obstacle_state(obstacle, state.presence)


def _apply_camera_states(building_copy, camera_states):

    for state in camera_states:

        camera = find_camera(building_copy, state.camera_id)

        if camera is None:
            continue

        apply_camera_state(camera, state.availability)


def _apply_detector_states(building_copy, detector_states):

    for state in detector_states:

        detector = find_detector(building_copy, state.detector_id)

        if detector is None:
            continue

        apply_detector_state(detector, state.availability)
