from scenario import DeviceAvailability, DoorState, PresenceState, StairAvailability

from scenario_runner import (
    apply_camera_state,
    apply_detector_state,
    apply_door_state,
    apply_exit_state,
    apply_obstacle_state,
    exclude_stair_edge,
    find_camera,
    find_detector,
    find_door,
    find_exit,
    find_obstacle,
    include_stair_edge,
)


# Per-target_type event application -- architecture doc
# docs/architecture/scenario_event_execution.md §6/§17: a pluggable
# {target_type: handler} dispatch table, not six special cases baked
# into the Executor's own advancement loop. Every handler below reuses
# scenario_runner's own public find_*()/apply_*_state()/
# exclude_stair_edge()/include_stair_edge() primitives verbatim -- no
# handler restates the door/exit/obstacle/camera/detector/stair field
# mapping a second time (§6's own compliance finding, resolved here by
# reuse, not duplication).
#
# `event.parameters` is where the resolved target value lives --
# event_type stays descriptive/provenance only, exactly as
# scenario_validator's own Event Validation already treats it. An id
# a handler cannot find on context.building/context.graph is silently
# skipped, not raised -- the identical "already validated upstream by
# scenario_validator's Building Validation, trust the input" behavior
# find_door()/etc. and _apply_door_states() themselves already use in
# scenario_runner (§1: "not a validator").
#
# "Future AI actions"/"Future firefighter actions" are NOT registered
# here -- deliberately (§17: no target_type/state-category convention
# exists for either anywhere in the frozen scenario/scenario_definition
# schema). An event.target_type with no registered handler is a hard
# error (UnsupportedEventTargetTypeError), not a silent skip -- a
# missing *capability* is a different failure mode from a missing
# *object* (the case above), and must not be masked the same way.


class UnsupportedEventTargetTypeError(LookupError):

    # Raised, never silently skipped -- mirrors
    # behaviour_profile_resolver.UnknownBehaviourProfileError's own
    # "no repair, anywhere" precedent: an event.target_type with no
    # registered handler is a genuine capability gap, not something
    # this Executor can paper over.

    pass


def _handle_door_event(context, event) -> None:

    door = find_door(context.building, event.target_id)

    if door is None:
        return

    apply_door_state(door, DoorState[event.parameters["state"]])


def _handle_exit_event(context, event) -> None:

    exit_obj = find_exit(context.building, event.target_id)

    if exit_obj is None:
        return

    apply_exit_state(exit_obj, bool(event.parameters["is_open"]))


def _handle_obstacle_event(context, event) -> None:

    obstacle = find_obstacle(context.building, event.target_id)

    if obstacle is None:
        return

    apply_obstacle_state(obstacle, PresenceState[event.parameters["presence"]])


def _handle_camera_event(context, event) -> None:

    camera = find_camera(context.building, event.target_id)

    if camera is None:
        return

    apply_camera_state(camera, DeviceAvailability[event.parameters["availability"]])


def _handle_detector_event(context, event) -> None:

    detector = find_detector(context.building, event.target_id)

    if detector is None:
        return

    apply_detector_state(detector, DeviceAvailability[event.parameters["availability"]])


def _handle_stair_event(context, event) -> None:

    availability = StairAvailability[event.parameters["availability"]]

    if availability == StairAvailability.CLOSED:
        exclude_stair_edge(context.graph, event.target_id)
    else:
        include_stair_edge(context.graph, context.building, event.target_id)


# The dispatch table -- architecture doc §17's own extension point.
# Adding a future target_type is adding one entry here; it never
# requires changing executor.py's own advancement loop.
EVENT_HANDLERS = {
    "door": _handle_door_event,
    "exit": _handle_exit_event,
    "obstacle": _handle_obstacle_event,
    "camera": _handle_camera_event,
    "detector": _handle_detector_event,
    "stair": _handle_stair_event,
}


def execute_event(context, event) -> None:

    try:
        handler = EVENT_HANDLERS[event.target_type]

    except KeyError:

        raise UnsupportedEventTargetTypeError(
            f"No handler is registered for ScenarioEvent.target_type "
            f"{event.target_type!r} (event_id={event.event_id!r})."
        ) from None

    handler(context, event)
