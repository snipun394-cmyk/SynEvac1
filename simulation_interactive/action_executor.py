from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Mapping, Optional, Tuple

from scenario.engineering_state import DoorState

from scenario_runner import (
    apply_door_state,
    apply_exit_state,
    exclude_stair_edge,
    find_door,
    find_exit,
    include_stair_edge,
)
from scenario_runner.context import SimulationContext

from simulation_interactive.route_manager import RouteManager


class InteractiveActionType(Enum):

    OPEN_DOOR = auto()
    CLOSE_DOOR = auto()
    OPEN_EXIT = auto()
    CLOSE_EXIT = auto()
    OPEN_STAIR = auto()
    CLOSE_STAIR = auto()

    # zone_id as target_id, parameters["exit_id"]/["stair_id"] as the
    # recommended edge. Vocabulary mirrors decision_policy's own
    # exit/stair recommendation shape (decision_policy/exit_policy.py,
    # stair_policy.py) without importing that package.
    RECOMMEND_EXIT = auto()
    RECOMMEND_STAIR = auto()

    # target_id/parameters["target_zone_ids"] select which zones; None
    # (the default) means every zone. Mirrors decision_policy.
    # zone_policy's EVACUATE_IMMEDIATELY/SHELTER_IN_PLACE vocabulary.
    BROADCAST_EVACUATE = auto()
    BROADCAST_SHELTER_IN_PLACE = auto()

    # No staff/firefighter entity exists anywhere in the simulated
    # model yet -- this action is recorded (so it reaches the
    # architecture, per this phase's Phase 3 scope) but never applied.
    DEPLOY_STAFF = auto()


@dataclass(frozen=True)
class Action:

    action_type: InteractiveActionType
    target_id: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResult:

    action: Action
    applied: bool
    changed_edge_ids: Tuple[str, ...] = ()
    reason: Optional[str] = None


class ActionExecutor:

    # Every engineering mutation here is imported straight from
    # scenario_runner's existing public re-export surface
    # (find_door/apply_door_state/find_exit/apply_exit_state/
    # include_stair_edge/exclude_stair_edge) -- zero reimplementation
    # of state-mutation logic. Recommendation/broadcast actions only
    # ever reach RouteManager's bookkeeping; they take effect the next
    # time an affected occupant is replanned (see route_manager.
    # maybe_replan).

    def __init__(self, context: SimulationContext, route_manager: RouteManager):

        self._context = context
        self._route_manager = route_manager

    # =====================================================

    def apply(self, action: Action, time: float) -> ActionResult:

        handler = self._HANDLERS.get(action.action_type)

        if handler is None:
            return ActionResult(
                action=action, applied=False,
                reason=f"unsupported action_type {action.action_type!r}",
            )

        return handler(self, action, time)

    # =====================================================

    def _open_door(self, action: Action, time: float) -> ActionResult:

        door = find_door(self._context.building, action.target_id)

        if door is None:
            return ActionResult(
                action=action, applied=False,
                reason=f"door {action.target_id!r} not found",
            )

        apply_door_state(door, DoorState.OPEN)
        self._route_manager.note_engineering_change([action.target_id])

        return ActionResult(action=action, applied=True, changed_edge_ids=(action.target_id,))

    # =====================================================

    def _close_door(self, action: Action, time: float) -> ActionResult:

        door = find_door(self._context.building, action.target_id)

        if door is None:
            return ActionResult(
                action=action, applied=False,
                reason=f"door {action.target_id!r} not found",
            )

        # LOCKED, not CLOSED: Edge.traversable (navigation/edge.py)
        # only reads door.locked/door.active, never door.normally_open
        # -- a merely CLOSED-but-unlocked door stays traversable.
        apply_door_state(door, DoorState.LOCKED)
        self._route_manager.note_engineering_change([action.target_id])

        return ActionResult(action=action, applied=True, changed_edge_ids=(action.target_id,))

    # =====================================================

    def _open_exit(self, action: Action, time: float) -> ActionResult:

        exit_obj = find_exit(self._context.building, action.target_id)

        if exit_obj is None:
            return ActionResult(
                action=action, applied=False,
                reason=f"exit {action.target_id!r} not found",
            )

        apply_exit_state(exit_obj, True)
        self._route_manager.note_engineering_change([action.target_id])

        return ActionResult(action=action, applied=True, changed_edge_ids=(action.target_id,))

    # =====================================================

    def _close_exit(self, action: Action, time: float) -> ActionResult:

        exit_obj = find_exit(self._context.building, action.target_id)

        if exit_obj is None:
            return ActionResult(
                action=action, applied=False,
                reason=f"exit {action.target_id!r} not found",
            )

        apply_exit_state(exit_obj, False)
        self._route_manager.note_engineering_change([action.target_id])

        return ActionResult(action=action, applied=True, changed_edge_ids=(action.target_id,))

    # =====================================================

    def _open_stair(self, action: Action, time: float) -> ActionResult:

        include_stair_edge(self._context.graph, self._context.building, action.target_id)
        self._route_manager.note_engineering_change([action.target_id])

        return ActionResult(action=action, applied=True, changed_edge_ids=(action.target_id,))

    # =====================================================

    def _close_stair(self, action: Action, time: float) -> ActionResult:

        exclude_stair_edge(self._context.graph, action.target_id)
        self._route_manager.note_engineering_change([action.target_id])

        return ActionResult(action=action, applied=True, changed_edge_ids=(action.target_id,))

    # =====================================================

    def _recommend_exit(self, action: Action, time: float) -> ActionResult:

        self._route_manager.note_recommendation(
            action.target_id, recommended_exit_edge_id=action.parameters.get("exit_id"),
        )

        return ActionResult(action=action, applied=True)

    # =====================================================

    def _recommend_stair(self, action: Action, time: float) -> ActionResult:

        self._route_manager.note_recommendation(
            action.target_id, recommended_stair_edge_id=action.parameters.get("stair_id"),
        )

        return ActionResult(action=action, applied=True)

    # =====================================================

    def _broadcast_evacuate(self, action: Action, time: float) -> ActionResult:

        self._route_manager.note_broadcast(
            "EVACUATE_IMMEDIATELY", target_zone_ids=action.parameters.get("target_zone_ids"),
        )

        return ActionResult(action=action, applied=True)

    # =====================================================

    def _broadcast_shelter_in_place(self, action: Action, time: float) -> ActionResult:

        self._route_manager.note_broadcast(
            "SHELTER_IN_PLACE", target_zone_ids=action.parameters.get("target_zone_ids"),
        )

        return ActionResult(action=action, applied=True)

    # =====================================================

    def _deploy_staff(self, action: Action, time: float) -> ActionResult:

        return ActionResult(
            action=action, applied=False,
            reason="no staff entity exists in the current simulated model; recorded for observability only",
        )

    # =====================================================

    _HANDLERS = {
        InteractiveActionType.OPEN_DOOR: _open_door,
        InteractiveActionType.CLOSE_DOOR: _close_door,
        InteractiveActionType.OPEN_EXIT: _open_exit,
        InteractiveActionType.CLOSE_EXIT: _close_exit,
        InteractiveActionType.OPEN_STAIR: _open_stair,
        InteractiveActionType.CLOSE_STAIR: _close_stair,
        InteractiveActionType.RECOMMEND_EXIT: _recommend_exit,
        InteractiveActionType.RECOMMEND_STAIR: _recommend_stair,
        InteractiveActionType.BROADCAST_EVACUATE: _broadcast_evacuate,
        InteractiveActionType.BROADCAST_SHELTER_IN_PLACE: _broadcast_shelter_in_place,
        InteractiveActionType.DEPLOY_STAFF: _deploy_staff,
    }
