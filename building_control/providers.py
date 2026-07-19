from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

from simulation_interactive.action_executor import Action, ActionExecutor, InteractiveActionType

from building_control.requests import ControlInstruction, ControlResult
from building_control.types import ControlAction, ControlSystemType


class BuildingControlProvider(ABC):

    # ControlInstruction -> provider -> ControlResult. The same
    # instruction/result shape must work regardless of which provider
    # receives it (Phase 12) -- SimulationControlProvider is the only
    # implementation in this milestone. A future ReplayControlProvider/
    # LiveBuildingControlProvider would implement this same interface;
    # no protocol library (Modbus/BACnet/MQTT/PLC/vendor SDK) may ever
    # be imported into this package, in this class or any subclass
    # defined here -- an authorized protocol adapter, if one is ever
    # built, lives entirely outside building_control.

    # False for every provider except an explicitly simulation-only
    # one. BuildingControlController refuses ApprovalMode.
    # AUTO_APPROVE_SIMULATION unless this is True (see that class's
    # constructor) -- the one guard preventing simulation auto-approval
    # from ever reaching a Live provider.
    is_simulation_only: bool = False

    @abstractmethod
    def execute(self, instruction: ControlInstruction) -> ControlResult:
        ...


# The only (system, action) pairs with a real, already-existing
# executable handler (simulation_interactive.action_executor.
# ActionExecutor._HANDLERS). Composed here, not reimplemented --
# see that module's own ActionExecutor docstring: "Every engineering
# mutation here is imported straight from scenario_runner's existing
# public re-export surface... zero reimplementation of state-mutation
# logic."
_DOOR_EXIT_ACTION_MAP = {
    (ControlSystemType.DOOR, ControlAction.OPEN): InteractiveActionType.OPEN_DOOR,
    (ControlSystemType.DOOR, ControlAction.CLOSE): InteractiveActionType.CLOSE_DOOR,
    (ControlSystemType.EXIT, ControlAction.OPEN): InteractiveActionType.OPEN_EXIT,
    (ControlSystemType.EXIT, ControlAction.CLOSE): InteractiveActionType.CLOSE_EXIT,
}

_STATE_ONLY_SYSTEMS = (
    ControlSystemType.STAIR_PRESSURIZATION,
    ControlSystemType.SMOKE_EXHAUST,
    ControlSystemType.DELUGE,
)


class SimulationControlProvider(BuildingControlProvider):

    # Door/Exit: composes ActionExecutor -- confirmation is whatever
    # ActionResult.applied that executor actually reports, never
    # assumed. action_executor=None (the default) honestly means "no
    # running simulation to mutate" (e.g. Command Center's historical
    # replay view, which has no live SimulationContext) -- Door/Exit
    # instructions then report confirmed=False with that reason, they
    # are never silently treated as CONFIRMED.
    #
    # Stair Pressurization/Smoke Exhaust/Deluge: no backing physics
    # anywhere in this codebase (Phase 1/8 investigation) -- confirmed
    # state is tracked here as a plain state-only record. The message
    # returned is deliberately narrow ("Control confirmed in
    # simulation: <target> <system> = <action>") and never claims a
    # physical effect (e.g. "smoke prevented from entering stair") that
    # this platform's hazard/fire simulation does not actually model.

    is_simulation_only = True

    def __init__(
        self,
        building,
        action_executor: Optional[ActionExecutor] = None,
        *,
        should_fail: Optional[Callable[[ControlInstruction], bool]] = None,
    ):
        self._building = building
        self._action_executor = action_executor
        self._should_fail = should_fail or (lambda instruction: False)
        self._state_only: Dict[Tuple[ControlSystemType, Optional[str]], ControlAction] = {}

    # =====================================================

    def execute(self, instruction: ControlInstruction) -> ControlResult:

        if self._should_fail(instruction):

            return ControlResult(
                instruction_id=instruction.instruction_id, confirmed=False,
                message="simulation provider reported failure for this instruction",
            )

        if instruction.system_type in (ControlSystemType.DOOR, ControlSystemType.EXIT):
            return self._execute_door_or_exit(instruction)

        return self._execute_state_only(instruction)

    # =====================================================

    def _execute_door_or_exit(self, instruction: ControlInstruction) -> ControlResult:

        if self._action_executor is None:

            return ControlResult(
                instruction_id=instruction.instruction_id, confirmed=False,
                message="no simulation action executor available for this session",
            )

        interactive_action_type = _DOOR_EXIT_ACTION_MAP[(instruction.system_type, instruction.action)]
        action = Action(action_type=interactive_action_type, target_id=instruction.target_id)

        result = self._action_executor.apply(action, time=instruction.dispatched_at)

        return ControlResult(
            instruction_id=instruction.instruction_id, confirmed=result.applied,
            message=result.reason or "confirmed by simulation action executor",
        )

    # =====================================================

    def _execute_state_only(self, instruction: ControlInstruction) -> ControlResult:

        self._state_only[(instruction.system_type, instruction.target_id)] = instruction.action

        return ControlResult(
            instruction_id=instruction.instruction_id, confirmed=True,
            message=(
                f"Control confirmed in simulation: {instruction.target_id} "
                f"{instruction.system_type.name} = {instruction.action.name}"
            ),
        )

    # =====================================================

    def state_only_states(self) -> Dict[Tuple[ControlSystemType, Optional[str]], ControlAction]:

        # Read-only view for BuildingControlController.snapshot() --
        # Door/Exit confirmed state is instead read straight off the
        # Building's own Door.locked/Exit.is_blocked fields, which
        # already are the single source of truth for those two systems.

        return dict(self._state_only)
