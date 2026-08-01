from abc import ABC, abstractmethod
from typing import Dict, Optional

from warden_notification.requests import WardenNotificationInstruction, WardenNotificationResult


class WardenNotificationProvider(ABC):

    # WardenNotificationInstruction -> provider -> WardenNotificationResult.
    # Mirrors building_control.providers.BuildingControlProvider exactly.
    # A future real notification transport (SMS/push/email/webhook) would
    # implement this same interface; no such transport library may ever
    # be imported into this package, in this class or any subclass
    # defined here.

    is_simulation_only: bool = False

    @abstractmethod
    def notify(self, instruction: WardenNotificationInstruction) -> WardenNotificationResult:
        ...


class SimulationWardenNotificationProvider(WardenNotificationProvider):

    # Pure bookkeeping -- no real SMS/push/email/webhook transport
    # exists anywhere in this codebase (confirmed by the Execution Layer
    # V1 architectural review). Records that a notification was "sent
    # to the simulated warden roster," always confirmed=True, with a
    # message deliberately narrow ("recorded in simulation") and never
    # claiming a real person was actually reached -- mirrors
    # building_control.providers.SimulationControlProvider's own
    # state-only branch and its "no backing physics" disclosure.

    is_simulation_only = True

    def __init__(self):

        self._roster: Dict[str, WardenNotificationInstruction] = {}

    # =====================================================

    def notify(self, instruction: WardenNotificationInstruction) -> WardenNotificationResult:

        self._roster[instruction.instruction_id] = instruction

        return WardenNotificationResult(
            instruction_id=instruction.instruction_id, confirmed=True,
            message=(
                f"Notification recorded in simulation: zone {instruction.zone_id} -- {instruction.reason}"
            ),
        )

    # =====================================================

    def roster(self) -> Dict[str, WardenNotificationInstruction]:

        # Read-only view -- e.g. for a future test/panel to inspect what
        # was "sent" in this simulation session.

        return dict(self._roster)
