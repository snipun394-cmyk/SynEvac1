# Building Control & Safety Systems Integration Framework -- a
# digital-twin control abstraction between SynEvac's Advisory System
# and building engineering systems (doors, exits, stair
# pressurization, smoke exhaust, deluge/suppression).
#
# This is NOT real hardware control. No Modbus/BACnet/MQTT/PLC/vendor
# protocol code exists anywhere in this package, and none of it talks
# to any real building system. Every control action that actually
# takes effect here only ever mutates already-simulated engineering
# state (composing simulation_interactive.ActionExecutor for Door/Exit,
# or this package's own honest state-only tracking for the three
# systems -- stair pressurization, smoke exhaust, deluge -- that have
# no backing physics anywhere in this codebase yet).
#
# Core distinction preserved throughout: RECOMMENDATION != COMMAND !=
# CONFIRMED PHYSICAL ACTION. See requests.py for the request lifecycle
# and providers.py for why CONFIRMED is only ever provider-reported,
# never assumed.

from building_control.types import (
    ApprovalMode,
    ControlAction,
    ControlSystemType,
    RequestSource,
    RequestStatus,
)
from building_control.requests import ControlInstruction, ControlRequest, ControlResult
from building_control.history import ControlEvent
from building_control.snapshot import ControlStateEntry, ControlStateSnapshot
from building_control.controller import BuildingControlController
from building_control.providers import BuildingControlProvider, SimulationControlProvider
from building_control.advisory_adapter import translate_recommendation, translate_report

__all__ = [
    "ApprovalMode",
    "ControlAction",
    "ControlSystemType",
    "RequestSource",
    "RequestStatus",
    "ControlInstruction",
    "ControlRequest",
    "ControlResult",
    "ControlEvent",
    "ControlStateEntry",
    "ControlStateSnapshot",
    "BuildingControlController",
    "BuildingControlProvider",
    "SimulationControlProvider",
    "translate_recommendation",
    "translate_report",
]
