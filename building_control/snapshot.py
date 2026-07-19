from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from building_control.types import ControlAction, ControlSystemType, RequestStatus


@dataclass(frozen=True)
class ControlStateEntry:

    # One CONFIRMED control's current simulated state -- never a
    # pending/requested one (see BuildingControlController.snapshot()).
    # BuildingState.control_status only ever exposes what a provider
    # has actually confirmed, per this milestone's core rule:
    # RECOMMENDATION != COMMAND != CONFIRMED PHYSICAL ACTION.

    system_type: ControlSystemType
    target_id: Optional[str]
    action: ControlAction
    confirmed_at: float

    def to_dict(self) -> Dict[str, Any]:

        return {
            "system_type": self.system_type.name,
            "target_id": self.target_id,
            "action": self.action.name,
            "confirmed_at": self.confirmed_at,
        }


@dataclass(frozen=True)
class ControlStateSnapshot:

    # BuildingState's additive view over BuildingControlController --
    # see building_state/models.py::BuildingState.control_status,
    # populated the same pass-through way facp_status already is
    # (BuildingStateEstimator never computes this itself).

    entries: Tuple[ControlStateEntry, ...] = field(default_factory=tuple)
    pending_count: int = 0

    def to_dict(self) -> Dict[str, Any]:

        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "pending_count": self.pending_count,
        }
