from dataclasses import dataclass, field
from time import time as _now
from typing import Any, Dict, Optional
from uuid import uuid4

from building_control.types import RequestStatus


@dataclass(frozen=True)
class ControlEvent:

    # One immutable entry per status transition a ControlRequest goes
    # through -- BuildingControlController's own history list is
    # append-only (Phase 6/11: "maintain append-only control history",
    # "preserve contradictory request history rather than silently
    # overwriting it"). Nothing in this package ever mutates or removes
    # a ControlEvent once recorded.

    request_id: str
    from_status: Optional[RequestStatus]
    to_status: RequestStatus
    actor: str
    note: str = ""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "from_status": self.from_status.name if self.from_status else None,
            "to_status": self.to_status.name,
            "actor": self.actor,
            "note": self.note,
        }
