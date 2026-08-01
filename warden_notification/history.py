from dataclasses import dataclass, field
from time import time as _now
from typing import Any, Dict, Optional
from uuid import uuid4

from warden_notification.types import WardenNotificationStatus


@dataclass(frozen=True)
class WardenNotificationEvent:

    # One immutable entry per status transition a WardenNotificationRequest
    # goes through -- mirrors building_control.history.ControlEvent
    # exactly. Append-only; nothing in this package ever mutates or
    # removes an entry once recorded.

    request_id: str
    from_status: Optional[WardenNotificationStatus]
    to_status: WardenNotificationStatus
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
