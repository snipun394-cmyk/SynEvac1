from dataclasses import dataclass, field
from time import time as _now
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass(frozen=True)
class WardenNotificationRequest:

    # Immutable by design, mirroring building_control.requests.
    # ControlRequest exactly: this is "SynEvac recommends that a warden
    # be notified" -- it never means the notification happened.
    # WardenNotificationController tracks the mutable lifecycle
    # (WardenNotificationStatus) and append-only history alongside this
    # record; the request itself never changes after creation.

    zone_id: Optional[str]
    reason: str

    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=_now)

    # The REAL recommendation_layer.models.Recommendation.recommendation_id
    # when this request was translated from one (see execution_layer.
    # adapters.warden_adapter.translate()) -- unlike building_control's
    # own source_recommendation_id (always a synthesized advisory_system
    # stand-in today), this field carries the genuine upstream id,
    # giving Warden Notification full, real Recommendation ->
    # Notification traceability from day one.
    source_recommendation_id: Optional[str] = None

    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "zone_id": self.zone_id,
            "reason": self.reason,
            "source_recommendation_id": self.source_recommendation_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class WardenNotificationInstruction:

    # Produced only once a WardenNotificationRequest is APPROVED --
    # mirrors building_control.requests.ControlInstruction exactly.

    request_id: str
    zone_id: Optional[str]
    reason: str

    instruction_id: str = field(default_factory=lambda: str(uuid4()))
    dispatched_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "instruction_id": self.instruction_id,
            "request_id": self.request_id,
            "zone_id": self.zone_id,
            "reason": self.reason,
            "dispatched_at": self.dispatched_at,
        }


@dataclass(frozen=True)
class WardenNotificationResult:

    # A provider's own report of what happened when it tried to notify
    # a warden. confirmed=True is the ONLY thing that may ever move a
    # request to WardenNotificationStatus.CONFIRMED -- mirrors
    # building_control.requests.ControlResult's own discipline exactly.

    instruction_id: str
    confirmed: bool
    message: str = ""
    confirmed_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "instruction_id": self.instruction_id,
            "confirmed": self.confirmed,
            "message": self.message,
            "confirmed_at": self.confirmed_at,
        }
