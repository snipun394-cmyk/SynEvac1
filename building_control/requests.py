from dataclasses import dataclass, field
from time import time as _now
from typing import Any, Dict, Optional
from uuid import uuid4

from building_control.types import ApprovalMode, ControlAction, ControlSystemType, RequestSource


@dataclass(frozen=True)
class ControlRequest:

    # Immutable by design: this is "SynEvac recommends or requests
    # that this action be considered" -- it never means the action
    # happened. BuildingControlController tracks the mutable lifecycle
    # (RequestStatus) and append-only history (ControlEvent) alongside
    # this record; the request itself never changes after creation.

    system_type: ControlSystemType
    target_id: Optional[str]
    requested_action: ControlAction
    reason: str
    source: RequestSource

    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=_now)

    # BuildingRecommendation (advisory_system/recommendation_models.py)
    # has no id field of its own -- when source is ADVISORY_ADAPTER,
    # this is a synthesized, stable id built from the recommendation's
    # own content (see advisory_adapter._synthesize_recommendation_id),
    # not a native identifier. None for operator-originated requests.
    source_recommendation_id: Optional[str] = None

    confidence: Optional[float] = None
    priority: int = 0

    approval_requirement: ApprovalMode = ApprovalMode.REQUIRES_APPROVAL

    def to_dict(self) -> Dict[str, Any]:

        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "system_type": self.system_type.name,
            "target_id": self.target_id,
            "requested_action": self.requested_action.name,
            "source": self.source.name,
            "source_recommendation_id": self.source_recommendation_id,
            "reason": self.reason,
            "confidence": self.confidence,
            "priority": self.priority,
            "approval_requirement": self.approval_requirement.name,
        }


@dataclass(frozen=True)
class ControlInstruction:

    # Produced only once a ControlRequest is APPROVED -- this is the
    # "Requested" step's own output per this milestone's architecture
    # diagram (BuildingControlController -> ControlInstruction ->
    # provider). Carries no approval/confidence metadata of its own:
    # that context lives on the originating ControlRequest, reachable
    # via request_id.

    request_id: str
    system_type: ControlSystemType
    target_id: Optional[str]
    action: ControlAction

    instruction_id: str = field(default_factory=lambda: str(uuid4()))
    dispatched_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "instruction_id": self.instruction_id,
            "request_id": self.request_id,
            "system_type": self.system_type.name,
            "target_id": self.target_id,
            "action": self.action.name,
            "dispatched_at": self.dispatched_at,
        }


@dataclass(frozen=True)
class ControlResult:

    # A provider's own report of what happened when it tried to carry
    # out a ControlInstruction. confirmed=True is the ONLY thing that
    # may ever move a request to RequestStatus.CONFIRMED -- ControlResult
    # exists specifically so nothing upstream of a provider can fabricate
    # that state (see BuildingControlController._dispatch).

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
