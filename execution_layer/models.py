from dataclasses import dataclass, field
from typing import Optional, Tuple


# =====================================================
# The Execution Layer -- the immutable output models this package
# produces. Same conventions as recommendation_layer.models: frozen
# dataclasses, plain string constants (not stdlib Enum).
#
# THIS PACKAGE IS AN ORCHESTRATION/COORDINATING LAYER, NOT A REPLACEMENT
# EXECUTION ENGINE. voice_evacuation.VoiceEvacuationController,
# building_control.BuildingControlController, dynamic_signage.
# DynamicSignageController, and warden_notification.
# WardenNotificationController remain the sole execution authority --
# the only classes that ever call a provider's own send/execute/
# apply/notify method. This package only reads what those controllers
# already recorded (for Voice/BuildingControl/Signage) and, for the one
# genuinely new category (Warden Notification), translates a
# recommendation_layer.Recommendation into a submission for the
# existing-shaped WardenNotificationController -- it never calls
# approve/notify itself; that stays an explicit operator action
# routed through command_center.live_operator_action_gateway.
# LiveOperatorActionGateway, exactly like the other three categories.
# See docs/architecture/execution_layer.md.
# =====================================================


class ExecutionCategory:

    VOICE_EVACUATION = "VOICE_EVACUATION"
    BUILDING_CONTROL = "BUILDING_CONTROL"
    DYNAMIC_SIGNAGE = "DYNAMIC_SIGNAGE"
    WARDEN_NOTIFICATION = "WARDEN_NOTIFICATION"


class ExecutionStatus:

    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    SUPERSEDED = "SUPERSEDED"
    DISPATCHED = "DISPATCHED"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"


class RecommendationIdProvenance:

    # Honest disclosure of WHICH id space `ExecutionRequest.
    # originating_recommendation_id` came from -- Voice/BuildingControl/
    # Signage's underlying requests today only ever carry advisory_
    # system.advisory_adapter's own SYNTHESIZED id (a content-derived
    # hash, never a real recommendation_layer.Recommendation.
    # recommendation_id), so this must never be reported as
    # RECOMMENDATION_LAYER for those three categories. Warden
    # Notification is the one category built correctly from day one --
    # see execution_layer.adapters.warden_adapter.

    RECOMMENDATION_LAYER = "recommendation_layer"
    ADVISORY_SYSTEM = "advisory_system"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ExecutionRequest:

    execution_request_id: str = ""
    category: str = ""
    status: str = ""

    # The real provider's own class name (e.g. "SimulationControlProvider")
    # -- an audit-facing string, not a live object reference.
    provider_source: str = ""

    originating_recommendation_id: Optional[str] = None
    recommendation_id_provenance: str = RecommendationIdProvenance.UNAVAILABLE

    target_description: str = ""

    created_at: Optional[float] = None
    approved_at: Optional[float] = None
    dispatched_at: Optional[float] = None
    completed_at: Optional[float] = None

    result_message: str = ""
    result_confirmed: Optional[bool] = None

    def to_dict(self) -> dict:

        return {
            "execution_request_id": self.execution_request_id,
            "category": self.category,
            "status": self.status,
            "provider_source": self.provider_source,
            "originating_recommendation_id": self.originating_recommendation_id,
            "recommendation_id_provenance": self.recommendation_id_provenance,
            "target_description": self.target_description,
            "created_at": self.created_at,
            "approved_at": self.approved_at,
            "dispatched_at": self.dispatched_at,
            "completed_at": self.completed_at,
            "result_message": self.result_message,
            "result_confirmed": self.result_confirmed,
        }


@dataclass(frozen=True)
class ExecutionSet:

    timestamp: float = 0.0
    requests: Tuple[ExecutionRequest, ...] = field(default_factory=tuple)

    def by_category(self, category: str) -> Tuple[ExecutionRequest, ...]:

        return tuple(r for r in self.requests if r.category == category)

    def for_recommendation(self, recommendation_id: str) -> Tuple[ExecutionRequest, ...]:

        return tuple(r for r in self.requests if r.originating_recommendation_id == recommendation_id)

    def to_dict(self) -> dict:

        return {
            "timestamp": self.timestamp,
            "requests": [r.to_dict() for r in self.requests],
        }
