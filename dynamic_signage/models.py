from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from models.dynamic_sign import SignIndication


# =====================================================
# Live Dynamic Evacuation Signage milestone -- the immutable output
# models this package produces. Same conventions as every prior
# live-intelligence package (evacuation_guidance.models/evacuation_
# recommendation.models): frozen dataclasses, Mapping fields
# MappingProxyType-wrapped, plain string constants (not stdlib Enum),
# "Optional/UNAVAILABLE means genuinely unavailable, never a fabricated
# value" everywhere.
#
# THE CRITICAL BOUNDARY THIS PACKAGE PRESERVES:
#
#   Recommendation ("which exit")        -- evacuation_recommendation/
#   Guidance ("how to reach it")         -- evacuation_guidance/
#   Signage Planning ("what each sign
#     should indicate")                 -- THIS package, PLANNING ONLY
#   Provider ("actual display state")    -- THIS package's own
#                                            provider.py, reached ONLY
#                                            through the operator-
#                                            approval path (command_
#                                            center.live_operator_
#                                            action_gateway) via
#                                            DynamicSignageController --
#                                            the Planner itself has ZERO
#                                            dispatch authority and never
#                                            calls the Controller/Provider.
# =====================================================


class TargetAssetType:

    DOOR = "Door"
    STAIR = "Stair"
    EXIT = "Exit"

    ALL = (DOOR, STAIR, EXIT)


class SignageStatus:

    # The honest lifecycle state of one sign's CURRENT instruction --
    # distinct from SignIndication (what it says) and distinct from
    # DynamicSignageController's own PENDING/APPROVED/REJECTED workflow
    # state (command_center-facing). This is the PLANNER's own honest
    # judgement of whether it produced a genuinely actionable
    # instruction at all.

    ACTIVE = "ACTIVE"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICT = "CONFLICT"


class SignageInconsistency:

    # Phase 21's own required, visible diagnostics -- never silent.

    VOICE_SIGNAGE_EXIT_MISMATCH = "VOICE_SIGNAGE_EXIT_MISMATCH"
    SIGNAGE_GUIDANCE_MISMATCH = "SIGNAGE_GUIDANCE_MISMATCH"
    STALE_SIGNAGE_REVISION = "STALE_SIGNAGE_REVISION"


@dataclass(frozen=True)
class DynamicSignageConfig:

    # The one configurable angular threshold Phase 4 asks for -- see
    # direction.py. Degrees on either side of dead-ahead that still
    # count as STRAIGHT rather than LEFT/RIGHT.
    straight_threshold_degrees: float = 20.0


@dataclass(frozen=True)
class SignageInstruction:

    # One sign's worth of one guidance revision's routing outcome --
    # Phase 6's own required shape. Immutable: a changed instruction is
    # always a NEW SignageInstruction with an incremented
    # signage_revision (see planner.py), never a mutation in place.

    sign_id: str
    floor_id: Optional[str] = None
    zone_id: Optional[str] = None

    recommended_exit_id: Optional[str] = None
    guidance_revision: int = 0

    indication: str = SignIndication.UNAVAILABLE

    target_asset_id: Optional[str] = None
    target_asset_type: Optional[str] = None
    route_step_index: Optional[int] = None

    status: str = SignageStatus.UNAVAILABLE

    confidence: Optional[float] = None
    reason: str = ""

    signage_revision: int = 0
    timestamp: float = 0.0

    def to_dict(self):

        return {
            "sign_id": self.sign_id, "floor_id": self.floor_id, "zone_id": self.zone_id,
            "recommended_exit_id": self.recommended_exit_id, "guidance_revision": self.guidance_revision,
            "indication": self.indication, "target_asset_id": self.target_asset_id,
            "target_asset_type": self.target_asset_type, "route_step_index": self.route_step_index,
            "status": self.status, "confidence": self.confidence, "reason": self.reason,
            "signage_revision": self.signage_revision, "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class SignageConflict:

    # Phase 7 -- reported, never arbitrarily resolved. A sign whose
    # zone_ids span more than one zone with genuinely valid but
    # DIFFERENT-indication guidance for this cycle.

    sign_id: str
    floor_id: Optional[str] = None
    conflicting_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class DynamicSignageSnapshot:

    timestamp: float = 0.0

    # One SignageInstruction per currently-known sign (Phase 8 -- every
    # sign gets its OWN instruction, never a single collapsed
    # building-wide state), keyed by sign_id.
    instructions: Mapping[str, SignageInstruction] = field(default_factory=dict)

    # Only signs with a genuine conflict this cycle (Phase 7), keyed by
    # sign_id. Absence means no conflict detected, never "not checked".
    conflicts: Mapping[str, SignageConflict] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "instructions", MappingProxyType(dict(self.instructions)))
        object.__setattr__(self, "conflicts", MappingProxyType(dict(self.conflicts)))

    # =====================================================

    def instruction(self, sign_id: str) -> Optional[SignageInstruction]:

        return self.instructions.get(sign_id)

    def conflict(self, sign_id: str) -> Optional[SignageConflict]:

        return self.conflicts.get(sign_id)
