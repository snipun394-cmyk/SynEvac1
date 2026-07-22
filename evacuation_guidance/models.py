from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from hazard.severity import HazardSeverity


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone -- the
# immutable output models this package produces. Same conventions as
# every prior live-intelligence package (evacuation_recommendation.
# models/trajectory_intelligence.models/emergency_response.models):
# frozen dataclasses, Mapping fields MappingProxyType-wrapped, plain
# string constants (not stdlib Enum), "Optional/UNKNOWN means genuinely
# unavailable, never a fabricated value" everywhere.
#
# THE CRITICAL BOUNDARY THIS PACKAGE PRESERVES:
#
#   Recommendation ("which exit")  -- evacuation_recommendation/, unchanged
#   Guidance ("how to reach it")   -- THIS package: a structured, graph-
#                                      valid route + human-readable
#                                      instructions, still PLANNING ONLY
#   Message Plan ("what to say")   -- THIS package's own message_planner.py:
#                                      a VoiceMessage-shaped candidate,
#                                      still never sent
#   Broadcast ("actually said")    -- existing voice_evacuation.
#                                      VoiceEvacuationController, reached
#                                      ONLY through the existing operator-
#                                      approval path (command_center.
#                                      live_operator_action_gateway) --
#                                      this package has ZERO execution
#                                      authority and never calls it.
# =====================================================


class RouteStatus:

    ROUTE_AVAILABLE = "ROUTE_AVAILABLE"
    ROUTE_UNAVAILABLE = "ROUTE_UNAVAILABLE"
    ROUTE_UNCERTAIN = "ROUTE_UNCERTAIN"
    NO_SAFE_EXIT = "NO_SAFE_EXIT"
    ALREADY_AT_EXIT = "ALREADY_AT_EXIT"


_VALID_ROUTE_STATUSES = (RouteStatus.ROUTE_AVAILABLE, RouteStatus.ALREADY_AT_EXIT)


class NavigationStepType:

    # Only step types the building model can genuinely support -- no
    # step type here can be produced without a real, corresponding
    # graph node/edge (Phase 7's own "do not create step types
    # unsupported by the building model" requirement).

    LEAVE_ZONE = "LEAVE_ZONE"
    PASS_THROUGH_DOOR = "PASS_THROUGH_DOOR"
    ENTER_ZONE = "ENTER_ZONE"
    USE_STAIR = "USE_STAIR"
    CONTINUE_TO_EXIT = "CONTINUE_TO_EXIT"
    EXIT_BUILDING = "EXIT_BUILDING"


class GuidanceInconsistency:

    # Phase 23's own required, visible diagnostics -- never silent.

    RECOMMENDED_EXIT_UNREACHABLE = "RECOMMENDED_EXIT_UNREACHABLE"
    RECOMMENDATION_ROUTE_MISMATCH = "RECOMMENDATION_ROUTE_MISMATCH"
    ROUTE_BECAME_UNSAFE = "ROUTE_BECAME_UNSAFE"
    NO_SPEAKER_COVERAGE = "NO_SPEAKER_COVERAGE"
    STALE_GUIDANCE = "STALE_GUIDANCE"


class DeliveryStatus:

    # Phase 17 -- delivery (speaker/output-hardware availability) is a
    # SEPARATE concern from route validity. A NO_SPEAKER_COVERAGE
    # delivery status never demotes an otherwise-valid RouteStatus.

    NOT_PLANNED = "NOT_PLANNED"
    PLANNED = "PLANNED"
    NO_SPEAKER_COVERAGE = "NO_SPEAKER_COVERAGE"


@dataclass(frozen=True)
class GuidanceConfig:

    # Mirrors trajectory_intelligence.models.TrajectoryConfig/
    # evacuation_recommendation.models.RecommendationConfig's own
    # independently-owned hazard floor exactly -- this package never
    # imports a sibling engine's config, it keeps its own copy (this
    # codebase's own established "no cross-package engine coupling"
    # convention).
    hazard_unsafe_severity_floor: HazardSeverity = HazardSeverity.HIGH

    # Phase 23 -- how long since the consumed EvacuationRecommendationSnapshot
    # was produced before this cycle's guidance is honestly flagged
    # STALE_GUIDANCE (informational only -- never invalidates the route
    # by itself).
    staleness_seconds: float = 20.0


@dataclass(frozen=True)
class NavigationStep:

    # A single structured step -- text is DERIVED from this, never the
    # reverse (Phase 7's own "text is derived from structure" rule).
    # Every field is Optional because different step types populate
    # different subsets (e.g. USE_STAIR has no door_id).

    step_type: str

    zone_id: Optional[str] = None
    floor_id: Optional[str] = None

    door_id: Optional[str] = None
    stair_id: Optional[str] = None
    exit_id: Optional[str] = None

    from_floor_id: Optional[str] = None
    to_floor_id: Optional[str] = None

    def to_dict(self):

        return {
            "step_type": self.step_type, "zone_id": self.zone_id, "floor_id": self.floor_id,
            "door_id": self.door_id, "stair_id": self.stair_id, "exit_id": self.exit_id,
            "from_floor_id": self.from_floor_id, "to_floor_id": self.to_floor_id,
        }


@dataclass(frozen=True)
class EvacuationGuidancePlan:

    zone_id: str
    floor_id: Optional[str] = None

    recommended_exit_id: Optional[str] = None
    route_status: str = RouteStatus.ROUTE_UNCERTAIN

    route_distance_m: Optional[float] = None

    # Phase 3 -- only fields genuinely derivable from the graph. Never
    # includes Node.OUTSIDE_NODE_ID itself (that is implicit in
    # route_status == ROUTE_AVAILABLE/ALREADY_AT_EXIT).
    ordered_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    ordered_door_ids: Tuple[str, ...] = field(default_factory=tuple)
    ordered_stair_ids: Tuple[str, ...] = field(default_factory=tuple)

    ordered_navigation_steps: Tuple[NavigationStep, ...] = field(default_factory=tuple)

    # Phase 8/9 -- one deterministic sentence per meaningful decision
    # point (door/stair/exit), never per zone transition.
    instructions: Tuple[str, ...] = field(default_factory=tuple)

    # Phase 23 -- visible diagnostics; empty means no inconsistency
    # detected, never that none was checked.
    inconsistencies: Tuple[str, ...] = field(default_factory=tuple)

    # Phase 12 -- deterministic, monotonic, per-zone sequence number.
    # Never a random UUID.
    revision: int = 0

    confidence: Optional[float] = None
    evidence_timestamp: float = 0.0

    def is_valid(self) -> bool:

        return self.route_status in _VALID_ROUTE_STATUSES

    def to_dict(self):

        return {
            "zone_id": self.zone_id, "floor_id": self.floor_id,
            "recommended_exit_id": self.recommended_exit_id, "route_status": self.route_status,
            "route_distance_m": self.route_distance_m,
            "ordered_zone_ids": list(self.ordered_zone_ids), "ordered_door_ids": list(self.ordered_door_ids),
            "ordered_stair_ids": list(self.ordered_stair_ids),
            "ordered_navigation_steps": [step.to_dict() for step in self.ordered_navigation_steps],
            "instructions": list(self.instructions), "inconsistencies": list(self.inconsistencies),
            "revision": self.revision, "confidence": self.confidence, "evidence_timestamp": self.evidence_timestamp,
        }


@dataclass(frozen=True)
class SpeakerCoverage:

    # Phase 13 -- deliberately independent of route validity. Computed
    # for the ORIGINATING zone only (route delivery, not every zone the
    # route merely passes through).

    zone_id: str
    speaker_ids: Tuple[str, ...] = field(default_factory=tuple)
    delivery_status: str = DeliveryStatus.NOT_PLANNED


@dataclass(frozen=True)
class VoiceGuidanceMessagePlan:

    # Phase 14 -- a PLANNED candidate only, never sent. Reconciled with
    # the existing CivilianAnnouncement -> VoiceMessage path by
    # advisory_system/evacuation_guidance_evidence.py (Phase 15) rather
    # than broadcasting independently -- see this package's own module
    # docstring in engine.py.

    zone_id: str
    message_text: str = ""

    recommended_exit_id: Optional[str] = None
    guidance_revision: int = 0

    speaker_ids: Tuple[str, ...] = field(default_factory=tuple)
    delivery_status: str = DeliveryStatus.NOT_PLANNED
    route_status: str = RouteStatus.ROUTE_UNCERTAIN

    timestamp: float = 0.0

    def to_dict(self):

        return {
            "zone_id": self.zone_id, "message_text": self.message_text,
            "recommended_exit_id": self.recommended_exit_id, "guidance_revision": self.guidance_revision,
            "speaker_ids": list(self.speaker_ids), "delivery_status": self.delivery_status,
            "route_status": self.route_status, "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class EvacuationGuidanceSnapshot:

    timestamp: float = 0.0

    # Phase 8's own "one recommendation per occupied zone" upstream
    # discipline flows through unchanged -- one guidance plan per zone
    # evacuation_recommendation itself produced a recommendation for.
    zones: Mapping[str, EvacuationGuidancePlan] = field(default_factory=dict)

    # Phase 15 -- only zones with a genuinely VALID route (ROUTE_AVAILABLE
    # or ALREADY_AT_EXIT) get a voice plan at all; never generated from
    # an invalid/uncertain/unavailable route.
    voice_plans: Mapping[str, VoiceGuidanceMessagePlan] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "zones", MappingProxyType(dict(self.zones)))
        object.__setattr__(self, "voice_plans", MappingProxyType(dict(self.voice_plans)))

    # =====================================================

    def zone(self, zone_id: str) -> Optional[EvacuationGuidancePlan]:

        return self.zones.get(zone_id)

    def voice_plan(self, zone_id: str) -> Optional[VoiceGuidanceMessagePlan]:

        return self.voice_plans.get(zone_id)
