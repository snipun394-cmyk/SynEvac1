from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple


# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone -- mirrors
# advisory_system.trajectory_evidence/emergency_response_evidence/
# crowd_evidence/evacuation_progress_evidence exactly: a small,
# immutable, PLAIN-VALUE reduction of evacuation_recommendation.models.
# EvacuationRecommendationSnapshot -- never the mutable engine, never
# the whole snapshot. This module imports NOTHING from
# evacuation_recommendation/ -- the same "advisory_system gains no new
# package dependency" discipline every sibling evidence module already
# establishes.
#
# THIS IS SUPPORTING EVIDENCE ONLY. Safety precedence:
# Decision Policy > Recommendation Ranking > AI support -- see
# docs/architecture/live_evacuation_recommendation_engine.md.
# =====================================================


@dataclass(frozen=True)
class ZoneRecommendationDetail:

    status: Optional[str] = None
    recommended_exit_id: Optional[str] = None
    alternative_exit_ids: Tuple[str, ...] = field(default_factory=tuple)
    confidence: Optional[float] = None
    reason_codes: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "status": self.status,
            "recommended_exit_id": self.recommended_exit_id,
            "alternative_exit_ids": list(self.alternative_exit_ids),
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class EvacuationRecommendationEvidence:

    # `available=False` (every other field left at its None/empty
    # default) is the honest "no recommendation evidence this cycle"
    # value.

    available: bool

    timestamp: Optional[float] = None

    zone_ids_with_recommendation: Tuple[str, ...] = field(default_factory=tuple)
    zone_ids_without_safe_exit: Tuple[str, ...] = field(default_factory=tuple)

    safe_exit_ids: Tuple[str, ...] = field(default_factory=tuple)

    zone_details: Mapping[str, ZoneRecommendationDetail] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "zone_details", MappingProxyType(dict(self.zone_details)))

    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "available": self.available,
            "timestamp": self.timestamp,
            "zone_ids_with_recommendation": list(self.zone_ids_with_recommendation),
            "zone_ids_without_safe_exit": list(self.zone_ids_without_safe_exit),
            "safe_exit_ids": list(self.safe_exit_ids),
            "zone_details": {k: v.to_dict() for k, v in self.zone_details.items()},
        }


UNAVAILABLE_EVACUATION_RECOMMENDATION_EVIDENCE = EvacuationRecommendationEvidence(available=False)
