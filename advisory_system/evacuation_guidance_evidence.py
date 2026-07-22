from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone -- mirrors
# advisory_system.evacuation_recommendation_evidence/trajectory_evidence
# exactly: a small, immutable, PLAIN-VALUE reduction of
# evacuation_guidance.models.EvacuationGuidanceSnapshot -- never the
# mutable engine, never the whole snapshot. This module imports NOTHING
# from evacuation_guidance/ -- the same "advisory_system gains no new
# package dependency" discipline every sibling evidence module already
# establishes.
#
# COMMANDER AWARENESS ONLY. This is NOT the operator-approval voice
# path -- that remains Command Center's own dedicated operator-action
# gateway (approve/reject a guidance message), reached only through an
# explicit operator click, never through Advisory or this module.
# =====================================================


@dataclass(frozen=True)
class ZoneGuidanceDetail:

    route_status: Optional[str] = None
    recommended_exit_id: Optional[str] = None
    revision: int = 0
    inconsistencies: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "route_status": self.route_status, "recommended_exit_id": self.recommended_exit_id,
            "revision": self.revision, "inconsistencies": list(self.inconsistencies),
        }


@dataclass(frozen=True)
class EvacuationGuidanceEvidence:

    # `available=False` (every other field left at its None/empty
    # default) is the honest "no guidance evidence this cycle" value.

    available: bool

    timestamp: Optional[float] = None

    zone_ids_with_valid_route: Tuple[str, ...] = field(default_factory=tuple)
    zone_ids_without_valid_route: Tuple[str, ...] = field(default_factory=tuple)
    zone_ids_with_inconsistency: Tuple[str, ...] = field(default_factory=tuple)
    zone_ids_missing_speaker_coverage: Tuple[str, ...] = field(default_factory=tuple)

    zone_details: Mapping[str, ZoneGuidanceDetail] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "zone_details", MappingProxyType(dict(self.zone_details)))

    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "available": self.available,
            "timestamp": self.timestamp,
            "zone_ids_with_valid_route": list(self.zone_ids_with_valid_route),
            "zone_ids_without_valid_route": list(self.zone_ids_without_valid_route),
            "zone_ids_with_inconsistency": list(self.zone_ids_with_inconsistency),
            "zone_ids_missing_speaker_coverage": list(self.zone_ids_missing_speaker_coverage),
            "zone_details": {k: v.to_dict() for k, v in self.zone_details.items()},
        }


UNAVAILABLE_EVACUATION_GUIDANCE_EVIDENCE = EvacuationGuidanceEvidence(available=False)
