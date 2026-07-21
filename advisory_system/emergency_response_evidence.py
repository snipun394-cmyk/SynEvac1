from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple


# =====================================================
# Live Emergency Response & Rescue Priority Intelligence milestone --
# Phase 16's own required "Emergency Response Evidence" structure,
# mirroring advisory_system.crowd_evidence/evacuation_progress_evidence
# exactly: a small, immutable, PLAIN-VALUE reduction of
# emergency_response.models.EmergencyResponseSnapshot -- never the
# mutable engine, never the whole snapshot. This module imports NOTHING
# from emergency_response/ -- the same "advisory_system gains no new
# package dependency" discipline every sibling evidence module already
# establishes.
#
# THIS IS SUPPORTING EVIDENCE ONLY (Phase 19's own explicit
# requirement) -- see docs/architecture/
# live_emergency_response_intelligence.md's own Safety Precedence
# section.
# =====================================================


@dataclass(frozen=True)
class ZoneResponseDetail:

    # Included in EmergencyResponseEvidence.zone_details ONLY for
    # zones already worth Advisory's attention (critical, high,
    # possible-assistance, or uncertain-search) -- never one entry per
    # zone in the building.

    priority_level: Optional[str] = None
    priority_score: Optional[float] = None
    known_occupant_count: int = 0
    possible_assistance_count: int = 0
    confirmed_assistance_count: int = 0
    # Live Human State & Assistance Perception Bridge milestone --
    # distinct from confirmed_assistance_count (help already underway).
    being_assisted_count: int = 0
    reason_codes: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "priority_level": self.priority_level, "priority_score": self.priority_score,
            "known_occupant_count": self.known_occupant_count,
            "possible_assistance_count": self.possible_assistance_count,
            "confirmed_assistance_count": self.confirmed_assistance_count,
            "being_assisted_count": self.being_assisted_count,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class EmergencyResponseEvidence:

    # `available=False` (every other field left at its None/empty
    # default) is the honest "no emergency-response evidence this
    # cycle" value.

    available: bool

    timestamp: Optional[float] = None

    critical_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    high_priority_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    possible_assistance_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    uncertain_search_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    observed_clear_zone_ids: Tuple[str, ...] = field(default_factory=tuple)

    # Live Human State & Assistance Perception Bridge milestone, Phase
    # 20 -- both purely additive, informational id sets, mirroring the
    # existing five above exactly.
    being_assisted_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    vulnerable_person_observed_zone_ids: Tuple[str, ...] = field(default_factory=tuple)

    highest_priority_zone_id: Optional[str] = None
    highest_priority_reason_codes: Tuple[str, ...] = field(default_factory=tuple)

    zone_details: Mapping[str, ZoneResponseDetail] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "zone_details", MappingProxyType(dict(self.zone_details)))

    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "available": self.available,
            "timestamp": self.timestamp,
            "critical_zone_ids": list(self.critical_zone_ids),
            "high_priority_zone_ids": list(self.high_priority_zone_ids),
            "possible_assistance_zone_ids": list(self.possible_assistance_zone_ids),
            "uncertain_search_zone_ids": list(self.uncertain_search_zone_ids),
            "observed_clear_zone_ids": list(self.observed_clear_zone_ids),
            "being_assisted_zone_ids": list(self.being_assisted_zone_ids),
            "vulnerable_person_observed_zone_ids": list(self.vulnerable_person_observed_zone_ids),
            "highest_priority_zone_id": self.highest_priority_zone_id,
            "highest_priority_reason_codes": list(self.highest_priority_reason_codes),
            "zone_details": {k: v.to_dict() for k, v in self.zone_details.items()},
        }


UNAVAILABLE_EMERGENCY_RESPONSE_EVIDENCE = EmergencyResponseEvidence(available=False)
