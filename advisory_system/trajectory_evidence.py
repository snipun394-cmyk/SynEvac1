from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple, Optional


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone, Phase 22 -- mirrors advisory_system.
# emergency_response_evidence/crowd_evidence/evacuation_progress_evidence
# exactly: a small, immutable, PLAIN-VALUE reduction of
# trajectory_intelligence.models.TrajectoryIntelligenceSnapshot -- never
# the mutable engine, never the whole snapshot. This module imports
# NOTHING from trajectory_intelligence/ -- the same "advisory_system
# gains no new package dependency" discipline every sibling evidence
# module already establishes.
#
# THIS IS SUPPORTING EVIDENCE ONLY. Route deviation is movement geometry,
# never a claim of panic, confusion, or non-compliance -- see
# docs/architecture/live_trajectory_route_deviation_intelligence.md's
# own "what this is not" section.
# =====================================================


@dataclass(frozen=True)
class TrajectoryZoneDetail:

    occupant_ids: Tuple[str, ...] = field(default_factory=tuple)
    anomaly_flags: Tuple[str, ...] = field(default_factory=tuple)
    highest_anomaly_severity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "occupant_ids": list(self.occupant_ids),
            "anomaly_flags": list(self.anomaly_flags),
            "highest_anomaly_severity": self.highest_anomaly_severity,
        }


@dataclass(frozen=True)
class TrajectoryDecisionEvidence:

    # `available=False` (every other field left at its None/empty
    # default) is the honest "no trajectory evidence this cycle" value.

    available: bool

    timestamp: Optional[float] = None

    route_deviation_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    movement_stalled_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    hazardous_zone_movement_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    no_safe_route_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    shared_route_deviation_zone_ids: Tuple[str, ...] = field(default_factory=tuple)

    highest_anomaly_severity: Optional[str] = None
    critical_anomaly_occupant_ids: Tuple[str, ...] = field(default_factory=tuple)

    zone_details: Mapping[str, TrajectoryZoneDetail] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "zone_details", MappingProxyType(dict(self.zone_details)))

    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "available": self.available,
            "timestamp": self.timestamp,
            "route_deviation_zone_ids": list(self.route_deviation_zone_ids),
            "movement_stalled_zone_ids": list(self.movement_stalled_zone_ids),
            "hazardous_zone_movement_zone_ids": list(self.hazardous_zone_movement_zone_ids),
            "no_safe_route_zone_ids": list(self.no_safe_route_zone_ids),
            "shared_route_deviation_zone_ids": list(self.shared_route_deviation_zone_ids),
            "highest_anomaly_severity": self.highest_anomaly_severity,
            "critical_anomaly_occupant_ids": list(self.critical_anomaly_occupant_ids),
            "zone_details": {k: v.to_dict() for k, v in self.zone_details.items()},
        }


UNAVAILABLE_TRAJECTORY_DECISION_EVIDENCE = TrajectoryDecisionEvidence(available=False)
