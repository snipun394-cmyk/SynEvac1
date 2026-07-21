from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple


# =====================================================
# Live Evacuation Progress, Flow & Clearance Intelligence milestone --
# Phase 13's own required "Evacuation Progress Evidence" structure,
# mirroring advisory_system.crowd_evidence.CrowdDecisionEvidence and
# advisory_system.ai_evidence.AIDecisionEvidence exactly in spirit: a
# small, immutable, PLAIN-VALUE reduction of evacuation_progress.models.
# EvacuationProgressSnapshot -- never the mutable engine, never the
# whole snapshot. This module imports NOTHING from evacuation_progress/
# (every field below is a plain bool/float/str/tuple/mapping-of-plain-
# values) -- the same "advisory_system gains no new package dependency"
# discipline every sibling evidence module already establishes. The
# adapter that reduces a real EvacuationProgressSnapshot to these plain
# values lives in live_system.live_advisory_gateway, exactly where the
# crowd/AI equivalents already live.
#
# THIS IS SUPPORTING EVIDENCE ONLY (Phase 13's own explicit final
# line) -- see docs/architecture/live_evacuation_progress.md's own
# Safety Precedence section for the full chain of reasoning.
# =====================================================


@dataclass(frozen=True)
class EvacuationZoneDetail:

    # Included in EvacuationProgressEvidence.zone_details ONLY for
    # zones already worth Advisory's attention (stalled, clearance-
    # unknown, or observed-clear) -- never one entry per zone in the
    # building.

    status: Optional[str] = None
    clearance_fraction: Optional[float] = None
    current_active_count: int = 0
    trend: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "status": self.status, "clearance_fraction": self.clearance_fraction,
            "current_active_count": self.current_active_count, "trend": self.trend,
        }


@dataclass(frozen=True)
class EvacuationExitDetail:

    unique_exited_count: int = 0
    queue_candidate_count: int = 0
    recent_flow_per_minute: Optional[float] = None
    trend: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "unique_exited_count": self.unique_exited_count, "queue_candidate_count": self.queue_candidate_count,
            "recent_flow_per_minute": self.recent_flow_per_minute, "trend": self.trend,
        }


@dataclass(frozen=True)
class EvacuationProgressEvidence:

    # `available=False` (every other field left at its None/empty
    # default) is the honest "no evacuation-progress evidence this
    # cycle" value -- UNAVAILABLE_EVACUATION_PROGRESS_EVIDENCE below is
    # the one canonical instance, never fabricated as an all-clear
    # placeholder.

    available: bool

    timestamp: Optional[float] = None

    overall_progress_fraction: Optional[float] = None
    overall_progress_trend: Optional[str] = None

    stalled_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    zones_observed_clear: Tuple[str, ...] = field(default_factory=tuple)
    zones_clearance_unknown: Tuple[str, ...] = field(default_factory=tuple)

    slow_flow_exit_ids: Tuple[str, ...] = field(default_factory=tuple)
    high_queue_low_flow_exit_ids: Tuple[str, ...] = field(default_factory=tuple)

    observability_fraction: Optional[float] = None

    zone_details: Mapping[str, EvacuationZoneDetail] = field(default_factory=dict)
    exit_details: Mapping[str, EvacuationExitDetail] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "zone_details", MappingProxyType(dict(self.zone_details)))
        object.__setattr__(self, "exit_details", MappingProxyType(dict(self.exit_details)))

    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "available": self.available,
            "timestamp": self.timestamp,
            "overall_progress_fraction": self.overall_progress_fraction,
            "overall_progress_trend": self.overall_progress_trend,
            "stalled_zone_ids": list(self.stalled_zone_ids),
            "zones_observed_clear": list(self.zones_observed_clear),
            "zones_clearance_unknown": list(self.zones_clearance_unknown),
            "slow_flow_exit_ids": list(self.slow_flow_exit_ids),
            "high_queue_low_flow_exit_ids": list(self.high_queue_low_flow_exit_ids),
            "observability_fraction": self.observability_fraction,
            "zone_details": {k: v.to_dict() for k, v in self.zone_details.items()},
            "exit_details": {k: v.to_dict() for k, v in self.exit_details.items()},
        }


UNAVAILABLE_EVACUATION_PROGRESS_EVIDENCE = EvacuationProgressEvidence(available=False)
