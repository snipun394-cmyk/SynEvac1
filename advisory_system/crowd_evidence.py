from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple


# =====================================================
# Live Crowd Intelligence -> Operational Advisory Integration milestone
# -- Phase 2/3's own required "Crowd Decision Evidence" structure,
# placed here in advisory_system (mirroring advisory_system.ai_evidence.
# AIDecisionEvidence exactly, field for field in spirit): a small,
# immutable, PLAIN-VALUE reduction of crowd_intelligence.models.
# CrowdIntelligenceSnapshot -- never the mutable runtime, never the
# whole snapshot, and this module itself imports NOTHING from
# crowd_intelligence/ (every field below is a plain bool/float/str/
# tuple/mapping-of-plain-values), the identical "advisory_system gains
# no new package dependency" discipline ai_evidence.py already
# established for AI. The adapter that actually reads a real
# CrowdIntelligenceSnapshot and reduces it to these plain values lives
# in live_system (see live_system.live_advisory_gateway.
# crowd_decision_evidence_from_snapshot()), exactly where
# ai_decision_evidence_from_prediction_snapshot() already lives for AI.
#
# UNLIKE AIDecisionEvidence, this type DOES carry per-asset/per-zone
# localization (congested_exit_ids, highest_density_zone_id, ...) --
# ai_evidence.py's own docstring explains at length why no such field
# exists on AIDecisionEvidence (no live-compatible localized signal
# existed anywhere in this codebase at the time that milestone shipped).
# crowd_intelligence/ is exactly the "future milestone that builds a
# real per-door congestion aggregation layer" that docstring anticipated
# -- so this type is deliberately the first Advisory-facing evidence
# object that CAN name a specific exit/stair/door/zone.
# =====================================================


@dataclass(frozen=True)
class CrowdZoneDetail:

    # A small, per-zone detail record -- included in CrowdDecisionEvidence.
    # zone_details ONLY for zones already worth Advisory's attention
    # (the highest-density zone, and any zone above the configured
    # density threshold) -- never one entry per zone in the building
    # (Phase 2's own "do not duplicate the entire CrowdIntelligenceSnapshot
    # unnecessarily").

    density_classification: Optional[str] = None
    density_people_per_m2: Optional[float] = None
    trend: Optional[str] = None
    position_coverage_fraction: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "density_classification": self.density_classification,
            "density_people_per_m2": self.density_people_per_m2,
            "trend": self.trend,
            "position_coverage_fraction": self.position_coverage_fraction,
        }


@dataclass(frozen=True)
class CrowdAssetDetail:

    # Mirrors CrowdZoneDetail one layer over, for a Door/Exit/Stair --
    # included in CrowdDecisionEvidence.asset_details only for assets
    # already flagged congested/queueing/trending (never all doors/
    # exits/stairs in the building).

    asset_type: Optional[str] = None
    congestion_level: Optional[str] = None
    trend: Optional[str] = None
    queue_candidate_count: int = 0
    approaching_count: int = 0
    position_available: bool = False

    def to_dict(self) -> Dict[str, Any]:

        return {
            "asset_type": self.asset_type,
            "congestion_level": self.congestion_level,
            "trend": self.trend,
            "queue_candidate_count": self.queue_candidate_count,
            "approaching_count": self.approaching_count,
            "position_available": self.position_available,
        }


@dataclass(frozen=True)
class CrowdDecisionEvidence:

    # `available=False` (every other field left at its None/empty
    # default) is the honest "no crowd evidence this cycle" value --
    # UNAVAILABLE_CROWD_DECISION_EVIDENCE below is the one canonical
    # instance, never fabricated as an all-clear/all-zero placeholder
    # (Phase 3's own explicit "do NOT fabricate LOW congestion, zero
    # queue, zero density when the data simply does not exist").

    available: bool

    timestamp: Optional[float] = None

    highest_density_zone_id: Optional[str] = None
    highest_density_level: Optional[str] = None

    most_congested_asset_id: Optional[str] = None
    most_congested_asset_type: Optional[str] = None
    most_congested_level: Optional[str] = None

    congested_exit_ids: Tuple[str, ...] = field(default_factory=tuple)
    congested_stair_ids: Tuple[str, ...] = field(default_factory=tuple)
    congested_door_ids: Tuple[str, ...] = field(default_factory=tuple)

    queue_detected_asset_ids: Tuple[str, ...] = field(default_factory=tuple)
    rising_congestion_asset_ids: Tuple[str, ...] = field(default_factory=tuple)
    clearing_congestion_asset_ids: Tuple[str, ...] = field(default_factory=tuple)

    # Phase 10/13's own honesty requirement, one layer up into Advisory:
    # an asset absent from congested_exit_ids/congested_stair_ids/
    # congested_door_ids means EITHER "confirmed not congested" OR
    # "no position coverage here at all, so genuinely unknown" -- this
    # tuple is what lets a caller (see advisory_engine.py's own "prefer
    # an alternate SAFE asset" logic) tell the two apart, so an asset
    # with zero coverage is never treated as a confirmed-clear
    # alternative.
    position_unavailable_asset_ids: Tuple[str, ...] = field(default_factory=tuple)

    # Building-wide observability -- Phase 10's own "carry observability
    # limitations into Advisory" requirement. None whenever there is no
    # honest denominator (e.g. zero occupants observed at all this
    # cycle) -- never a fabricated 1.0/0.0.
    position_coverage_fraction: Optional[float] = None

    zones_above_density_threshold: Tuple[str, ...] = field(default_factory=tuple)

    zone_details: Mapping[str, CrowdZoneDetail] = field(default_factory=dict)
    asset_details: Mapping[str, CrowdAssetDetail] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "zone_details", MappingProxyType(dict(self.zone_details)))
        object.__setattr__(self, "asset_details", MappingProxyType(dict(self.asset_details)))

    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "available": self.available,
            "timestamp": self.timestamp,
            "highest_density_zone_id": self.highest_density_zone_id,
            "highest_density_level": self.highest_density_level,
            "most_congested_asset_id": self.most_congested_asset_id,
            "most_congested_asset_type": self.most_congested_asset_type,
            "most_congested_level": self.most_congested_level,
            "congested_exit_ids": list(self.congested_exit_ids),
            "congested_stair_ids": list(self.congested_stair_ids),
            "congested_door_ids": list(self.congested_door_ids),
            "queue_detected_asset_ids": list(self.queue_detected_asset_ids),
            "rising_congestion_asset_ids": list(self.rising_congestion_asset_ids),
            "clearing_congestion_asset_ids": list(self.clearing_congestion_asset_ids),
            "position_unavailable_asset_ids": list(self.position_unavailable_asset_ids),
            "position_coverage_fraction": self.position_coverage_fraction,
            "zones_above_density_threshold": list(self.zones_above_density_threshold),
            "zone_details": {k: v.to_dict() for k, v in self.zone_details.items()},
            "asset_details": {k: v.to_dict() for k, v in self.asset_details.items()},
        }

    # =====================================================
    # Small, honest query helpers -- used by advisory_engine.py so it
    # never has to reach into congested_exit_ids/congested_stair_ids/
    # congested_door_ids by hand and risk picking the wrong tuple for a
    # given asset_type.

    def is_congested(self, asset_type: str, asset_id: str) -> bool:

        if asset_type == "Exit":
            return asset_id in self.congested_exit_ids

        if asset_type == "Stair":
            return asset_id in self.congested_stair_ids

        if asset_type == "Door":
            return asset_id in self.congested_door_ids

        return False


UNAVAILABLE_CROWD_DECISION_EVIDENCE = CrowdDecisionEvidence(available=False)
