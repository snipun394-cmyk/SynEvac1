from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple


# =====================================================
# The Recommendation Layer -- the immutable output models this package
# produces. Same conventions as every prior live-intelligence package
# (evacuation_recommendation.models/emergency_response.models/crowd_
# intelligence.models): frozen dataclasses, Mapping fields
# MappingProxyType-wrapped, plain string constants (not stdlib Enum),
# "Optional means genuinely unavailable, never a fabricated value."
#
# This package is a downstream, read-only ADAPTER over evacuation_
# recommendation/evacuation_guidance/emergency_response/crowd_
# intelligence/advisory_system -- it normalizes, ranks, deduplicates,
# and manages the lifecycle of their already-computed output. It
# never recomputes routing, hazard, congestion, or priority logic
# itself. See recommendation_layer/adapters/'s own module docstrings
# and docs/architecture/recommendation_layer.md.
# =====================================================


class RecommendationType:

    OCCUPANT_ROUTING = "OCCUPANT_ROUTING"
    HAZARD_AVOIDANCE = "HAZARD_AVOIDANCE"
    CONGESTION_MITIGATION = "CONGESTION_MITIGATION"
    EXIT_UTILIZATION = "EXIT_UTILIZATION"
    WARDEN_DISPATCH = "WARDEN_DISPATCH"
    SYSTEM_WARNING = "SYSTEM_WARNING"


class RecommendationPriority:

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


# Sort ordinal -- higher means more urgent. Used by RecommendationManager's
# own ranking, never recomputed anywhere else.
PRIORITY_ORDINAL = {
    RecommendationPriority.INFO: 0,
    RecommendationPriority.LOW: 1,
    RecommendationPriority.MEDIUM: 2,
    RecommendationPriority.HIGH: 3,
    RecommendationPriority.CRITICAL: 4,
}


class RecommendationStatus:

    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"


class TriggerCondition:

    # A controlled vocabulary every adapter draws from -- two candidates
    # sharing the same trigger_condition for the same zone/exit are
    # making the SAME real-world claim from (possibly) different
    # providers; this is exactly what RecommendationManager's dedup key
    # relies on to merge them into one Recommendation instead of two.

    ZONE_EXIT_RECOMMENDED = "ZONE_EXIT_RECOMMENDED"
    ZONE_NO_SAFE_EXIT = "ZONE_NO_SAFE_EXIT"
    ZONE_HAZARD_PRESENT = "ZONE_HAZARD_PRESENT"
    ZONE_HIGH_CONGESTION = "ZONE_HIGH_CONGESTION"
    EXIT_OVERUTILIZED = "EXIT_OVERUTILIZED"
    EXIT_UNDERUTILIZED_ALTERNATIVE = "EXIT_UNDERUTILIZED_ALTERNATIVE"
    ZONE_RESPONSE_ELEVATED = "ZONE_RESPONSE_ELEVATED"
    ZONE_ASSISTANCE_REQUIRED = "ZONE_ASSISTANCE_REQUIRED"
    RECOMMENDATION_LOW_CONFIDENCE = "RECOMMENDATION_LOW_CONFIDENCE"
    RECOMMENDATION_AI_BOTTLENECK_RISK = "RECOMMENDATION_AI_BOTTLENECK_RISK"
    GUIDANCE_INCONSISTENCY = "GUIDANCE_INCONSISTENCY"
    BUILDING_NO_SAFE_EXIT_WIDESPREAD = "BUILDING_NO_SAFE_EXIT_WIDESPREAD"


# The internal-provider names every adapter/manager tags provenance
# with -- a plain, controlled vocabulary (not the Python module name),
# so `primary_source`/`supporting_sources` read cleanly in a UI or log
# regardless of this package's own internal file layout.
class RecommendationSource:

    EVACUATION_RECOMMENDATION = "evacuation_recommendation"
    EVACUATION_GUIDANCE = "evacuation_guidance"
    EMERGENCY_RESPONSE = "emergency_response"
    CROWD_INTELLIGENCE = "crowd_intelligence"
    ADVISORY_SYSTEM = "advisory_system"

    # The one category with no upstream provider of its own -- a
    # building-wide redistribution opportunity derived purely from
    # already-ranked evacuation_recommendation data.
    RECOMMENDATION_LAYER = "recommendation_layer"


# Fixed provider-priority order for same-cycle dedup collisions --
# always-live sources outrank the advisory-gated one, and the self-
# derived category is the lowest-priority tiebreak. RecommendationManager
# is the ONLY place this order is consulted.
PROVIDER_PRIORITY_ORDER = (
    RecommendationSource.EVACUATION_RECOMMENDATION,
    RecommendationSource.EMERGENCY_RESPONSE,
    RecommendationSource.EVACUATION_GUIDANCE,
    RecommendationSource.CROWD_INTELLIGENCE,
    RecommendationSource.ADVISORY_SYSTEM,
    RecommendationSource.RECOMMENDATION_LAYER,
)


@dataclass(frozen=True)
class Recommendation:

    recommendation_id: str = ""

    type: str = ""
    priority: str = RecommendationPriority.INFO
    severity: Optional[str] = None
    status: str = RecommendationStatus.ACTIVE

    created_at: float = 0.0
    updated_at: float = 0.0

    # None while ACTIVE and currently reappearing every cycle; set the
    # first cycle a previously-active recommendation fails to reappear
    # (RecommendationManager's own grace-period clock), cleared again if
    # it reappears before this deadline. Never means "already expired" --
    # status is the authoritative field for that.
    expires_at: Optional[float] = None

    trigger_condition: str = ""

    affected_zones: Tuple[str, ...] = field(default_factory=tuple)
    affected_exits: Tuple[str, ...] = field(default_factory=tuple)

    confidence: Optional[float] = None

    explanation: str = ""
    technical_reason: str = ""

    supporting_evidence: Mapping[str, Any] = field(default_factory=dict)
    recommended_action: str = ""

    # Provenance -- populated by the adapter that first emits a
    # candidate (primary_source only) and then, when RecommendationManager
    # merges same-cycle candidates that share a dedup key, expanded with
    # every OTHER provider that independently corroborated the same
    # real-world claim this cycle (supporting_sources), plus which
    # provider each supporting_evidence key actually came from
    # (evidence_origin). Adapters never set supporting_sources/
    # evidence_origin themselves -- see manager.py.
    primary_source: str = ""
    supporting_sources: Tuple[str, ...] = field(default_factory=tuple)
    evidence_origin: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "supporting_evidence", MappingProxyType(dict(self.supporting_evidence)))
        object.__setattr__(self, "evidence_origin", MappingProxyType(dict(self.evidence_origin)))

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "recommendation_id": self.recommendation_id,
            "type": self.type,
            "priority": self.priority,
            "severity": self.severity,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "trigger_condition": self.trigger_condition,
            "affected_zones": list(self.affected_zones),
            "affected_exits": list(self.affected_exits),
            "confidence": self.confidence,
            "explanation": self.explanation,
            "technical_reason": self.technical_reason,
            "supporting_evidence": dict(self.supporting_evidence),
            "recommended_action": self.recommended_action,
            "primary_source": self.primary_source,
            "supporting_sources": list(self.supporting_sources),
            "evidence_origin": dict(self.evidence_origin),
        }


@dataclass(frozen=True)
class RecommendationSet:

    # A whole-collection aggregate (mirrors advisory_system.
    # recommendation_models.AdvisoryReport's own shape), not per-zone-
    # keyed like evacuation_recommendation.EvacuationRecommendationSnapshot --
    # one zone can carry multiple recommendation types simultaneously
    # (e.g. OCCUPANT_ROUTING and WARDEN_DISPATCH for the same zone_id).

    timestamp: float = 0.0

    # Already priority-sorted by RecommendationManager -- never
    # re-sorted by a consumer.
    recommendations: Tuple[Recommendation, ...] = field(default_factory=tuple)

    def active(self) -> Tuple[Recommendation, ...]:

        return tuple(r for r in self.recommendations if r.status == RecommendationStatus.ACTIVE)

    def by_type(self, type_: str) -> Tuple[Recommendation, ...]:

        return tuple(r for r in self.recommendations if r.type == type_)

    def for_zone(self, zone_id: str) -> Tuple[Recommendation, ...]:

        return tuple(r for r in self.recommendations if zone_id in r.affected_zones)

    def to_dict(self) -> dict:

        return {
            "timestamp": self.timestamp,
            "recommendations": [r.to_dict() for r in self.recommendations],
        }
