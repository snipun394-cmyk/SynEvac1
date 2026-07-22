from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from hazard.severity import HazardSeverity


# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone -- the
# immutable output models this package produces. Same conventions as
# every prior live-intelligence package (crowd_intelligence.models/
# evacuation_progress.models/emergency_response.models/
# trajectory_intelligence.models): frozen dataclasses, Mapping fields
# MappingProxyType-wrapped, plain string constants (not stdlib Enum)
# for values also written into advisory_system's own plain-value
# evidence, "Optional/UNKNOWN means genuinely unavailable, never a
# fabricated value" everywhere.
#
# This engine sits AFTER deterministic safety evaluation, never before
# and never in place of it. It NEVER scores or recommends an exit that
# deterministic safety logic (BuildingState.hazard_summary + structural
# Edge.traversable) considers unsafe -- see ranking.SafeExitDistanceCalculator's
# own docstring for the mechanical guarantee. It has no execution
# authority whatsoever.
# =====================================================


class RecommendationStatus:

    RECOMMENDED = "RECOMMENDED"
    NO_SAFE_EXIT_AVAILABLE = "NO_SAFE_EXIT_AVAILABLE"


class RecommendationReason:

    # Phase 5's own required "every recommendation explains itself" --
    # every one of these names a SPECIFIC, disclosed contribution the
    # scoring ladder in scoring.py actually applies; never an
    # unexplained/opaque label.

    SHORTEST_SAFE_ROUTE = "SHORTEST_SAFE_ROUTE"
    LOW_CONGESTION = "LOW_CONGESTION"
    HIGH_CONGESTION = "HIGH_CONGESTION"
    QUEUE_PRESENT = "QUEUE_PRESENT"
    HIGH_THROUGHPUT = "HIGH_THROUGHPUT"
    LOW_THROUGHPUT = "LOW_THROUGHPUT"
    TRAJECTORY_PROGRESSING_NORMALLY = "TRAJECTORY_PROGRESSING_NORMALLY"
    TRAJECTORY_MOVING_AWAY = "TRAJECTORY_MOVING_AWAY"
    EMERGENCY_RESPONSE_ZONE_ELEVATED = "EMERGENCY_RESPONSE_ZONE_ELEVATED"

    # Phase 7/13's own "AI only supports, never drives ranking" --
    # applied as the SAME uniform contribution to every candidate for a
    # given zone this cycle (see scoring.score_candidate()), so its
    # presence can never, by construction, change the relative order of
    # two candidates -- only the absolute score/confidence context.
    AI_BOTTLENECK_RISK_LOW = "AI_BOTTLENECK_RISK_LOW"
    AI_BOTTLENECK_RISK_ELEVATED = "AI_BOTTLENECK_RISK_ELEVATED"

    GOOD_COVERAGE = "GOOD_COVERAGE"
    POOR_COVERAGE = "POOR_COVERAGE"

    NO_SAFE_EXIT_REACHABLE = "NO_SAFE_EXIT_REACHABLE"


@dataclass(frozen=True)
class RecommendationWeights:

    # Every weight below is a documented, configurable project
    # assumption -- same disclosure discipline as emergency_response.
    # models.ResponseWeights/trajectory_intelligence.models.AnomalyWeights.
    # None is a validated life-safety standard. Score is an explicitly
    # RELATIVE ranking value (never clamped to [0, 1], never presented
    # as a probability).

    route_distance_weight: float = 0.40
    congestion_weight: float = 0.20
    queue_weight: float = 0.10
    throughput_weight: float = 0.10
    trajectory_weight: float = 0.10
    emergency_response_weight: float = 0.05

    # Phase 7/13 -- deliberately the smallest weight, and applied
    # uniformly across every candidate in a zone (see scoring.py) so it
    # can never, by construction, be the deciding factor between two
    # otherwise-tied candidates' relative order.
    ai_support_weight: float = 0.05


@dataclass(frozen=True)
class RecommendationConfig:

    # Phase 3/6's own hard-safety floor -- mirrors trajectory_
    # intelligence.models.TrajectoryConfig.hazard_unsafe_severity_floor
    # exactly (an independently-owned copy, per this codebase's own
    # "sibling live-intelligence packages never import each other's
    # engine internals" convention -- see ranking.py's own docstring).
    hazard_unsafe_severity_floor: HazardSeverity = HazardSeverity.HIGH

    congestion_normalization_level: str = "CRITICAL"  # crowd_intelligence.models.IntensityLevel name representing full penalty
    queue_normalization_count: int = 5
    throughput_normalization_per_minute: float = 20.0

    # emergency_response.models.ResponsePriorityLevel names that
    # trigger EMERGENCY_RESPONSE_ZONE_ELEVATED for a candidate exit
    # whose own zone carries that priority level.
    emergency_response_penalty_levels: Tuple[str, ...] = ("HIGH", "CRITICAL")


@dataclass(frozen=True)
class ExitCandidate:

    # One safe exit's own ranked evidence for one zone -- frozen, never
    # mutated after ranking.rank_exits_for_zone() constructs it.

    exit_id: str
    route_distance_m: float

    congestion_level: Optional[str] = None
    queue_candidate_count: Optional[int] = None
    throughput_per_minute: Optional[float] = None
    trajectory_support: Optional[str] = None  # "PROGRESSING" | "MOVING_AWAY" | None (no/mixed evidence)
    emergency_response_elevated: bool = False

    score: float = 0.0
    reason_codes: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self):

        return {
            "exit_id": self.exit_id,
            "route_distance_m": self.route_distance_m,
            "congestion_level": self.congestion_level,
            "queue_candidate_count": self.queue_candidate_count,
            "throughput_per_minute": self.throughput_per_minute,
            "trajectory_support": self.trajectory_support,
            "emergency_response_elevated": self.emergency_response_elevated,
            "score": self.score,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ZoneEvacuationRecommendation:

    zone_id: str
    floor_id: Optional[str] = None

    status: str = RecommendationStatus.NO_SAFE_EXIT_AVAILABLE

    recommended_exit_id: Optional[str] = None

    # Phase 6 -- full ranked exit id list (recommended exit first),
    # never omitting an unsafe exit by mistake (unsafe exits are simply
    # never candidates at all -- see ranking.py). alternative_exit_ids
    # is ranked_exit_ids[1:], kept as its own field purely for a
    # convenient "everything except the top pick" read.
    ranked_exit_ids: Tuple[str, ...] = field(default_factory=tuple)
    alternative_exit_ids: Tuple[str, ...] = field(default_factory=tuple)

    candidates: Tuple[ExitCandidate, ...] = field(default_factory=tuple)

    reason_codes: Tuple[str, ...] = field(default_factory=tuple)
    explanation: str = ""

    # Phase 7 -- evidence QUALITY, never AI certainty. None means
    # genuinely no basis to estimate confidence at all (never a
    # fabricated 0.5).
    confidence: Optional[float] = None
    coverage_fraction: Optional[float] = None

    occupant_count: int = 0
    timestamp: float = 0.0

    def to_dict(self):

        return {
            "zone_id": self.zone_id,
            "floor_id": self.floor_id,
            "status": self.status,
            "recommended_exit_id": self.recommended_exit_id,
            "ranked_exit_ids": list(self.ranked_exit_ids),
            "alternative_exit_ids": list(self.alternative_exit_ids),
            "candidates": [c.to_dict() for c in self.candidates],
            "reason_codes": list(self.reason_codes),
            "explanation": self.explanation,
            "confidence": self.confidence,
            "coverage_fraction": self.coverage_fraction,
            "occupant_count": self.occupant_count,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class EvacuationRecommendationSnapshot:

    timestamp: float = 0.0

    # Phase 8 -- one entry per OCCUPIED zone only; an unoccupied zone
    # is simply absent from this mapping, never present with a
    # fabricated "nobody to recommend anything to" placeholder.
    zones: Mapping[str, ZoneEvacuationRecommendation] = field(default_factory=dict)

    # Building-wide currently-safe Exit ids this cycle -- the seam
    # live_system.orchestrator's own SAFE_EXIT_CHANGED transition
    # detection (a building-wide event, distinct from the per-zone
    # RECOMMENDATION_CHANGED) reads.
    safe_exit_ids: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):

        object.__setattr__(self, "zones", MappingProxyType(dict(self.zones)))

    # =====================================================

    def zone(self, zone_id: str) -> Optional[ZoneEvacuationRecommendation]:

        return self.zones.get(zone_id)
