from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from hazard.severity import HazardSeverity


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone -- the immutable output models this package
# produces. Same conventions as crowd_intelligence.models/
# evacuation_progress.models/emergency_response.models throughout:
# frozen dataclasses, Mapping fields MappingProxyType-wrapped, plain
# string constants (not stdlib Enum) for values also written into
# advisory_system's own plain-value evidence, "Optional/UNKNOWN means
# genuinely unavailable, never a fabricated value" everywhere.
#
# This package OBSERVES MOVEMENT GEOMETRY ONLY. It never infers panic,
# confusion, or non-compliance -- see docs/architecture/
# live_trajectory_route_deviation_intelligence.md's own "what this is
# not" section. It never recommends an exit/stair that deterministic
# safety logic (BuildingState.hazard_summary / structural Edge.
# traversable) considers unsafe, and it never executes/dispatches/
# broadcasts anything.
# =====================================================


class MovementStatus:

    # Purely a physical-movement classification derived from an
    # occupant's own recent position samples (speed) -- deliberately
    # NOT the same concept as RouteProgressStatus (whether that motion
    # is USEFUL evacuation progress) or AnomalyFlag.MOVEMENT_STALLED
    # (whether route progress has stalled over time). A person can be
    # MOVING while still MOVEMENT_STALLED in route terms.

    MOVING = "MOVING"
    STATIONARY = "STATIONARY"
    UNKNOWN = "UNKNOWN"


class RouteProgressStatus:

    # Graph-aware only -- never inferred from Euclidean distance alone
    # when the Navigation Graph provides connectivity (Phase 5's own
    # explicit "moving closer through a wall must not count" rule).

    PROGRESSING_TOWARD_EXIT = "PROGRESSING_TOWARD_EXIT"
    NOT_PROGRESSING = "NOT_PROGRESSING"
    MOVING_AWAY_FROM_EXIT = "MOVING_AWAY_FROM_EXIT"
    ROUTE_UNCERTAIN = "ROUTE_UNCERTAIN"
    NO_SAFE_ROUTE = "NO_SAFE_ROUTE"
    EXIT_APPROACHING = "EXIT_APPROACHING"


class RouteDistanceTrend:

    DECREASING = "DECREASING"
    STABLE = "STABLE"
    INCREASING = "INCREASING"
    UNKNOWN = "UNKNOWN"


class AnomalyFlag:

    # Every flag is DECISION-SUPPORT EVIDENCE ONLY (Phase 8's own
    # explicit "not proof of confusion" boundary) -- geometry-derived,
    # deterministic, never a psychological or compliance judgment.

    MOVING_AWAY_FROM_SAFE_EXIT = "MOVING_AWAY_FROM_SAFE_EXIT"
    REPEATED_ROUTE_REVERSAL = "REPEATED_ROUTE_REVERSAL"
    MOVEMENT_STALLED = "MOVEMENT_STALLED"
    ENTERED_HAZARDOUS_ZONE = "ENTERED_HAZARDOUS_ZONE"
    REMAINS_IN_HAZARDOUS_ZONE = "REMAINS_IN_HAZARDOUS_ZONE"
    AGAINST_DOMINANT_FLOW = "AGAINST_DOMINANT_FLOW"
    NO_SAFE_ROUTE = "NO_SAFE_ROUTE"
    SHARED_ROUTE_DEVIATION = "SHARED_ROUTE_DEVIATION"
    MULTI_OCCUPANT_REVERSAL = "MULTI_OCCUPANT_REVERSAL"


class AnomalySeverity:

    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


_SEVERITY_RANK = {
    HazardSeverity.NONE: 0,
    HazardSeverity.LOW: 1,
    HazardSeverity.MODERATE: 2,
    HazardSeverity.HIGH: 3,
    HazardSeverity.CRITICAL: 4,
}


def hazard_severity_at_least(severity: Optional[HazardSeverity], floor: HazardSeverity) -> bool:

    if severity is None:
        return False

    return _SEVERITY_RANK[severity] >= _SEVERITY_RANK[floor]


@dataclass(frozen=True)
class TrajectoryConfig:

    # Every threshold below is a documented, configurable project
    # assumption -- same disclosure discipline as emergency_response.
    # models.ResponseWeights/crowd_intelligence.models.DensityThresholds.
    # None is a validated life-safety standard.

    # Phase 4 -- bounded position history. LiveOccupant.history already
    # retains position_samples/velocity_samples/zone_transitions bounded
    # by LiveOccupantManager's own history_length (default 30) -- this
    # engine reuses that directly rather than duplicating occupant
    # identity/lifecycle ownership. max_route_distance_samples bounds
    # the ONE piece of state this engine itself owns (see history.py).
    max_route_distance_samples: int = 30

    # Phase 7 -- route distance trend.
    route_distance_tolerance_m: float = 1.0
    trend_window_samples: int = 3

    # Phase 5 -- "very close to a safe exit" overrides trend noise.
    exit_approaching_distance_m: float = 5.0

    # Phase 8 -- persistent moving-away requires BOTH a minimum sample
    # count and a minimum duration, never a single noisy frame.
    moving_away_min_samples: int = 3
    moving_away_min_duration_seconds: float = 6.0

    # Phase 9 -- reversal detection. reversal_window_transitions bounds
    # how far back into OccupantHistory.zone_transitions/position_samples
    # a reversal search looks; reversal_min_events is how many distinct
    # A-B-A patterns must be found before flagging (never one reversal).
    reversal_window_transitions: int = 8
    reversal_min_events: int = 2
    reversal_min_angle_degrees: float = 150.0  # world-space fallback only

    # Phase 10 -- movement stall: insufficient ROUTE progress for this
    # long, independent of physical movement/RecognizedBehavior.
    movement_stall_duration_seconds: float = 15.0

    # Phase 6/11 -- hazard-aware safety. hazard_unsafe_severity_floor is
    # the floor at/above which a zone is excluded entirely from safe-
    # route candidates (a hard exclusion, never a soft cost penalty --
    # Phase 23's own "never recommend an unsafe route" requirement).
    # hazard_zone_entry_floor is the (documented, deliberately lower)
    # floor at/above which ENTERED_HAZARDOUS_ZONE/REMAINS_IN_HAZARDOUS_
    # ZONE are reported -- zone-level only, since live BuildingState
    # exposes no live edge-level hazard reading beyond structural Edge.
    # traversable (see docs/architecture/
    # live_trajectory_route_deviation_intelligence.md's own investigation
    # notes) -- MOVES_TOWARD_HAZARD is deliberately NOT implemented for
    # the same reason: no gradient exists to honestly support it.
    hazard_unsafe_severity_floor: HazardSeverity = HazardSeverity.HIGH
    hazard_zone_entry_severity_floor: HazardSeverity = HazardSeverity.MODERATE

    # Phase 12 -- against-flow. Building-local (per-floor) dominant
    # direction, computed only once genuine evidence exists.
    against_flow_min_occupants: int = 5
    against_flow_min_coverage_fraction: float = 0.5
    against_flow_angle_threshold_degrees: float = 135.0
    stationary_speed_threshold_m_s: float = 0.15  # excluded from flow direction sampling

    # Phase 13 -- group/shared anomalies.
    group_min_size: int = 2

    # Phase 15/17 -- staleness (camera-offline / long blind interval).
    staleness_seconds: float = 20.0


@dataclass(frozen=True)
class AnomalyWeights:

    # Phase 14's own required transparency -- every contribution is a
    # documented, disclosed weight, mirroring emergency_response.models.
    # ResponseWeights exactly. Never clamped to [0, 1], never presented
    # as a probability.

    moving_away_weight: float = 0.30
    reversal_weight: float = 0.20
    movement_stalled_weight: float = 0.25
    hazardous_zone_weight: float = 0.35
    against_flow_weight: float = 0.20
    no_safe_route_weight: float = 0.30
    shared_deviation_weight: float = 0.15


@dataclass(frozen=True)
class AnomalySeverityThresholds:

    critical_at: float = 0.60
    high_at: float = 0.40
    moderate_at: float = 0.20
    low_at: float = 0.05

    def classify(self, score: float) -> str:

        if score >= self.critical_at:
            return AnomalySeverity.CRITICAL

        if score >= self.high_at:
            return AnomalySeverity.HIGH

        if score >= self.moderate_at:
            return AnomalySeverity.MODERATE

        if score >= self.low_at:
            return AnomalySeverity.LOW

        return AnomalySeverity.NONE


@dataclass(frozen=True)
class TrajectoryResult:

    # One occupant's own current trajectory/route-deviation picture --
    # frozen, exactly like live_occupants.occupant.LiveOccupant itself.
    # Every field is either honestly derivable from LiveOccupant/
    # NavigationGraph/BuildingState this cycle, or an explicit
    # UNKNOWN/None -- Phase 3's own "do not add fields that cannot be
    # honestly calculated" requirement.

    occupant_id: str

    floor_id: Optional[str] = None
    zone_id: Optional[str] = None

    current_position: Optional[Tuple[float, float]] = None
    previous_position: Optional[Tuple[float, float]] = None

    distance_travelled: Optional[float] = None
    net_displacement: Optional[float] = None

    movement_direction: Optional[float] = None  # radians, previous -> current position
    current_speed: Optional[float] = None  # m/s

    trajectory_duration: Optional[float] = None  # seconds spanned by retained position samples

    zone_transition_history: Tuple[str, ...] = field(default_factory=tuple)

    movement_status: str = MovementStatus.UNKNOWN

    route_progress_status: str = RouteProgressStatus.ROUTE_UNCERTAIN
    route_distance_m: Optional[float] = None
    route_distance_trend: str = RouteDistanceTrend.UNKNOWN
    nearest_safe_exit_id: Optional[str] = None

    anomaly_flags: Tuple[str, ...] = field(default_factory=tuple)
    anomaly_severity: str = AnomalySeverity.NONE

    position_available: bool = False
    position_sample_count: int = 0

    stale: bool = False

    last_updated_at: float = 0.0

    def to_dict(self):

        return {
            "occupant_id": self.occupant_id,
            "floor_id": self.floor_id,
            "zone_id": self.zone_id,
            "current_position": self.current_position,
            "previous_position": self.previous_position,
            "distance_travelled": self.distance_travelled,
            "net_displacement": self.net_displacement,
            "movement_direction": self.movement_direction,
            "current_speed": self.current_speed,
            "trajectory_duration": self.trajectory_duration,
            "zone_transition_history": list(self.zone_transition_history),
            "movement_status": self.movement_status,
            "route_progress_status": self.route_progress_status,
            "route_distance_m": self.route_distance_m,
            "route_distance_trend": self.route_distance_trend,
            "nearest_safe_exit_id": self.nearest_safe_exit_id,
            "anomaly_flags": list(self.anomaly_flags),
            "anomaly_severity": self.anomaly_severity,
            "position_available": self.position_available,
            "position_sample_count": self.position_sample_count,
            "stale": self.stale,
            "last_updated_at": self.last_updated_at,
        }


@dataclass(frozen=True)
class GroupAnomalyRecord:

    # Phase 13's own required aggregate-only evidence -- produced ONLY
    # when at least TrajectoryConfig.group_min_size occupants genuinely
    # share the pattern (never a single occupant, never a fabricated
    # cause -- "SHARED_ROUTE_DEVIATION"/"MULTI_OCCUPANT_REVERSAL", never
    # "PANIC"/"HERDING").

    anomaly_type: str
    occupant_ids: Tuple[str, ...]
    zone_id: Optional[str]
    severity: str


@dataclass(frozen=True)
class TrajectoryIntelligenceSnapshot:

    timestamp: float = 0.0

    occupants: Mapping[str, TrajectoryResult] = field(default_factory=dict)
    group_anomalies: Tuple[GroupAnomalyRecord, ...] = field(default_factory=tuple)

    # Phase 12 -- per-floor dominant flow direction (radians), only for
    # floors with genuinely sufficient evidence. Absent (not present in
    # the mapping) means "insufficient evidence this cycle," never 0.0.
    dominant_flow_direction_by_floor: Mapping[str, float] = field(default_factory=dict)
    dominant_flow_coverage_by_floor: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "occupants", MappingProxyType(dict(self.occupants)))
        object.__setattr__(self, "dominant_flow_direction_by_floor", MappingProxyType(dict(self.dominant_flow_direction_by_floor)))
        object.__setattr__(self, "dominant_flow_coverage_by_floor", MappingProxyType(dict(self.dominant_flow_coverage_by_floor)))

    # =====================================================

    def occupant(self, occupant_id: str) -> Optional[TrajectoryResult]:

        return self.occupants.get(occupant_id)

    def occupants_with_flag(self, flag: str) -> Tuple[str, ...]:

        return tuple(sorted(
            occupant_id for occupant_id, result in self.occupants.items() if flag in result.anomaly_flags
        ))
