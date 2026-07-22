from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple


# =====================================================
# Live Emergency Response & Rescue Priority Intelligence milestone --
# the immutable output models this package produces. Same conventions
# as crowd_intelligence.models/evacuation_progress.models throughout:
# frozen dataclasses, Mapping fields MappingProxyType-wrapped, "Optional
# means genuinely unavailable, never a fabricated value" everywhere.
#
# This package RANKS AND EXPLAINS operational priority. It never
# dispatches, never executes, never overrides a deterministic safety
# decision -- see docs/architecture/live_emergency_response_intelligence.md's
# own Safety Precedence section.
# =====================================================


class ResponsePriorityLevel:

    # Plain string constants (matching evacuation_progress.models'
    # ExitEvidenceLevel/EvacuationProgressTrend convention) rather than
    # a stdlib Enum, since these values are also written directly into
    # EmergencyResponseEvidence (advisory_system-facing, plain-value-
    # only). UNKNOWN is reserved for a zone with genuinely ZERO
    # evidence of any kind (Phase 3's own honest floor) -- NOT the same
    # as "poor observability with zero occupants," which still gets a
    # real, actionable MODERATE-or-higher score (Phase 6/9's own
    # explicit "poor observability must not automatically mean LOW
    # priority" requirement) -- see engine.py's own scoring ladder.

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class ResponseReason:

    # Phase 7's own required "reason codes alongside the score" --
    # every one of these names a SPECIFIC, disclosed contribution the
    # scoring ladder in engine.py actually applies; never an
    # unexplained/opaque label.

    KNOWN_OCCUPANTS_PRESENT = "KNOWN_OCCUPANTS_PRESENT"
    POSSIBLE_ASSISTANCE_REQUIRED = "POSSIBLE_ASSISTANCE_REQUIRED"
    CONFIRMED_ASSISTANCE_REQUIRED = "CONFIRMED_ASSISTANCE_REQUIRED"

    # Live Human State & Assistance Perception Bridge milestone -- a
    # DISTINCT reason from CONFIRMED_ASSISTANCE_REQUIRED: HumanState.
    # BEING_ASSISTED means help is ALREADY underway for this occupant
    # (Phase 12/23 test 33's own "BEING_ASSISTED distinguishable from
    # unassisted FALLEN" requirement) -- still genuinely elevated
    # priority (the situation is not resolved yet), but never conflated
    # with an occupant who is fallen/crawling and NOT yet being helped.
    ASSISTANCE_IN_PROGRESS = "ASSISTANCE_IN_PROGRESS"

    # A conservative, disclosed, INFORMATIONAL-weight-only reason --
    # Phase 12's own explicit "be conservative... may affect assistance-
    # awareness, not fabricate incapacity" requirement. Never implies
    # CHILD=trapped, ELDERLY=injured, or WHEELCHAIR_USER=incapable of
    # self-evacuation; this reason exists purely to raise operator
    # awareness that a zone contains an occupant with one of these
    # observed classifications, nothing more.
    VULNERABLE_PERSON_OBSERVED = "VULNERABLE_PERSON_OBSERVED"

    EVACUATION_STALLED = "EVACUATION_STALLED"
    HAZARD_PRESENT = "HAZARD_PRESENT"
    HIGH_CONGESTION_RESTRICTING_EVACUATION = "HIGH_CONGESTION_RESTRICTING_EVACUATION"
    UNCERTAIN_OCCUPANCY = "UNCERTAIN_OCCUPANCY"
    FACP_ALARM_ACTIVE = "FACP_ALARM_ACTIVE"
    OBSERVED_CLEAR = "OBSERVED_CLEAR"

    # Live Occupant Trajectory, Movement Anomaly & Route-Deviation
    # Intelligence milestone, Phase 21 -- additive. Deliberately a
    # SINGLE, conservative reason covering every severe trajectory
    # anomaly case (HIGH/CRITICAL anomaly_severity, or NO_SAFE_ROUTE) --
    # route deviation is decision-support evidence, never proof of
    # confusion/panic/non-compliance (trajectory_intelligence.models's
    # own documented boundary), so it never becomes its own assistance
    # tier here either; it only ever nudges search/response priority.
    SEVERE_ROUTE_ANOMALY = "SEVERE_ROUTE_ANOMALY"


@dataclass(frozen=True)
class ResponseWeights:

    # Every weight below is a documented, configurable project
    # assumption -- same disclosure discipline as crowd_intelligence.
    # models.DensityThresholds/evacuation_progress's own
    # _PROGRESS_FINDING_CONFIDENCE; none is a validated life-safety
    # standard. priority_score is therefore an explicitly RELATIVE
    # ranking value (never clamped to [0, 1], never presented as a
    # probability) -- see engine.py's own compute_zone_priority() for
    # exactly how each weight is applied.

    occupants_weight: float = 0.25
    occupants_normalization_count: float = 3.0  # 3+ known occupants reaches this factor's full contribution

    possible_assistance_weight: float = 0.20
    confirmed_assistance_weight: float = 0.35  # deliberately stronger than possible (Phase 22 test 6)

    # Live Human State & Assistance Perception Bridge milestone --
    # deliberately WEAKER than confirmed_assistance_weight (help is
    # already underway, Phase 12/23 test 33) but still clearly stronger
    # than possible_assistance_weight (this is CONFIRMED evidence, not a
    # hedged heuristic).
    being_assisted_weight: float = 0.25

    # A conservative, small, disclosed contribution -- deliberately the
    # SMALLEST assistance-adjacent weight of all (never approaching
    # possible/confirmed/being-assisted's own magnitude), reflecting
    # "assistance-awareness, not fabricated incapacity" (Phase 12).
    vulnerable_classification_weight: float = 0.10

    stalled_weight: float = 0.20
    hazard_weight: float = 0.30
    congestion_restricting_weight: float = 0.15
    uncertainty_weight: float = 0.20
    facp_alarm_weight: float = 0.15

    # Live Occupant Trajectory, Movement Anomaly & Route-Deviation
    # Intelligence milestone, Phase 21 -- additive, deliberately modest
    # (smaller than confirmed_assistance_weight/hazard_weight): route
    # anomaly evidence may raise a zone's search/response priority, it
    # never substitutes for genuine assistance/hazard evidence.
    severe_route_anomaly_weight: float = 0.20


@dataclass(frozen=True)
class ResponsePriorityThresholds:

    critical_at: float = 0.65
    high_at: float = 0.40
    moderate_at: float = 0.15

    def classify(self, score: Optional[float]) -> str:

        if score is None:
            return ResponsePriorityLevel.UNKNOWN

        if score >= self.critical_at:
            return ResponsePriorityLevel.CRITICAL

        if score >= self.high_at:
            return ResponsePriorityLevel.HIGH

        if score >= self.moderate_at:
            return ResponsePriorityLevel.MODERATE

        return ResponsePriorityLevel.LOW


@dataclass(frozen=True)
class OccupantAssistanceSignal:

    # One occupant's own observed assistance signal -- Phase 4/5's own
    # careful boundary: `possible` is set by RecognizedBehavior.
    # POSSIBLY_FALLEN (a hedged geometric heuristic). `confirmed`/
    # `being_assisted` are set from genuine HumanState evidence -- as of
    # the Live Human State & Assistance Perception Bridge milestone,
    # this is now PRIMARILY LiveOccupant.human_state itself (the
    # reconciled, live-pipeline-sourced field human_evidence.
    # reconciliation now populates), with the caller-supplied
    # human_state_by_occupant_id override (engine.py's own compute()
    # docstring) consulted only when the occupant carries no live-
    # sourced state of its own -- see engine.py's own _assistance_signal()
    # for the exact precedence. `confirmed` (FALLEN/CRAWLING -- not yet
    # assisted) and `being_assisted` (BEING_ASSISTED -- help already
    # underway) are mutually exclusive tiers, both distinct from
    # `possible` -- Phase 5's own explicit "must NOT become
    # CONFIRMED_FALLEN" requirement, preserved structurally: nothing
    # here ever promotes `possible` into `confirmed`/`being_assisted`.

    occupant_id: str
    zone_id: Optional[str]
    possible: bool = False
    confirmed: bool = False
    being_assisted: bool = False


@dataclass(frozen=True)
class ZoneResponsePriority:

    zone_id: str
    floor_id: Optional[str] = None

    priority_level: str = ResponsePriorityLevel.UNKNOWN
    priority_score: Optional[float] = None

    known_occupant_count: int = 0
    possible_assistance_count: int = 0
    confirmed_assistance_count: int = 0

    # Live Human State & Assistance Perception Bridge milestone --
    # distinct from confirmed_assistance_count (Phase 12/23 test 33).
    being_assisted_count: int = 0
    vulnerable_person_observed: bool = False

    evacuation_stalled: bool = False
    hazard_severity: Optional[str] = None  # hazard.severity.HazardSeverity name, as a plain string
    clearance_status: Optional[str] = None  # evacuation_progress.models.ZoneClearanceStatus value
    observability_fraction: Optional[float] = None

    reason_codes: Tuple[str, ...] = field(default_factory=tuple)
    explanation: str = ""

    # Phase 18/19's own "explain human evidence per zone" requirement --
    # one entry per occupant currently in this zone who carries ANY
    # genuine human-evidence signal (a known classification, a known
    # state, or the POSSIBLY_FALLEN behavior heuristic) -- never one
    # entry per occupant in the building regardless of relevance.
    occupant_evidence: Tuple["OccupantEvidenceSummary", ...] = field(default_factory=tuple)

    timestamp: float = 0.0


@dataclass(frozen=True)
class OccupantEvidenceSummary:

    # A small, display-oriented, plain-value reduction of one occupant's
    # own currently-held human evidence -- Command Center/Advisory
    # facing, never a second copy of LiveOccupant itself.

    occupant_id: str
    human_state: Optional[str] = None  # perception.models.human_observation.HumanState name, as a plain string
    classification: Optional[str] = None  # HumanClassification name, None only when genuinely UNKNOWN
    possible_assistance: bool = False  # RecognizedBehavior.POSSIBLY_FALLEN


@dataclass(frozen=True)
class FloorResponseSummary:

    floor_id: str

    critical_zone_count: int = 0
    high_zone_count: int = 0
    moderate_zone_count: int = 0
    known_occupants_remaining: int = 0
    possible_assistance_count: int = 0
    being_assisted_count: int = 0


@dataclass(frozen=True)
class EmergencyResponseSnapshot:

    timestamp: float = 0.0

    zones: Mapping[str, ZoneResponsePriority] = field(default_factory=dict)
    floors: Mapping[str, FloorResponseSummary] = field(default_factory=dict)

    # Phase 11's own required deterministic ordered list -- highest
    # response priority first. Tie-breaking: priority_score (descending),
    # then floor display_order, then zone_id -- see engine.py's own
    # _order_zones().
    response_priority_order: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):

        object.__setattr__(self, "zones", MappingProxyType(dict(self.zones)))
        object.__setattr__(self, "floors", MappingProxyType(dict(self.floors)))

    # =====================================================

    def zone(self, zone_id: str) -> Optional[ZoneResponsePriority]:

        return self.zones.get(zone_id)

    def floor(self, floor_id: str) -> Optional[FloorResponseSummary]:

        return self.floors.get(floor_id)

    def highest_priority_zone_id(self) -> Optional[str]:

        return self.response_priority_order[0] if self.response_priority_order else None
