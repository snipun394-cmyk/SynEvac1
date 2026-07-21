from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple


# =====================================================
# Live Evacuation Progress, Flow & Clearance Intelligence milestone --
# the immutable output models this package produces. Same conventions
# as crowd_intelligence.models throughout: frozen dataclasses, Mapping
# fields MappingProxyType-wrapped, "Optional means genuinely
# unavailable, never a fabricated value" everywhere.
#
# THE central distinction this whole package holds to, everywhere:
#
#   CURRENT OCCUPANCY        -- live_occupants.manager.LiveOccupantManager's
#                                own active_occupants() right now (unchanged,
#                                already exists).
#   OBSERVED EVACUATION      -- THIS package: what has been directly,
#   PROGRESS                    deterministically observed so far, always
#                                phrased against a known-observed
#                                denominator, never the true building
#                                population (which this package cannot
#                                know and never claims to).
#   PREDICTED EVACUATION     -- live_system.live_ai_gateway's own,
#   TIME                        pre-existing, EXPERIMENTAL
#                                evacuation_time_experimental field. This
#                                package builds no prediction of any kind
#                                and never touches that field.
# =====================================================


class ExitEvidenceLevel:

    # Deliberately NOT an enum with a CONFIRMED member -- Phase 3's own
    # explicit "only add this distinction if the current data genuinely
    # supports it" instruction. Investigated directly: no turnstile,
    # door sensor, or any hard-confirmation signal exists anywhere in
    # this codebase for an exit crossing -- live_occupants.state.
    # OccupantStatus.EXITED is itself already documented as "an honest,
    # geometry-based guess ... not a certainty," recoverable the moment
    # real evidence contradicts it. This package therefore only ever
    # reports LIKELY_EXITED (never a stronger, unsupported CONFIRMED)
    # or UNKNOWN (lost track away from any modeled exit -- outcome
    # genuinely unknown, never assumed to be evacuation).

    LIKELY_EXITED = "LIKELY_EXITED"
    UNKNOWN = "UNKNOWN"


class EvacuationProgressTrend:

    # Phase 6's own required classification -- a plain string constant
    # group (matching this module's own ExitEvidenceLevel convention)
    # rather than a stdlib Enum, since these values are also written
    # directly into EvacuationProgressEvidence (advisory_system-facing,
    # plain-value-only, mirroring crowd_evidence's own discipline).

    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    SLOWING = "SLOWING"
    STALLED = "STALLED"
    UNKNOWN = "UNKNOWN"


class ZoneClearanceStatus:

    UNKNOWN = "UNKNOWN"
    OCCUPIED = "OCCUPIED"
    CLEARING = "CLEARING"
    NEARLY_CLEAR = "NEARLY_CLEAR"
    OBSERVED_CLEAR = "OBSERVED_CLEAR"
    STALLED = "STALLED"


@dataclass(frozen=True)
class EvacuationProgressConfig:

    # Every threshold below is a documented, configurable project
    # assumption -- same disclosure discipline as crowd_intelligence.
    # models.DensityThresholds/CongestionThresholds; none is a validated
    # life-safety standard.

    nearly_clear_fraction: float = 0.8
    flow_window_seconds: float = 60.0

    def __post_init__(self):

        if not (0.0 < self.nearly_clear_fraction <= 1.0):
            raise ValueError("nearly_clear_fraction must be in (0.0, 1.0]")

        if self.flow_window_seconds <= 0.0:
            raise ValueError("flow_window_seconds must be > 0")


@dataclass(frozen=True)
class ZoneClearance:

    zone_id: str

    # "Observed" (never true-population) baseline -- the count of
    # distinct occupant_ids EVER seen in this zone since tracking began
    # (grows monotonically; never re-derived from a fabricated design
    # occupancy). 0 means genuinely nobody has ever been tracked here.
    baseline_observed_count: int = 0

    current_active_count: int = 0
    cleared_count: int = 0
    clearance_fraction: Optional[float] = None

    status: str = ZoneClearanceStatus.UNKNOWN
    trend: str = EvacuationProgressTrend.UNKNOWN

    # Phase 7's own critical distinction -- whether at least one active
    # camera currently covers this zone (from BuildingState.
    # camera_observations, never from crowd_intelligence's own coverage
    # fraction, which is honestly None for a zone with zero occupants
    # and therefore useless for THIS specific question).
    observable: bool = False

    last_decrease_time: Optional[float] = None


@dataclass(frozen=True)
class ExitFlow:

    exit_id: str

    unique_exited_count: int = 0
    recent_flow_per_minute: Optional[float] = None

    # Reused directly from crowd_intelligence -- never recomputed
    # (Phase 10's own explicit "do not duplicate density/queue/approach/
    # congestion" requirement).
    queue_candidate_count: int = 0
    approaching_count: int = 0
    congestion_level: Optional[str] = None

    flow_active: bool = False
    trend: str = EvacuationProgressTrend.UNKNOWN

    # Honest degrade signal (Phase 16 test 16) -- True only when at
    # least one occupant near this exit currently has a usable world
    # position; False means exit-crossing attribution for this exit is
    # unavailable, not that nobody is using it.
    position_available: bool = False


@dataclass(frozen=True)
class ObservabilitySummary:

    total_observed_occupants: int = 0
    calibrated_occupant_count: int = 0
    position_coverage_fraction: Optional[float] = None

    zones_with_camera_coverage: Tuple[str, ...] = field(default_factory=tuple)
    zones_without_camera_coverage: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvacuationProgressSnapshot:

    timestamp: float = 0.0

    known_active_occupants: int = 0
    known_exited_occupants: int = 0
    known_total_observed_occupants: int = 0

    # None only when known_total_observed_occupants == 0 (no honest
    # denominator) -- never a fabricated 0%/100%.
    evacuation_progress_fraction: Optional[float] = None

    zones: Mapping[str, ZoneClearance] = field(default_factory=dict)
    exits: Mapping[str, ExitFlow] = field(default_factory=dict)

    overall_progress_trend: str = EvacuationProgressTrend.UNKNOWN

    stalled_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    low_flow_exit_ids: Tuple[str, ...] = field(default_factory=tuple)

    observability: ObservabilitySummary = field(default_factory=ObservabilitySummary)

    def __post_init__(self):

        object.__setattr__(self, "zones", MappingProxyType(dict(self.zones)))
        object.__setattr__(self, "exits", MappingProxyType(dict(self.exits)))

    # =====================================================

    def zone(self, zone_id: str) -> Optional[ZoneClearance]:

        return self.zones.get(zone_id)

    def exit(self, exit_id: str) -> Optional[ExitFlow]:

        return self.exits.get(exit_id)
