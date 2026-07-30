from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from stair_flow.models import StairFlowMetrics


# =====================================================
# Live Occupancy, Crowd Density & Congestion Intelligence milestone --
# the immutable output models this package produces. Every convention
# here mirrors what this codebase already established for HazardSnapshot/
# OccupancySnapshot/BuildingState/LiveAIPredictionSnapshot: frozen
# dataclasses, Mapping fields MappingProxyType-wrapped in __post_init__,
# and "Optional means genuinely unavailable, never a fabricated zero"
# throughout. Nothing here computes anything -- see density.py/flow.py/
# congestion.py/queue.py/trends.py/engine.py for that; this module only
# defines the shapes.
# =====================================================


class IntensityLevel(Enum):

    # Deliberately reused for BOTH density classification (Phase 4) and
    # congestion classification (Phase 6) -- the same neutral,
    # engineering-judgment scale applies to either "how packed is this
    # zone" or "how congested is this asset", and giving them two
    # separate enums with identical members would only invite the two
    # to drift apart for no reason. NOT a validated life-safety standard
    # -- see DensityThresholds' own docstring below for exactly what
    # this is and is not.

    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    VERY_HIGH = auto()
    CRITICAL = auto()


class TrendDirection(Enum):

    # Phase 8's own required classification for temporal trends (density/
    # queue/flow/congestion). UNKNOWN is the honest default whenever
    # there is not yet enough bounded history to compare against (a
    # single sample, or an empty window) -- never guessed as STABLE.

    RISING = auto()
    STABLE = auto()
    FALLING = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class DensityThresholds:

    # Phase 4's own explicit requirement: "Do NOT hardcode undocumented
    # safety limits as universal truth... Thresholds must be
    # configurable and clearly documented as project assumptions unless
    # backed by an existing standard already represented in the
    # repository." No existing standard for people/m^2 crowd density is
    # represented anywhere in this repository (confirmed directly --
    # hazard/severity.py's own HazardSeverity.from_score() cutoffs are
    # the closest precedent, and those are themselves documented as "a
    # placeholder classification, not a validated life-safety threshold
    # model"; simulator/congestion.py's DefaultCongestionModel and
    # simulator/capacity.py's DefaultCapacityModel carry the identical
    # disclosure). These defaults are therefore an equally explicit
    # PROJECT ASSUMPTION, not a fire-egress code value, and are a plain
    # dataclass (not a classmethod-with-fixed-cutoffs the way
    # HazardSeverity.from_score() is) specifically so a caller CAN
    # reconfigure them per deployment without editing this file.
    #
    # Units: people per square meter. Boundaries are inclusive on the
    # lower bound of each named level (a density exactly AT a cutoff
    # belongs to the higher level), mirroring HazardSeverity.from_score()'s
    # own >= convention.

    moderate_at: float = 1.0
    high_at: float = 2.0
    very_high_at: float = 3.0
    critical_at: float = 4.0

    def classify(self, density: Optional[float]) -> Optional[IntensityLevel]:

        if density is None:
            return None

        if density >= self.critical_at:
            return IntensityLevel.CRITICAL

        if density >= self.very_high_at:
            return IntensityLevel.VERY_HIGH

        if density >= self.high_at:
            return IntensityLevel.HIGH

        if density >= self.moderate_at:
            return IntensityLevel.MODERATE

        return IntensityLevel.LOW


@dataclass(frozen=True)
class ZoneCrowdMetrics:

    # Phase 3's own required per-zone shape. occupant_count/density are
    # always computed from the SAME canonical global-occupant source
    # every other zone-level metric in this codebase already uses
    # (live_occupants.manager.LiveOccupantManager.active_occupants()) --
    # never raw per-camera detections (Phase 12).
    #
    # temporarily_lost_count is DELIBERATELY excluded from occupant_count/
    # density -- it counts occupants whose LAST KNOWN zone was this one
    # but who are not currently active (live_occupants.state.
    # OccupantStatus.TEMPORARILY_LOST), an honest diagnostic signal, not
    # a second, disagreeing headcount (Phase 12's "no double-count"
    # discipline applies here too: a temporarily-lost occupant must
    # never be counted in BOTH this field and occupant_count/density at
    # once).
    #
    # position_coverage_fraction is None if occupant_count is 0 (there is
    # nothing to have a fraction of) -- never a fabricated 1.0 or 0.0.

    zone_id: str

    occupant_count: int = 0

    zone_area: Optional[float] = None
    density_people_per_m2: Optional[float] = None
    density_classification: Optional[IntensityLevel] = None

    moving_count: int = 0
    stationary_count: int = 0
    running_count: int = 0
    temporarily_lost_count: int = 0

    mean_speed: Optional[float] = None

    position_coverage_count: int = 0
    position_coverage_fraction: Optional[float] = None

    trend: TrendDirection = TrendDirection.UNKNOWN


@dataclass(frozen=True)
class AssetApproachMetrics:

    # Phase 5/6/7's shared shape for a Door, Exit, or Stair -- one
    # instance per asset id, regardless of which of the three
    # door_metrics/exit_metrics/stair_metrics mappings it lives in.
    #
    # position_available=False means this asset's approach/queue fields
    # (approaching_count, queue_candidate_count, estimated_queue_length,
    # mean_approach_speed) are honestly None/0 because NO occupant near
    # this asset currently has a usable world_position -- never silently
    # computed from zone occupancy alone (Phase 7's own explicit "never
    # fabricate a queue from zone occupancy alone").
    #
    # simulation_style_capacity is the SAME number simulator.capacity.
    # DefaultCapacityModel/StairCapacityModel would derive for this
    # asset's own width/explicit capacity -- reused directly (Phase 6),
    # but named and documented so it is never confused with a measured
    # live flow rate: it is a design/engineering capacity ESTIMATE, not
    # an empirically observed one.

    asset_id: str
    asset_type: str  # "Door" | "Exit" | "Stair" -- matches navigation.edge.Edge's own EDGE_TYPES vocabulary

    position_available: bool = False

    approaching_count: int = 0
    queue_candidate_count: int = 0
    estimated_queue_length: Optional[int] = None
    mean_approach_speed: Optional[float] = None

    simulation_style_capacity: Optional[int] = None

    congestion_level: Optional[IntensityLevel] = None

    trend: TrendDirection = TrendDirection.UNKNOWN

    # Observable Asset Perception Framework milestone (generalized from
    # the Observable Stair Perception milestone's own observed_on_
    # stair_count, which had no Stair-specific logic behind it at all) --
    # a SEPARATE, DISTINCT measurement from approaching_count/
    # queue_candidate_count above, never a replacement for either: those
    # describe occupants NEAR this asset (landing-proximity, world_
    # position-based); this describes occupants PHYSICALLY WITHIN the
    # asset's own observable region (see camera_calibration.projection.
    # WorldProjection.asset_id and observable_assets.models.
    # AssetObservation). Only ever populated for asset_type == "Stair"
    # today -- CrowdIntelligenceEngine's own building index
    # (_index_building()) still only walks floor.stairs; a future Door/
    # Exit integration would extend that same index, not this field's
    # shape, which is already type-agnostic. None means "not measured" --
    # either this asset's type has no such measurement wired up yet, or
    # it does but no calibrated camera covers its observable region this
    # cycle (see observable_assets.models.ObservationStatus.UNKNOWN) --
    # an int (including honestly 0) means a calibrated camera DOES cover
    # it and this is what it currently sees. Same "None means no reading,
    # never a fabricated zero" convention as occupancy.observation.
    # OccupancyObservation.occupant_count already establishes elsewhere
    # in this codebase.
    observed_occupant_count: Optional[int] = None


@dataclass(frozen=True)
class BuildingCrowdSummary:

    # Phase 9's building-wide roll-up -- purely a report of already-
    # computed zone/asset metrics (max/filter over them), never a new
    # computation of its own, and never an evacuation decision (Phase 9's
    # own "this package reports state, it does not decide").

    total_observed_occupants: int = 0
    calibrated_occupant_count: int = 0
    position_coverage_fraction: Optional[float] = None

    highest_density_zone: Optional[str] = None
    most_congested_asset_id: Optional[str] = None
    most_congested_asset_type: Optional[str] = None

    zones_above_configured_density_threshold: Tuple[str, ...] = field(default_factory=tuple)

    congested_doors: Tuple[str, ...] = field(default_factory=tuple)
    congested_exits: Tuple[str, ...] = field(default_factory=tuple)
    congested_stairs: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CrowdIntelligenceSnapshot:

    # Phase 3's top-level output -- the ONE object crowd_intelligence.
    # engine.CrowdIntelligenceEngine.compute() produces each cycle, and
    # the one object live_system.crowd_intelligence_gateway wires into
    # LiveBuildingSnapshot.crowd_intelligence (Phase 11). Immutable, same
    # "replace, never mutate" convention as every sibling snapshot type.

    timestamp: float = 0.0

    zone_metrics: Mapping[str, ZoneCrowdMetrics] = field(default_factory=dict)
    door_metrics: Mapping[str, AssetApproachMetrics] = field(default_factory=dict)
    exit_metrics: Mapping[str, AssetApproachMetrics] = field(default_factory=dict)
    stair_metrics: Mapping[str, AssetApproachMetrics] = field(default_factory=dict)

    # Live Stair Flow & Movement Direction Intelligence milestone
    # (additive) -- a GENUINELY separate measurement from stair_metrics
    # above (AssetApproachMetrics.observed_occupant_count is a point-in-
    # time occupancy fact; StairFlowMetrics is a WINDOWED throughput fact
    # -- entries/exits/rate/direction), never merged into
    # AssetApproachMetrics itself (that shape is explicitly shared across
    # Door/Exit/Stair, and entry/exit/direction semantics do not apply to
    # Door/Exit today -- see stair_flow/models.py's own docstring). Keyed
    # by stair_id, same "one mapping per asset kind" convention door_
    # metrics/exit_metrics/stair_metrics already establish. Computed by
    # stair_flow.compute.compute_stair_flow_snapshot(), called from
    # CrowdIntelligenceEngine.compute() itself (Phase 10's own "prefer
    # extending the existing Crowd Intelligence snapshot rather than
    # creating another intelligence engine") -- never a second, separately-
    # scheduled engine. Empty (never fabricated entries) for any stair
    # this cycle genuinely has no evidence for.
    stair_flow_metrics: Mapping[str, StairFlowMetrics] = field(default_factory=dict)

    building_summary: BuildingCrowdSummary = field(default_factory=BuildingCrowdSummary)

    def __post_init__(self):

        object.__setattr__(self, "zone_metrics", MappingProxyType(dict(self.zone_metrics)))
        object.__setattr__(self, "door_metrics", MappingProxyType(dict(self.door_metrics)))
        object.__setattr__(self, "exit_metrics", MappingProxyType(dict(self.exit_metrics)))
        object.__setattr__(self, "stair_metrics", MappingProxyType(dict(self.stair_metrics)))
        object.__setattr__(self, "stair_flow_metrics", MappingProxyType(dict(self.stair_flow_metrics)))

    # =====================================================

    def zone(self, zone_id: str) -> Optional[ZoneCrowdMetrics]:

        return self.zone_metrics.get(zone_id)

    def door(self, door_id: str) -> Optional[AssetApproachMetrics]:

        return self.door_metrics.get(door_id)

    def exit(self, exit_id: str) -> Optional[AssetApproachMetrics]:

        return self.exit_metrics.get(exit_id)

    def stair(self, stair_id: str) -> Optional[AssetApproachMetrics]:

        return self.stair_metrics.get(stair_id)

    def stair_flow(self, stair_id: str) -> Optional[StairFlowMetrics]:

        return self.stair_flow_metrics.get(stair_id)
