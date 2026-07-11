from dataclasses import dataclass, field
from types import MappingProxyType
from typing import List, Mapping, Optional
from uuid import uuid4

from hazard.severity import HazardSeverity

from pathfinding.route import Route


@dataclass(frozen=True)
class ZoneRecommendation:

    # One zone's evacuation picture for a single decision cycle --
    # reuses Route as-is (never redefines the shape of a path) and
    # HazardSeverity as-is (never invents a second severity scale).
    # is_reachable is False exactly when PathfindingEngine.nearest_exit
    # found no Route at all under current hazard conditions -- distinct
    # from is_unsafe, since a zone can be one without the other (a
    # CRITICAL zone with a clear route out, or a clear zone stranded by
    # someone else's blocked corridor).

    zone_id: str
    severity: HazardSeverity
    occupant_count: Optional[float]
    is_unsafe: bool
    is_reachable: bool
    recommended_exit_edge_id: Optional[str] = None
    recommended_route: Optional[Route] = None


@dataclass(frozen=True)
class PriorityEvacuationEntry:

    # rank is 1-based and unique within one DecisionRecommendation --
    # the order EvacuationPriorityRule.rank() returned, made explicit
    # rather than left implicit in list position, so a consumer that
    # only cares about "top 3" doesn't need to re-derive rank from
    # index.

    rank: int
    zone_id: str
    reason: str


@dataclass(frozen=True)
class BlockedRoute:

    # An edge HazardSnapshot currently asserts is not traversable --
    # stranded_zone_ids is populated only when this edge is also
    # structurally critical under BuildingAnalysisEngine.
    # critical_connectors() (i.e. its loss actually strands zones, not
    # merely one of several redundant paths) -- empty, not None, when
    # it isn't, so callers never branch on None vs [].

    edge_id: str
    edge_type: str
    stranded_zone_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class VoiceAnnouncement:

    zone_id: str
    message: str
    severity: HazardSeverity


@dataclass(frozen=True)
class FirefighterPriority:

    # target_type distinguishes "go to this zone" from "this blocked
    # connector is worth reopening" -- two different kinds of action a
    # single flat list of zone ids couldn't express. rank is 1-based
    # and unique within its own target_type-agnostic ordering.

    rank: int
    target_type: str  # "zone" | "connector"
    target_id: str
    reason: str


@dataclass(frozen=True)
class DecisionRecommendation:

    # The one output shape AIDecisionEngine (and any future RL engine
    # implementing DecisionEngine) ever produces -- immutable, so a
    # recommendation already handed to a consumer can't be mutated out
    # from under it, same convention HazardSnapshot/HazardContribution/
    # OccupancySnapshot already use. recommendation_id mirrors
    # HazardSnapshot.snapshot_id/OccupancySnapshot.snapshot_id's role:
    # a stable identity for replay/logging, independent of timestamp
    # or content.

    recommendation_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0

    zone_recommendations: Mapping[str, ZoneRecommendation] = field(default_factory=dict)

    priority_evacuation_order: List[PriorityEvacuationEntry] = field(default_factory=list)
    unsafe_zone_ids: List[str] = field(default_factory=list)
    blocked_routes: List[BlockedRoute] = field(default_factory=list)
    voice_announcements: List[VoiceAnnouncement] = field(default_factory=list)
    firefighter_priorities: List[FirefighterPriority] = field(default_factory=list)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(
            self, "zone_recommendations", MappingProxyType(dict(self.zone_recommendations)),
        )

    # =====================================================

    def zone_recommendation(self, zone_id) -> Optional[ZoneRecommendation]:

        return self.zone_recommendations.get(zone_id)
