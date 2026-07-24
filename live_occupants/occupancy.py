from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Mapping, Sequence, Tuple

from live_occupants.occupant import LiveOccupant


# =====================================================
# Canonical Live Occupancy Source of Truth milestone -- Phase 1's own
# investigation (docs/architecture/canonical_live_occupancy.md) found
# FOUR independent, hand-written "filter NEW/ACTIVE occupants, group by
# current_zone_id" loops (live_perception.providers.
# LiveOccupantObservationProvider, crowd_intelligence.density.
# compute_zone_metrics via crowd_intelligence.engine.CrowdIntelligenceEngine,
# evacuation_progress.engine.EvacuationProgressEngine, emergency_response.
# engine.EmergencyResponseIntelligenceEngine) that happened to agree
# today only because every one of them was independently written against
# the SAME live_occupants.manager.LiveOccupantManager.active_occupants()
# filter -- never mechanically tied together. OccupancyFacts is the
# SMALLEST possible immutable value object that collapses the GROUPING
# step (not the lifecycle-inclusion decision, which remains
# LiveOccupantManager.active_occupants()'s own job, unchanged) into one
# place, computed once per cycle (see LiveOccupantManager.
# canonical_occupancy()) and consumed everywhere.
#
# Deliberately narrow (Phase 4's own "do not add prediction or capacity
# semantics -- this is observed occupancy only" instruction): no
# density, no capacity, no confidence, no trend, no world position --
# every one of those stays a downstream, consumer-specific concern
# (Crowd Intelligence's own density/queue/congestion math, Evacuation
# Progress's own ledger, Emergency Response's own assistance scoring).
# Occupant IDs (not counts alone) are carried on every grouping so a
# caller can always resolve back to the full live_occupants.occupant.
# LiveOccupant object via LiveOccupantManager.get(occupant_id) --
# Phase 4's own explicit auditability/double-count-debugging
# requirement.
# =====================================================


@dataclass(frozen=True)
class OccupancyFacts:

    timestamp: float

    occupant_ids_by_zone: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    occupant_ids_by_floor: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    # Canonical Live Occupancy Source of Truth milestone, Phase 13 --
    # an occupant currently NEW/ACTIVE but with no resolved
    # current_zone_id (no reliable zone localization this cycle) is
    # NEVER silently dropped from total_observed_occupant_ids, and is
    # NEVER assigned a fabricated zone -- it appears here instead,
    # honestly distinct from every zone's own occupant_ids_by_zone
    # entry.
    unlocalized_occupant_ids: Tuple[str, ...] = ()

    # Every NEW/ACTIVE occupant this cycle, localized or not -- the
    # ONE canonical "how many people does this live session currently
    # know about" total. occupant_ids_by_zone's values, UNIONED with
    # unlocalized_occupant_ids, always exactly reconstructs this set
    # (Phase 13's own required invariant, proven in
    # tests/test_canonical_live_occupancy.py).
    total_observed_occupant_ids: Tuple[str, ...] = ()

    def __post_init__(self):

        # MappingProxyType-wrapped so the snapshot stays genuinely
        # immutable even though dict/dataclass equality still works --
        # the same technique occupancy.snapshot.OccupancySnapshot/
        # hazard.snapshot.HazardSnapshot already establish.
        object.__setattr__(
            self, "occupant_ids_by_zone",
            MappingProxyType({zone_id: tuple(ids) for zone_id, ids in self.occupant_ids_by_zone.items()}),
        )
        object.__setattr__(
            self, "occupant_ids_by_floor",
            MappingProxyType({floor_id: tuple(ids) for floor_id, ids in self.occupant_ids_by_floor.items()}),
        )

    # =====================================================

    @property
    def total_observed_count(self) -> int:

        return len(self.total_observed_occupant_ids)

    # =====================================================

    @property
    def unlocalized_count(self) -> int:

        return len(self.unlocalized_occupant_ids)

    # =====================================================

    def zone_count(self, zone_id: str) -> int:

        return len(self.occupant_ids_by_zone.get(zone_id, ()))

    # =====================================================

    def floor_count(self, floor_id: str) -> int:

        return len(self.occupant_ids_by_floor.get(floor_id, ()))


# =====================================================


def compute_occupancy_facts(occupants: Sequence[LiveOccupant], timestamp: float) -> OccupancyFacts:

    # A pure grouping function -- it does NOT decide who counts (that
    # remains live_occupants.manager.LiveOccupantManager.
    # active_occupants()'s own NEW/ACTIVE lifecycle filter, applied by
    # the one caller, LiveOccupantManager.canonical_occupancy(), below).
    # Kept a free function, separate from LiveOccupantManager, purely so
    # it stays trivially unit-testable against a hand-built occupant
    # list without constructing a whole manager (mirrors crowd_
    # intelligence.density.compute_zone_metrics's own "pure function,
    # caller supplies the pre-filtered list" convention, which this
    # function's own output now directly feeds).

    by_zone: Dict[str, list] = {}
    by_floor: Dict[str, list] = {}
    unlocalized: list = []
    total_ids: list = []

    for occupant in occupants:

        total_ids.append(occupant.occupant_id)

        if occupant.current_zone_id is not None:
            by_zone.setdefault(occupant.current_zone_id, []).append(occupant.occupant_id)
        else:
            unlocalized.append(occupant.occupant_id)

        if occupant.current_floor_id is not None:
            by_floor.setdefault(occupant.current_floor_id, []).append(occupant.occupant_id)

    return OccupancyFacts(
        timestamp=timestamp,
        occupant_ids_by_zone={zone_id: tuple(ids) for zone_id, ids in by_zone.items()},
        occupant_ids_by_floor={floor_id: tuple(ids) for floor_id, ids in by_floor.items()},
        unlocalized_occupant_ids=tuple(unlocalized),
        total_observed_occupant_ids=tuple(total_ids),
    )
