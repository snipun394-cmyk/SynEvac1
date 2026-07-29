import dataclasses
from typing import Dict, List, Optional, Tuple

from crowd_intelligence.capacity import door_capacity, exit_capacity, stair_capacity
from crowd_intelligence.congestion import CongestionThresholds, compute_congestion_level
from crowd_intelligence.density import compute_zone_metrics
from crowd_intelligence.flow import door_sides, exit_sides, stair_sides
from crowd_intelligence.models import (
    AssetApproachMetrics, BuildingCrowdSummary, CrowdIntelligenceSnapshot, DensityThresholds, IntensityLevel,
)
from crowd_intelligence.queue import DEFAULT_APPROACH_REGION_DEPTH, compute_queue_metrics
from crowd_intelligence.trends import TrendConfig, TrendTracker


# =====================================================
# Live Occupancy, Crowd Density & Congestion Intelligence milestone,
# Phase 10 -- the ONE object LiveRuntime owns (mirrors sensor_fusion.
# engine.SensorFusionEngine's own "exactly one shared instance per live
# session" role). compute(time) is the single entry point: reads
# LiveOccupantManager (Phase 12's canonical occupant source) + Building
# geometry (read-only, never mutated), and produces one
# CrowdIntelligenceSnapshot. Never makes an evacuation decision, never
# broadcasts, never executes a control, never touches FACP -- purely a
# deterministic reporting layer (Phase 9/18).
# =====================================================


class CrowdIntelligenceEngine:

    def __init__(
        self,
        building,
        live_occupant_manager,
        density_thresholds: Optional[DensityThresholds] = None,
        congestion_thresholds: Optional[CongestionThresholds] = None,
        trend_config: Optional[TrendConfig] = None,
        approach_region_depth: float = DEFAULT_APPROACH_REGION_DEPTH,
    ):

        self.building = building
        self.live_occupant_manager = live_occupant_manager

        self.density_thresholds = density_thresholds if density_thresholds is not None else DensityThresholds()
        self.congestion_thresholds = congestion_thresholds if congestion_thresholds is not None else CongestionThresholds()
        self.approach_region_depth = approach_region_depth

        self._trend_tracker = TrendTracker(trend_config)

        self._zones, self._doors, self._exits, self._stairs = self._index_building()

    # =====================================================

    def _index_building(self):

        zones = []
        doors = []
        exits = []
        stairs = []

        for floor in self.building.ordered_floors():

            zones.extend(floor.zones)
            doors.extend(floor.doors)
            exits.extend(floor.exits)
            stairs.extend(floor.stairs)

        return zones, doors, exits, stairs

    # =====================================================

    def compute(
        self, time: float, observed_stair_occupancy: Optional[Dict[str, Optional[int]]] = None,
    ) -> CrowdIntelligenceSnapshot:

        # Observable Stair Perception milestone -- OPTIONAL, additive.
        # `observed_stair_occupancy`: stair_id -> observed occupant count
        # (or None, meaning "not measured this cycle" -- see
        # AssetApproachMetrics.observed_on_stair_count's own docstring).
        # Deliberately a plain Dict[str, Optional[int]], never a
        # stair_perception.models.StairOccupancySnapshot import here --
        # this package's own architecture guard
        # (tests/test_crowd_intelligence_architecture_guards.py) keeps a
        # deliberately short, documented import allow-list, and a caller
        # (e.g. live_system's own orchestration glue) already has
        # everything needed to reduce a StairOccupancySnapshot to this
        # simple shape before calling compute() -- see
        # docs/architecture/live_stair_perception.md Sec 11. None (the
        # default) means "not supplied at all" -- every existing caller
        # keeps its exact pre-milestone behavior, every Stair's
        # observed_on_stair_count simply stays None (honestly
        # "not measured"), identical to how a Door/Exit's own
        # observed_on_stair_count already always does.
        observed_stair_occupancy = observed_stair_occupancy or {}

        # Canonical Live Occupancy Source of Truth milestone -- the ONE
        # per-cycle canonical grouping (memoized by LiveOccupantManager
        # itself, shared with the BuildingState occupancy provider,
        # Evacuation Progress, and Emergency Response -- see docs/
        # architecture/canonical_live_occupancy.md), fetched once here
        # and threaded through compute_zone_metrics()/_compute_building_
        # summary() below instead of either independently re-grouping
        # active_occupants() by zone.
        facts = self.live_occupant_manager.canonical_occupancy(time)

        active_occupants = self.live_occupant_manager.active_occupants()
        all_occupants = self.live_occupant_manager.all_occupants()

        zone_areas = {zone.id: (zone.area if zone.area > 0 else None) for zone in self._zones}

        zone_metrics = compute_zone_metrics(
            facts.occupant_ids_by_zone, active_occupants, all_occupants, zone_areas, self.density_thresholds,
        )

        zone_metrics = {
            zone_id: dataclasses.replace(
                metrics, trend=self._trend_tracker.observe(f"zone_density:{zone_id}", time, metrics.density_people_per_m2),
            )
            for zone_id, metrics in zone_metrics.items()
        }

        door_metrics = {
            door.id: self._compute_asset_metrics(
                "Door", door.id, door_sides(door), active_occupants, door_capacity(door), time,
            )
            for door in self._doors
        }

        exit_metrics = {
            exit_obj.id: self._compute_asset_metrics(
                "Exit", exit_obj.id, exit_sides(exit_obj), active_occupants, exit_capacity(exit_obj), time,
            )
            for exit_obj in self._exits
        }

        stair_metrics = {
            stair.id: self._compute_asset_metrics(
                "Stair", stair.id, stair_sides(stair), active_occupants, stair_capacity(stair, self.building), time,
                observed_on_stair_count=observed_stair_occupancy.get(stair.id),
            )
            for stair in self._stairs
        }

        building_summary = self._compute_building_summary(
            zone_metrics, door_metrics, exit_metrics, stair_metrics, active_occupants, facts,
        )

        return CrowdIntelligenceSnapshot(
            timestamp=time, zone_metrics=zone_metrics, door_metrics=door_metrics,
            exit_metrics=exit_metrics, stair_metrics=stair_metrics, building_summary=building_summary,
        )

    # =====================================================

    def _compute_asset_metrics(
        self, asset_type: str, asset_id: str, sides, active_occupants, simulation_style_capacity, time: float,
        observed_on_stair_count: Optional[int] = None,
    ) -> AssetApproachMetrics:

        # position_available is False only when there is at least one
        # ACTIVE occupant known (via current_zone_id/current_floor_id)
        # to be on one of this asset's own floors, but whose
        # world_position is None -- a "known occupant, unknown precise
        # position" case (Phase 13) that would make approaching_count/
        # queue_candidate_count an honest undercount, not a confirmed
        # reading. Vacuously True when nobody at all is currently known
        # to be on this asset's floor(s) -- a real, legitimate zero,
        # not a coverage gap.
        relevant_floor_ids = {side.floor_id for side in sides}
        occupants_on_relevant_floors = [o for o in active_occupants if o.current_floor_id in relevant_floor_ids]
        position_available = all(o.world_position is not None for o in occupants_on_relevant_floors)

        if not position_available:

            return AssetApproachMetrics(
                asset_id=asset_id, asset_type=asset_type, position_available=False,
                simulation_style_capacity=simulation_style_capacity,
                trend=self._trend_tracker.observe(f"asset_congestion:{asset_id}", time, None),
                observed_on_stair_count=observed_on_stair_count,
            )

        queue_metrics = compute_queue_metrics(active_occupants, sides, self.approach_region_depth)
        congestion_level = compute_congestion_level(queue_metrics, simulation_style_capacity, self.congestion_thresholds)

        congestion_value = float(congestion_level.value) if congestion_level is not None else None
        trend = self._trend_tracker.observe(f"asset_congestion:{asset_id}", time, congestion_value)

        return AssetApproachMetrics(
            asset_id=asset_id, asset_type=asset_type, position_available=True,
            approaching_count=queue_metrics.approaching_count,
            queue_candidate_count=queue_metrics.queue_candidate_count,
            estimated_queue_length=queue_metrics.estimated_queue_length,
            mean_approach_speed=queue_metrics.mean_approach_speed,
            simulation_style_capacity=simulation_style_capacity,
            congestion_level=congestion_level,
            trend=trend,
            observed_on_stair_count=observed_on_stair_count,
        )

    # =====================================================

    def _compute_building_summary(
        self, zone_metrics, door_metrics, exit_metrics, stair_metrics, active_occupants, facts,
    ) -> BuildingCrowdSummary:

        # Canonical Live Occupancy Source of Truth milestone -- both the
        # total headcount and the already-fetched active_occupants list
        # are reused here, never re-queried from live_occupant_manager a
        # second/third time within the same compute() call (a genuine,
        # if minor, redundant-work fix alongside the main duplication
        # removal -- see Phase 17's own performance goal).
        total_observed_occupants = facts.total_observed_count
        calibrated_occupant_count = sum(1 for o in active_occupants if o.world_position is not None)
        position_coverage_fraction = (
            (calibrated_occupant_count / total_observed_occupants) if total_observed_occupants > 0 else None
        )

        highest_density_zone = self._argmax(zone_metrics, lambda m: m.density_people_per_m2)

        all_assets = (
            [("Door", m) for m in door_metrics.values()]
            + [("Exit", m) for m in exit_metrics.values()]
            + [("Stair", m) for m in stair_metrics.values()]
        )

        most_congested = self._argmax_asset(all_assets)

        zone_by_id = {zone.id: zone for zone in self._zones}

        zones_above_threshold = tuple(sorted(
            zone_id for zone_id, metrics in zone_metrics.items()
            if self._zone_above_threshold(metrics, zone_by_id.get(zone_id))
        ))

        congested_doors = tuple(sorted(
            asset_id for asset_id, m in door_metrics.items()
            if m.congestion_level is not None and m.congestion_level.value >= IntensityLevel.HIGH.value
        ))
        congested_exits = tuple(sorted(
            asset_id for asset_id, m in exit_metrics.items()
            if m.congestion_level is not None and m.congestion_level.value >= IntensityLevel.HIGH.value
        ))
        congested_stairs = tuple(sorted(
            asset_id for asset_id, m in stair_metrics.items()
            if m.congestion_level is not None and m.congestion_level.value >= IntensityLevel.HIGH.value
        ))

        return BuildingCrowdSummary(
            total_observed_occupants=total_observed_occupants,
            calibrated_occupant_count=calibrated_occupant_count,
            position_coverage_fraction=position_coverage_fraction,
            highest_density_zone=highest_density_zone,
            most_congested_asset_id=most_congested[0] if most_congested is not None else None,
            most_congested_asset_type=most_congested[1] if most_congested is not None else None,
            zones_above_configured_density_threshold=zones_above_threshold,
            congested_doors=congested_doors,
            congested_exits=congested_exits,
            congested_stairs=congested_stairs,
        )

    # =====================================================

    def _zone_above_threshold(self, metrics, zone) -> bool:

        # Two independent, honestly-available signals, either sufficient
        # on its own: the configurable global IntensityLevel scale
        # (Phase 4), OR the zone's own Designer-authored Zone.max_occupancy
        # -- already an established "this zone's own capacity" value
        # elsewhere in this codebase (see navigation.node.Node.capacity,
        # which reads Zone.max_occupancy for exactly this meaning), never
        # a NEW invented interpretation of that field.

        if metrics.density_classification is not None and metrics.density_classification.value >= IntensityLevel.HIGH.value:
            return True

        if zone is not None and zone.max_occupancy > 0 and metrics.occupant_count > zone.max_occupancy:
            return True

        return False

    # =====================================================

    @staticmethod
    def _argmax(metrics_by_id: Dict[str, object], key) -> Optional[str]:

        best_id = None
        best_value = None

        for object_id, metrics in sorted(metrics_by_id.items()):

            value = key(metrics)

            if value is None:
                continue

            if best_value is None or value > best_value:
                best_value = value
                best_id = object_id

        return best_id

    # =====================================================

    @staticmethod
    def _argmax_asset(all_assets: List[Tuple[str, AssetApproachMetrics]]) -> Optional[Tuple[str, str]]:

        best = None
        best_value = None

        for asset_type, metrics in sorted(all_assets, key=lambda pair: (pair[1].asset_id, pair[0])):

            if metrics.congestion_level is None:
                continue

            value = metrics.congestion_level.value

            if best_value is None or value > best_value:
                best_value = value
                best = (metrics.asset_id, asset_type)

        return best
