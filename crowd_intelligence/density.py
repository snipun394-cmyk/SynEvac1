from typing import Dict, Optional, Sequence

from behavior_recognition.observation import RecognizedBehavior

from live_occupants.occupant import LiveOccupant
from live_occupants.state import OccupantStatus

from crowd_intelligence.models import DensityThresholds, IntensityLevel, ZoneCrowdMetrics


# =====================================================
# Phase 4 -- zone density, computed ONLY from live_occupants.manager.
# LiveOccupantManager.active_occupants() (Phase 12's canonical source --
# never raw per-camera detections, never multi_camera_fusion's own
# FusedTrack count independently). Phase 3's per-zone breakdown
# (moving/stationary/running/mean_speed/position coverage) is computed
# here too, in the same single pass over active_occupants(), since both
# need the identical "group by current_zone_id" step.
# =====================================================


def compute_zone_density(occupant_count: int, zone_area: Optional[float]) -> Optional[float]:

    # None whenever there is no honest area to divide by -- a zone with
    # zero/unknown area never reports a fabricated infinite or zero
    # density (Phase 4's own "do not invent precision").

    if zone_area is None or zone_area <= 0.0:
        return None

    return occupant_count / zone_area


# =====================================================


def compute_zone_metrics(
    active_occupants: Sequence[LiveOccupant],
    all_occupants: Sequence[LiveOccupant],
    zone_areas: Dict[str, Optional[float]],
    thresholds: DensityThresholds,
) -> Dict[str, ZoneCrowdMetrics]:

    # `all_occupants` is used ONLY to find TEMPORARILY_LOST occupants for
    # temporarily_lost_count (Phase 3) -- active_occupants (NEW/ACTIVE)
    # remains the sole source for occupant_count/density/moving/
    # stationary/running/mean_speed/position coverage, so the two counts
    # can never be conflated (Phase 12).

    by_zone: Dict[str, list] = {}

    for occupant in active_occupants:

        if occupant.current_zone_id is None:
            continue

        by_zone.setdefault(occupant.current_zone_id, []).append(occupant)

    lost_by_zone: Dict[str, int] = {}

    for occupant in all_occupants:

        if occupant.status != OccupantStatus.TEMPORARILY_LOST:
            continue

        if occupant.current_zone_id is None:
            continue

        lost_by_zone[occupant.current_zone_id] = lost_by_zone.get(occupant.current_zone_id, 0) + 1

    zone_ids = set(by_zone.keys()) | set(lost_by_zone.keys()) | set(zone_areas.keys())

    metrics: Dict[str, ZoneCrowdMetrics] = {}

    for zone_id in zone_ids:

        occupants = by_zone.get(zone_id, ())
        occupant_count = len(occupants)

        zone_area = zone_areas.get(zone_id)
        density = compute_zone_density(occupant_count, zone_area)
        classification = thresholds.classify(density)

        moving_count = sum(1 for o in occupants if o.behavior == RecognizedBehavior.WALKING or o.behavior == RecognizedBehavior.RUNNING)
        stationary_count = sum(1 for o in occupants if o.behavior == RecognizedBehavior.STATIONARY)
        running_count = sum(1 for o in occupants if o.behavior == RecognizedBehavior.RUNNING)

        speeds = [o.world_velocity for o in occupants if o.world_velocity is not None]
        mean_speed = (sum(speeds) / len(speeds)) if speeds else None

        position_coverage_count = sum(1 for o in occupants if o.world_position is not None)
        position_coverage_fraction = (position_coverage_count / occupant_count) if occupant_count > 0 else None

        metrics[zone_id] = ZoneCrowdMetrics(
            zone_id=zone_id,
            occupant_count=occupant_count,
            zone_area=zone_area,
            density_people_per_m2=density,
            density_classification=classification,
            moving_count=moving_count,
            stationary_count=stationary_count,
            running_count=running_count,
            temporarily_lost_count=lost_by_zone.get(zone_id, 0),
            mean_speed=mean_speed,
            position_coverage_count=position_coverage_count,
            position_coverage_fraction=position_coverage_fraction,
        )

    return metrics
