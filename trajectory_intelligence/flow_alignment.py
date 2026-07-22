import math
from typing import Dict, Mapping, Set, Tuple

from trajectory_intelligence.models import TrajectoryConfig


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone, Phase 12 -- dominant local (per-floor)
# evacuation flow direction, estimated purely from this cycle's own
# occupant movement_direction values (trajectory.py's own output) --
# never imports/duplicates crowd_intelligence's own approach-to-asset
# evidence (Phase 20's own "keep the raw trajectory result independent"
# boundary). Movement geometry only -- the AGAINST_DOMINANT_FLOW flag
# this module contributes is never a claim of panic, confusion, or
# non-compliance (Phase 31's own required distinction).
# =====================================================


def compute_flow_alignment(
    occupants, movement_directions: Mapping[str, float], speeds: Mapping[str, float], config: TrajectoryConfig,
) -> Tuple[Dict[str, float], Dict[str, float], Set[str]]:

    # Returns (dominant_direction_by_floor, coverage_by_floor,
    # against_flow_occupant_ids). A floor is present in the first two
    # mappings ONLY when Phase 12's hard requirements are genuinely met
    # -- absence means "insufficient evidence," never a fabricated 0.0.

    by_floor_moving: Dict[str, list] = {}
    by_floor_total: Dict[str, int] = {}

    for occupant in occupants:

        floor_id = occupant.current_floor_id

        if floor_id is None:
            continue

        by_floor_total[floor_id] = by_floor_total.get(floor_id, 0) + 1

        direction = movement_directions.get(occupant.occupant_id)
        speed = speeds.get(occupant.occupant_id)

        if direction is None or speed is None or speed <= config.stationary_speed_threshold_m_s:
            continue

        by_floor_moving.setdefault(floor_id, []).append((occupant.occupant_id, direction))

    dominant_direction_by_floor: Dict[str, float] = {}
    coverage_by_floor: Dict[str, float] = {}

    for floor_id, entries in by_floor_moving.items():

        total = by_floor_total.get(floor_id, 0)
        coverage = len(entries) / total if total > 0 else 0.0

        if len(entries) < config.against_flow_min_occupants:
            continue

        if coverage < config.against_flow_min_coverage_fraction:
            continue

        sin_sum = sum(math.sin(direction) for _occupant_id, direction in entries)
        cos_sum = sum(math.cos(direction) for _occupant_id, direction in entries)

        dominant_direction_by_floor[floor_id] = math.atan2(sin_sum, cos_sum)
        coverage_by_floor[floor_id] = coverage

    against_flow_occupant_ids: Set[str] = set()
    threshold_radians = math.radians(config.against_flow_angle_threshold_degrees)

    for floor_id, entries in by_floor_moving.items():

        dominant = dominant_direction_by_floor.get(floor_id)

        if dominant is None:
            continue

        for occupant_id, direction in entries:

            delta = abs(direction - dominant)
            delta = min(delta, 2 * math.pi - delta)

            if delta >= threshold_radians:
                against_flow_occupant_ids.add(occupant_id)

    return dominant_direction_by_floor, coverage_by_floor, against_flow_occupant_ids
