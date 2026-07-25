import math
from typing import Any, Dict, Sequence, Set

from crowd_intelligence.models import IntensityLevel

from navigation.edge import Edge


# =====================================================
# Large-Scale Predictive Dataset Campaign & Validation milestone,
# Phase 9 -- mechanical data-quality checks over an already-built
# candidate dataset, run at full campaign scale (not just the small
# hand-built fixtures the unit tests use). Every check here is a
# structural/statistical sanity check over already-generated rows --
# none of it re-derives or re-simulates anything.
# =====================================================

VALID_CANDIDATE_TYPES = set(Edge.EDGE_TYPES)
VALID_CONGESTION_LEVELS = {level.name for level in IntensityLevel} | {None}


def run_quality_checks(rows: Sequence[Dict[str, Any]], known_candidate_ids: Set[str]) -> Dict[str, Any]:

    row_key_counts: Dict[tuple, int] = {}
    identity_key_counts: Dict[tuple, int] = {}

    invalid_candidate_ids = set()
    missing_target_key_count = 0
    nan_value_rows = 0
    invalid_ranges: Dict[str, int] = {}
    invalid_candidate_type_count = 0
    invalid_congestion_level_count = 0
    inconsistent_currently_congested_vs_target = 0

    for row in rows:

        row_key = tuple(sorted(row.items(), key=lambda item: item[0]))
        row_key_counts[row_key] = row_key_counts.get(row_key, 0) + 1

        identity_key = (row.get("scenario_id"), row.get("observation_time"), row.get("candidate_id"), row.get("prediction_horizon"))
        identity_key_counts[identity_key] = identity_key_counts.get(identity_key, 0) + 1

        if row.get("candidate_id") not in known_candidate_ids:
            invalid_candidate_ids.add(row.get("candidate_id"))

        if "target" not in row:
            missing_target_key_count += 1

        for field_name, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                nan_value_rows += 1
                break

        invalid_ranges.update(_check_ranges(row, invalid_ranges))

        if row.get("candidate_type") not in VALID_CANDIDATE_TYPES:
            invalid_candidate_type_count += 1

        if row.get("candidate_congestion_level") not in VALID_CONGESTION_LEVELS:
            invalid_congestion_level_count += 1

        currently_congested = row.get("currently_congested")
        target = row.get("target")
        if currently_congested is True and target is not None:
            inconsistent_currently_congested_vs_target += 1
        if currently_congested is False and target is None:
            inconsistent_currently_congested_vs_target += 1

    duplicate_row_count = sum(count - 1 for count in row_key_counts.values() if count > 1)
    duplicate_identity_count = sum(count - 1 for count in identity_key_counts.values() if count > 1)

    return {
        "total_rows_checked": len(rows),
        "duplicate_exact_rows": duplicate_row_count,
        "duplicate_identity_keys": duplicate_identity_count,
        "invalid_candidate_ids": sorted(str(cid) for cid in invalid_candidate_ids),
        "missing_target_key_count": missing_target_key_count,
        "nan_value_row_count": nan_value_rows,
        "invalid_ranges": invalid_ranges,
        "invalid_candidate_type_count": invalid_candidate_type_count,
        "invalid_congestion_level_count": invalid_congestion_level_count,
        "currently_congested_target_inconsistencies": inconsistent_currently_congested_vs_target,
    }


# =====================================================


def _check_ranges(row: Dict[str, Any], running_counts: Dict[str, int]) -> Dict[str, int]:

    updates = dict(running_counts)

    def flag(name: str):
        updates[name] = updates.get(name, 0) + 1

    for field_name in ("candidate_queue_length", "candidate_approaching_count", "total_active_occupant_count", "candidate_adjacent_zone_occupancy"):
        value = row.get(field_name)
        if value is not None and value < 0:
            flag(f"{field_name}_negative")

    capacity = row.get("candidate_capacity")
    if capacity is not None and capacity <= 0:
        flag("candidate_capacity_not_positive")

    distance = row.get("candidate_walking_distance")
    if distance is not None and distance < 0:
        flag("candidate_walking_distance_negative")

    observation_time = row.get("observation_time")
    if observation_time is not None and observation_time < 0:
        flag("observation_time_negative")

    horizon = row.get("prediction_horizon")
    if horizon is not None and horizon <= 0:
        flag("prediction_horizon_not_positive")

    return updates


def duplicate_scenario_ids(scenario_metadata: Sequence[Dict[str, Any]]) -> int:

    seen: Dict[str, int] = {}

    for entry in scenario_metadata:
        scenario_id = entry.get("scenario_id")
        seen[scenario_id] = seen.get(scenario_id, 0) + 1

    return sum(count - 1 for count in seen.values() if count > 1)
