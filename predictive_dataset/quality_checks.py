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

    # Predictive Dataset V2 milestone -- keyed by hash(...) rather than
    # the raw tuple itself. At V1's ~2.5M-row scale a dict keyed by the
    # full ~19-field row tuple was survivable; at V2's ~9.6M rows it
    # retains millions of large tuple objects simultaneously and drove
    # this process to 8GB+ RSS on a 7.3GB-RAM machine (discovered via a
    # real campaign run, not a hypothetical). Hashing collapses each key
    # to a single int -- the temporary tuple is discarded immediately
    # after hashing instead of being retained as a dict key. A 64-bit
    # hash collision across ~9.6M rows is astronomically unlikely
    # (~2.5e-6 by the birthday bound) and, even if it occurred, would
    # only ever *undercount* an exact-duplicate diagnostic by merging
    # two coincidentally-colliding counts -- never a correctness issue
    # for the actual dataset rows themselves.
    row_key_counts: Dict[int, int] = {}
    identity_key_counts: Dict[int, int] = {}

    invalid_candidate_ids = set()
    missing_target_key_count = 0
    nan_value_rows = 0
    invalid_ranges: Dict[str, int] = {}
    invalid_candidate_type_count = 0
    invalid_congestion_level_count = 0
    inconsistent_currently_congested_vs_target = 0

    for row in rows:

        row_key = hash(tuple(sorted(row.items(), key=lambda item: item[0])))
        row_key_counts[row_key] = row_key_counts.get(row_key, 0) + 1

        identity_key = hash((row.get("scenario_id"), row.get("observation_time"), row.get("candidate_id"), row.get("prediction_horizon")))
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


# =====================================================
# Predictive Dataset V2 milestone, Phase 1/16 -- the mechanical guard
# that would have caught V1's Stair bug. candidate_walking_distance is
# a STRUCTURAL, per-building geometry property -- being constant for a
# given candidate_id across an entire campaign is completely expected
# and NOT itself a defect (the building's geometry doesn't change
# scenario to scenario). What IS a defect is that constant being
# exactly 0.0: a real Door/Exit/Stair candidate always occupies some
# positive physical distance, and this exact degenerate value is what
# ai_registry.training_scenario.make_training_building()'s Staircase(...)
# silently produced by never setting from_floor_id (see
# docs/architecture/predictive_dataset_campaign_v2.md Section 1 for the
# full traced mechanism -- Staircase.vertical_height() -> travel_distance()
# -> Edge.walking_distance -> simulator.coordinator's admission duration,
# all silently collapsing to zero). This check flags any candidate_id
# whose walking_distance is 0.0 on EVERY row it appears in.
# =====================================================


def zero_walking_distance_candidates(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    distances_by_candidate: Dict[str, Set[float]] = {}

    for row in rows:

        candidate_id = row.get("candidate_id")
        distance = row.get("candidate_walking_distance")

        if candidate_id is None or distance is None:
            continue

        distances_by_candidate.setdefault(candidate_id, set()).add(distance)

    flagged = sorted(
        candidate_id for candidate_id, distances in distances_by_candidate.items()
        if distances == {0.0}
    )

    return {
        "candidates_checked": sorted(distances_by_candidate.keys()),
        "flagged_zero_distance_candidate_ids": flagged,
        "explanation": (
            "candidate_walking_distance being CONSTANT per candidate_id is expected "
            "(structural geometry never changes scenario to scenario) -- being constant "
            "at exactly 0.0 is not: it silently zeroes simulated traversal duration for "
            "that candidate (see predictive_dataset_campaign_v2.md Section 1)."
        ),
    }


def duplicate_scenario_ids(scenario_metadata: Sequence[Dict[str, Any]]) -> int:

    seen: Dict[str, int] = {}

    for entry in scenario_metadata:
        scenario_id = entry.get("scenario_id")
        seen[scenario_id] = seen.get(scenario_id, 0) + 1

    return sum(count - 1 for count in seen.values() if count > 1)
