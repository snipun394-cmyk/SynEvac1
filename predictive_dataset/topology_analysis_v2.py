from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Sequence


# =====================================================
# Predictive Dataset V2 milestone, Phases 13/14/15 -- analysis specific
# to the multi-topology campaign: family/floor/exit/stair distribution,
# stair feature repair verification, and multi-bottleneck candidate-type
# combination coverage. Mirrors the existing predictive_dataset analysis
# modules' own shape (plain functions over rows/scenario_metadata lists,
# no model training, no simulation) -- kept as a separate module rather
# than added to diversity.py/operational_coverage.py since every
# function here is specific to V2's multi-topology-family structure
# those V1-era modules never had.
# =====================================================


def topology_family_distribution(scenario_metadata: Sequence[Dict[str, Any]]) -> Dict[str, int]:

    return dict(Counter(entry["topology_family"] for entry in scenario_metadata))


def structural_distribution(scenario_metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    return {
        "floor_count": dict(Counter(entry["floor_count"] for entry in scenario_metadata)),
        "exit_count": dict(Counter(entry["exit_count"] for entry in scenario_metadata)),
        "stair_count": dict(Counter(entry["stair_count"] for entry in scenario_metadata)),
        "door_count": dict(Counter(entry["door_count"] for entry in scenario_metadata)),
    }


def occupancy_strata(scenario_metadata: Sequence[Dict[str, Any]], *, low_max: int = 10, high_min: int = 30) -> Dict[str, int]:

    def bucket(total_occupants: int) -> str:
        if total_occupants <= low_max:
            return "LOW"
        if total_occupants >= high_min:
            return "HIGH"
        return "MEDIUM"

    return dict(Counter(bucket(entry["total_occupants"]) for entry in scenario_metadata))


def total_lockout_scenarios(scenario_metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    lockouts = [entry for entry in scenario_metadata if entry["blocked_exit_count"] >= entry["exit_count"]]
    lockouts_with_rows = [entry for entry in lockouts if entry.get("contributed_rows")]
    lockouts_zero_rows = [entry for entry in lockouts if not entry.get("contributed_rows")]

    return {
        "total_lockout_scenario_count": len(lockouts),
        "total_lockout_with_candidate_rows": len(lockouts_with_rows),
        "total_lockout_zero_rows": len(lockouts_zero_rows),
        "sample_scenario_ids_with_rows": [entry["scenario_id"] for entry in lockouts_with_rows[:5]],
    }


# =====================================================


def stair_feature_repair_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Direct Stair-candidate feature-variation check -- Phase 14's own
    "demonstrate that Stair features now genuinely vary" requirement.
    Compare these numbers against docs/architecture/
    predictive_dataset_campaign_v1.md Section 5/9 (V1: candidate_queue_length
    constant 0 across all 501,696 stair rows, candidate_walking_distance
    constant 0.0, currently_congested constant False)."""

    stair_rows = [row for row in rows if row["candidate_type"] == "Stair"]

    if not stair_rows:
        return {"stair_row_count": 0}

    queue_lengths = [row["candidate_queue_length"] for row in stair_rows]
    approaching_counts = [row["candidate_approaching_count"] for row in stair_rows]
    distances = {row["candidate_walking_distance"] for row in stair_rows}
    currently_congested_true = sum(1 for row in stair_rows if row["currently_congested"] is True)

    trainable_stair_rows = [row for row in stair_rows if row["target"] is not None]
    positive_stair_rows = sum(1 for row in trainable_stair_rows if row["target"] is True)

    return {
        "stair_row_count": len(stair_rows),
        "distinct_walking_distance_values": len(distances),
        "walking_distance_is_zero_for_all_rows": distances == {0.0},
        "queue_length_nonzero_row_count": sum(1 for value in queue_lengths if value > 0),
        "queue_length_max": max(queue_lengths),
        "approaching_count_nonzero_row_count": sum(1 for value in approaching_counts if value > 0),
        "approaching_count_max": max(approaching_counts),
        "currently_congested_true_count": currently_congested_true,
        "trainable_stair_row_count": len(trainable_stair_rows),
        "positive_stair_row_count": positive_stair_rows,
        "positive_rate": (positive_stair_rows / len(trainable_stair_rows)) if trainable_stair_rows else None,
    }


# =====================================================


def multi_bottleneck_candidate_type_combinations(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """For every (scenario_id, observation_time, prediction_horizon) with
    2+ distinct candidates showing target=True, tally which CANDIDATE-TYPE
    pairs co-occur (Door+Door, Door+Exit, Door+Stair, Exit+Stair, ...) --
    Phase 7/15's own "measure how frequently these occur... include
    combinations such as Door+Door, Door+Exit, Door+Stair, Stair+Exit"."""

    positive_types_by_bucket: Dict[tuple, List[str]] = {}

    for row in rows:

        if row["target"] is not True:
            continue

        bucket_key = (row["scenario_id"], row["observation_time"], row["prediction_horizon"])
        positive_types_by_bucket.setdefault(bucket_key, []).append(row["candidate_type"])

    single_bottleneck_count = 0
    multiple_bottleneck_count = 0
    combination_counts: Counter = Counter()

    for candidate_types in positive_types_by_bucket.values():

        if len(candidate_types) < 2:
            single_bottleneck_count += 1
            continue

        multiple_bottleneck_count += 1

        distinct_types = sorted(set(candidate_types))
        for type_a, type_b in combinations(distinct_types, 2):
            combination_counts[f"{type_a}+{type_b}"] += 1

        if len(distinct_types) == 1:
            # 2+ candidates of the SAME type simultaneously bottlenecked
            # (e.g. Door+Door on two different doors) -- still a genuine
            # multiple-simultaneous-bottleneck bucket, tallied separately
            # from the cross-type combinations above.
            combination_counts[f"{distinct_types[0]}+{distinct_types[0]}"] += 1

    return {
        "no_or_single_bottleneck_bucket_count": single_bottleneck_count,
        "multiple_bottleneck_bucket_count": multiple_bottleneck_count,
        "candidate_type_combination_counts": dict(combination_counts),
    }


# =====================================================


def verify_coverage_targets(actual_counts: Dict[str, int], targets: Dict[str, Any]) -> Dict[str, Any]:
    """Mechanically checks every predictive_dataset.campaign_config_v2.
    COVERAGE_TARGETS entry against what the campaign actually produced --
    Phase 2's own "do not rely on random generation alone to hopefully
    produce them" instruction, closed the loop."""

    report = {}

    for name, target in targets.items():

        actual = actual_counts.get(name)
        minimum = target.minimum_count if hasattr(target, "minimum_count") else target["minimum_count"]

        report[name] = {
            "description": target.description if hasattr(target, "description") else target["description"],
            "minimum_required": minimum,
            "actual": actual,
            "met": (actual is not None and actual >= minimum),
        }

    return report
