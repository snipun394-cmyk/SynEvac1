from collections import Counter
from typing import Any, Dict, Sequence

from predictive_dataset.label_analysis import HIGH_OCCUPANCY_MIN, LOW_OCCUPANCY_MAX, occupancy_bucket, severity_bucket


# =====================================================
# Large-Scale Predictive Dataset Campaign & Validation milestone,
# Phase 10 -- whether key OPERATIONAL situations are represented at
# all, not just whether the dataset is large. Reuses label_analysis.py's
# own occupancy/severity bucketing (never a second, differently-tuned
# definition of "low occupancy" or "severe fire" for this phase).
# =====================================================


def operational_coverage_report(
    rows: Sequence[Dict[str, Any]], scenario_metadata: Sequence[Dict[str, Any]], candidate_type_counts: Dict[str, int],
) -> Dict[str, Any]:

    occupancy_counts = Counter(occupancy_bucket(entry["total_occupants"]) for entry in scenario_metadata)
    severity_counts = Counter(severity_bucket(entry.get("fire_growth_time_seconds")) for entry in scenario_metadata)

    blocked_exit_scenarios = sum(1 for entry in scenario_metadata if entry["blocked_exit_count"] > 0)
    blocked_door_scenarios = sum(1 for entry in scenario_metadata if entry["blocked_door_count"] > 0)
    blocked_stair_scenarios = sum(1 for entry in scenario_metadata if entry["unavailable_stair_count"] > 0)
    fully_open_scenarios = sum(
        1 for entry in scenario_metadata
        if entry["blocked_exit_count"] == 0 and entry["blocked_door_count"] == 0 and entry["unavailable_stair_count"] == 0
    )

    bottleneck_candidates_by_scenario = _distinct_congested_candidates_by_scenario(rows)

    no_bottleneck_scenarios = sum(1 for count in bottleneck_candidates_by_scenario.values() if count == 0)
    single_bottleneck_scenarios = sum(1 for count in bottleneck_candidates_by_scenario.values() if count == 1)
    multiple_bottleneck_scenarios = sum(1 for count in bottleneck_candidates_by_scenario.values() if count >= 2)

    # Scenarios with zero congestion events at all never appear in
    # bottleneck_candidates_by_scenario (it is only populated from rows
    # with a real scenario_id, which every scenario contributes) -- but
    # scenario_metadata is the authoritative total count, so scenarios
    # absent from that dict (shouldn't happen, but checked) are also
    # "no bottleneck".
    total_scenarios = len(scenario_metadata)
    accounted = len(bottleneck_candidates_by_scenario)
    no_bottleneck_scenarios += max(total_scenarios - accounted, 0)

    return {
        "occupancy_level_scenario_counts": dict(occupancy_counts),
        "occupancy_thresholds": {"low_max": LOW_OCCUPANCY_MAX, "high_min": HIGH_OCCUPANCY_MIN},
        "fire_severity_scenario_counts": dict(severity_counts),
        "fire_severity_note": (
            "Proxy only, from fire_growth_time -- direct smoke/hazard-score coverage cannot be "
            "checked here, since hazard is deliberately excluded from the deployable candidate "
            "feature schema (SIMULATION_ONLY, see docs/architecture/localized_predictive_ai_dataset.md §7)."
        ),
        "blocked_exit_scenarios": blocked_exit_scenarios,
        "blocked_door_scenarios": blocked_door_scenarios,
        "blocked_stair_scenarios": blocked_stair_scenarios,
        "fully_open_scenarios": fully_open_scenarios,
        "no_bottleneck_scenarios": no_bottleneck_scenarios,
        "single_bottleneck_scenarios": single_bottleneck_scenarios,
        "multiple_bottleneck_scenarios": multiple_bottleneck_scenarios,
        "candidate_type_row_counts": candidate_type_counts,
        "building_topology_note": _building_topology_note(scenario_metadata),
    }


def _building_topology_note(scenario_metadata: Sequence[Dict[str, Any]]) -> str:
    """Predictive Dataset V2 milestone -- this note used to be a single
    hardcoded string asserting "every scenario uses the SAME fixed
    Building... single-exit topologies are NOT represented", written
    when this function only ever saw V1's one-Building campaign. Called
    on a V2 campaign (multiple distinct topology_family values,
    including single_exit_lowrise), that hardcoded claim is simply
    false -- a real reporting bug, not a hypothetical one, caught by
    actually running this function against V2 data. Derived from the
    scenario_metadata actually passed in instead, so it stays accurate
    for whichever campaign calls it."""

    topology_families = sorted({
        entry["topology_family"] for entry in scenario_metadata if entry.get("topology_family")
    })
    single_exit_present = any(entry.get("exit_count") == 1 for entry in scenario_metadata)

    if len(topology_families) <= 1:
        return (
            "Every scenario in this campaign uses the SAME fixed Building. "
            f"Single-exit building topologies are {'' if single_exit_present else 'NOT '}represented."
        )

    return (
        f"This campaign draws from {len(topology_families)} distinct topology families "
        f"({', '.join(topology_families)}), not a single fixed Building. "
        f"Single-exit building topologies are {'' if single_exit_present else 'NOT '}represented."
    )


def _distinct_congested_candidates_by_scenario(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:

    congested_candidates: Dict[str, set] = {}

    for row in rows:

        scenario_id = row["scenario_id"]
        congested_candidates.setdefault(scenario_id, set())

        if row.get("currently_congested"):
            congested_candidates[scenario_id].add(row["candidate_id"])

    return {scenario_id: len(candidates) for scenario_id, candidates in congested_candidates.items()}
