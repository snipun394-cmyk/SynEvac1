import statistics
from collections import Counter
from typing import Any, Dict, Sequence


# =====================================================
# Large-Scale Predictive Dataset Campaign & Validation milestone,
# Phase 4 -- scenario diversity, measured from per-scenario metadata
# (NOT from candidate-time rows -- diversity is a property of the
# underlying SCENARIOS, and a scenario contributes many rows regardless
# of how different it is from its neighbors, so measuring diversity off
# rows would just double-count long scenarios). `scenario_metadata` is
# one dict per generated scenario: scenario_id, total_occupants,
# ignition_zone_id, fire_growth_time_seconds, blocked_door_count,
# blocked_exit_count, unavailable_stair_count, evacuation_duration,
# unreachable_occupant_count -- produced by the campaign script
# alongside (not derived from) the candidate dataset.
# =====================================================


def scenario_diversity_report(scenario_metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    if not scenario_metadata:
        return {"scenario_count": 0}

    occupant_counts = [entry["total_occupants"] for entry in scenario_metadata]
    ignition_zones = Counter(entry["ignition_zone_id"] for entry in scenario_metadata if entry.get("ignition_zone_id"))
    blocked_doors = [entry["blocked_door_count"] for entry in scenario_metadata]
    blocked_exits = [entry["blocked_exit_count"] for entry in scenario_metadata]
    unavailable_stairs = [entry["unavailable_stair_count"] for entry in scenario_metadata]
    durations = [entry["evacuation_duration"] for entry in scenario_metadata if entry.get("evacuation_duration") is not None]
    growth_times = [entry["fire_growth_time_seconds"] for entry in scenario_metadata if entry.get("fire_growth_time_seconds") is not None]

    return {
        "scenario_count": len(scenario_metadata),
        "distinct_ignition_zones": len(ignition_zones),
        "ignition_zone_distribution": dict(ignition_zones),
        "occupant_count": _numeric_spread(occupant_counts),
        "blocked_door_count": _numeric_spread(blocked_doors),
        "blocked_exit_count": _numeric_spread(blocked_exits),
        "unavailable_stair_count": _numeric_spread(unavailable_stairs),
        "fire_growth_time_seconds": _numeric_spread(growth_times),
        "evacuation_duration_seconds": _numeric_spread(durations),
        "fraction_scenarios_with_any_blocked_route": (
            sum(1 for entry in scenario_metadata if entry["blocked_door_count"] + entry["blocked_exit_count"] + entry["unavailable_stair_count"] > 0)
            / len(scenario_metadata)
        ),
    }


def _numeric_spread(values) -> Dict[str, Any]:

    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "stddev": None, "distinct_values": 0}

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "distinct_values": len(set(values)),
    }


# =====================================================


def candidate_utilization_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    # Phase 4's "candidate utilization" -- for each candidate id, the
    # fraction of its own rows where it showed ANY recorded demand
    # (queue/approach) -- an unused candidate across the whole campaign
    # would be a genuine diversity gap (see also operational_coverage.py).

    totals: Dict[str, int] = Counter()
    active: Dict[str, int] = Counter()
    candidate_types: Dict[str, str] = {}

    for row in rows:

        candidate_id = row["candidate_id"]
        totals[candidate_id] += 1
        candidate_types[candidate_id] = row["candidate_type"]

        queue = row.get("candidate_queue_length") or 0
        approaching = row.get("candidate_approaching_count") or 0

        if queue > 0 or approaching > 0:
            active[candidate_id] += 1

    return {
        candidate_id: {
            "candidate_type": candidate_types[candidate_id],
            "total_observations": totals[candidate_id],
            "active_observations": active.get(candidate_id, 0),
            "active_fraction": (active.get(candidate_id, 0) / totals[candidate_id]) if totals[candidate_id] else None,
        }
        for candidate_id in totals
    }
