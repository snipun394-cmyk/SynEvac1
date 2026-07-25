from collections import defaultdict
from typing import Any, Dict, Sequence


# =====================================================
# Large-Scale Predictive Dataset Campaign & Validation milestone,
# Phase 6/7 -- label bias by scenario-level context (occupancy, fire
# severity) and by evacuation PHASE (early/mid/late, relative to each
# scenario's own duration -- deliberately not an absolute time bucket,
# since scenarios run for very different total lengths and "early" in
# a 60s scenario is a different observation_time than "early" in a
# 400s one). Requires `scenario_metadata` (see diversity.py's own
# docstring for its shape) joined onto each row by scenario_id.
# =====================================================

# Documented, disclosed thresholds -- same "project assumption, not a
# validated standard" discipline crowd_intelligence.models.
# DensityThresholds/CongestionThresholds already establish elsewhere in
# this codebase.
LOW_OCCUPANCY_MAX = 10
HIGH_OCCUPANCY_MIN = 20

FAST_FIRE_GROWTH_MAX_SECONDS = 150.0   # faster growth = more severe
SLOW_FIRE_GROWTH_MIN_SECONDS = 300.0


def occupancy_bucket(total_occupants: int) -> str:

    if total_occupants <= LOW_OCCUPANCY_MAX:
        return "LOW"
    if total_occupants >= HIGH_OCCUPANCY_MIN:
        return "HIGH"
    return "MEDIUM"


def severity_bucket(growth_time_seconds) -> str:

    if growth_time_seconds is None:
        return "UNKNOWN"
    if growth_time_seconds <= FAST_FIRE_GROWTH_MAX_SECONDS:
        return "FAST_GROWTH_MORE_SEVERE"
    if growth_time_seconds >= SLOW_FIRE_GROWTH_MIN_SECONDS:
        return "SLOW_GROWTH_LESS_SEVERE"
    return "MODERATE_GROWTH"


def label_bias_report(rows: Sequence[Dict[str, Any]], scenario_metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    metadata_by_scenario = {entry["scenario_id"]: entry for entry in scenario_metadata}

    by_occupancy: Dict[str, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
    by_severity: Dict[str, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})

    for row in rows:

        if row["target"] is None:
            continue

        meta = metadata_by_scenario.get(row["scenario_id"])
        if meta is None:
            continue

        key = "positive" if row["target"] is True else "negative"

        by_occupancy[occupancy_bucket(meta["total_occupants"])][key] += 1
        by_severity[severity_bucket(meta.get("fire_growth_time_seconds"))][key] += 1

    return {
        "by_building_occupancy_level": _with_rates(by_occupancy),
        "by_fire_severity": _with_rates(by_severity),
    }


def _with_rates(buckets: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:

    result = {}

    for bucket, counts in buckets.items():

        total = counts["positive"] + counts["negative"]

        result[bucket] = {
            "positive": counts["positive"],
            "negative": counts["negative"],
            "positive_rate": (counts["positive"] / total) if total else None,
        }

    return result


# =====================================================
# Phase 7 -- temporal coverage. EARLY/MID/LATE thirds of each
# scenario's OWN evacuation_duration (falls back to UNKNOWN when a
# scenario's duration isn't available, e.g. nobody ever arrived).
# =====================================================


def temporal_coverage_report(rows: Sequence[Dict[str, Any]], scenario_metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    metadata_by_scenario = {entry["scenario_id"]: entry for entry in scenario_metadata}

    by_phase: Dict[str, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})

    for row in rows:

        if row["target"] is None:
            continue

        meta = metadata_by_scenario.get(row["scenario_id"])
        duration = meta.get("evacuation_duration") if meta else None

        phase = _phase_of(row["observation_time"], duration)

        key = "positive" if row["target"] is True else "negative"
        by_phase[phase][key] += 1

    return _with_rates(by_phase)


def _phase_of(observation_time: float, duration) -> str:

    if not duration or duration <= 0:
        return "UNKNOWN"

    fraction = observation_time / duration

    if fraction < 1.0 / 3.0:
        return "EARLY"
    if fraction < 2.0 / 3.0:
        return "MID"
    return "LATE"
