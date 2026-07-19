import re

from collections import Counter
from typing import Any, Callable, Dict, List, Optional


# =====================================================
# Phase 3 -- Distribution Analysis.
#
# Every function below takes only an already-loaded, duck-typed
# iterable of samples (each carrying .scenario_features/
# .simulation_outcome/.zone_results/.timeline/.ground_truth/
# .decision_policy, the shape training_dataset.SimulationSample
# already has) -- this module imports nothing from training_dataset or
# any simulation-stack package, so it has no dependency beyond the
# Python standard library.
# =====================================================

ZONE_SMOKE_COLUMN_PATTERN = re.compile(r"^Zone_\d+_Smoke$")

# A zone is "smoke affected" once its own Zone_N_Smoke reading crosses
# this fraction of its [0, 1] range -- the same kind of disclosed
# policy constant ground_truth/recommendations.py's own
# HIGH_RISK_THRESHOLD already establishes, not a fabricated data value.
SMOKE_AFFECTED_THRESHOLD = 0.5

DEFAULT_EVACUATION_TIME_BUCKET = 30.0
DEFAULT_TRAPPED_OCCUPANT_BUCKET = 1.0


def compute_distributions(
    samples,
    *,
    evacuation_time_bucket: float = DEFAULT_EVACUATION_TIME_BUCKET,
    trapped_occupant_bucket: float = DEFAULT_TRAPPED_OCCUPANT_BUCKET,
) -> Dict[str, Any]:

    samples = list(samples)

    return {
        "evacuation_time": _numeric_distribution(
            samples, lambda s: s.simulation_outcome.get("total_evacuation_time"),
            evacuation_time_bucket,
        ),
        "trapped_occupants": _numeric_distribution(
            samples, lambda s: s.simulation_outcome.get("people_trapped"),
            trapped_occupant_bucket,
        ),
        "bottleneck_frequency": _bottleneck_frequency_summary(samples),
        "bottleneck_locations": _bottleneck_locations(samples),
        "exit_utilization": _distribution(
            samples, lambda s: s.simulation_outcome.get("most_congested_exit"),
        ),
        "stair_utilization": _distribution(
            samples, lambda s: s.simulation_outcome.get("most_congested_stair"),
        ),
        "smoke_spread": _smoke_spread_statistics(samples),
        "recommendation_frequencies": _recommendation_frequencies(samples),
        "decision_policy_frequencies": _decision_policy_frequencies(samples),
    }


# =====================================================
# Generic helpers.
# =====================================================


def _distribution(samples, getter: Callable[[Any], Any]) -> Dict[str, int]:

    counter: Counter = Counter()

    for sample in samples:

        value = getter(sample)

        if value is None or value == "":
            continue

        counter[str(value)] += 1

    return dict(counter.most_common())


def _numeric_values(samples, getter: Callable[[Any], Any]) -> List[float]:

    values = []

    for sample in samples:

        value = getter(sample)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(value)

    return values


def _numeric_summary(values: List[float]) -> Dict[str, Optional[float]]:

    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}

    ordered = sorted(values)
    count = len(ordered)
    midpoint = count // 2

    median = (
        ordered[midpoint]
        if count % 2 == 1
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )

    return {
        "count": count,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / count,
        "median": median,
    }


def _histogram(values: List[float], bucket_size: float) -> Dict[float, int]:

    if not values or bucket_size <= 0:
        return {}

    counter: Counter = Counter()

    for value in values:
        bucket = int(value // bucket_size) * bucket_size
        counter[bucket] += 1

    return dict(sorted(counter.items()))


def _numeric_distribution(samples, getter, bucket_size: float) -> Dict[str, Any]:

    values = _numeric_values(samples, getter)

    return {
        "summary": _numeric_summary(values),
        "histogram": _histogram(values, bucket_size),
    }


# =====================================================
# Domain-specific aggregations.
# =====================================================


def _bottleneck_frequency_summary(samples) -> Dict[str, Any]:

    total = len(samples)
    with_bottleneck = sum(
        1
        for sample in samples
        if sample.ground_truth and sample.ground_truth.get("doors_that_became_bottlenecks")
    )

    return {
        "scenarios_with_a_bottleneck": with_bottleneck,
        "total_scenarios": total,
        "fraction": (with_bottleneck / total) if total else None,
    }


def _bottleneck_locations(samples) -> Dict[str, int]:

    counter: Counter = Counter()

    for sample in samples:

        if not sample.ground_truth:
            continue

        for door_id in sample.ground_truth.get("doors_that_became_bottlenecks") or []:
            counter[door_id] += 1

    return dict(counter.most_common())


def _smoke_spread_statistics(samples) -> Dict[str, Any]:

    peak_smoke_values = []
    affected_zone_counts = []

    for sample in samples:

        peak_by_column: Dict[str, float] = {}

        for row in sample.timeline:
            for key, value in row.items():

                if not ZONE_SMOKE_COLUMN_PATTERN.match(key) or not isinstance(value, (int, float)):
                    continue

                if value > peak_by_column.get(key, float("-inf")):
                    peak_by_column[key] = value

        if not peak_by_column:
            continue

        peak_smoke_values.append(max(peak_by_column.values()))
        affected_zone_counts.append(
            sum(1 for value in peak_by_column.values() if value >= SMOKE_AFFECTED_THRESHOLD)
        )

    return {
        "peak_smoke_summary": _numeric_summary(peak_smoke_values),
        "affected_zone_count_summary": _numeric_summary(affected_zone_counts),
    }


def _recommendation_frequencies(samples) -> Dict[str, int]:

    counter: Counter = Counter()

    for sample in samples:

        if not sample.ground_truth:
            continue

        for recommendation in sample.ground_truth.get("recommendations") or []:
            counter[recommendation.get("target_type", "unknown")] += 1

    return dict(counter.most_common())


def _decision_policy_frequencies(samples) -> Dict[str, Dict[str, int]]:

    zone_actions: Counter = Counter()
    exit_statuses: Counter = Counter()
    stair_statuses: Counter = Counter()
    announcement_priorities: Counter = Counter()

    for sample in samples:

        decision_policy = sample.decision_policy

        if not decision_policy:
            continue

        for entry in decision_policy.get("zone_decisions") or []:
            zone_actions[entry.get("action", "unknown")] += 1

        for entry in decision_policy.get("exit_decisions") or []:
            exit_statuses[entry.get("status", "unknown")] += 1

        for entry in decision_policy.get("stair_decisions") or []:
            stair_statuses[entry.get("status", "unknown")] += 1

        for entry in decision_policy.get("announcements") or []:
            announcement_priorities[entry.get("priority", "unknown")] += 1

    return {
        "zone_actions": dict(zone_actions.most_common()),
        "exit_statuses": dict(exit_statuses.most_common()),
        "stair_statuses": dict(stair_statuses.most_common()),
        "announcement_priorities": dict(announcement_priorities.most_common()),
    }
