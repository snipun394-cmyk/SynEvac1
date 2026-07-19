import re

from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional


# =====================================================
# Dataset Statistics (Phase 3) -- engineering statistics over an
# entire campaign, computed purely from already-loaded SimulationSamples
# (loader.CampaignDataset or any iterable of loader.SimulationSample).
# The goal (objective's own words) is detecting dataset bias before
# training -- every function here is a plain aggregation, never a
# model or a learned statistic.
# =====================================================

DETECTOR_STATE_COLUMN_PATTERN = re.compile(r"^Detector_\d+_State$")
CAMERA_STATE_COLUMN_PATTERN = re.compile(r"^Camera_\d+_State$")

FAILED_DEVICE_STATE = "FAILED"

DEFAULT_EVACUATION_TIME_BUCKET = 30.0
DEFAULT_HAZARD_SCORE_BUCKET = 0.1


def compute_statistics(
    dataset: Iterable,
    *,
    evacuation_time_bucket: float = DEFAULT_EVACUATION_TIME_BUCKET,
    hazard_score_bucket: float = DEFAULT_HAZARD_SCORE_BUCKET,
) -> Dict[str, Any]:

    samples = list(dataset)

    return {
        "num_scenarios": len(samples),
        "fire_profile_distribution": _distribution(
            samples, lambda s: s.scenario_features.get("fire_profile"),
        ),
        "ignition_zone_distribution": _distribution(
            samples, lambda s: s.scenario_features.get("ignition_zone"),
        ),
        "occupancy_distribution": _numeric_summary(
            samples, lambda s: s.scenario_features.get("total_occupants"),
        ),
        "evacuation_time_histogram": _histogram(
            _numeric_values(samples, lambda s: s.simulation_outcome.get("total_evacuation_time")),
            evacuation_time_bucket,
        ),
        "trapped_occupant_distribution": _numeric_summary(
            samples, lambda s: s.simulation_outcome.get("people_trapped"),
        ),
        "bottleneck_frequency": _bottleneck_frequency(samples),
        "exit_utilization": _distribution(
            samples, lambda s: s.simulation_outcome.get("most_congested_exit"),
        ),
        "stair_utilization": _distribution(
            samples, lambda s: s.simulation_outcome.get("most_congested_stair"),
        ),
        "detector_failure_rate": _device_failure_rate(samples, DETECTOR_STATE_COLUMN_PATTERN),
        "camera_failure_rate": _device_failure_rate(samples, CAMERA_STATE_COLUMN_PATTERN),
        "hazard_growth_statistics": _hazard_growth_statistics(samples, hazard_score_bucket),
        "ground_truth_recommendation_frequencies": _recommendation_frequencies(samples),
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


def _numeric_summary(samples, getter: Callable[[Any], Any]) -> Dict[str, Optional[float]]:

    return _numeric_summary_from_values(_numeric_values(samples, getter))


def _numeric_summary_from_values(values: List[float]) -> Dict[str, Optional[float]]:

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


# =====================================================
# Domain-specific aggregations.
# =====================================================


def _bottleneck_frequency(samples) -> Dict[str, int]:

    counter: Counter = Counter()

    for sample in samples:

        ground_truth = sample.ground_truth

        if not ground_truth:
            continue

        for door_id in ground_truth.get("doors_that_became_bottlenecks") or []:
            counter[door_id] += 1

    return dict(counter.most_common())


def _device_failure_rate(samples, pattern: "re.Pattern") -> Dict[str, Optional[float]]:

    total = 0
    failed = 0

    for sample in samples:

        for key, value in sample.scenario_features.items():

            if pattern.match(key):

                total += 1

                if value == FAILED_DEVICE_STATE:
                    failed += 1

    return {
        "total_devices_observed": total,
        "failed": failed,
        "failure_rate": (failed / total) if total else None,
    }


def _hazard_growth_statistics(samples, bucket_size: float) -> Dict[str, Any]:

    peak_hazard_scores = []

    for sample in samples:

        scores = [
            row.get("current_hazard_score")
            for row in sample.timeline
            if isinstance(row.get("current_hazard_score"), (int, float))
        ]

        if scores:
            peak_hazard_scores.append(max(scores))

    return {
        "peak_hazard_score_summary": _numeric_summary_from_values(peak_hazard_scores),
        "peak_hazard_score_histogram": _histogram(peak_hazard_scores, bucket_size),
    }


def _recommendation_frequencies(samples) -> Dict[str, int]:

    counter: Counter = Counter()

    for sample in samples:

        ground_truth = sample.ground_truth

        if not ground_truth:
            continue

        for recommendation in ground_truth.get("recommendations") or []:
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
