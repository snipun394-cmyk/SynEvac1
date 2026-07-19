import math

from collections import Counter
from typing import List, Optional

from campaign_analytics.analyzer import Finding


# =====================================================
# Phase 2 -- Engineering Validation (statistical-anomaly half).
#
# Same duck-typed-samples-only contract as engineering_checks.py --
# this module imports nothing beyond the standard library and
# campaign_analytics.analyzer.Finding.
# =====================================================

# A total_evacuation_time value repeating this many times (or more)
# across the campaign, or in more than this fraction of scenarios --
# whichever is larger -- is unusually frequent for a value that should
# be effectively continuous (a byproduct of varying fire growth/
# occupant placement per scenario).
DUPLICATE_EVACUATION_TIME_MIN_COUNT = 3
DUPLICATE_EVACUATION_TIME_FRACTION_THRESHOLD = 0.10

# Outlier fence multiplier on the interquartile range -- the same
# "3x IQR" convention commonly used for "far outlier" (as opposed to
# the more sensitive 1.5x "mild outlier") boundaries, chosen here to
# keep this check quiet on ordinary variance and only flag values a
# fire engineer would actually want to look at by hand.
OUTLIER_IQR_MULTIPLIER = 3.0
MINIMUM_SAMPLES_FOR_OUTLIER_CHECK = 5


def run_checks(samples) -> List[Finding]:

    samples = list(samples)

    findings: List[Finding] = []

    findings.extend(_check_repeated_evacuation_times(samples))
    findings.extend(_check_constant_occupant_count(samples))
    findings.extend(_check_evacuation_time_outliers(samples))

    return findings


# =====================================================


def _percentile(sorted_values: List[float], percentile: float) -> Optional[float]:

    if not sorted_values:
        return None

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)

    if lower_index == upper_index:
        return sorted_values[int(rank)]

    lower_weight = sorted_values[lower_index] * (upper_index - rank)
    upper_weight = sorted_values[upper_index] * (rank - lower_index)

    return lower_weight + upper_weight


def _check_repeated_evacuation_times(samples) -> List[Finding]:

    total = len(samples)

    if total < 2:
        return []

    values = [
        round(sample.simulation_outcome.get("total_evacuation_time"), 6)
        for sample in samples
        if isinstance(sample.simulation_outcome.get("total_evacuation_time"), (int, float))
    ]

    if not values:
        return []

    counter = Counter(values)
    threshold = max(
        DUPLICATE_EVACUATION_TIME_MIN_COUNT,
        math.ceil(total * DUPLICATE_EVACUATION_TIME_FRACTION_THRESHOLD),
    )

    findings = []

    for value, count in counter.most_common():

        if count >= threshold:
            findings.append(
                Finding(
                    "warning", "repeated_evacuation_time",
                    f"total_evacuation_time == {value} occurs in {count}/{total} scenarios -- "
                    f"unusually frequent for a continuous quantity; check that fire/occupant "
                    f"inputs actually vary across the campaign.",
                )
            )

    return findings


def _check_constant_occupant_count(samples) -> List[Finding]:

    values = [
        sample.scenario_features.get("total_occupants")
        for sample in samples
        if isinstance(sample.scenario_features.get("total_occupants"), (int, float))
    ]

    if len(values) < 2 or len(set(values)) != 1:
        return []

    return [
        Finding(
            "warning", "constant_occupant_count",
            f"total_occupants is exactly {values[0]} in every one of {len(values)} scenarios -- "
            f"the occupancy distribution never varies across this campaign.",
        )
    ]


def _check_evacuation_time_outliers(samples) -> List[Finding]:

    entries = [
        (sample.scenario_id, sample.simulation_outcome.get("total_evacuation_time"))
        for sample in samples
        if isinstance(sample.simulation_outcome.get("total_evacuation_time"), (int, float))
    ]

    if len(entries) < MINIMUM_SAMPLES_FOR_OUTLIER_CHECK:
        return []

    values = sorted(value for _, value in entries)

    q1 = _percentile(values, 25)
    q3 = _percentile(values, 75)
    interquartile_range = q3 - q1

    if interquartile_range == 0:
        return []

    lower_fence = q1 - OUTLIER_IQR_MULTIPLIER * interquartile_range
    upper_fence = q3 + OUTLIER_IQR_MULTIPLIER * interquartile_range

    return [
        Finding(
            "warning", "evacuation_time_outlier",
            f"total_evacuation_time = {value} is a strong outlier "
            f"(outside [{lower_fence:.2f}, {upper_fence:.2f}]).",
            scenario_id=scenario_id,
        )
        for scenario_id, value in entries
        if value < lower_fence or value > upper_fence
    ]
