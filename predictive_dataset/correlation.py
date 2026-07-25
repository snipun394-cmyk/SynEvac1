import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =====================================================
# Large-Scale Predictive Dataset Campaign & Validation milestone,
# Phase 8 -- simple, stdlib-only correlation/association checks. NO
# MODEL IS TRAINED HERE -- this is Pearson's r over paired (feature,
# target) values, the same descriptive statistic a human would compute
# by hand, used only to catch broken/constant/redundant features and
# flag anything suspiciously close to |r|=1 for manual leakage review
# (a statistical smell test, not a leakage proof -- the actual leakage
# boundary is enforced mechanically elsewhere, see
# tests/test_predictive_dataset_architecture_guards.py and
# tests/test_predictive_dataset_leakage_guards.py).
# =====================================================

NUMERIC_FEATURES = (
    "total_active_occupant_count",
    "candidate_capacity",
    "candidate_walking_distance",
    "candidate_adjacent_zone_occupancy",
    "candidate_queue_length",
    "candidate_approaching_count",
)

REDUNDANCY_THRESHOLD = 0.95
LEAKAGE_REVIEW_THRESHOLD = 0.9


def pearson_correlation(xs: List[float], ys: List[float]) -> Optional[float]:

    n = len(xs)

    if n < 2:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denominator = math.sqrt(var_x * var_y)

    if denominator == 0:
        return None

    return cov / denominator


# =====================================================


def feature_target_correlations(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    trainable = [row for row in rows if row["target"] is not None]
    y = [1.0 if row["target"] else 0.0 for row in trainable]

    correlations = {}

    for feature in NUMERIC_FEATURES:

        paired_x, paired_y = _paired_non_missing([row.get(feature) for row in trainable], y)
        correlations[feature] = pearson_correlation(paired_x, paired_y)

    flagged_for_leakage_review = sorted(
        feature for feature, r in correlations.items()
        if r is not None and abs(r) >= LEAKAGE_REVIEW_THRESHOLD
    )

    return {
        "trainable_row_count": len(trainable),
        "correlation_with_target": correlations,
        "flagged_for_leakage_review": flagged_for_leakage_review,
    }


def categorical_target_association(rows: Sequence[Dict[str, Any]], field_name: str) -> Dict[str, Any]:

    trainable = [row for row in rows if row["target"] is not None]

    counts: Dict[str, Dict[str, int]] = {}

    for row in trainable:

        category = str(row.get(field_name))
        counts.setdefault(category, {"positive": 0, "negative": 0})

        key = "positive" if row["target"] is True else "negative"
        counts[category][key] += 1

    rates = {}
    for category, tally in counts.items():
        total = tally["positive"] + tally["negative"]
        rates[category] = (tally["positive"] / total) if total else None

    usable_rates = [rate for rate in rates.values() if rate is not None]
    spread = (max(usable_rates) - min(usable_rates)) if usable_rates else None

    return {
        "positive_rate_by_category": rates,
        "counts_by_category": counts,
        "positive_rate_spread": spread,
    }


# =====================================================


def redundant_feature_pairs(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:

    redundant = []

    for feature_a, feature_b in combinations(NUMERIC_FEATURES, 2):

        xs, ys = _paired_non_missing(
            [row.get(feature_a) for row in rows], [row.get(feature_b) for row in rows],
        )

        r = pearson_correlation(xs, ys)

        if r is not None and abs(r) >= REDUNDANCY_THRESHOLD:
            redundant.append({"feature_a": feature_a, "feature_b": feature_b, "correlation": r})

    return redundant


# =====================================================


def _paired_non_missing(xs: List[Any], ys: List[Any]) -> Tuple[List[float], List[float]]:

    paired_x = []
    paired_y = []

    for x, y in zip(xs, ys):

        if x is None or y is None:
            continue

        if isinstance(x, bool) or isinstance(y, bool):
            continue

        paired_x.append(float(x))
        paired_y.append(float(y))

    return paired_x, paired_y
