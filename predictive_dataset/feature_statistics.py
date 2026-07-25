import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Sequence

from predictive_dataset.schema import CANDIDATE_FEATURE_NAMES


# =====================================================
# Large-Scale Predictive Dataset Campaign & Validation milestone,
# Phase 5 -- per-feature distribution statistics over an already-built
# candidate dataset. Pure descriptive statistics, stdlib only (no
# numpy/pandas dependency added to this package) -- min/max/mean/
# median/stddev/%missing, plus three data-quality flags this milestone
# explicitly asks for: CONSTANT (exactly one distinct observed value),
# NEAR_CONSTANT (one value accounts for >= NEAR_CONSTANT_THRESHOLD of
# non-missing observations), and a simple 3-sigma OUTLIER count.
# =====================================================

NEAR_CONSTANT_THRESHOLD = 0.99
OUTLIER_SIGMA = 3.0

# candidate_type is categorical by construction (schema.py's own
# dtype="str") -- reported as a value-count table, never coerced into
# numeric stats. candidate_congestion_level/candidate_traversable are
# likewise non-numeric.
CATEGORICAL_FEATURES = {"candidate_type", "candidate_congestion_level", "candidate_traversable"}


def feature_distribution_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    report: Dict[str, Any] = {}

    for name in CANDIDATE_FEATURE_NAMES:

        values = [row.get(name) for row in rows]

        if name in CATEGORICAL_FEATURES:
            report[name] = _categorical_summary(values)
        else:
            report[name] = _numeric_summary(values)

    return report


# =====================================================


def _categorical_summary(values: List[Any]) -> Dict[str, Any]:

    total = len(values)
    non_missing = [value for value in values if value is not None]
    missing = total - len(non_missing)

    counts: Dict[str, int] = {}
    for value in non_missing:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1

    distinct = len(counts)
    is_constant = distinct == 1

    dominant_fraction = (max(counts.values()) / len(non_missing)) if non_missing else None

    return {
        "kind": "categorical",
        "count": total,
        "missing_count": missing,
        "missing_fraction": (missing / total) if total else None,
        "distinct_values": distinct,
        "value_counts": counts,
        "is_constant": is_constant,
        "is_near_constant": (dominant_fraction is not None and dominant_fraction >= NEAR_CONSTANT_THRESHOLD and not is_constant),
    }


def _numeric_summary(values: List[Any]) -> Dict[str, Any]:

    total = len(values)

    # bool is a subclass of int in Python -- candidate_traversable is
    # handled as categorical above, so any bool reaching here would be
    # a genuine schema mismatch, not something to silently coerce.
    non_missing = [
        float(value) for value in values
        if value is not None and not isinstance(value, bool) and not (isinstance(value, float) and math.isnan(value))
    ]
    nan_count = sum(1 for value in values if isinstance(value, float) and math.isnan(value))
    missing = total - len(non_missing) - nan_count

    if not non_missing:

        return {
            "kind": "numeric",
            "count": total,
            "missing_count": missing,
            "missing_fraction": (missing / total) if total else None,
            "nan_count": nan_count,
            "min": None, "max": None, "mean": None, "median": None, "stddev": None,
            "is_constant": None, "is_near_constant": None,
            "outlier_count": None, "outlier_fraction": None,
        }

    value_counts = Counter(non_missing)
    is_constant = len(value_counts) == 1

    dominant_count = value_counts.most_common(1)[0][1]
    is_near_constant = (
        (dominant_count / len(non_missing)) >= NEAR_CONSTANT_THRESHOLD
        and not is_constant
    )

    mean = statistics.fmean(non_missing)
    stddev = statistics.pstdev(non_missing) if len(non_missing) > 1 else 0.0

    outlier_count = 0
    if stddev > 0:
        outlier_count = sum(1 for value in non_missing if abs(value - mean) > OUTLIER_SIGMA * stddev)

    return {
        "kind": "numeric",
        "count": total,
        "missing_count": missing,
        "missing_fraction": (missing / total) if total else None,
        "nan_count": nan_count,
        "min": min(non_missing),
        "max": max(non_missing),
        "mean": mean,
        "median": statistics.median(non_missing),
        "stddev": stddev,
        "is_constant": is_constant,
        "is_near_constant": is_near_constant,
        "outlier_count": outlier_count,
        "outlier_fraction": (outlier_count / len(non_missing)) if non_missing else None,
    }
