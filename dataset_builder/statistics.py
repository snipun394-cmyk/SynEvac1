from collections import Counter
from typing import Any, Iterable, Optional


# =====================================================
# Small, pure numeric helpers shared by feature_extractor.py and
# labels.py. No statistics computed here look beyond the values they
# are handed -- nothing samples, simulates, or fills in a missing
# value from anywhere else.
# =====================================================


def mean(values: Iterable[float]) -> Optional[float]:

    values = [value for value in values if value is not None]

    if not values:
        return None

    return sum(values) / len(values)


def mode(values: Iterable[Any]) -> Optional[Any]:

    # Most frequently occurring non-None value. Ties are broken by
    # whichever value was encountered first, so the result is
    # deterministic for a given input order rather than depending on
    # Counter's internal tie-breaking.

    values = [value for value in values if value is not None]

    if not values:
        return None

    counts = Counter(values)
    best_count = max(counts.values())

    for value in values:
        if counts[value] == best_count:
            return value

    return None


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:

    if not denominator:
        return None

    return numerator / denominator
