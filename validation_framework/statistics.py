from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from scipy import stats as scipy_stats

from research_framework.statistics import (
    ANOVAResult,
    ArmComparisonReport,
    ConfidenceInterval,
    DistributionShiftResult,
    EffectSize,
    PairedComparisonResult,
    SensitivityResult,
    compare_arms,
    confidence_interval,
    distribution_shift,
    effect_size_cohens_d,
    feature_sensitivity,
    one_way_anova,
    paired_comparison,
)

# =====================================================
# Phase 5 -- Statistical Analysis. Every primitive research_framework.
# statistics already provides (CI, Cohen's d, ANOVA, paired t-test,
# Pearson sensitivity, KS distribution-shift, the bundled compare_arms()
# report) is re-exported here rather than reimplemented. Bootstrap
# resampling is the one genuinely missing primitive anywhere in this
# codebase -- added below, same scipy.stats-based style and same
# frozen-dataclass-with-.to_dict()/"never fabricate a result for a
# degenerate input" discipline as its research_framework siblings.
# =====================================================

__all__ = [
    "ANOVAResult", "ArmComparisonReport", "ConfidenceInterval", "DistributionShiftResult",
    "EffectSize", "PairedComparisonResult", "SensitivityResult",
    "compare_arms", "confidence_interval", "distribution_shift", "effect_size_cohens_d",
    "feature_sensitivity", "one_way_anova", "paired_comparison",
    "BootstrapResult", "bootstrap_confidence_interval",
]


@dataclass(frozen=True)
class BootstrapResult:

    statistic: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    confidence: float
    n_resamples: int
    n: int

    def to_dict(self) -> Dict[str, Any]:

        return {
            "statistic": self.statistic, "lower": self.lower, "upper": self.upper,
            "confidence": self.confidence, "n_resamples": self.n_resamples, "n": self.n,
        }


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    statistic: Callable[..., float] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    random_state: int = 0,
) -> BootstrapResult:

    cleaned = [float(v) for v in values]
    n = len(cleaned)

    if n < 2:

        point = cleaned[0] if n == 1 else None

        return BootstrapResult(
            statistic=point, lower=None, upper=None,
            confidence=confidence, n_resamples=n_resamples, n=n,
        )

    if len(set(cleaned)) == 1:

        # Zero variance -- every resample is identical to the original
        # sample, so the interval collapses to a point, same "sem == 0"
        # defensive case confidence_interval() already handles.
        return BootstrapResult(
            statistic=cleaned[0], lower=cleaned[0], upper=cleaned[0],
            confidence=confidence, n_resamples=n_resamples, n=n,
        )

    array = np.asarray(cleaned)

    result = scipy_stats.bootstrap(
        (array,), statistic, confidence_level=confidence, n_resamples=n_resamples,
        method="basic", random_state=np.random.default_rng(random_state),
    )

    return BootstrapResult(
        statistic=float(statistic(array)),
        lower=float(result.confidence_interval.low),
        upper=float(result.confidence_interval.high),
        confidence=confidence, n_resamples=n_resamples, n=n,
    )
