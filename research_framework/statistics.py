from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from scipy import stats as scipy_stats


# =====================================================
# Phase 3 -- Statistical Analysis. Pure functions over plain sequences
# of floats -- no dependency on any simulation package, no pandas (this
# codebase's own established "plain dict/list data shapes, no pandas"
# convention -- see campaign_analytics/ai_training). scipy is already
# an installed, if previously unused, dependency (confirmed present for
# every existing environment this platform runs in) -- no new
# requirement is introduced.
#
# Every result type below carries `n`/sample sizes alongside the
# statistic itself, and every function returns a documented "could not
# be computed" value (None, or an explicit `ok=False`) rather than
# raising or fabricating a result for degenerate inputs (fewer than 2
# samples, zero variance, unequal-length paired samples, ...) -- the
# same "never fabricate, honestly absent instead" discipline
# ground_truth/dataset_builder already apply throughout this codebase.
# =====================================================


@dataclass(frozen=True)
class ConfidenceInterval:

    mean: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    confidence: float
    n: int

    def to_dict(self) -> Dict[str, Any]:

        return {
            "mean": self.mean, "lower": self.lower, "upper": self.upper,
            "confidence": self.confidence, "n": self.n,
        }


def confidence_interval(values: Sequence[float], confidence: float = 0.95) -> ConfidenceInterval:

    values = [float(v) for v in values]
    n = len(values)

    if n == 0:
        return ConfidenceInterval(mean=None, lower=None, upper=None, confidence=confidence, n=0)

    mean = sum(values) / n

    if n < 2:
        return ConfidenceInterval(mean=mean, lower=None, upper=None, confidence=confidence, n=n)

    sem = scipy_stats.sem(values)

    if sem == 0:
        return ConfidenceInterval(mean=mean, lower=mean, upper=mean, confidence=confidence, n=n)

    half_width = sem * scipy_stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1)

    return ConfidenceInterval(
        mean=mean, lower=mean - half_width, upper=mean + half_width, confidence=confidence, n=n,
    )


# =====================================================


@dataclass(frozen=True)
class EffectSize:

    cohens_d: Optional[float]
    n_a: int
    n_b: int

    def to_dict(self) -> Dict[str, Any]:

        return {"cohens_d": self.cohens_d, "n_a": self.n_a, "n_b": self.n_b}


def effect_size_cohens_d(a: Sequence[float], b: Sequence[float]) -> EffectSize:

    a = [float(v) for v in a]
    b = [float(v) for v in b]

    if len(a) < 2 or len(b) < 2:
        return EffectSize(cohens_d=None, n_a=len(a), n_b=len(b))

    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    var_a = scipy_stats.tvar(a)
    var_b = scipy_stats.tvar(b)

    pooled_df = len(a) + len(b) - 2
    pooled_variance = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / pooled_df

    if pooled_variance == 0:
        return EffectSize(cohens_d=0.0 if mean_a == mean_b else None, n_a=len(a), n_b=len(b))

    return EffectSize(cohens_d=(mean_a - mean_b) / (pooled_variance ** 0.5), n_a=len(a), n_b=len(b))


# =====================================================


@dataclass(frozen=True)
class ANOVAResult:

    f_statistic: Optional[float]
    p_value: Optional[float]
    group_sizes: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:

        return {
            "f_statistic": self.f_statistic, "p_value": self.p_value, "group_sizes": list(self.group_sizes),
        }


def one_way_anova(*groups: Sequence[float]) -> ANOVAResult:

    cleaned = [[float(v) for v in group] for group in groups]
    sizes = tuple(len(group) for group in cleaned)

    if len(cleaned) < 2 or any(size < 2 for size in sizes):
        return ANOVAResult(f_statistic=None, p_value=None, group_sizes=sizes)

    result = scipy_stats.f_oneway(*cleaned)

    return ANOVAResult(f_statistic=float(result.statistic), p_value=float(result.pvalue), group_sizes=sizes)


# =====================================================


@dataclass(frozen=True)
class PairedComparisonResult:

    t_statistic: Optional[float]
    p_value: Optional[float]
    mean_difference: Optional[float]
    n: int

    def to_dict(self) -> Dict[str, Any]:

        return {
            "t_statistic": self.t_statistic, "p_value": self.p_value,
            "mean_difference": self.mean_difference, "n": self.n,
        }


def paired_comparison(a: Sequence[float], b: Sequence[float]) -> PairedComparisonResult:

    # A true paired test requires a and b to be the *same length*,
    # entry-by-entry corresponding observations (e.g. the same seed
    # index's outcome under two arms) -- unequal lengths mean the
    # caller handed in two independent, unpaired samples, which this
    # function honestly refuses to pair up arbitrarily.

    if len(a) != len(b):

        raise ValueError(
            f"paired_comparison() requires equal-length sequences (same seed index in "
            f"each arm), got {len(a)} and {len(b)}.",
        )

    a = [float(v) for v in a]
    b = [float(v) for v in b]
    n = len(a)

    if n < 2:
        return PairedComparisonResult(t_statistic=None, p_value=None, mean_difference=None, n=n)

    differences = [x - y for x, y in zip(a, b)]
    mean_difference = sum(differences) / n

    result = scipy_stats.ttest_rel(a, b)

    return PairedComparisonResult(
        t_statistic=float(result.statistic), p_value=float(result.pvalue),
        mean_difference=mean_difference, n=n,
    )


# =====================================================


@dataclass(frozen=True)
class SensitivityResult:

    pearson_r: Optional[float]
    p_value: Optional[float]
    slope: Optional[float]
    n: int

    def to_dict(self) -> Dict[str, Any]:

        return {
            "pearson_r": self.pearson_r, "p_value": self.p_value, "slope": self.slope, "n": self.n,
        }


def feature_sensitivity(x_values: Sequence[float], y_values: Sequence[float]) -> SensitivityResult:

    if len(x_values) != len(y_values):

        raise ValueError(
            f"feature_sensitivity() requires equal-length sequences, got "
            f"{len(x_values)} and {len(y_values)}.",
        )

    x_values = [float(v) for v in x_values]
    y_values = [float(v) for v in y_values]
    n = len(x_values)

    if n < 2 or len(set(x_values)) < 2:
        # A constant swept parameter has no slope to measure against.
        return SensitivityResult(pearson_r=None, p_value=None, slope=None, n=n)

    regression = scipy_stats.linregress(x_values, y_values)

    return SensitivityResult(
        pearson_r=float(regression.rvalue), p_value=float(regression.pvalue),
        slope=float(regression.slope), n=n,
    )


# =====================================================


@dataclass(frozen=True)
class DistributionShiftResult:

    ks_statistic: Optional[float]
    p_value: Optional[float]
    n_a: int
    n_b: int

    def to_dict(self) -> Dict[str, Any]:

        return {
            "ks_statistic": self.ks_statistic, "p_value": self.p_value, "n_a": self.n_a, "n_b": self.n_b,
        }


def distribution_shift(a: Sequence[float], b: Sequence[float]) -> DistributionShiftResult:

    a = [float(v) for v in a]
    b = [float(v) for v in b]

    if len(a) < 2 or len(b) < 2:
        return DistributionShiftResult(ks_statistic=None, p_value=None, n_a=len(a), n_b=len(b))

    result = scipy_stats.ks_2samp(a, b)

    return DistributionShiftResult(
        ks_statistic=float(result.statistic), p_value=float(result.pvalue), n_a=len(a), n_b=len(b),
    )


# =====================================================


@dataclass(frozen=True)
class ArmComparisonReport:

    metric_name: str
    anova: ANOVAResult
    confidence_intervals: Mapping[str, ConfidenceInterval] = field(default_factory=dict)
    effect_sizes_vs_baseline: Mapping[str, EffectSize] = field(default_factory=dict)
    baseline_arm: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "metric_name": self.metric_name,
            "anova": self.anova.to_dict(),
            "confidence_intervals": {
                name: ci.to_dict() for name, ci in self.confidence_intervals.items()
            },
            "effect_sizes_vs_baseline": {
                name: effect.to_dict() for name, effect in self.effect_sizes_vs_baseline.items()
            },
            "baseline_arm": self.baseline_arm,
        }


def compare_arms(
    values_by_arm: Mapping[str, Sequence[float]], metric_name: str,
    baseline_arm: Optional[str] = None, confidence: float = 0.95,
) -> ArmComparisonReport:

    anova = one_way_anova(*values_by_arm.values())

    confidence_intervals = {
        name: confidence_interval(values, confidence=confidence)
        for name, values in values_by_arm.items()
    }

    baseline_arm = baseline_arm if baseline_arm is not None else next(iter(values_by_arm), None)

    effect_sizes_vs_baseline = {}

    if baseline_arm is not None and baseline_arm in values_by_arm:

        baseline_values = values_by_arm[baseline_arm]

        for name, values in values_by_arm.items():

            if name == baseline_arm:
                continue

            effect_sizes_vs_baseline[name] = effect_size_cohens_d(values, baseline_values)

    return ArmComparisonReport(
        metric_name=metric_name, anova=anova, confidence_intervals=confidence_intervals,
        effect_sizes_vs_baseline=effect_sizes_vs_baseline, baseline_arm=baseline_arm,
    )
