import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List

from calibration_benchmark.harness import CalibrationBenchmarkResult, MetricComparison


# =====================================================
# Calibration Benchmark Framework V1 -- "never overwrite SynEvac
# defaults. Calibration candidates must remain experimental until
# proven beneficial" (this milestone's own brief). This module produces
# an Adoption, never performs one -- there is no function anywhere in
# this package that writes to behaviour_profile_resolver/registry.py,
# simulator/capacity.py, simulator/congestion.py, or
# crowd_intelligence/models.py. A human reads an AdoptionRecommendation
# and decides; nothing here can decide for them.
#
# "Beneficial" is judged per metric, never by a single combined score --
# a candidate that shortens evacuation time but also increases queue
# length is a genuine engineering trade-off a recommendation must
# surface, not average away.
# =====================================================


SIGNIFICANCE_ALPHA = 0.05

# Metrics where a LOWER candidate value is the desirable direction.
_LOWER_IS_BETTER = {
    "evacuation_time", "congestion_duration", "queue_length", "peak_occupancy_ratio",
}
# Metrics where a HIGHER candidate value is the desirable direction.
_HIGHER_IS_BETTER = {
    "exit_utilization_balance", "recommendation_effectiveness", "prediction_accuracy",
}


class Verdict(Enum):

    ADOPT = auto()
    REJECT = auto()
    INCONCLUSIVE = auto()
    NOT_APPLICABLE = auto()


@dataclass(frozen=True)
class MetricVerdict:

    metric_name: str
    verdict: Verdict
    reason: str

    def to_dict(self) -> dict:
        return {"metric_name": self.metric_name, "verdict": self.verdict.name, "reason": self.reason}


@dataclass(frozen=True)
class AdoptionRecommendation:

    candidate_name: str
    overall_verdict: Verdict
    metric_verdicts: List[MetricVerdict]
    summary: str

    def to_dict(self) -> dict:

        return {
            "candidate_name": self.candidate_name,
            "overall_verdict": self.overall_verdict.name,
            "metric_verdicts": [v.to_dict() for v in self.metric_verdicts],
            "summary": self.summary,
        }


# =====================================================


def _judge_metric(metric_name: str, comparison: MetricComparison) -> MetricVerdict:

    if comparison.n_pairs < 2 or comparison.paired.p_value is None:

        return MetricVerdict(
            metric_name, Verdict.NOT_APPLICABLE,
            f"Fewer than 2 paired scenarios produced a value for this metric (n={comparison.n_pairs}).",
        )

    if math.isnan(comparison.paired.p_value):

        # scipy's paired t-test returns NaN (not a small p-value) when
        # every paired difference is exactly zero -- a real, definitive
        # "no measurable difference" result, not a degenerate/failed
        # test. Worded separately from the ordinary not-significant case
        # so a reader never sees a bare, unexplained "p=nan".
        return MetricVerdict(
            metric_name, Verdict.INCONCLUSIVE,
            f"Baseline and candidate produced identical values in every one of the "
            f"{comparison.n_pairs} paired scenarios -- no variance for a significance test "
            f"to measure, and no evidence of a difference either.",
        )

    significant = comparison.paired.p_value < SIGNIFICANCE_ALPHA

    if not significant:

        return MetricVerdict(
            metric_name, Verdict.INCONCLUSIVE,
            f"p={comparison.paired.p_value:.4f} (n={comparison.n_pairs}) -- no statistically "
            f"significant difference from the current default at alpha={SIGNIFICANCE_ALPHA}.",
        )

    mean_difference = comparison.paired.mean_difference  # baseline - candidate

    if metric_name in _LOWER_IS_BETTER:
        improved = mean_difference is not None and mean_difference > 0
    elif metric_name in _HIGHER_IS_BETTER:
        improved = mean_difference is not None and mean_difference < 0
    else:
        return MetricVerdict(
            metric_name, Verdict.INCONCLUSIVE,
            f"No adopt/reject direction is defined for metric {metric_name!r}.",
        )

    cohens_d = comparison.effect_size.cohens_d
    cohens_d_text = f"{cohens_d:.3f}" if cohens_d is not None else "n/a"

    if improved:
        return MetricVerdict(
            metric_name, Verdict.ADOPT,
            f"p={comparison.paired.p_value:.4f} (n={comparison.n_pairs}), Cohen's d="
            f"{cohens_d_text}; candidate mean "
            f"({comparison.candidate_mean:.3f}) is significantly better than baseline "
            f"({comparison.baseline_mean:.3f}).",
        )

    return MetricVerdict(
        metric_name, Verdict.REJECT,
        f"p={comparison.paired.p_value:.4f} (n={comparison.n_pairs}), Cohen's d="
        f"{cohens_d_text}; candidate mean "
        f"({comparison.candidate_mean:.3f}) is significantly WORSE than baseline "
        f"({comparison.baseline_mean:.3f}).",
    )


def recommend(result: CalibrationBenchmarkResult) -> AdoptionRecommendation:

    all_comparisons: Dict[str, MetricComparison] = {**result.comparisons, **result.additional_comparisons}
    metric_verdicts = [_judge_metric(name, comparison) for name, comparison in all_comparisons.items()]

    any_reject = any(v.verdict == Verdict.REJECT for v in metric_verdicts)
    any_adopt = any(v.verdict == Verdict.ADOPT for v in metric_verdicts)

    if any_reject:

        overall = Verdict.REJECT
        summary = (
            f"REJECT: candidate {result.candidate.name!r} significantly worsens at least one "
            f"metric. A candidate is never adopted piecemeal -- see the per-metric verdicts below "
            f"for exactly which trade-off is unacceptable."
        )

    elif any_adopt:

        overall = Verdict.ADOPT
        adopted_metrics = [v.metric_name for v in metric_verdicts if v.verdict == Verdict.ADOPT]
        summary = (
            f"ADOPT (candidate): candidate {result.candidate.name!r} significantly improves "
            f"{', '.join(adopted_metrics)} with no significant regression on any other measured "
            f"metric, across {result.n_completed_pairs} paired scenarios. This is a recommendation "
            f"for a human reviewer to act on -- SynEvac's own default "
            f"({result.candidate.current_value} {result.candidate.unit}) is NOT changed by "
            f"running this benchmark."
        )

    else:

        overall = Verdict.INCONCLUSIVE
        summary = (
            f"INCONCLUSIVE: no metric showed a statistically significant difference at "
            f"alpha={SIGNIFICANCE_ALPHA} across {result.n_completed_pairs} paired scenarios. "
            f"Either the sample size is too small to detect a real effect, or this candidate "
            f"genuinely makes no measurable difference to SynEvac's own simulated outcomes."
        )

    return AdoptionRecommendation(
        candidate_name=result.candidate.name, overall_verdict=overall,
        metric_verdicts=metric_verdicts, summary=summary,
    )
