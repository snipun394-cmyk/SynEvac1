import unittest

from research_framework.statistics import ConfidenceInterval, EffectSize, PairedComparisonResult

from calibration_benchmark.candidates import WalkingSpeedCandidate
from calibration_benchmark.harness import CalibrationBenchmarkResult, MetricComparison
from calibration_benchmark.recommendation import Verdict, recommend


def _comparison(metric_name, p_value, mean_difference, baseline_mean=10.0, candidate_mean=8.0, n=10):

    return MetricComparison(
        metric_name=metric_name, metric_label=metric_name, n_pairs=n,
        baseline_mean=baseline_mean, candidate_mean=candidate_mean,
        paired=PairedComparisonResult(t_statistic=3.0, p_value=p_value, mean_difference=mean_difference, n=n),
        effect_size=EffectSize(cohens_d=0.8, n_a=n, n_b=n),
        baseline_ci=ConfidenceInterval(mean=baseline_mean, lower=baseline_mean - 1, upper=baseline_mean + 1, confidence=0.95, n=n),
        candidate_ci=ConfidenceInterval(mean=candidate_mean, lower=candidate_mean - 1, upper=candidate_mean + 1, confidence=0.95, n=n),
    )


def _result(comparisons, additional=None):

    candidate = WalkingSpeedCandidate("Adult_Default", 0.6, "test", "test")
    return CalibrationBenchmarkResult(
        candidate=candidate, n_scenarios_requested=10, n_completed_pairs=10,
        baseline_samples=(), candidate_samples=(), comparisons=comparisons,
        additional_comparisons=additional or {},
    )


class RecommendationTests(unittest.TestCase):

    def test_significant_improvement_on_evacuation_time_with_no_regression_is_adopt(self):

        result = _result({
            "evacuation_time": _comparison("evacuation_time", p_value=0.001, mean_difference=2.0),  # baseline - candidate > 0 -> improved
            "queue_length": _comparison("queue_length", p_value=0.5, mean_difference=0.1),
        })

        recommendation = recommend(result)

        self.assertEqual(recommendation.overall_verdict, Verdict.ADOPT)

    def test_significant_regression_on_any_metric_is_reject_even_if_others_improve(self):

        result = _result({
            "evacuation_time": _comparison("evacuation_time", p_value=0.001, mean_difference=2.0),
            "queue_length": _comparison("queue_length", p_value=0.001, mean_difference=-5.0),  # candidate worse
        })

        recommendation = recommend(result)

        self.assertEqual(recommendation.overall_verdict, Verdict.REJECT)

    def test_no_significant_difference_anywhere_is_inconclusive(self):

        result = _result({
            "evacuation_time": _comparison("evacuation_time", p_value=0.5, mean_difference=0.1),
        })

        recommendation = recommend(result)

        self.assertEqual(recommendation.overall_verdict, Verdict.INCONCLUSIVE)

    def test_too_few_pairs_is_not_applicable_for_that_metric(self):

        result = _result({
            "evacuation_time": _comparison("evacuation_time", p_value=None, mean_difference=None, n=1),
        })

        recommendation = recommend(result)

        self.assertEqual(recommendation.metric_verdicts[0].verdict, Verdict.NOT_APPLICABLE)

    def test_higher_is_better_metric_direction_is_respected(self):

        # exit_utilization_balance: HIGHER candidate value is better --
        # mean_difference (baseline - candidate) < 0 means candidate > baseline -> improved.
        result = _result({
            "exit_utilization_balance": _comparison("exit_utilization_balance", p_value=0.01, mean_difference=-0.3),
        })

        recommendation = recommend(result)

        self.assertEqual(recommendation.metric_verdicts[0].verdict, Verdict.ADOPT)


if __name__ == "__main__":
    unittest.main()
