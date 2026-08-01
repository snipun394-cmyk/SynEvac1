import unittest
from types import SimpleNamespace

from crowd_intelligence.models import DensityThresholds

from calibration_benchmark.optional_metrics import PredictionAccuracyMetric, RecommendationEffectivenessMetric


def _ground_truth(**overrides):

    defaults = dict(
        doors_that_became_bottlenecks=(), exits_exceeding_capacity=(), stairs_exceeding_capacity=(),
        recommendations=(),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class RecommendationEffectivenessMetricTests(unittest.TestCase):

    def setUp(self):
        self.metric = RecommendationEffectivenessMetric()

    def test_returns_none_when_no_problem_exists(self):

        ground_truth = _ground_truth()

        self.assertIsNone(self.metric.compute(ground_truth, None, None, None, None))

    def test_returns_one_when_every_problem_target_is_addressed(self):

        ground_truth = _ground_truth(
            doors_that_became_bottlenecks=("door-a",),
            recommendations=({"target_id": "door-a", "action": "Close Door door-a"},),
        )

        self.assertEqual(self.metric.compute(ground_truth, None, None, None, None), 1.0)

    def test_returns_zero_when_no_problem_target_is_addressed(self):

        ground_truth = _ground_truth(
            exits_exceeding_capacity=("exit-a",),
            recommendations=({"target_id": "zone-1", "action": "Additional detector"},),
        )

        self.assertEqual(self.metric.compute(ground_truth, None, None, None, None), 0.0)

    def test_partial_coverage_is_a_fraction(self):

        ground_truth = _ground_truth(
            doors_that_became_bottlenecks=("door-a",), exits_exceeding_capacity=("exit-a",),
            recommendations=({"target_id": "door-a", "action": "Close Door door-a"},),
        )

        self.assertEqual(self.metric.compute(ground_truth, None, None, None, None), 0.5)


class PredictionAccuracyMetricTests(unittest.TestCase):

    def setUp(self):
        self.metric = PredictionAccuracyMetric()
        self.thresholds = DensityThresholds()  # moderate_at=1.0, high_at=2.0, ...

    def test_returns_none_without_a_peak_occupancy_ratio(self):

        self.assertIsNone(self.metric.compute(_ground_truth(), None, None, None, self.thresholds))

    def test_correct_positive_prediction_scores_one(self):

        ground_truth = _ground_truth(doors_that_became_bottlenecks=("door-a",))

        result = self.metric.compute(ground_truth, None, None, 2.5, self.thresholds)  # HIGH -> congested
        self.assertEqual(result, 1.0)

    def test_false_negative_scores_zero(self):

        ground_truth = _ground_truth(doors_that_became_bottlenecks=("door-a",))

        result = self.metric.compute(ground_truth, None, None, 1.0, self.thresholds)  # MODERATE -> not congested
        self.assertEqual(result, 0.0)

    def test_correct_negative_prediction_scores_one(self):

        ground_truth = _ground_truth()

        result = self.metric.compute(ground_truth, None, None, 0.2, self.thresholds)  # LOW -> not congested
        self.assertEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()
