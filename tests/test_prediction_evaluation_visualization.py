import os
import shutil
import tempfile
import unittest

from prediction_evaluation.classification_metrics import compute_classification_metrics
from prediction_evaluation.condition_analysis import analyze_by_condition
from prediction_evaluation.horizon_analysis import analyze_by_horizon
from prediction_evaluation.registry import PredictionRegistry
from prediction_evaluation.timeline import GroundTruthTimeline, match_predictions
from prediction_evaluation.models import GroundTruthSample
from prediction_evaluation import visualization as viz


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 7 --
# evaluation-only plots. Smoke tests: every function must produce a real
# file and never crash against realistic (including degenerate/empty)
# inputs -- content correctness is implicitly covered by the metrics
# tests these plots simply visualize.
# =====================================================


class _FakeBottleneck:
    def __init__(self, probability, predicted_occurrence):
        self.probability = probability
        self.predicted_occurrence = predicted_occurrence


class _FakeEvacuationTime:
    def __init__(self, predicted_seconds):
        self.predicted_seconds = predicted_seconds


class _FakeSnapshot:
    def __init__(self, timestamp, bottleneck=None, evacuation_time_experimental=None):
        self.timestamp = timestamp
        self.bottleneck = bottleneck
        self.evacuation_time_experimental = evacuation_time_experimental
        self.feature_schema_version = "1.0"


class PredictionEvaluationVisualizationTests(unittest.TestCase):

    def setUp(self):

        self.tmp_dir = tempfile.mkdtemp(prefix="pred_eval_viz_")

        registry = PredictionRegistry()
        predictions = []
        gt_samples = []

        for i in range(6):

            t = float(i * 10)
            prob = 0.2 + 0.1 * i
            occurred = prob >= 0.5
            actual = i % 2 == 0

            predictions.append(registry.record(
                timestamp=t, prediction_horizon_seconds=20.0,
                payload=_FakeSnapshot(
                    t, bottleneck=_FakeBottleneck(prob, occurred),
                    evacuation_time_experimental=_FakeEvacuationTime(100.0 + i * 5),
                ),
                source="simulation",
            ))
            gt_samples.append(GroundTruthSample(
                timestamp=t + 20.0, source="simulation", congestion_detected=actual,
                evacuation_time_seconds=95.0 + i * 5, total_occupant_count=10 + i,
            ))

        timeline = GroundTruthTimeline(gt_samples)
        self.evaluations = match_predictions(predictions, timeline, tolerance_seconds=1.0)

        y_true = [e.ground_truth.congestion_detected for e in self.evaluations]
        y_pred = [e.prediction.payload.bottleneck.predicted_occurrence for e in self.evaluations]
        y_proba = [e.prediction.payload.bottleneck.probability for e in self.evaluations]
        self.classification_metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    def tearDown(self):

        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _path(self, name):
        return os.path.join(self.tmp_dir, name)

    def test_predicted_vs_actual_regression(self):

        path = viz.plot_predicted_vs_actual_regression(self.evaluations, self._path("pred_vs_actual.png"))
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_error_over_time(self):

        path = viz.plot_error_over_time(self.evaluations, self._path("error_over_time.png"))
        self.assertTrue(os.path.exists(path))

    def test_accuracy_by_horizon(self):

        results = analyze_by_horizon(self.evaluations, horizon_buckets_seconds=(20.0,))
        path = viz.plot_accuracy_by_horizon(results, self._path("by_horizon.png"))
        self.assertTrue(os.path.exists(path))

    def test_confusion_matrix(self):

        path = viz.plot_confusion_matrix(self.classification_metrics, self._path("confusion.png"))
        self.assertTrue(os.path.exists(path))

    def test_calibration_curve(self):

        path = viz.plot_calibration_curve(self.classification_metrics, self._path("calibration.png"))
        self.assertTrue(os.path.exists(path))

    def test_ground_truth_metric_over_time_occupancy(self):

        path = viz.plot_ground_truth_metric_over_time(self.evaluations, "total_occupant_count", self._path("occupancy.png"))
        self.assertTrue(os.path.exists(path))

    def test_ground_truth_metric_over_time_congestion(self):

        path = viz.plot_ground_truth_metric_over_time(self.evaluations, "congestion_detected", self._path("congestion.png"))
        self.assertTrue(os.path.exists(path))

    def test_unsupported_metric_raises(self):

        with self.assertRaises(ValueError):
            viz.plot_ground_truth_metric_over_time(self.evaluations, "not_a_real_metric", self._path("bad.png"))

    def test_plots_never_crash_on_empty_input(self):

        empty_metrics = compute_classification_metrics([], [])

        self.assertTrue(os.path.exists(viz.plot_predicted_vs_actual_regression((), self._path("empty1.png"))))
        self.assertTrue(os.path.exists(viz.plot_error_over_time((), self._path("empty2.png"))))
        self.assertTrue(os.path.exists(viz.plot_confusion_matrix(empty_metrics, self._path("empty3.png"))))
        self.assertTrue(os.path.exists(viz.plot_calibration_curve(empty_metrics, self._path("empty4.png"))))

    def test_creates_missing_parent_directories(self):

        nested_path = os.path.join(self.tmp_dir, "nested", "sub", "plot.png")
        result = viz.plot_error_over_time(self.evaluations, nested_path)
        self.assertTrue(os.path.exists(result))


if __name__ == "__main__":
    unittest.main()
