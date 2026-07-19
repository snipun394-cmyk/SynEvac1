import math
import unittest

import numpy as np

from ai_training.metrics import (
    classification_metrics,
    multioutput_regression_metrics,
    regression_metrics,
)


class RegressionMetricsTests(unittest.TestCase):

    def test_perfect_predictions_give_zero_error_and_r2_of_one(self):

        y_true = [1.0, 2.0, 3.0, 4.0]

        metrics = regression_metrics(y_true, y_true)

        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)

    def test_known_errors_match_hand_computed_values(self):

        y_true = [10.0, 20.0, 30.0]
        y_pred = [12.0, 18.0, 33.0]

        # errors: 2, -2, 3 -- MAE = (2+2+3)/3 = 7/3
        # squared errors: 4, 4, 9 -- MSE = 17/3, RMSE = sqrt(17/3)
        metrics = regression_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["mae"], 7.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(17.0 / 3.0))

    def test_r2_is_nan_with_fewer_than_two_samples(self):

        metrics = regression_metrics([5.0], [5.0])

        self.assertTrue(math.isnan(metrics["r2"]))


class MultioutputRegressionMetricsTests(unittest.TestCase):

    def test_per_output_and_macro_average_are_computed_independently(self):

        y_true = [[0.0, 10.0], [2.0, 10.0]]
        y_pred = [[0.0, 10.0], [0.0, 10.0]]  # first output off by 2 on row 2, second perfect

        result = multioutput_regression_metrics(y_true, y_pred, ["exit_a", "exit_b"])

        self.assertAlmostEqual(result["per_output"]["exit_a"]["mae"], 1.0)
        self.assertAlmostEqual(result["per_output"]["exit_b"]["mae"], 0.0)
        self.assertAlmostEqual(result["macro_average"]["mae"], 0.5)


class ClassificationMetricsTests(unittest.TestCase):

    def test_perfect_binary_predictions(self):

        y_true = [True, False, True, False]

        metrics = classification_metrics(y_true, y_true)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_known_accuracy_for_partially_wrong_predictions(self):

        y_true = [True, True, False, False]
        y_pred = [True, False, False, False]  # 3 of 4 correct

        metrics = classification_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["accuracy"], 0.75)

    def test_roc_auc_is_none_without_probabilities(self):

        metrics = classification_metrics([True, False], [True, False])

        self.assertIsNone(metrics["roc_auc"])

    def test_roc_auc_is_none_when_only_one_class_present(self):

        y_true = [True, True, True]
        y_proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.05, 0.95]])

        metrics = classification_metrics(y_true, y_true, y_proba)

        self.assertIsNone(metrics["roc_auc"])

    def test_roc_auc_computed_for_binary_probabilities(self):

        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

        metrics = classification_metrics(y_true, y_pred, y_proba)

        self.assertAlmostEqual(metrics["roc_auc"], 1.0)

    def test_string_labels_work_the_same_as_bool_labels(self):

        y_true = ["zone-1", "zone-2", "zone-1"]
        y_pred = ["zone-1", "zone-1", "zone-1"]

        metrics = classification_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["accuracy"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
