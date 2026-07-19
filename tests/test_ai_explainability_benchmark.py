import unittest

from ai_explainability.benchmark import (
    BenchmarkResult,
    benchmark_algorithms,
    best_result,
    flatten_metrics,
    to_table_rows,
)

from tests.ai_training_fixtures import RealCampaignTestCase


class BenchmarkAlgorithmsTests(RealCampaignTestCase):

    def test_unknown_model_name_raises(self):

        with self.assertRaises(ValueError):
            benchmark_algorithms(self.dataset, "not_a_real_model", ["random_forest"])

    def test_regression_benchmark_returns_one_result_per_algorithm(self):

        results = benchmark_algorithms(
            self.dataset, "evacuation_time", ["random_forest", "gradient_boosting"],
        )

        self.assertEqual(len(results), 2)

        for result in results:

            self.assertIsInstance(result, BenchmarkResult)
            self.assertEqual(set(result.metrics.keys()), {"mae", "rmse", "r2"})
            self.assertGreaterEqual(result.training_time_seconds, 0.0)
            self.assertGreaterEqual(result.prediction_time_seconds, 0.0)
            self.assertGreater(result.model_size_bytes, 0)
            self.assertGreater(result.train_size, 0)
            self.assertGreater(result.test_size, 0)

    def test_algorithm_labels_override_the_default_label(self):

        results = benchmark_algorithms(
            self.dataset, "evacuation_time", ["random_forest"],
            algorithm_labels={"random_forest": "Random Forest"},
        )

        self.assertEqual(results[0].label, "Random Forest")
        self.assertEqual(results[0].algorithm, "random_forest")

    def test_classification_benchmark(self):

        results = benchmark_algorithms(
            self.dataset, "bottleneck", ["random_forest", "gradient_boosting"],
            model_kwargs={"target": "occurrence"},
        )

        for result in results:
            self.assertEqual(
                set(result.metrics.keys()), {"accuracy", "precision", "recall", "f1", "roc_auc"},
            )

    def test_multioutput_regression_benchmark(self):

        results = benchmark_algorithms(self.dataset, "exit_usage", ["random_forest"])

        for result in results:
            self.assertEqual(set(result.metrics.keys()), {"per_output", "macro_average"})

    def test_every_algorithm_trains_against_the_same_split(self):

        results = benchmark_algorithms(
            self.dataset, "evacuation_time", ["random_forest", "gradient_boosting"], random_state=3,
        )

        self.assertEqual(results[0].train_size, results[1].train_size)
        self.assertEqual(results[0].test_size, results[1].test_size)

    def test_xgboost_algorithm_is_available(self):

        results = benchmark_algorithms(self.dataset, "evacuation_time", ["xgboost"])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].algorithm, "xgboost")


class ToTableRowsTests(RealCampaignTestCase):

    def test_flattens_regression_metrics_into_the_row(self):

        results = benchmark_algorithms(self.dataset, "evacuation_time", ["random_forest"])
        rows = to_table_rows(results)

        self.assertEqual(len(rows), 1)
        self.assertIn("mae", rows[0])
        self.assertIn("training_time_seconds", rows[0])
        self.assertIn("model_size_bytes", rows[0])

    def test_flattens_multioutput_macro_average_with_a_prefix(self):

        results = benchmark_algorithms(self.dataset, "exit_usage", ["random_forest"])
        rows = to_table_rows(results)

        self.assertIn("macro_mae", rows[0])
        self.assertNotIn("per_output", rows[0])


class BestResultTests(RealCampaignTestCase):

    def setUp(self):

        self.results = benchmark_algorithms(
            self.dataset, "evacuation_time", ["random_forest", "gradient_boosting", "xgboost"],
        )

    def test_minimize_picks_the_lowest_mae(self):

        winner = best_result(self.results, "mae", minimize=True)

        expected = min(self.results, key=lambda r: flatten_metrics(r.metrics)["mae"])
        self.assertIs(winner, expected)

    def test_maximize_picks_the_highest_r2(self):

        winner = best_result(self.results, "r2", minimize=False)

        expected = max(self.results, key=lambda r: flatten_metrics(r.metrics)["r2"])
        self.assertIs(winner, expected)


if __name__ == "__main__":
    unittest.main()
