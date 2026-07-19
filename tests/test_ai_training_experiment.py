import os
import unittest

from ai_training.experiment import ExperimentConfig, ExperimentRunner

from tests.ai_training_fixtures import RealCampaignTestCase


class ExperimentRunnerRunTests(RealCampaignTestCase):

    def setUp(self):

        self.runner = ExperimentRunner()

    def test_unknown_model_name_raises(self):

        config = ExperimentConfig(name="bad", model_name="not_a_real_model")

        with self.assertRaises(ValueError):
            self.runner.run(self.dataset, config)

    def test_evacuation_time_experiment_produces_regression_metrics(self):

        config = ExperimentConfig(
            name="exp-001", model_name="evacuation_time", algorithm="random_forest",
            feature_set="scenario_features",
        )

        result = self.runner.run(self.dataset, config)

        self.assertEqual(set(result.metrics.keys()), {"mae", "rmse", "r2"})
        self.assertGreater(result.train_size, 0)
        self.assertGreater(result.test_size_actual, 0)
        self.assertEqual(result.train_size + result.test_size_actual, len(self.dataset))
        self.assertEqual(result.model.metadata["experiment_name"], "exp-001")
        self.assertEqual(result.model.metadata["task"], "regression")

    def test_bottleneck_experiment_with_gradient_boosting(self):

        config = ExperimentConfig(
            name="exp-002", model_name="bottleneck", algorithm="gradient_boosting",
            model_kwargs={"target": "location"},
        )

        result = self.runner.run(self.dataset, config)

        self.assertEqual(
            set(result.metrics.keys()), {"accuracy", "precision", "recall", "f1", "roc_auc"},
        )
        self.assertEqual(result.model.target, "location")

    def test_exit_usage_experiment_produces_multioutput_metrics(self):

        config = ExperimentConfig(name="exp-003", model_name="exit_usage")

        result = self.runner.run(self.dataset, config)

        self.assertEqual(set(result.metrics.keys()), {"per_output", "macro_average"})

    def test_smoke_prediction_experiment_uses_group_aware_split(self):

        config = ExperimentConfig(name="exp-004", model_name="smoke_prediction")

        result = self.runner.run(self.dataset, config)

        self.assertEqual(
            set(result.metrics.keys()), {"accuracy", "precision", "recall", "f1", "roc_auc"},
        )

    def test_run_is_deterministic_for_a_fixed_random_state(self):

        config = ExperimentConfig(
            name="exp-005", model_name="evacuation_time", random_state=11, test_size=0.25,
        )

        first = self.runner.run(self.dataset, config)
        second = self.runner.run(self.dataset, config)

        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.train_size, second.train_size)


class ExperimentRunnerPersistenceTests(RealCampaignTestCase):

    def setUp(self):

        self.runner = ExperimentRunner()

    def test_save_and_load_result_round_trips_metrics_and_predictions(self):

        config = ExperimentConfig(name="exp-persist", model_name="evacuation_time")
        result = self.runner.run(self.dataset, config)

        directory = os.path.join(self._tmp_dir, "exp-persist")
        paths = self.runner.save_result(result, directory)

        self.assertTrue(os.path.isfile(paths["model"]))
        self.assertTrue(os.path.isfile(paths["manifest"]))

        loaded = self.runner.load_result(directory)

        self.assertEqual(loaded.metrics, result.metrics)
        self.assertEqual(loaded.config, result.config)
        self.assertEqual(loaded.train_size, result.train_size)

        from ai_training.models.evacuation_time_model import EvacuationTimeModel

        X_rows, _y, _extra = EvacuationTimeModel.build_table(self.dataset)

        import numpy as np

        np.testing.assert_allclose(
            result.model.predict(X_rows), loaded.model.predict(X_rows),
        )

    def test_manifest_json_is_human_readable_and_contains_config(self):

        import json

        config = ExperimentConfig(name="exp-manifest", model_name="bottleneck")
        result = self.runner.run(self.dataset, config)

        directory = os.path.join(self._tmp_dir, "exp-manifest")
        self.runner.save_result(result, directory)

        with open(os.path.join(directory, "manifest.json"), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.assertEqual(manifest["config"]["name"], "exp-manifest")
        self.assertEqual(manifest["config"]["model_name"], "bottleneck")
        self.assertIn("metrics", manifest)
        self.assertIn("model_metadata", manifest)


if __name__ == "__main__":
    unittest.main()
