import os

from ai_training.experiment import ExperimentConfig, ExperimentRunner

from tests.ai_training_fixtures import RealCampaignTestCase


class RealSavedModelsTestCase(RealCampaignTestCase):

    # Extends RealCampaignTestCase (a real generated campaign, built
    # once per test class) with a set of real models trained AND saved
    # to disk via ai_training.experiment.ExperimentRunner -- exactly
    # the artifact shape (model.joblib + manifest.json) ai_inference's
    # loader.py consumes. Built once per test class; every directory
    # lives under cls._tmp_dir, cleaned up by the base class's own
    # tearDownClass().

    @classmethod
    def setUpClass(cls):

        super().setUpClass()

        runner = ExperimentRunner()
        cls.runner = runner

        cls.evac_dirs = {}

        for algorithm in ("random_forest", "gradient_boosting", "xgboost"):

            config = ExperimentConfig(
                name=f"evac-{algorithm}", model_name="evacuation_time", algorithm=algorithm,
            )
            result = runner.run(cls.dataset, config)
            directory = os.path.join(cls._tmp_dir, f"evac_{algorithm}")
            runner.save_result(result, directory)
            cls.evac_dirs[algorithm] = directory

        cls.evac_rf_dir = cls.evac_dirs["random_forest"]

        bottleneck_location_config = ExperimentConfig(
            name="bottleneck-location", model_name="bottleneck", model_kwargs={"target": "location"},
        )
        bottleneck_location_result = runner.run(cls.dataset, bottleneck_location_config)
        cls.bottleneck_location_dir = os.path.join(cls._tmp_dir, "bottleneck_location")
        runner.save_result(bottleneck_location_result, cls.bottleneck_location_dir)

        bottleneck_occurrence_config = ExperimentConfig(
            name="bottleneck-occurrence", model_name="bottleneck", model_kwargs={"target": "occurrence"},
        )
        bottleneck_occurrence_result = runner.run(cls.dataset, bottleneck_occurrence_config)
        cls.bottleneck_occurrence_dir = os.path.join(cls._tmp_dir, "bottleneck_occurrence")
        runner.save_result(bottleneck_occurrence_result, cls.bottleneck_occurrence_dir)

        exit_usage_config = ExperimentConfig(name="exit-usage", model_name="exit_usage")
        exit_usage_result = runner.run(cls.dataset, exit_usage_config)
        cls.exit_usage_dir = os.path.join(cls._tmp_dir, "exit_usage")
        runner.save_result(exit_usage_result, cls.exit_usage_dir)

        smoke_config = ExperimentConfig(name="smoke-prediction", model_name="smoke_prediction")
        smoke_result = runner.run(cls.dataset, smoke_config)
        cls.smoke_dir = os.path.join(cls._tmp_dir, "smoke_prediction")
        runner.save_result(smoke_result, cls.smoke_dir)
