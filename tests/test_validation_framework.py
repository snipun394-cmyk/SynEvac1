import os
import tempfile
import unittest

import numpy as np

from ai_training.dataset import load_campaign_dataset

from rl_training.environment import SynEvacGymEnv
from rl_training.evaluator import EpisodeMetrics
from rl_training.trainer import TrainerConfig

from tests.training_dataset_fixtures import make_building, make_campaign, make_definition

from validation_framework import benchmark_runner, report
from validation_framework.ai_augmented_policy import AIAugmentedObservationWrapper, fit_ai_prediction_bundle
from validation_framework.metrics import rl_policy_metrics
from validation_framework.prediction_validator import validate_bottleneck_model, validate_evacuation_time_model
from validation_framework.recommendation_validator import validate_recommendations
from validation_framework.statistics import bootstrap_confidence_interval
from validation_framework import figures as vf_figures


# =====================================================
# validation_framework.statistics
# =====================================================


class BootstrapConfidenceIntervalTests(unittest.TestCase):

    def test_known_distribution_interval_contains_true_mean(self):

        rng = np.random.default_rng(0)
        values = rng.normal(loc=50.0, scale=5.0, size=200).tolist()

        result = bootstrap_confidence_interval(values, n_resamples=500, random_state=0)

        self.assertLess(result.lower, 50.0)
        self.assertGreater(result.upper, 50.0)
        self.assertEqual(result.n, 200)

    def test_fewer_than_two_values_returns_none_bounds(self):

        result = bootstrap_confidence_interval([5.0])

        self.assertEqual(result.statistic, 5.0)
        self.assertIsNone(result.lower)
        self.assertIsNone(result.upper)

    def test_zero_variance_collapses_to_a_point(self):

        result = bootstrap_confidence_interval([3.0, 3.0, 3.0])

        self.assertEqual(result.lower, 3.0)
        self.assertEqual(result.upper, 3.0)


# =====================================================
# validation_framework.metrics
# =====================================================


class RLPolicyMetricsTests(unittest.TestCase):

    def test_aggregates_across_episodes(self):

        episodes = (
            EpisodeMetrics(
                scenario_id="s1", total_reward=10.0, total_evacuation_time=100.0,
                people_trapped=1, people_evacuated=9, peak_congestion_value=3,
                exits_underutilized=(), exits_exceeding_capacity=(),
            ),
            EpisodeMetrics(
                scenario_id="s2", total_reward=20.0, total_evacuation_time=200.0,
                people_trapped=0, people_evacuated=10, peak_congestion_value=5,
                exits_underutilized=(), exits_exceeding_capacity=(),
            ),
        )

        metrics = rl_policy_metrics(episodes)

        self.assertEqual(metrics["episode_count"], 2)
        self.assertAlmostEqual(metrics["reward_mean"], 15.0)
        self.assertAlmostEqual(metrics["evacuation_time_mean"], 150.0)
        self.assertAlmostEqual(metrics["occupants_saved_mean"], 9.5)
        self.assertIsNone(metrics["smoke_exposure_mean"])
        self.assertIsNone(metrics["travel_distance_mean"])

    def test_empty_episodes_returns_all_none(self):

        metrics = rl_policy_metrics(())

        self.assertEqual(metrics["episode_count"], 0)
        self.assertIsNone(metrics["reward_mean"])


# =====================================================
# validation_framework.figures
# =====================================================


class FiguresTests(unittest.TestCase):

    def test_save_evacuation_curve_writes_a_nonempty_png(self):

        timeline_rows = [
            {"simulation_time": 0.0, "people_remaining": 10, "people_evacuated": 0},
            {"simulation_time": 10.0, "people_remaining": 4, "people_evacuated": 6},
            {"simulation_time": 20.0, "people_remaining": 0, "people_evacuated": 10},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = vf_figures.save_evacuation_curve(timeline_rows, os.path.join(tmp_dir, "curve.png"))

            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_save_occupancy_heatmap_writes_a_nonempty_png(self):

        building = make_building()

        timeline_rows = [
            {"simulation_time": 0.0, "Zone_1_Occupancy": 2, "Zone_2_Occupancy": 1, "Zone_3_Occupancy": 0},
            {"simulation_time": 10.0, "Zone_1_Occupancy": 1, "Zone_2_Occupancy": 0, "Zone_3_Occupancy": 0},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = vf_figures.save_occupancy_heatmap(
                building, timeline_rows, os.path.join(tmp_dir, "heatmap.png"),
            )

            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_save_recommendation_timeline_requires_at_least_one_change(self):

        with tempfile.TemporaryDirectory() as tmp_dir:

            with self.assertRaises(ValueError):
                vf_figures.save_recommendation_timeline((), os.path.join(tmp_dir, "timeline.png"))


# =====================================================
# validation_framework.benchmark_runner -- graceful skipping
# =====================================================


class BenchmarkRunnerGracefulSkipTests(unittest.TestCase):

    def test_no_inputs_supplied_skips_every_phase_without_raising(self):

        result = benchmark_runner.run_full_validation()

        self.assertIsNone(result.campaign_dir)
        self.assertIsNone(result.dataset_statistics)
        self.assertIsNone(result.campaign_analysis)
        self.assertEqual(result.prediction_results, {})
        self.assertIsNone(result.policy_comparison)
        self.assertIsNone(result.recommendation_validation)
        self.assertEqual(result.figure_paths, {})
        self.assertGreaterEqual(len(result.limitations), 3)


class ValidationReportTests(unittest.TestCase):

    def test_report_contains_every_expected_section_header(self):

        result = benchmark_runner.run_full_validation()
        text = report.generate_validation_report(result)

        for header in (
            "# SynEvac Validation & Benchmark Report",
            "## Simulation Performance",
            "## AI Performance",
            "## RL Performance",
            "## Recommendation Performance",
            "## Command Center Performance",
            "## Dataset Statistics",
            "## Known Limitations",
            "## Future Improvements",
        ):
            self.assertIn(header, text)

    def test_write_validation_report_creates_a_file(self):

        result = benchmark_runner.run_full_validation()

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = report.write_validation_report(result, os.path.join(tmp_dir, "report.md"))

            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)


# =====================================================
# End-to-end against a real (tiny) campaign -- reuses training_dataset_
# fixtures.make_campaign(), the same real-pipeline fixture ai_training's
# own tests already rely on, rather than a hand-authored CSV/JSON that
# could silently drift from the real artifact shape.
# =====================================================


class PredictionValidatorRealCampaignTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls._tmp_dir = tempfile.TemporaryDirectory()
        make_campaign(cls._tmp_dir.name, count=6, master_seed=42)
        cls.dataset = load_campaign_dataset(cls._tmp_dir.name, strict=False)

    @classmethod
    def tearDownClass(cls):

        cls._tmp_dir.cleanup()

    def test_validate_evacuation_time_model_returns_regression_metrics(self):

        result = validate_evacuation_time_model(self.dataset)

        self.assertIn("mae", result.metrics)
        self.assertIn("rmse", result.metrics)
        self.assertIn("r2", result.metrics)
        self.assertIn("mae", result.bootstrap_ci)

    def test_validate_bottleneck_model_returns_classification_metrics(self):

        result = validate_bottleneck_model(self.dataset, target="occurrence")

        self.assertIn("accuracy", result.metrics)
        self.assertIn("f1", result.metrics)
        self.assertIn("accuracy", result.bootstrap_ci)


class AIAugmentedObservationWrapperTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls._tmp_dir = tempfile.TemporaryDirectory()
        make_campaign(cls._tmp_dir.name, count=6, master_seed=42)
        cls.dataset = load_campaign_dataset(cls._tmp_dir.name, strict=False)
        cls.building = make_building()
        cls.definition = make_definition()
        cls.bundle = fit_ai_prediction_bundle(cls.dataset, cls.building, max_steps=10, dt=10.0)

    @classmethod
    def tearDownClass(cls):

        cls._tmp_dir.cleanup()

    def _make_wrapped_env(self):

        env = SynEvacGymEnv(self.building, self.definition, "def-test", 1, dt=10.0, max_steps=10)

        return AIAugmentedObservationWrapper(env, self.building, self.definition, "def-test", 1, self.bundle)

    def test_observation_space_is_base_space_plus_ai_feature_count(self):

        wrapped = self._make_wrapped_env()
        base_size = wrapped.env.observation_space.shape[0]

        self.assertEqual(wrapped.observation_space.shape[0], base_size + self.bundle.feature_count)

    def test_same_seed_produces_identical_ai_features_across_fresh_instances(self):

        wrapped_a = self._make_wrapped_env()
        observation_a, _info_a = wrapped_a.reset(seed=100)

        wrapped_b = self._make_wrapped_env()
        observation_b, _info_b = wrapped_b.reset(seed=100)

        np.testing.assert_array_equal(observation_a, observation_b)


class RecommendationValidatorRealCampaignTests(unittest.TestCase):

    def test_runs_end_to_end_without_raising(self):

        building = make_building()
        definition = make_definition()

        result = validate_recommendations(
            building, definition, "def-test", master_seed=5, scenario_count=1, dt=10.0, max_steps=10,
        )

        self.assertGreaterEqual(result.scenarios_evaluated, 0)
        self.assertGreaterEqual(result.total_ticks, 0)
        self.assertIsInstance(result.stability_score, float)
        self.assertEqual(result.false_redirects, ())


class RunFullValidationSmokeTest(unittest.TestCase):

    def test_full_pipeline_with_every_input_supplied(self):

        with tempfile.TemporaryDirectory() as tmp_dir:

            make_campaign(tmp_dir, count=6, master_seed=42)
            building = make_building()
            definition = make_definition()

            result = benchmark_runner.run_full_validation(
                campaign_dir=tmp_dir, building=building, definition=definition, definition_id="def-test",
                master_seed=7, rl_episode_count=1, recommendation_scenario_count=1,
                ai_augmented=True, rl_total_timesteps=100,
                rl_trainer_config=TrainerConfig(algorithm="PPO"),
                dt=10.0, max_steps=10, output_dir=os.path.join(tmp_dir, "results"),
            )

            self.assertIsNotNone(result.dataset_statistics)
            self.assertIsNotNone(result.campaign_analysis)
            self.assertIn("evacuation_time", result.prediction_results)
            self.assertIsNotNone(result.policy_comparison)
            self.assertIn("no_intervention", result.policy_comparison.reports)
            self.assertIn("decision_policy", result.policy_comparison.reports)
            self.assertIn("rl_policy", result.policy_comparison.reports)
            self.assertIsNotNone(result.recommendation_validation)

            text = report.generate_validation_report(result)
            self.assertIn("## Simulation Performance", text)


if __name__ == "__main__":
    unittest.main()
