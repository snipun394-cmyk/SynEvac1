import shutil
import tempfile
import unittest

from training_dataset.loader import load_campaign
from training_dataset.statistics import compute_statistics

from tests.training_dataset_fixtures import make_campaign


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="training_dataset_statistics_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


class StatisticsShapeTests(unittest.TestCase):

    def setUp(self):

        self.output_dir_ctx = _TempOutputDir()
        self.output_dir = self.output_dir_ctx.__enter__()

        make_campaign(self.output_dir, count=5, master_seed=101)
        self.dataset = load_campaign(self.output_dir)
        self.stats = compute_statistics(self.dataset)

    def tearDown(self):
        self.output_dir_ctx.__exit__(None, None, None)

    def test_num_scenarios_matches_loaded_sample_count(self):

        self.assertEqual(self.stats["num_scenarios"], len(self.dataset))

    def test_every_expected_statistic_key_is_present(self):

        expected_keys = {
            "num_scenarios",
            "fire_profile_distribution",
            "ignition_zone_distribution",
            "occupancy_distribution",
            "evacuation_time_histogram",
            "trapped_occupant_distribution",
            "bottleneck_frequency",
            "exit_utilization",
            "stair_utilization",
            "detector_failure_rate",
            "camera_failure_rate",
            "hazard_growth_statistics",
            "ground_truth_recommendation_frequencies",
            "decision_policy_frequencies",
        }
        self.assertEqual(set(self.stats.keys()), expected_keys)

    def test_ignition_zone_distribution_only_counts_known_zones_and_sums_to_scenario_count(self):

        distribution = self.stats["ignition_zone_distribution"]

        self.assertTrue(set(distribution).issubset({"zone-1", "zone-2", "zone-3"}))
        self.assertEqual(sum(distribution.values()), len(self.dataset))

    def test_occupancy_distribution_summary_matches_raw_values(self):

        raw_values = [sample.scenario_features["total_occupants"] for sample in self.dataset]

        summary = self.stats["occupancy_distribution"]

        self.assertEqual(summary["count"], len(raw_values))
        self.assertEqual(summary["min"], min(raw_values))
        self.assertEqual(summary["max"], max(raw_values))
        self.assertAlmostEqual(summary["mean"], sum(raw_values) / len(raw_values))

    def test_decision_policy_frequencies_totals_match_number_of_zones_exits_stairs(self):

        frequencies = self.stats["decision_policy_frequencies"]

        total_zone_decisions = sum(frequencies["zone_actions"].values())
        total_exit_decisions = sum(frequencies["exit_statuses"].values())
        total_stair_decisions = sum(frequencies["stair_statuses"].values())

        # Building fixture: 3 zones, 1 exit, 1 stair -- per scenario.
        self.assertEqual(total_zone_decisions, 3 * len(self.dataset))
        self.assertEqual(total_exit_decisions, 1 * len(self.dataset))
        self.assertEqual(total_stair_decisions, 1 * len(self.dataset))

    def test_detector_and_camera_failure_rate_denominators_match_device_count(self):

        # Building fixture: 1 detector, 1 camera -- per scenario.
        self.assertEqual(
            self.stats["detector_failure_rate"]["total_devices_observed"], len(self.dataset),
        )
        self.assertEqual(
            self.stats["camera_failure_rate"]["total_devices_observed"], len(self.dataset),
        )

    def test_evacuation_time_histogram_bucket_counts_sum_to_number_of_finite_evacuation_times(self):

        finite_times = [
            sample.simulation_outcome["total_evacuation_time"]
            for sample in self.dataset
            if isinstance(sample.simulation_outcome.get("total_evacuation_time"), (int, float))
        ]

        histogram = self.stats["evacuation_time_histogram"]

        self.assertEqual(sum(histogram.values()), len(finite_times))


class StatisticsEmptyDatasetTests(unittest.TestCase):

    def test_statistics_over_zero_samples_do_not_raise(self):

        stats = compute_statistics([])

        self.assertEqual(stats["num_scenarios"], 0)
        self.assertEqual(stats["fire_profile_distribution"], {})
        self.assertEqual(stats["occupancy_distribution"]["count"], 0)
        self.assertIsNone(stats["occupancy_distribution"]["mean"])
        self.assertEqual(stats["evacuation_time_histogram"], {})
        self.assertEqual(stats["detector_failure_rate"]["failure_rate"], None)


if __name__ == "__main__":
    unittest.main()
