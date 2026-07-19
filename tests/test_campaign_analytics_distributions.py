import shutil
import tempfile
import unittest

from training_dataset.loader import load_campaign

from campaign_analytics.distributions import compute_distributions

from tests.training_dataset_fixtures import make_campaign


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="campaign_analytics_distributions_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


class DistributionShapeTests(unittest.TestCase):

    def setUp(self):

        self.output_dir_ctx = _TempOutputDir()
        self.output_dir = self.output_dir_ctx.__enter__()

        make_campaign(self.output_dir, count=6, master_seed=21)

        self.samples = list(load_campaign(self.output_dir))
        self.distributions = compute_distributions(self.samples)

    def tearDown(self):
        self.output_dir_ctx.__exit__(None, None, None)

    def test_every_expected_key_is_present(self):

        expected = {
            "evacuation_time",
            "trapped_occupants",
            "bottleneck_frequency",
            "bottleneck_locations",
            "exit_utilization",
            "stair_utilization",
            "smoke_spread",
            "recommendation_frequencies",
            "decision_policy_frequencies",
        }
        self.assertEqual(set(self.distributions.keys()), expected)

    def test_evacuation_time_histogram_sums_to_finite_value_count(self):

        finite_count = sum(
            1
            for sample in self.samples
            if isinstance(sample.simulation_outcome.get("total_evacuation_time"), (int, float))
        )

        histogram = self.distributions["evacuation_time"]["histogram"]

        self.assertEqual(sum(histogram.values()), finite_count)
        self.assertEqual(self.distributions["evacuation_time"]["summary"]["count"], finite_count)

    def test_bottleneck_frequency_total_matches_sample_count(self):

        summary = self.distributions["bottleneck_frequency"]

        self.assertEqual(summary["total_scenarios"], len(self.samples))
        self.assertLessEqual(summary["scenarios_with_a_bottleneck"], summary["total_scenarios"])

    def test_decision_policy_frequencies_totals_match_building_shape(self):

        frequencies = self.distributions["decision_policy_frequencies"]

        # Building fixture: 3 zones, 1 exit, 1 stair -- per scenario.
        self.assertEqual(sum(frequencies["zone_actions"].values()), 3 * len(self.samples))
        self.assertEqual(sum(frequencies["exit_statuses"].values()), 1 * len(self.samples))
        self.assertEqual(sum(frequencies["stair_statuses"].values()), 1 * len(self.samples))

    def test_smoke_spread_summary_count_is_at_most_sample_count(self):

        smoke = self.distributions["smoke_spread"]

        self.assertLessEqual(smoke["peak_smoke_summary"]["count"], len(self.samples))
        self.assertLessEqual(smoke["affected_zone_count_summary"]["count"], len(self.samples))

    def test_recommendation_frequencies_only_contains_known_target_types(self):

        known_target_types = {"exit", "door", "stair", "zone"}

        for target_type in self.distributions["recommendation_frequencies"]:
            self.assertIn(target_type, known_target_types)


class DistributionEmptyDatasetTests(unittest.TestCase):

    def test_computing_distributions_over_zero_samples_does_not_raise(self):

        distributions = compute_distributions([])

        self.assertEqual(distributions["evacuation_time"]["summary"]["count"], 0)
        self.assertEqual(distributions["bottleneck_frequency"]["total_scenarios"], 0)
        self.assertIsNone(distributions["bottleneck_frequency"]["fraction"])
        self.assertEqual(distributions["bottleneck_locations"], {})
        self.assertEqual(distributions["smoke_spread"]["peak_smoke_summary"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
