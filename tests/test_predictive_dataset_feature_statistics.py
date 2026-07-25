import unittest

from predictive_dataset.feature_statistics import feature_distribution_report


def make_row(**overrides):

    base = {
        "total_active_occupant_count": 5,
        "candidate_type": "Exit",
        "candidate_capacity": 2,
        "candidate_walking_distance": 8.0,
        "candidate_traversable": True,
        "candidate_adjacent_zone_occupancy": 1,
        "candidate_queue_length": 0,
        "candidate_approaching_count": 0,
        "candidate_congestion_level": "LOW",
    }
    base.update(overrides)
    return base


class NumericSummaryTests(unittest.TestCase):

    def test_constant_feature_is_flagged(self):

        rows = [make_row(candidate_capacity=2) for _ in range(10)]
        report = feature_distribution_report(rows)

        self.assertTrue(report["candidate_capacity"]["is_constant"])
        self.assertEqual(report["candidate_capacity"]["min"], 2)
        self.assertEqual(report["candidate_capacity"]["max"], 2)

    def test_near_constant_feature_is_flagged_but_not_constant(self):

        rows = [make_row(candidate_queue_length=0) for _ in range(99)] + [make_row(candidate_queue_length=5)]
        report = feature_distribution_report(rows)

        summary = report["candidate_queue_length"]
        self.assertFalse(summary["is_constant"])
        self.assertTrue(summary["is_near_constant"])

    def test_missing_values_reported_as_fraction_not_zero(self):

        rows = [make_row(candidate_capacity=2) for _ in range(3)] + [make_row(candidate_capacity=None)]
        report = feature_distribution_report(rows)

        summary = report["candidate_capacity"]
        self.assertEqual(summary["missing_count"], 1)
        self.assertAlmostEqual(summary["missing_fraction"], 0.25)
        # mean/min/max computed only over the 3 non-missing values, not fabricated as 0.
        self.assertEqual(summary["mean"], 2)

    def test_outlier_detection(self):

        rows = [make_row(candidate_queue_length=1) for _ in range(50)] + [make_row(candidate_queue_length=1000)]
        report = feature_distribution_report(rows)

        summary = report["candidate_queue_length"]
        self.assertGreaterEqual(summary["outlier_count"], 1)

    def test_all_missing_produces_none_stats_not_a_crash(self):

        rows = [make_row(candidate_capacity=None) for _ in range(5)]
        report = feature_distribution_report(rows)

        summary = report["candidate_capacity"]
        self.assertIsNone(summary["mean"])
        self.assertEqual(summary["missing_count"], 5)


class CategoricalSummaryTests(unittest.TestCase):

    def test_candidate_type_reports_value_counts(self):

        rows = [make_row(candidate_type="Exit") for _ in range(3)] + [make_row(candidate_type="Door")]
        report = feature_distribution_report(rows)

        summary = report["candidate_type"]
        self.assertEqual(summary["value_counts"], {"Exit": 3, "Door": 1})
        self.assertFalse(summary["is_constant"])

    def test_constant_categorical_is_flagged(self):

        rows = [make_row(candidate_traversable=True) for _ in range(5)]
        report = feature_distribution_report(rows)

        self.assertTrue(report["candidate_traversable"]["is_constant"])

    def test_missing_congestion_level_counted_as_missing_not_a_category(self):

        rows = [make_row(candidate_congestion_level="LOW") for _ in range(2)] + [make_row(candidate_congestion_level=None)]
        report = feature_distribution_report(rows)

        summary = report["candidate_congestion_level"]
        self.assertEqual(summary["missing_count"], 1)
        self.assertNotIn("None", summary["value_counts"])


if __name__ == "__main__":
    unittest.main()
