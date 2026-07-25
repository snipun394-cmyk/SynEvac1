import unittest

from predictive_dataset.correlation import (
    categorical_target_association, feature_target_correlations, pearson_correlation, redundant_feature_pairs,
)


class PearsonCorrelationTests(unittest.TestCase):

    def test_perfect_positive_correlation(self):

        self.assertAlmostEqual(pearson_correlation([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)

    def test_perfect_negative_correlation(self):

        self.assertAlmostEqual(pearson_correlation([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_no_correlation_constant_y(self):

        self.assertIsNone(pearson_correlation([1, 2, 3, 4], [5, 5, 5, 5]))

    def test_too_few_points_returns_none(self):

        self.assertIsNone(pearson_correlation([1.0], [1.0]))


def make_row(**overrides):

    base = {
        "candidate_type": "Exit", "candidate_congestion_level": "LOW",
        "total_active_occupant_count": 5, "candidate_capacity": 2,
        "candidate_walking_distance": 8.0, "candidate_adjacent_zone_occupancy": 1,
        "candidate_queue_length": 0, "candidate_approaching_count": 0,
        "target": False,
    }
    base.update(overrides)
    return base


class FeatureTargetCorrelationTests(unittest.TestCase):

    def test_feature_that_tracks_target_shows_high_correlation(self):

        rows = [
            make_row(candidate_queue_length=0, target=False),
            make_row(candidate_queue_length=0, target=False),
            make_row(candidate_queue_length=5, target=True),
            make_row(candidate_queue_length=5, target=True),
        ]
        report = feature_target_correlations(rows)

        self.assertAlmostEqual(report["correlation_with_target"]["candidate_queue_length"], 1.0)
        self.assertIn("candidate_queue_length", report["flagged_for_leakage_review"])

    def test_currently_congested_rows_excluded_from_correlation(self):

        rows = [make_row(target=None)]
        report = feature_target_correlations(rows)

        self.assertEqual(report["trainable_row_count"], 0)

    def test_unrelated_feature_shows_no_flag(self):

        rows = [
            make_row(candidate_walking_distance=8.0, target=False),
            make_row(candidate_walking_distance=8.0, target=True),
            make_row(candidate_walking_distance=8.0, target=False),
            make_row(candidate_walking_distance=8.0, target=True),
        ]
        report = feature_target_correlations(rows)

        self.assertIsNone(report["correlation_with_target"]["candidate_walking_distance"])
        self.assertNotIn("candidate_walking_distance", report["flagged_for_leakage_review"])


class CategoricalAssociationTests(unittest.TestCase):

    def test_positive_rate_spread_across_categories(self):

        rows = [
            make_row(candidate_type="Door", target=True),
            make_row(candidate_type="Door", target=True),
            make_row(candidate_type="Stair", target=False),
            make_row(candidate_type="Stair", target=False),
        ]
        report = categorical_target_association(rows, "candidate_type")

        self.assertEqual(report["positive_rate_by_category"]["Door"], 1.0)
        self.assertEqual(report["positive_rate_by_category"]["Stair"], 0.0)
        self.assertAlmostEqual(report["positive_rate_spread"], 1.0)


class RedundantFeaturePairTests(unittest.TestCase):

    def test_perfectly_correlated_features_flagged_as_redundant(self):

        rows = [
            make_row(candidate_queue_length=q, candidate_approaching_count=q)
            for q in range(10)
        ]
        redundant = redundant_feature_pairs(rows)

        pairs = {(entry["feature_a"], entry["feature_b"]) for entry in redundant}
        self.assertIn(("candidate_queue_length", "candidate_approaching_count"), pairs)

    def test_unrelated_features_not_flagged(self):

        rows = [
            make_row(candidate_capacity=2, candidate_walking_distance=8.0),
            make_row(candidate_capacity=4, candidate_walking_distance=8.0),
            make_row(candidate_capacity=2, candidate_walking_distance=8.0),
        ]
        redundant = redundant_feature_pairs(rows)

        pairs = {(entry["feature_a"], entry["feature_b"]) for entry in redundant}
        self.assertNotIn(("candidate_capacity", "candidate_walking_distance"), pairs)


if __name__ == "__main__":
    unittest.main()
