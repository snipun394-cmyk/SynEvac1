import unittest

import numpy as np
import pandas as pd

from predictive_model.feature_prep import build_feature_matrix
from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix


def _synthetic_frame(n=40, seed=0):

    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        rows.append({
            "scenario_id": f"scn-{i % 4}",
            "observation_time": float(i),
            "candidate_id": f"door-{i % 2}",
            "candidate_type": ["Door", "Exit", "Stair"][i % 3],
            "total_active_occupant_count": int(rng.integers(0, 30)),
            "candidate_capacity": int(rng.integers(1, 20)),
            "candidate_walking_distance": float(rng.uniform(0, 60)),
            "candidate_traversable": bool(rng.integers(0, 2)),
            "candidate_adjacent_zone_occupancy": None if i % 5 == 0 else float(rng.integers(0, 20)),
            "candidate_queue_length": int(rng.integers(0, 5)),
            "candidate_approaching_count": int(rng.integers(0, 5)),
            "candidate_congestion_level": ["LOW", "MODERATE", "HIGH", "VERY_HIGH", "CRITICAL"][i % 5],
            "candidate_recent_flow_rate": int(rng.integers(0, 10)),
            "candidate_congestion_trend": ["RISING", "STABLE", "FALLING", "UNKNOWN"][i % 4],
            "candidate_alternative_route_count": int(rng.integers(0, 5)),
            "target": bool(rng.integers(0, 2)),
        })

    return pd.DataFrame(rows)


class ExperimentalFeaturePrepTests(unittest.TestCase):

    def setUp(self):
        self.frame = _synthetic_frame(n=60, seed=1)

    def test_experimental_matrix_has_more_columns_than_baseline(self):

        baseline = build_feature_matrix(self.frame)
        experimental = build_experimental_feature_matrix(self.frame)

        # +2 numeric (flow_rate, alternative_route_count) +4 one-hot trend categories
        self.assertEqual(experimental.X.shape[1], baseline.X.shape[1] + 2 + 4)

    def test_new_feature_names_present(self):

        experimental = build_experimental_feature_matrix(self.frame)

        self.assertIn("candidate_recent_flow_rate", experimental.feature_names)
        self.assertIn("candidate_alternative_route_count", experimental.feature_names)
        for category in ("RISING", "STABLE", "FALLING", "UNKNOWN"):
            self.assertIn(f"candidate_congestion_trend={category}", experimental.feature_names)

    def test_trend_one_hot_is_mutually_exclusive_and_exhaustive(self):

        experimental = build_experimental_feature_matrix(self.frame)
        trend_columns = [
            i for i, name in enumerate(experimental.feature_names)
            if name.startswith("candidate_congestion_trend=")
        ]

        row_sums = experimental.X[:, trend_columns].sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(len(self.frame)))

    def test_row_count_and_y_unchanged_from_baseline(self):

        baseline = build_feature_matrix(self.frame)
        experimental = build_experimental_feature_matrix(self.frame)

        self.assertEqual(len(baseline.y), len(experimental.y))
        np.testing.assert_array_equal(baseline.y, experimental.y)

    def test_feature_names_stable_across_independently_built_frames(self):

        frame_a = _synthetic_frame(n=30, seed=2)
        frame_b = _synthetic_frame(n=45, seed=3)

        feat_a = build_experimental_feature_matrix(frame_a)
        feat_b = build_experimental_feature_matrix(frame_b)

        self.assertEqual(feat_a.feature_names, feat_b.feature_names)

    def test_no_nan_in_output_matrix(self):

        experimental = build_experimental_feature_matrix(self.frame)
        self.assertFalse(np.any(np.isnan(experimental.X)))


if __name__ == "__main__":
    unittest.main()
