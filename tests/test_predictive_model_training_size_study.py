import unittest

import numpy as np
import pandas as pd

from predictive_model.feature_prep import build_feature_matrix
from predictive_model.tree_models import HistGradientBoostingModel
from predictive_model.training_size_study import training_size_study


def _synthetic_frame(n_scenarios, rows_per_scenario, seed):

    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scenarios):
        for r in range(rows_per_scenario):
            rows.append({
                "scenario_id": f"scn-{s}",
                "candidate_type": ["Door", "Exit", "Stair"][r % 3],
                "total_active_occupant_count": int(rng.integers(0, 30)),
                "candidate_capacity": int(rng.integers(1, 20)),
                "candidate_walking_distance": float(rng.uniform(0, 60)),
                "candidate_traversable": bool(rng.integers(0, 2)),
                "candidate_adjacent_zone_occupancy": float(rng.integers(0, 20)),
                "candidate_queue_length": int(rng.integers(0, 5)),
                "candidate_approaching_count": int(rng.integers(0, 5)),
                "candidate_congestion_level": ["LOW", "MODERATE", "HIGH", "VERY_HIGH", "CRITICAL"][r % 5],
                "target": bool(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


class TrainingSizeStudyTests(unittest.TestCase):

    def setUp(self):
        self.train_trainable = _synthetic_frame(n_scenarios=40, rows_per_scenario=10, seed=1)
        val_frame = _synthetic_frame(n_scenarios=10, rows_per_scenario=10, seed=2)
        self.val_feat = build_feature_matrix(val_frame)

    def _factory(self):
        return HistGradientBoostingModel(seed=1, max_iter=10)

    def test_returns_one_result_per_fraction(self):

        result = training_size_study(
            self.train_trainable, self.val_feat, self._factory, fractions=(0.1, 0.5, 1.0), seed=1,
        )
        self.assertEqual(set(result.keys()), {"0.1", "0.5", "1.0"})

    def test_scenario_counts_scale_with_fraction(self):

        result = training_size_study(
            self.train_trainable, self.val_feat, self._factory, fractions=(0.25, 0.5, 1.0), seed=1,
        )

        self.assertLess(result["0.25"]["train_scenario_count"], result["0.5"]["train_scenario_count"])
        self.assertLess(result["0.5"]["train_scenario_count"], result["1.0"]["train_scenario_count"])
        self.assertEqual(result["1.0"]["train_scenario_count"], 40)

    def test_full_fraction_uses_every_train_scenario(self):

        result = training_size_study(
            self.train_trainable, self.val_feat, self._factory, fractions=(1.0,), seed=1,
        )
        self.assertEqual(result["1.0"]["train_scenario_count"], 40)
        self.assertEqual(result["1.0"]["train_row_count"], len(self.train_trainable))

    def test_subsampling_preserves_scenario_boundaries(self):
        """A fraction < 1.0 must select whole scenarios only -- every row
        for a selected scenario_id is included, none partially."""

        result = training_size_study(
            self.train_trainable, self.val_feat, self._factory, fractions=(0.25,), seed=1,
        )
        expected_scenarios = result["0.25"]["train_scenario_count"]
        expected_rows = expected_scenarios * 10  # rows_per_scenario in the fixture

        self.assertEqual(result["0.25"]["train_row_count"], expected_rows)

    def test_deterministic_given_same_seed(self):

        first = training_size_study(self.train_trainable, self.val_feat, self._factory, fractions=(0.5,), seed=7)
        second = training_size_study(self.train_trainable, self.val_feat, self._factory, fractions=(0.5,), seed=7)

        self.assertEqual(first["0.5"]["train_scenario_count"], second["0.5"]["train_scenario_count"])
        self.assertEqual(first["0.5"]["train_row_count"], second["0.5"]["train_row_count"])

    def test_val_metrics_are_present_and_in_range(self):

        result = training_size_study(self.train_trainable, self.val_feat, self._factory, fractions=(1.0,), seed=1)
        entry = result["1.0"]

        self.assertIsNotNone(entry["val_roc_auc"])
        self.assertIsNotNone(entry["val_pr_auc"])
        self.assertGreaterEqual(entry["val_roc_auc"], 0.0)
        self.assertLessEqual(entry["val_roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
