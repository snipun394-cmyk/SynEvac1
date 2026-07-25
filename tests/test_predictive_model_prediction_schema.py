import unittest

import numpy as np
import pandas as pd

from predictive_model.baselines import build_baselines
from predictive_model.feature_prep import build_feature_matrix
from predictive_model.tree_models import build_tree_models, library_availability_report


def _synthetic_frame(n=60, seed=0):

    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n):
        rows.append({
            "scenario_id": f"scn-{i % 6}",
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
            "currently_congested": False,
            "had_any_activity_in_window": True,
            "target": bool(rng.integers(0, 2)),
        })

    return pd.DataFrame(rows)


class PredictionSchemaTests(unittest.TestCase):
    """This package's model/prediction contract (Phase 15's "prediction
    schema compatibility"): every model, baseline or tree-based, exposes
    fit(X, y, sample_weight=None) and predict_proba(X) -> a 1D array of
    length len(X) with values in [0, 1]. Deliberately synthetic, tiny
    data here -- these tests must never require retraining the real
    campaign-scale model."""

    def setUp(self):
        self.train_frame = _synthetic_frame(n=80, seed=1)
        self.test_frame = _synthetic_frame(n=20, seed=2)

    def _all_models(self):
        models = {**build_baselines(seed=1), **build_tree_models(seed=1)}
        return models

    def test_feature_names_are_identical_across_independently_built_frames(self):
        """The columns a model is trained on and the columns a later,
        independently-built frame produces (e.g. a future val/test split,
        or eventually a live-parity frame) must match exactly in name AND
        order -- otherwise predict_proba(X) would silently score against
        the wrong column semantics."""

        train_prepared = build_feature_matrix(self.train_frame)
        test_prepared = build_feature_matrix(self.test_frame)

        self.assertEqual(train_prepared.feature_names, test_prepared.feature_names)
        self.assertEqual(train_prepared.X.shape[1], test_prepared.X.shape[1])

    def test_every_model_predict_proba_has_expected_shape_and_range(self):

        train_prepared = build_feature_matrix(self.train_frame)
        test_prepared = build_feature_matrix(self.test_frame)

        for name, model in self._all_models().items():
            with self.subTest(model=name):

                model.fit(train_prepared.X, train_prepared.y)
                probabilities = model.predict_proba(test_prepared.X)

                self.assertEqual(probabilities.shape, (len(test_prepared.X),))
                self.assertTrue(np.all(probabilities >= 0.0))
                self.assertTrue(np.all(probabilities <= 1.0))
                self.assertFalse(np.any(np.isnan(probabilities)))

    def test_library_availability_report_matches_importable_modules(self):

        report = library_availability_report()

        try:
            import xgboost  # noqa: F401
            self.assertTrue(report["xgboost"])
        except ImportError:
            self.assertFalse(report["xgboost"])

        self.assertTrue(report["random_forest_sklearn"])
        self.assertTrue(report["gradient_boosting_sklearn_hist"])


if __name__ == "__main__":
    unittest.main()
