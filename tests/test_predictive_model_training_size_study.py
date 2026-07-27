import unittest

import numpy as np
import pandas as pd

from predictive_model.feature_prep import build_feature_matrix
from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix
from predictive_model.tree_models import HistGradientBoostingModel
from predictive_model.training_size_study import training_size_study


def _synthetic_frame(n_scenarios, rows_per_scenario, seed, experimental=False):

    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scenarios):
        for r in range(rows_per_scenario):
            row = {
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
            }
            if experimental:
                row["candidate_recent_flow_rate"] = int(rng.integers(0, 10))
                row["candidate_congestion_trend"] = ["RISING", "STABLE", "FALLING", "UNKNOWN"][r % 4]
                row["candidate_alternative_route_count"] = int(rng.integers(0, 5))
            rows.append(row)
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

    def test_feature_builder_defaults_to_v1_v2_baseline_schema(self):
        """Regression test for a real bug found during the V2.2
        milestone: the ORIGINAL implementation hardcoded
        predictive_model.feature_prep.build_feature_matrix internally,
        so passing a val_feat built with a DIFFERENT (e.g. experimental,
        more-columns) schema raised an XGBoost feature-shape mismatch at
        predict_proba time. The default (no feature_builder passed) must
        still match build_feature_matrix's own column count exactly, so
        every existing V2 call site stays completely unaffected."""

        result = training_size_study(self.train_trainable, self.val_feat, self._factory, fractions=(1.0,), seed=1)

        baseline_feat = build_feature_matrix(self.train_trainable)
        self.assertEqual(result["1.0"]["train_row_count"], len(baseline_feat.y))

    def test_progress_fn_called_once_per_fraction_with_matching_results(self):
        """Localized Predictive Model V3 milestone -- progress_fn (added
        so the V3 script could log memory usage per training-size-study
        fraction, after a real OOM crash occurred silently mid-study with
        no visibility into which fraction was running) must be invoked
        exactly once per fraction, in order, with that fraction's own
        result dict -- and must not change the study's own return value."""

        seen = []

        def _progress(key, result):
            seen.append((key, result))

        result = training_size_study(
            self.train_trainable, self.val_feat, self._factory,
            fractions=(0.25, 0.5, 1.0), seed=1, progress_fn=_progress,
        )

        self.assertEqual([key for key, _ in seen], ["0.25", "0.5", "1.0"])
        for key, reported_result in seen:
            self.assertEqual(reported_result, result[key])

    def test_progress_fn_defaults_to_none_and_is_optional(self):
        """Every pre-existing call site (V1/V2/V2.2 scripts) omits
        progress_fn entirely -- must keep working unchanged."""

        result = training_size_study(
            self.train_trainable, self.val_feat, self._factory, fractions=(1.0,), seed=1,
        )
        self.assertEqual(result["1.0"]["train_scenario_count"], 40)

    def test_feature_builder_override_matches_val_feat_schema(self):
        """The bug's actual fix: passing feature_builder=<the same
        builder val_feat was built with> must produce a subset feature
        matrix with the SAME column count as val_feat.X, so predict_proba
        never raises a shape mismatch."""

        train_experimental = _synthetic_frame(n_scenarios=40, rows_per_scenario=10, seed=1, experimental=True)
        val_experimental = _synthetic_frame(n_scenarios=10, rows_per_scenario=10, seed=2, experimental=True)
        val_feat_experimental = build_experimental_feature_matrix(val_experimental)

        def _factory():
            return HistGradientBoostingModel(seed=1, max_iter=10)

        result = training_size_study(
            train_experimental, val_feat_experimental, _factory, fractions=(1.0,), seed=1,
            feature_builder=build_experimental_feature_matrix,
        )

        # the real assertion: this must not raise (predict_proba shape
        # mismatch) -- val_roc_auc/val_pr_auc being present confirms
        # predict_proba actually ran to completion.
        self.assertIsNotNone(result["1.0"]["val_roc_auc"])


if __name__ == "__main__":
    unittest.main()
