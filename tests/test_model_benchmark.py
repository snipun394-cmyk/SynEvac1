import unittest

from ai_training.models.base import build_classifier, build_regressor
from ai_training.models.bottleneck_model import BottleneckModel
from ai_training.models.evacuation_time_model import EvacuationTimeModel

from model_benchmark.algorithms import (
    CLASSIFICATION_ALGORITHMS, CLASSIFICATION_PARAM_GRIDS, EXCLUDED_ALGORITHMS,
    REGRESSION_ALGORITHMS, REGRESSION_PARAM_GRIDS, display_name,
)
from model_benchmark.feature_importance import (
    correlation_analysis, native_importance, permutation_importance_classification,
    permutation_importance_regression,
)
from model_benchmark.robustness import (
    build_condition_tags, congestion_tier, exit_block_tier, floor_of_ignition, occupancy_tier,
    robustness_report_classification, robustness_report_regression,
)
from model_benchmark.search import grid_search_classification, grid_search_regression


# =====================================================
# Predictive Model Development & Benchmark Campaign milestone -- unit
# coverage for model_benchmark/ and the ai_training.models.base algorithm
# factory extension (decision_tree/logistic_regression/linear_regression/
# mlp/dummy), using small hand-built synthetic data rather than the full
# 11,997-scenario benchmark campaign (which lives entirely outside the
# repo/tests -- regenerable via scripts/run_model_benchmark_campaign.py,
# never committed, matching the established "datasets are never
# committed" convention).
# =====================================================


def _make_classification_rows(n=60, seed=0):

    import random
    rng = random.Random(seed)

    X_rows = []
    y = []

    for i in range(n):

        occupancy = rng.randint(1, 30)
        camera_active = rng.randint(0, 5)
        row = {
            "total_occupant_count": occupancy,
            "camera_active_count": camera_active,
            "facp_panel_state": rng.choice(["NORMAL", "ALARM"]),
            "building_alarm_status": rng.choice([True, False]),
        }
        X_rows.append(row)
        y.append(occupancy > 20 and camera_active < 2)  # a learnable-ish rule

    return X_rows, y, [f"scenario-{i}" for i in range(n)]


def _make_regression_rows(n=60, seed=1):

    import random
    rng = random.Random(seed)

    X_rows = []
    y = []

    for i in range(n):

        occupancy = rng.randint(1, 30)
        row = {"total_occupant_count": occupancy, "camera_active_count": rng.randint(0, 5)}
        X_rows.append(row)
        y.append(30.0 + occupancy * 4.0 + rng.uniform(-5, 5))

    return X_rows, y, [f"scenario-{i}" for i in range(n)]


class AlgorithmFactoryExtensionTests(unittest.TestCase):

    def test_every_new_classification_algorithm_builds_and_fits(self):

        X_rows, y, _groups = _make_classification_rows()

        for algorithm in CLASSIFICATION_ALGORITHMS:

            model = BottleneckModel(config={"algorithm": algorithm}, target="occurrence")
            model.fit(X_rows, y)
            predictions = model.predict(X_rows)

            self.assertEqual(len(predictions), len(y))

    def test_every_new_regression_algorithm_builds_and_fits(self):

        X_rows, y, _groups = _make_regression_rows()

        for algorithm in REGRESSION_ALGORITHMS:

            model = EvacuationTimeModel(config={"algorithm": algorithm})
            model.fit(X_rows, y)
            predictions = model.predict(X_rows)

            self.assertEqual(len(predictions), len(y))

    def test_unknown_classification_algorithm_raises(self):

        with self.assertRaises(ValueError):
            build_classifier("not_a_real_algorithm")

    def test_unknown_regression_algorithm_raises(self):

        with self.assertRaises(ValueError):
            build_regressor("not_a_real_algorithm")

    def test_lightgbm_explicitly_excluded_and_disclosed(self):

        self.assertIn("lightgbm", EXCLUDED_ALGORITHMS)
        self.assertNotIn("lightgbm", CLASSIFICATION_ALGORITHMS)
        self.assertNotIn("lightgbm", REGRESSION_ALGORITHMS)

    def test_display_name_covers_every_algorithm(self):

        for algorithm in set(CLASSIFICATION_ALGORITHMS) | set(REGRESSION_ALGORITHMS):
            self.assertNotEqual(display_name(algorithm), "")


class HyperparameterSearchTests(unittest.TestCase):

    def test_classification_grid_search_is_deterministic(self):

        X_rows, y, groups = _make_classification_rows(n=80, seed=5)

        def factory(params):
            return BottleneckModel(config={"algorithm": "random_forest", "algorithm_kwargs": params}, target="occurrence")

        grid = {"n_estimators": [50, 100]}

        result_a = grid_search_classification(factory, X_rows, y, groups, grid, n_folds=3)
        result_b = grid_search_classification(factory, X_rows, y, groups, grid, n_folds=3)

        self.assertEqual(result_a.best_params, result_b.best_params)
        self.assertEqual(len(result_a.trials), 2)

    def test_regression_grid_search_is_deterministic(self):

        X_rows, y, groups = _make_regression_rows(n=80, seed=6)

        def factory(params):
            return EvacuationTimeModel(config={"algorithm": "random_forest", "algorithm_kwargs": params})

        grid = {"n_estimators": [50, 100]}

        result_a = grid_search_regression(factory, X_rows, y, groups, grid, n_folds=3)
        result_b = grid_search_regression(factory, X_rows, y, groups, grid, n_folds=3)

        self.assertEqual(result_a.best_params, result_b.best_params)

    def test_every_declared_grid_is_a_valid_dict(self):

        for algorithm in CLASSIFICATION_ALGORITHMS:
            self.assertIsInstance(CLASSIFICATION_PARAM_GRIDS[algorithm], dict)

        for algorithm in REGRESSION_ALGORITHMS:
            self.assertIsInstance(REGRESSION_PARAM_GRIDS[algorithm], dict)


class RobustnessTaggingTests(unittest.TestCase):

    def test_occupancy_tier_buckets(self):

        self.assertEqual(occupancy_tier(5), "low")
        self.assertEqual(occupancy_tier(15), "medium")
        self.assertEqual(occupancy_tier(25), "high")
        self.assertEqual(occupancy_tier(None), "unknown")

    def test_floor_of_ignition(self):

        self.assertEqual(floor_of_ignition("floor-2"), "floor_2_upper")
        self.assertEqual(floor_of_ignition("floor-1"), "floor_1_ground")
        self.assertEqual(floor_of_ignition(None), "unknown")

    def test_exit_block_tier(self):

        self.assertEqual(exit_block_tier({"Exit_1_State": True, "Exit_2_State": True}), "all_open")
        self.assertEqual(exit_block_tier({"Exit_1_State": False, "Exit_2_State": True}), "blocked")
        self.assertEqual(exit_block_tier({}), "unknown")

    def test_congestion_tier(self):

        self.assertEqual(congestion_tier({"exits_exceeding_capacity": ("exit-1",), "stairs_exceeding_capacity": ()}), "heavy_congestion")
        self.assertEqual(congestion_tier({"exits_exceeding_capacity": (), "stairs_exceeding_capacity": ()}), "normal")
        self.assertEqual(congestion_tier(None), "unknown")

    def test_robustness_report_classification_covers_every_axis(self):

        tags = [build_condition_tags(
            {"total_occupants": 5, "ignition_zone": "zone-lobby", "ignition_floor": "floor-1",
             "Exit_1_State": True, "Exit_2_State": True},
            {"exits_exceeding_capacity": (), "stairs_exceeding_capacity": ()},
        ) for _ in range(10)]

        y_true = [True, False] * 5
        y_pred = [True, False] * 5
        y_proba = [0.9, 0.1] * 5

        results = robustness_report_classification(y_true, y_pred, y_proba, tags)

        axes_covered = set(r.axis for r in results)
        self.assertEqual(axes_covered, {"occupancy_tier", "floor_of_ignition", "exit_block_tier", "congestion_tier", "fire_origin_zone"})

    def test_robustness_report_regression_covers_every_axis(self):

        tags = [build_condition_tags(
            {"total_occupants": 25, "ignition_zone": "zone-upper", "ignition_floor": "floor-2",
             "Exit_1_State": False, "Exit_2_State": True},
            None,
        ) for _ in range(10)]

        y_true = [100.0 + i for i in range(10)]
        y_pred = [98.0 + i for i in range(10)]

        results = robustness_report_regression(y_true, y_pred, tags)

        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertIsNotNone(r.regression_metrics)


class FeatureImportanceTests(unittest.TestCase):

    def test_native_importance_available_for_tree_model(self):

        X_rows, y, _ = _make_classification_rows(n=50, seed=2)
        model = BottleneckModel(config={"algorithm": "random_forest"}, target="occurrence")
        model.fit(X_rows, y)

        importance = native_importance(model)

        self.assertIsNotNone(importance)
        self.assertGreater(len(importance), 0)
        self.assertTrue(all(isinstance(v, float) for v in importance.values()))

    def test_native_importance_none_for_mlp(self):

        X_rows, y, _ = _make_classification_rows(n=50, seed=3)
        model = BottleneckModel(config={"algorithm": "mlp"}, target="occurrence")
        model.fit(X_rows, y)

        self.assertIsNone(native_importance(model))

    def test_permutation_importance_classification_returns_every_feature(self):

        X_rows, y, _ = _make_classification_rows(n=50, seed=4)
        model = BottleneckModel(config={"algorithm": "random_forest"}, target="occurrence")
        model.fit(X_rows, y)

        feature_names = ["total_occupant_count", "camera_active_count", "facp_panel_state", "building_alarm_status"]
        importance = permutation_importance_classification(model, X_rows, y, feature_names, n_repeats=2)

        self.assertEqual(set(importance.keys()), set(feature_names))

    def test_permutation_importance_regression_returns_every_feature(self):

        X_rows, y, _ = _make_regression_rows(n=50, seed=7)
        model = EvacuationTimeModel(config={"algorithm": "random_forest"})
        model.fit(X_rows, y)

        feature_names = ["total_occupant_count", "camera_active_count"]
        importance = permutation_importance_regression(model, X_rows, y, feature_names, n_repeats=2)

        self.assertEqual(set(importance.keys()), set(feature_names))
        # the genuinely predictive feature should show up as more important
        # than the noise feature in this synthetic dataset
        self.assertGreater(importance["total_occupant_count"], importance["camera_active_count"])

    def test_correlation_analysis_finds_a_known_correlated_pair(self):

        rows = [{"a": float(i), "b": float(i) * 2.0 + 1.0, "c": float((i * 37) % 11)} for i in range(30)]

        pairs, numeric_cols = correlation_analysis(rows, threshold=0.8)

        self.assertEqual(set(numeric_cols), {"a", "b", "c"})
        pair_features = {(p.feature_a, p.feature_b) for p in pairs}
        self.assertIn(("a", "b"), pair_features)

    def test_correlation_analysis_handles_fewer_than_two_numeric_columns(self):

        rows = [{"a": float(i)} for i in range(10)]

        pairs, numeric_cols = correlation_analysis(rows)

        self.assertEqual(pairs, ())


if __name__ == "__main__":
    unittest.main()
