import unittest

import numpy as np

from ai_explainability.feature_importance import (
    RankedFeature,
    extract_raw_importances,
    get_feature_importances,
    rank_feature_importances,
)

from tests.ai_explainability_fixtures import RealTrainedModelsTestCase


class _StubPreprocessor:

    def __init__(self, names):
        self._names = names

    def feature_names_out(self):
        return self._names


class _StubEstimator:

    def __init__(self, importances):
        self.feature_importances_ = importances


class _StubModel:

    def __init__(self, names, importances, preprocessor=True):
        self.preprocessor = _StubPreprocessor(names) if preprocessor else None
        self.estimator = _StubEstimator(importances)


class ExtractRawImportancesTests(unittest.TestCase):

    def test_reads_feature_importances_directly_off_the_estimator(self):

        estimator = _StubEstimator([0.1, 0.7, 0.2])

        result = extract_raw_importances(estimator)

        np.testing.assert_allclose(result, [0.1, 0.7, 0.2])

    def test_averages_across_sub_estimators_for_a_multioutput_wrapper(self):

        class _MultiOutputStub:
            def __init__(self, sub_estimators):
                self.estimators_ = sub_estimators

        wrapper = _MultiOutputStub([_StubEstimator([0.2, 0.8]), _StubEstimator([0.4, 0.6])])

        result = extract_raw_importances(wrapper)

        np.testing.assert_allclose(result, [0.3, 0.7])

    def test_raises_for_an_estimator_without_feature_importances(self):

        class _Unsupported:
            pass

        with self.assertRaises(TypeError):
            extract_raw_importances(_Unsupported())


class GetFeatureImportancesTests(unittest.TestCase):

    def test_aggregates_one_hot_encoded_columns_back_to_the_original_name(self):

        names = ["growth", "state=OPEN", "state=CLOSED", "state=LOCKED"]
        importances = [0.4, 0.1, 0.2, 0.3]
        model = _StubModel(names, importances)

        result = get_feature_importances(model)

        self.assertAlmostEqual(result["growth"], 0.4)
        self.assertAlmostEqual(result["state"], 0.6)
        self.assertNotIn("state=OPEN", result)

    def test_aggregate_encoded_false_keeps_one_hot_columns_separate(self):

        names = ["growth", "state=OPEN", "state=CLOSED"]
        importances = [0.5, 0.3, 0.2]
        model = _StubModel(names, importances)

        result = get_feature_importances(model, aggregate_encoded=False)

        self.assertEqual(set(result.keys()), {"growth", "state=OPEN", "state=CLOSED"})

    def test_raises_when_model_is_not_fit(self):

        model = _StubModel(["a"], [1.0], preprocessor=False)

        with self.assertRaises(RuntimeError):
            get_feature_importances(model)


class RankFeatureImportancesTests(unittest.TestCase):

    def test_ties_are_broken_alphabetically_by_feature_name(self):

        model = _StubModel(["zebra", "apple", "middle"], [0.5, 0.5, 0.5])

        ranked = rank_feature_importances(model, aggregate_encoded=False)

        self.assertEqual([entry.feature for entry in ranked], ["apple", "middle", "zebra"])
        self.assertEqual([entry.rank for entry in ranked], [1, 2, 3])

    def test_sorted_descending_by_importance(self):

        model = _StubModel(["a", "b", "c"], [0.1, 0.9, 0.5])

        ranked = rank_feature_importances(model, aggregate_encoded=False)

        self.assertEqual([entry.feature for entry in ranked], ["b", "c", "a"])

    def test_top_n_limits_the_result_length(self):

        model = _StubModel(["a", "b", "c", "d"], [0.1, 0.4, 0.3, 0.2])

        ranked = rank_feature_importances(model, top_n=2, aggregate_encoded=False)

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].feature, "b")


class RankFeatureImportancesRealModelTests(RealTrainedModelsTestCase):

    def test_returns_ranked_feature_instances(self):

        ranked = rank_feature_importances(self.evac_model)

        self.assertGreater(len(ranked), 0)
        for entry in ranked:
            self.assertIsInstance(entry, RankedFeature)

    def test_importances_sum_to_approximately_one(self):

        importances = get_feature_importances(self.evac_model)

        self.assertAlmostEqual(sum(importances.values()), 1.0, places=4)

    def test_is_deterministic_across_repeated_calls(self):

        first = rank_feature_importances(self.evac_model)
        second = rank_feature_importances(self.evac_model)

        self.assertEqual(first, second)

    def test_classification_model_also_supports_feature_importances(self):

        ranked = rank_feature_importances(self.bottleneck_model)

        self.assertGreater(len(ranked), 0)


if __name__ == "__main__":
    unittest.main()
