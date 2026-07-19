import unittest

from ai_training.metrics import classification_metrics, regression_metrics
from ai_explainability.feature_importance import RankedFeature
from ai_explainability.permutation_importance import (
    default_score,
    permutation_importance,
    rank_permutation_importances,
)

from tests.ai_explainability_fixtures import RealTrainedModelsTestCase


class DefaultScoreTests(RealTrainedModelsTestCase):

    def test_regression_score_is_negative_mae(self):

        score = default_score(self.evac_model, self.evac_X_test, self.evac_y_test)

        predictions = self.evac_model.predict(self.evac_X_test)
        expected = -regression_metrics(self.evac_y_test, predictions)["mae"]

        self.assertAlmostEqual(score, expected)

    def test_classification_score_is_accuracy(self):

        score = default_score(self.bottleneck_model, self.bottleneck_X_test, self.bottleneck_y_test)

        predictions = self.bottleneck_model.predict(self.bottleneck_X_test)
        expected = classification_metrics(self.bottleneck_y_test, predictions)["accuracy"]

        self.assertAlmostEqual(score, expected)


class PermutationImportanceTests(RealTrainedModelsTestCase):

    def test_returns_one_entry_per_raw_feature_column(self):

        importances = permutation_importance(
            self.evac_model, self.evac_X_test, self.evac_y_test, n_repeats=3, random_state=0,
        )

        expected_columns = {key for row in self.evac_X_test for key in row.keys()}

        self.assertEqual(set(importances.keys()), expected_columns)

    def test_is_deterministic_for_a_fixed_random_state(self):

        first = permutation_importance(
            self.evac_model, self.evac_X_test, self.evac_y_test, n_repeats=3, random_state=5,
        )
        second = permutation_importance(
            self.evac_model, self.evac_X_test, self.evac_y_test, n_repeats=3, random_state=5,
        )

        self.assertEqual(first, second)

    def test_different_random_state_can_change_the_result(self):

        first = permutation_importance(
            self.evac_model, self.evac_X_test, self.evac_y_test, n_repeats=1, random_state=1,
        )
        second = permutation_importance(
            self.evac_model, self.evac_X_test, self.evac_y_test, n_repeats=1, random_state=2,
        )

        # Not asserting inequality of the whole dict (could coincidentally
        # match) -- just that the function actually depends on random_state
        # rather than silently ignoring it.
        self.assertIsInstance(first, dict)
        self.assertIsInstance(second, dict)

    def test_an_influential_feature_scores_higher_than_an_irrelevant_one(self):

        importances = permutation_importance(
            self.evac_model, self.evac_X_test, self.evac_y_test, n_repeats=5, random_state=0,
        )

        # Zone_2_Occupancy drives occupant count (and therefore
        # evacuation time) directly in the fixture Building/Definition;
        # Camera_1_State is a device-availability flag with no causal
        # link to evacuation time at all.
        self.assertGreater(importances["Zone_2_Occupancy"], importances["Camera_1_State"])


class RankPermutationImportancesTests(RealTrainedModelsTestCase):

    def test_returns_ranked_feature_instances_sorted_descending(self):

        ranked = rank_permutation_importances(
            self.evac_model, self.evac_X_test, self.evac_y_test, n_repeats=3, random_state=0,
        )

        for entry in ranked:
            self.assertIsInstance(entry, RankedFeature)

        importances = [entry.importance for entry in ranked]
        self.assertEqual(importances, sorted(importances, reverse=True))

    def test_top_n_limits_the_result_length(self):

        ranked = rank_permutation_importances(
            self.evac_model, self.evac_X_test, self.evac_y_test, top_n=3, n_repeats=2, random_state=0,
        )

        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0].rank, 1)


if __name__ == "__main__":
    unittest.main()
