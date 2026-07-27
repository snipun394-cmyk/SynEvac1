import unittest

import numpy as np

from predictive_model.duplicate_groups import feature_vector_group_ids, shuffle_within_groups


class FeatureVectorGroupIdsTests(unittest.TestCase):

    def test_identical_rows_get_the_same_group_id(self):

        X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        groups = feature_vector_group_ids(X)

        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])

    def test_all_unique_rows_get_distinct_group_ids(self):

        X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        groups = feature_vector_group_ids(X)

        self.assertEqual(len(set(groups.tolist())), 3)

    def test_group_id_count_matches_number_of_unique_rows(self):

        X = np.array([[1.0], [1.0], [1.0], [2.0], [3.0], [3.0]], dtype=np.float32)
        groups = feature_vector_group_ids(X)

        self.assertEqual(len(set(groups.tolist())), 3)
        self.assertEqual(len(groups), 6)

    def test_deterministic_given_same_input(self):

        rng = np.random.default_rng(0)
        X = rng.integers(0, 3, size=(200, 4)).astype(np.float32)

        first = feature_vector_group_ids(X)
        second = feature_vector_group_ids(X)

        np.testing.assert_array_equal(first, second)

    def test_row_order_within_a_group_does_not_affect_grouping(self):
        """Two independently-constructed matrices with the SAME
        multiset of rows in a DIFFERENT order must produce a grouping
        where the same-row-content pairs are still grouped together --
        this is what makes the duplicate-audit's cross-scenario/cross-
        topology row counts trustworthy regardless of source row order."""

        X1 = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]], dtype=np.float32)
        X2 = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)

        groups1 = feature_vector_group_ids(X1)
        groups2 = feature_vector_group_ids(X2)

        self.assertEqual(groups1[0], groups1[2])
        self.assertEqual(groups2[0], groups2[1])

    def test_rejects_non_2d_input(self):

        with self.assertRaises(ValueError):
            feature_vector_group_ids(np.array([1.0, 2.0, 3.0]))


class ShuffleWithinGroupsTests(unittest.TestCase):

    def test_preserves_per_group_positive_count_exactly(self):
        """The defining property this milestone's Phase 5 shuffle-
        battery variants B/C/D/E rely on: shuffling within a group must
        NEVER change how many positives that specific group has --
        only WHICH member rows get them."""

        y = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=int)
        groups = np.array(["a", "a", "a", "b", "b", "b", "c", "c"])

        shuffled = shuffle_within_groups(y, groups, seed=1)

        for group_value in ("a", "b", "c"):
            mask = groups == group_value
            self.assertEqual(int(y[mask].sum()), int(shuffled[mask].sum()))

    def test_overall_positive_count_unchanged(self):

        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, size=500)
        groups = rng.integers(0, 20, size=500)

        shuffled = shuffle_within_groups(y, groups, seed=42)

        self.assertEqual(int(y.sum()), int(shuffled.sum()))

    def test_deterministic_given_same_seed(self):

        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, size=300)
        groups = rng.integers(0, 10, size=300)

        first = shuffle_within_groups(y, groups, seed=7)
        second = shuffle_within_groups(y, groups, seed=7)

        np.testing.assert_array_equal(first, second)

    def test_different_seeds_usually_produce_different_shuffles(self):

        rng = np.random.default_rng(9)
        y = rng.integers(0, 2, size=500)
        groups = rng.integers(0, 5, size=500)

        first = shuffle_within_groups(y, groups, seed=1)
        second = shuffle_within_groups(y, groups, seed=2)

        self.assertFalse(np.array_equal(first, second))

    def test_works_with_string_group_keys(self):
        """A naive np.diff-based boundary detector fails on string
        arrays (TypeError: unsupported operand for '-') -- this is a
        regression test for exactly that bug, found and fixed during
        this milestone's Phase 5 shuffle battery."""

        y = np.array([1, 0, 0, 1, 1, 0], dtype=int)
        groups = np.array(["scn-1", "scn-1", "scn-2", "scn-2", "scn-3", "scn-3"])

        shuffled = shuffle_within_groups(y, groups, seed=0)
        self.assertEqual(len(shuffled), len(y))
        for group_value in ("scn-1", "scn-2", "scn-3"):
            mask = groups == group_value
            self.assertEqual(int(y[mask].sum()), int(shuffled[mask].sum()))

    def test_single_group_is_a_global_shuffle(self):

        y = np.array([1, 1, 0, 0, 0], dtype=int)
        groups = np.array([1, 1, 1, 1, 1])

        shuffled = shuffle_within_groups(y, groups, seed=2)
        self.assertEqual(int(shuffled.sum()), int(y.sum()))

    def test_mismatched_lengths_raise(self):

        with self.assertRaises(ValueError):
            shuffle_within_groups(np.array([1, 0]), np.array([1, 1, 1]), seed=0)


if __name__ == "__main__":
    unittest.main()
