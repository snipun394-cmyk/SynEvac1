import unittest

from ai_training.split import apply_split, make_split

from tests.ai_training_fixtures import RealCampaignTestCase


class MakeSplitDeterminismTests(unittest.TestCase):

    def test_same_inputs_always_produce_the_same_split(self):

        first = make_split(50, test_size=0.2, random_state=3)
        second = make_split(50, test_size=0.2, random_state=3)

        self.assertEqual(first, second)

    def test_different_random_state_can_produce_a_different_split(self):

        first = make_split(50, test_size=0.2, random_state=3)
        second = make_split(50, test_size=0.2, random_state=4)

        self.assertNotEqual(first.test_indices, second.test_indices)

    def test_train_and_test_indices_are_disjoint_and_cover_every_row(self):

        split = make_split(40, test_size=0.25, random_state=0)

        train_set = set(split.train_indices)
        test_set = set(split.test_indices)

        self.assertEqual(train_set & test_set, set())
        self.assertEqual(train_set | test_set, set(range(40)))

    def test_test_size_is_approximately_respected(self):

        split = make_split(100, test_size=0.3, random_state=0)

        self.assertEqual(len(split.test_indices), 30)
        self.assertEqual(len(split.train_indices), 70)

    def test_validation_split_carves_out_of_the_training_portion(self):

        split = make_split(100, test_size=0.2, val_size=0.2, random_state=0)

        self.assertTrue(split.has_validation)
        self.assertEqual(len(split.test_indices), 20)
        self.assertEqual(len(split.val_indices), 20)
        self.assertEqual(len(split.train_indices), 60)

        train_set = set(split.train_indices)
        val_set = set(split.val_indices)
        test_set = set(split.test_indices)

        self.assertEqual(train_set & val_set, set())
        self.assertEqual(train_set & test_set, set())
        self.assertEqual(val_set & test_set, set())

    def test_no_validation_by_default(self):

        split = make_split(20, test_size=0.2, random_state=0)

        self.assertFalse(split.has_validation)
        self.assertEqual(split.val_indices, ())


class MakeSplitGroupAwarenessTests(unittest.TestCase):

    def test_rows_sharing_a_group_never_straddle_train_and_test(self):

        # 5 groups, 4 rows each -- a group-unaware split would almost
        # certainly put some of a group's rows on each side.
        groups = [group for group in range(5) for _ in range(4)]

        split = make_split(len(groups), groups=groups, test_size=0.4, random_state=0)

        train_groups = {groups[i] for i in split.train_indices}
        test_groups = {groups[i] for i in split.test_indices}

        self.assertEqual(train_groups & test_groups, set())

    def test_group_aware_split_is_deterministic(self):

        groups = [group for group in range(6) for _ in range(3)]

        first = make_split(len(groups), groups=groups, test_size=0.3, random_state=1)
        second = make_split(len(groups), groups=groups, test_size=0.3, random_state=1)

        self.assertEqual(first, second)


class ApplySplitTests(unittest.TestCase):

    def test_apply_split_partitions_rows_by_index(self):

        rows = list("abcdefghij")
        split = make_split(len(rows), test_size=0.2, random_state=0)

        train, val, test = apply_split(rows, split)

        self.assertEqual(len(train) + len(val) + len(test), len(rows))
        self.assertEqual(set(train) | set(test), set(rows))
        self.assertEqual(val, [])


class SplitOnRealDatasetTests(RealCampaignTestCase):

    def test_splitting_real_scenario_rows_is_deterministic_and_complete(self):

        rows = self.dataset.scenario_feature_rows()

        split_a = make_split(len(rows), test_size=0.25, random_state=5)
        split_b = make_split(len(rows), test_size=0.25, random_state=5)

        self.assertEqual(split_a, split_b)

        train, _val, test = apply_split(rows, split_a)

        self.assertEqual(len(train) + len(test), len(rows))


if __name__ == "__main__":
    unittest.main()
