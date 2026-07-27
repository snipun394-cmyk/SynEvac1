import unittest

import pandas as pd

from predictive_model.topology_holdout import (
    TopologyHoldoutSplit,
    apply_topology_holdout,
    assert_no_holdout_overlap,
    build_topology_holdout_splits,
    topology_families,
)


def _scenario_metadata(counts):
    """counts: {family_name: n_scenarios}"""

    metadata = []
    for family, n in counts.items():
        for i in range(n):
            metadata.append({"scenario_id": f"{family}-{i}", "topology_family": family})
    return metadata


class TopologyHoldoutTests(unittest.TestCase):

    def setUp(self):
        self.counts = {"alpha": 5, "beta": 4, "gamma": 3, "delta": 2}
        self.metadata = _scenario_metadata(self.counts)

    def test_topology_families_are_sorted_and_distinct(self):

        families = topology_families(self.metadata)
        self.assertEqual(families, ("alpha", "beta", "delta", "gamma"))

    def test_one_split_per_family(self):

        splits = build_topology_holdout_splits(self.metadata)
        held_out_names = {s.held_out_family for s in splits}

        self.assertEqual(held_out_names, set(self.counts.keys()))
        self.assertEqual(len(splits), 4)

    def test_held_out_family_scenarios_are_exactly_the_test_set(self):

        splits = build_topology_holdout_splits(self.metadata)
        alpha_split = next(s for s in splits if s.held_out_family == "alpha")

        expected_test_ids = {f"alpha-{i}" for i in range(5)}
        self.assertEqual(set(alpha_split.test_scenario_ids), expected_test_ids)

    def test_train_set_excludes_held_out_family_entirely(self):

        splits = build_topology_holdout_splits(self.metadata)
        alpha_split = next(s for s in splits if s.held_out_family == "alpha")

        self.assertFalse(any(sid.startswith("alpha-") for sid in alpha_split.train_scenario_ids))
        # every OTHER family's scenarios must all be present in train
        expected_train_count = sum(n for family, n in self.counts.items() if family != "alpha")
        self.assertEqual(len(alpha_split.train_scenario_ids), expected_train_count)

    def test_no_scenario_in_both_train_and_test(self):

        splits = build_topology_holdout_splits(self.metadata)
        for split in splits:
            with self.subTest(family=split.held_out_family):
                overlap = set(split.train_scenario_ids) & set(split.test_scenario_ids)
                self.assertEqual(overlap, set())

    def test_apply_topology_holdout_filters_frame_rows(self):

        frame = pd.DataFrame({
            "scenario_id": [f"alpha-{i}" for i in range(5)] + [f"beta-{i}" for i in range(4)],
            "value": range(9),
        })

        split = next(s for s in build_topology_holdout_splits(self.metadata) if s.held_out_family == "alpha")
        train_df, test_df = apply_topology_holdout(frame, split)

        self.assertTrue((test_df["scenario_id"].str.startswith("alpha-")).all())
        self.assertFalse((train_df["scenario_id"].str.startswith("alpha-")).any())

    def test_assert_no_holdout_overlap_passes_for_valid_split(self):

        frame = pd.DataFrame({
            "scenario_id": [f"alpha-{i}" for i in range(5)] + [f"beta-{i}" for i in range(4)]
                           + [f"gamma-{i}" for i in range(3)] + [f"delta-{i}" for i in range(2)],
        })
        split = next(s for s in build_topology_holdout_splits(self.metadata) if s.held_out_family == "alpha")
        train_df, test_df = apply_topology_holdout(frame, split)

        assert_no_holdout_overlap(split, train_df, test_df)  # must not raise

    def test_assert_no_holdout_overlap_detects_id_set_overlap(self):

        bad_split = TopologyHoldoutSplit(
            held_out_family="alpha",
            train_scenario_ids=("alpha-0", "beta-0"),
            test_scenario_ids=("alpha-0",),
        )
        empty_frame = pd.DataFrame({"scenario_id": []})

        with self.assertRaises(AssertionError):
            assert_no_holdout_overlap(bad_split, empty_frame, empty_frame)

    def test_assert_no_holdout_overlap_detects_row_level_overlap(self):

        split = next(s for s in build_topology_holdout_splits(self.metadata) if s.held_out_family == "alpha")

        # deliberately construct row-level overlap even though the ID sets themselves don't overlap
        train_df = pd.DataFrame({"scenario_id": ["alpha-0"]})
        test_df = pd.DataFrame({"scenario_id": ["alpha-0"]})

        with self.assertRaises(AssertionError):
            assert_no_holdout_overlap(split, train_df, test_df)

    def test_splits_are_deterministic_given_same_metadata(self):

        first = build_topology_holdout_splits(self.metadata)
        second = build_topology_holdout_splits(list(reversed(self.metadata)))

        for a, b in zip(first, second):
            self.assertEqual(a.held_out_family, b.held_out_family)
            self.assertEqual(a.train_scenario_ids, b.train_scenario_ids)
            self.assertEqual(a.test_scenario_ids, b.test_scenario_ids)


if __name__ == "__main__":
    unittest.main()
