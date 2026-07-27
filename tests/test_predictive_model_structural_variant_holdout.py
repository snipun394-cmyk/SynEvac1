import unittest

import pandas as pd

from predictive_model.structural_variant_holdout import (
    StructuralVariantHoldoutSplit,
    apply_structural_variant_holdout,
    assert_no_variant_holdout_overlap,
    build_structural_variant_holdout_splits,
    structural_variants,
)


def _scenario_metadata(counts):
    """counts: {(family, variant): n_scenarios}"""

    metadata = []
    for (family, variant), n in counts.items():
        for i in range(n):
            metadata.append({
                "scenario_id": f"{variant}-{i}", "topology_family": family, "structural_variant_id": variant,
            })
    return metadata


class StructuralVariantHoldoutTests(unittest.TestCase):

    def setUp(self):
        self.counts = {
            ("fam_a", "fam_a_base"): 5, ("fam_a", "fam_a_variant2"): 4,
            ("fam_b", "fam_b_base"): 3, ("fam_b", "fam_b_variant2"): 2,
        }
        self.metadata = _scenario_metadata(self.counts)

    def test_structural_variants_are_sorted_and_distinct(self):

        variants = structural_variants(self.metadata)
        self.assertEqual(variants, ("fam_a_base", "fam_a_variant2", "fam_b_base", "fam_b_variant2"))

    def test_one_split_per_variant(self):

        splits = build_structural_variant_holdout_splits(self.metadata)
        held_out = {s.held_out_variant for s in splits}

        self.assertEqual(held_out, {variant for (_, variant) in self.counts.keys()})
        self.assertEqual(len(splits), 4)

    def test_held_out_variant_scenarios_are_exactly_the_test_set(self):

        splits = build_structural_variant_holdout_splits(self.metadata)
        split = next(s for s in splits if s.held_out_variant == "fam_a_base")

        expected_test_ids = {f"fam_a_base-{i}" for i in range(5)}
        self.assertEqual(set(split.test_scenario_ids), expected_test_ids)
        self.assertEqual(split.held_out_family, "fam_a")

    def test_train_set_includes_other_variants_of_the_SAME_family(self):
        """The key difference from family-level holdout: training data
        for a held-out variant still includes OTHER variants of that
        same family (e.g. fam_a_variant2 scenarios train toward
        fam_a_base's test set) -- the model has seen the family, just
        not this exact structural shape."""

        splits = build_structural_variant_holdout_splits(self.metadata)
        split = next(s for s in splits if s.held_out_variant == "fam_a_base")

        self.assertTrue(any(sid.startswith("fam_a_variant2-") for sid in split.train_scenario_ids))
        self.assertFalse(any(sid.startswith("fam_a_base-") for sid in split.train_scenario_ids))

        expected_train_count = sum(n for (fam, variant), n in self.counts.items() if variant != "fam_a_base")
        self.assertEqual(len(split.train_scenario_ids), expected_train_count)

    def test_no_scenario_in_both_train_and_test(self):

        splits = build_structural_variant_holdout_splits(self.metadata)
        for split in splits:
            with self.subTest(variant=split.held_out_variant):
                overlap = set(split.train_scenario_ids) & set(split.test_scenario_ids)
                self.assertEqual(overlap, set())

    def test_apply_structural_variant_holdout_filters_frame_rows(self):

        frame = pd.DataFrame({
            "scenario_id": [f"fam_a_base-{i}" for i in range(5)] + [f"fam_a_variant2-{i}" for i in range(4)],
            "value": range(9),
        })

        split = next(
            s for s in build_structural_variant_holdout_splits(self.metadata) if s.held_out_variant == "fam_a_base"
        )
        train_df, test_df = apply_structural_variant_holdout(frame, split)

        self.assertTrue((test_df["scenario_id"].str.startswith("fam_a_base-")).all())
        self.assertFalse((train_df["scenario_id"].str.startswith("fam_a_base-")).any())

    def test_assert_no_variant_holdout_overlap_passes_for_valid_split(self):

        frame = pd.DataFrame({
            "scenario_id": (
                [f"fam_a_base-{i}" for i in range(5)] + [f"fam_a_variant2-{i}" for i in range(4)]
                + [f"fam_b_base-{i}" for i in range(3)] + [f"fam_b_variant2-{i}" for i in range(2)]
            ),
        })
        split = next(
            s for s in build_structural_variant_holdout_splits(self.metadata) if s.held_out_variant == "fam_a_base"
        )
        train_df, test_df = apply_structural_variant_holdout(frame, split)

        assert_no_variant_holdout_overlap(split, train_df, test_df)  # must not raise

    def test_assert_no_variant_holdout_overlap_detects_id_set_overlap(self):

        bad_split = StructuralVariantHoldoutSplit(
            held_out_variant="fam_a_base", held_out_family="fam_a",
            train_scenario_ids=("fam_a_base-0", "fam_a_variant2-0"),
            test_scenario_ids=("fam_a_base-0",),
        )
        empty_frame = pd.DataFrame({"scenario_id": []})

        with self.assertRaises(AssertionError):
            assert_no_variant_holdout_overlap(bad_split, empty_frame, empty_frame)

    def test_splits_are_deterministic_given_same_metadata(self):

        first = build_structural_variant_holdout_splits(self.metadata)
        second = build_structural_variant_holdout_splits(list(reversed(self.metadata)))

        for a, b in zip(first, second):
            self.assertEqual(a.held_out_variant, b.held_out_variant)
            self.assertEqual(a.train_scenario_ids, b.train_scenario_ids)
            self.assertEqual(a.test_scenario_ids, b.test_scenario_ids)


if __name__ == "__main__":
    unittest.main()
