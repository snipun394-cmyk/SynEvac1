import unittest

import pandas as pd

from predictive_model.scenario_split import (
    apply_split,
    assert_no_scenario_overlap,
    split_scenarios,
)


class ScenarioSplitTests(unittest.TestCase):

    def _scenario_ids(self, n):
        return [f"scn-{i:04d}" for i in range(n)]

    def test_split_covers_every_scenario_exactly_once(self):

        scenario_ids = self._scenario_ids(200)
        split = split_scenarios(scenario_ids, seed=42)

        all_assigned = list(split.train_scenario_ids) + list(split.val_scenario_ids) + list(split.test_scenario_ids)

        self.assertEqual(sorted(all_assigned), sorted(scenario_ids))
        self.assertEqual(len(all_assigned), len(set(all_assigned)))

    def test_split_is_deterministic_for_same_seed(self):

        scenario_ids = self._scenario_ids(150)

        first = split_scenarios(scenario_ids, seed=7)
        second = split_scenarios(scenario_ids, seed=7)

        self.assertEqual(first.train_scenario_ids, second.train_scenario_ids)
        self.assertEqual(first.val_scenario_ids, second.val_scenario_ids)
        self.assertEqual(first.test_scenario_ids, second.test_scenario_ids)

    def test_split_is_order_independent(self):
        """Shuffling the INPUT scenario_ids order must not change the split
        (Phase 2's own determinism requirement -- sorted() before shuffling
        in split_scenarios)."""

        scenario_ids = self._scenario_ids(150)
        reversed_ids = list(reversed(scenario_ids))

        first = split_scenarios(scenario_ids, seed=99)
        second = split_scenarios(reversed_ids, seed=99)

        self.assertEqual(first.train_scenario_ids, second.train_scenario_ids)
        self.assertEqual(first.val_scenario_ids, second.val_scenario_ids)
        self.assertEqual(first.test_scenario_ids, second.test_scenario_ids)

    def test_split_respects_ratios_approximately(self):

        scenario_ids = self._scenario_ids(2000)
        split = split_scenarios(scenario_ids, seed=1, ratios=(0.70, 0.15, 0.15))

        self.assertAlmostEqual(len(split.train_scenario_ids) / 2000, 0.70, delta=0.01)
        self.assertAlmostEqual(len(split.val_scenario_ids) / 2000, 0.15, delta=0.01)
        self.assertAlmostEqual(len(split.test_scenario_ids) / 2000, 0.15, delta=0.01)

    def test_split_rejects_ratios_not_summing_to_one(self):

        with self.assertRaises(ValueError):
            split_scenarios(self._scenario_ids(10), seed=1, ratios=(0.5, 0.3, 0.3))

    def test_apply_split_and_assert_no_overlap_on_real_frame(self):

        scenario_ids = self._scenario_ids(60)
        split = split_scenarios(scenario_ids, seed=5)

        rows = []
        for scenario_id in scenario_ids:
            for tick in range(3):
                rows.append({"scenario_id": scenario_id, "observation_time": float(tick), "value": 1})
        frame = pd.DataFrame(rows)

        train_df, val_df, test_df = apply_split(frame, split)

        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(frame))
        assert_no_scenario_overlap(split, train_df, val_df, test_df)  # must not raise

    def test_assert_no_scenario_overlap_detects_id_set_overlap(self):

        scenario_ids = self._scenario_ids(30)
        split = split_scenarios(scenario_ids, seed=3)

        # Manually corrupt the split so one scenario appears in two ID sets.
        overlapping_id = split.train_scenario_ids[0]
        corrupted_val_ids = split.val_scenario_ids + (overlapping_id,)

        from dataclasses import replace
        corrupted_split = replace(split, val_scenario_ids=corrupted_val_ids)

        rows = [{"scenario_id": sid, "observation_time": 0.0} for sid in scenario_ids]
        frame = pd.DataFrame(rows)
        train_df, val_df, test_df = apply_split(frame, corrupted_split)

        with self.assertRaises(AssertionError):
            assert_no_scenario_overlap(corrupted_split, train_df, val_df, test_df)

    def test_no_row_from_one_scenario_leaks_into_another_split(self):
        """Mechanical, row-level proof: for every scenario_id that appears
        in the train frame's rows, it must never also appear in val or
        test frame rows (and vice versa) -- the exact property Phase 2
        asks to prove."""

        scenario_ids = self._scenario_ids(90)
        split = split_scenarios(scenario_ids, seed=11)

        rows = []
        for scenario_id in scenario_ids:
            for tick in range(5):
                rows.append({"scenario_id": scenario_id, "observation_time": float(tick)})
        frame = pd.DataFrame(rows)

        train_df, val_df, test_df = apply_split(frame, split)

        train_ids = set(train_df["scenario_id"])
        val_ids = set(val_df["scenario_id"])
        test_ids = set(test_df["scenario_id"])

        self.assertEqual(train_ids & val_ids, set())
        self.assertEqual(train_ids & test_ids, set())
        self.assertEqual(val_ids & test_ids, set())


if __name__ == "__main__":
    unittest.main()
