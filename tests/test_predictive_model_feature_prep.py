import unittest

import numpy as np
import pandas as pd

from predictive_model.feature_prep import (
    CANDIDATE_TYPE_CATEGORIES,
    CONGESTION_LEVEL_CATEGORIES,
    build_feature_matrix,
    trainable_rows,
)


def _make_row(**overrides):
    row = {
        "scenario_id": "scn-1",
        "observation_time": 5.0,
        "candidate_id": "door-1",
        "candidate_type": "Door",
        "total_active_occupant_count": 10,
        "candidate_capacity": 5,
        "candidate_walking_distance": 12.0,
        "candidate_traversable": True,
        "candidate_adjacent_zone_occupancy": 3.0,
        "candidate_queue_length": 1,
        "candidate_approaching_count": 2,
        "candidate_congestion_level": "LOW",
        "currently_congested": False,
        "had_any_activity_in_window": True,
        "target": False,
    }
    row.update(overrides)
    return row


class FeaturePrepTests(unittest.TestCase):

    def test_trainable_rows_excludes_null_target(self):

        frame = pd.DataFrame([_make_row(target=False), _make_row(target=None), _make_row(target=True)])
        trainable = trainable_rows(frame)

        self.assertEqual(len(trainable), 2)
        self.assertTrue(trainable["target"].notna().all())

    def test_feature_matrix_shape_is_stable_regardless_of_missingness(self):
        """A frame with NO missing values and a frame WITH missing values
        (for a nullable field) must produce the SAME feature_names/column
        count -- otherwise a model trained on one split's columns
        couldn't score another split whose missingness pattern differs."""

        frame_no_missing = pd.DataFrame([_make_row(), _make_row(candidate_adjacent_zone_occupancy=7.0)])
        frame_with_missing = pd.DataFrame([_make_row(), _make_row(candidate_adjacent_zone_occupancy=None)])

        prepared_no_missing = build_feature_matrix(frame_no_missing)
        prepared_with_missing = build_feature_matrix(frame_with_missing)

        self.assertEqual(prepared_no_missing.feature_names, prepared_with_missing.feature_names)
        self.assertEqual(prepared_no_missing.X.shape[1], prepared_with_missing.X.shape[1])

    def test_missing_numeric_value_is_imputed_with_sentinel_and_flagged(self):

        frame = pd.DataFrame([_make_row(candidate_adjacent_zone_occupancy=None)])
        prepared = build_feature_matrix(frame)

        occupancy_index = prepared.feature_names.index("candidate_adjacent_zone_occupancy")
        missing_flag_index = prepared.feature_names.index("candidate_adjacent_zone_occupancy_missing")

        self.assertEqual(prepared.X[0, occupancy_index], -1.0)
        self.assertEqual(prepared.X[0, missing_flag_index], 1.0)

    def test_categorical_one_hot_covers_all_known_categories(self):

        frame = pd.DataFrame([_make_row(candidate_type=category) for category in CANDIDATE_TYPE_CATEGORIES])
        prepared = build_feature_matrix(frame)

        for category in CANDIDATE_TYPE_CATEGORIES:
            self.assertIn(f"candidate_type={category}", prepared.feature_names)

        # Row i (candidate_type == CANDIDATE_TYPE_CATEGORIES[i]) must have a
        # 1.0 in exactly its own one-hot column and 0.0 in every other.
        for row_index, category in enumerate(CANDIDATE_TYPE_CATEGORIES):
            for other_category in CANDIDATE_TYPE_CATEGORIES:
                col_index = prepared.feature_names.index(f"candidate_type={other_category}")
                expected = 1.0 if other_category == category else 0.0
                self.assertEqual(prepared.X[row_index, col_index], expected)

    def test_congestion_level_missing_gets_its_own_indicator(self):

        frame = pd.DataFrame([_make_row(candidate_congestion_level=None)])
        prepared = build_feature_matrix(frame)

        self.assertIn("candidate_congestion_level=__missing__", prepared.feature_names)
        missing_index = prepared.feature_names.index("candidate_congestion_level=__missing__")
        self.assertEqual(prepared.X[0, missing_index], 1.0)

        for category in CONGESTION_LEVEL_CATEGORIES:
            col_index = prepared.feature_names.index(f"candidate_congestion_level={category}")
            self.assertEqual(prepared.X[0, col_index], 0.0)

    def test_y_and_scenario_ids_align_with_rows(self):

        frame = pd.DataFrame([
            _make_row(scenario_id="scn-a", target=True),
            _make_row(scenario_id="scn-b", target=False),
        ])
        prepared = build_feature_matrix(frame)

        np.testing.assert_array_equal(prepared.y, np.array([1, 0]))
        np.testing.assert_array_equal(prepared.scenario_ids, np.array(["scn-a", "scn-b"]))


if __name__ == "__main__":
    unittest.main()
