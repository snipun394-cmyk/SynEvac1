import unittest

import numpy as np
import pandas as pd

from predictive_model.operational_slices import (
    annotate_operational_slices,
    build_operational_slice_report,
    slice_report,
    slice_report_by_candidate_type,
)


def _scenario_metadata():
    return [
        {"scenario_id": "scn-low-single", "total_occupants": 5, "exit_count": 1},
        {"scenario_id": "scn-high-multi", "total_occupants": 40, "exit_count": 3},
        {"scenario_id": "scn-med-multi", "total_occupants": 15, "exit_count": 2},
    ]


def _full_horizon_frame():
    """Ground truth for simultaneous_bottleneck_counts: at (scn-high-multi,
    10.0), two DISTINCT candidates (door-1, exit-1) both have target=True
    -- a real multi-bottleneck tick. Every other (scenario, time) has at
    most one positive candidate."""

    rows = [
        {"scenario_id": "scn-low-single", "observation_time": 10.0, "candidate_id": "door-1", "target": False},
        {"scenario_id": "scn-high-multi", "observation_time": 10.0, "candidate_id": "door-1", "target": True},
        {"scenario_id": "scn-high-multi", "observation_time": 10.0, "candidate_id": "exit-1", "target": True},
        {"scenario_id": "scn-med-multi", "observation_time": 10.0, "candidate_id": "stair-1", "target": True},
    ]
    return pd.DataFrame(rows)


def _test_trainable_frame():
    return pd.DataFrame([
        {"scenario_id": "scn-low-single", "observation_time": 10.0, "candidate_id": "door-1", "candidate_type": "Door"},
        {"scenario_id": "scn-high-multi", "observation_time": 10.0, "candidate_id": "door-1", "candidate_type": "Door"},
        {"scenario_id": "scn-high-multi", "observation_time": 10.0, "candidate_id": "exit-1", "candidate_type": "Exit"},
        {"scenario_id": "scn-med-multi", "observation_time": 10.0, "candidate_id": "stair-1", "candidate_type": "Stair"},
    ])


class OperationalSlicesTests(unittest.TestCase):

    def setUp(self):
        self.test_trainable = _test_trainable_frame()
        self.y_true = np.array([0, 1, 1, 1])
        self.y_prob = np.array([0.1, 0.9, 0.8, 0.7])
        self.threshold = 0.5
        self.scenario_metadata = _scenario_metadata()
        self.full_horizon_frame = _full_horizon_frame()

    def test_occupancy_bucket_assignment(self):

        annotated = annotate_operational_slices(
            self.test_trainable, self.y_true, self.y_prob, self.threshold,
            self.scenario_metadata, self.full_horizon_frame,
        )

        by_scenario = annotated.set_index("scenario_id")["occupancy_bucket"].to_dict()
        self.assertEqual(by_scenario["scn-low-single"], "LOW")
        self.assertEqual(by_scenario["scn-high-multi"], "HIGH")
        self.assertEqual(by_scenario["scn-med-multi"], "MEDIUM")

    def test_exit_topology_assignment(self):

        annotated = annotate_operational_slices(
            self.test_trainable, self.y_true, self.y_prob, self.threshold,
            self.scenario_metadata, self.full_horizon_frame,
        )

        by_scenario = annotated.set_index("scenario_id")["exit_topology"].to_dict()
        self.assertEqual(by_scenario["scn-low-single"], "single_exit")
        self.assertEqual(by_scenario["scn-high-multi"], "multi_exit")
        self.assertEqual(by_scenario["scn-med-multi"], "multi_exit")

    def test_bottleneck_category_assignment(self):

        annotated = annotate_operational_slices(
            self.test_trainable, self.y_true, self.y_prob, self.threshold,
            self.scenario_metadata, self.full_horizon_frame,
        )

        rows_by_scenario_candidate = annotated.set_index(["scenario_id", "candidate_id"])["bottleneck_category"].to_dict()

        self.assertEqual(rows_by_scenario_candidate[("scn-low-single", "door-1")], "none")
        self.assertEqual(rows_by_scenario_candidate[("scn-high-multi", "door-1")], "multiple")
        self.assertEqual(rows_by_scenario_candidate[("scn-high-multi", "exit-1")], "multiple")
        self.assertEqual(rows_by_scenario_candidate[("scn-med-multi", "stair-1")], "single")

    def test_slice_report_has_full_metrics_per_value(self):

        annotated = annotate_operational_slices(
            self.test_trainable, self.y_true, self.y_prob, self.threshold,
            self.scenario_metadata, self.full_horizon_frame,
        )
        report = slice_report(annotated, "occupancy_bucket", self.threshold)

        for bucket in ("LOW", "MEDIUM", "HIGH"):
            self.assertIn(bucket, report)
            self.assertIn("false_negative_rate", report[bucket])
            self.assertIn("false_positive_rate", report[bucket])
            self.assertIn("precision", report[bucket])
            self.assertIn("recall", report[bucket])

    def test_slice_report_by_candidate_type_is_nested(self):

        annotated = annotate_operational_slices(
            self.test_trainable, self.y_true, self.y_prob, self.threshold,
            self.scenario_metadata, self.full_horizon_frame,
        )
        report = slice_report_by_candidate_type(annotated, "bottleneck_category", self.threshold)

        self.assertIn("multiple", report)
        self.assertIn("Door", report["multiple"])
        self.assertIn("Exit", report["multiple"])
        self.assertNotIn("Stair", report["multiple"])  # stair-1 is in "single", not "multiple"

    def test_build_operational_slice_report_top_level_keys(self):

        full_report = build_operational_slice_report(
            self.test_trainable, self.y_true, self.y_prob, self.threshold,
            self.scenario_metadata, self.full_horizon_frame,
        )

        for key in (
            "by_bottleneck_category", "by_bottleneck_category_and_candidate_type",
            "by_occupancy_bucket", "by_occupancy_bucket_and_candidate_type",
            "by_exit_topology", "by_exit_topology_and_candidate_type",
        ):
            self.assertIn(key, full_report)

    def test_false_negative_rate_computation(self):
        """scn-med-multi's stair-1 row (y_true=1, y_prob=0.7 >= 0.5 threshold
        -> predicted positive) is a true positive, not a false negative --
        false_negative_rate for the 'single' bottleneck category (which
        contains only this one row) must be 0.0, not None."""

        annotated = annotate_operational_slices(
            self.test_trainable, self.y_true, self.y_prob, self.threshold,
            self.scenario_metadata, self.full_horizon_frame,
        )
        report = slice_report(annotated, "bottleneck_category", self.threshold)

        self.assertEqual(report["single"]["false_negative_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
