import unittest

from predictive_dataset.campaign_config_v2 import CoverageTarget
from predictive_dataset.topology_analysis_v2 import (
    occupancy_strata,
    stair_feature_repair_report,
    structural_distribution,
    topology_family_distribution,
    total_lockout_scenarios,
    verify_coverage_targets,
)


# =====================================================
# Predictive Dataset V2 milestone -- the analysis functions in
# topology_analysis_v2.py this milestone's coverage-target verification
# and stair-repair evidence directly depend on had no isolated unit
# tests (only exercised indirectly through the full campaign script and
# multi_bottleneck_candidate_type_combinations in
# test_predictive_dataset_campaign_v2_pipeline.py). These tests close
# that gap for the functions that decide the milestone's own pass/fail
# verdicts (coverage_target_verification, stair_feature_repair).
# =====================================================


def make_scenario_entry(**overrides):

    base = {
        "scenario_id": "scn-1", "topology_family": "single_exit_lowrise",
        "floor_count": 1, "exit_count": 1, "stair_count": 0, "door_count": 1,
        "total_occupants": 5, "blocked_door_count": 0, "blocked_exit_count": 0,
        "unavailable_stair_count": 0, "contributed_rows": True,
    }
    base.update(overrides)
    return base


def make_stair_row(**overrides):

    base = {
        "candidate_type": "Stair", "candidate_queue_length": 0, "candidate_approaching_count": 0,
        "candidate_walking_distance": 5.0, "currently_congested": False, "target": False,
    }
    base.update(overrides)
    return base


class TopologyFamilyDistributionTests(unittest.TestCase):

    def test_counts_scenarios_per_family(self):

        metadata = [
            make_scenario_entry(topology_family="single_exit_lowrise"),
            make_scenario_entry(topology_family="single_exit_lowrise"),
            make_scenario_entry(topology_family="twin_stair_highrise"),
        ]

        self.assertEqual(
            topology_family_distribution(metadata),
            {"single_exit_lowrise": 2, "twin_stair_highrise": 1},
        )


class StructuralDistributionTests(unittest.TestCase):

    def test_reports_floor_exit_stair_door_counts(self):

        metadata = [
            make_scenario_entry(floor_count=1, exit_count=1, stair_count=0, door_count=1),
            make_scenario_entry(floor_count=3, exit_count=2, stair_count=2, door_count=3),
        ]

        report = structural_distribution(metadata)

        self.assertEqual(report["floor_count"], {1: 1, 3: 1})
        self.assertEqual(report["exit_count"], {1: 1, 2: 1})
        self.assertEqual(report["stair_count"], {0: 1, 2: 1})
        self.assertEqual(report["door_count"], {1: 1, 3: 1})


class OccupancyStrataTests(unittest.TestCase):

    def test_buckets_low_medium_high_at_the_default_thresholds(self):

        metadata = [
            make_scenario_entry(total_occupants=5),   # LOW (<= 10)
            make_scenario_entry(total_occupants=20),  # MEDIUM
            make_scenario_entry(total_occupants=30),  # HIGH (>= 30)
        ]

        self.assertEqual(occupancy_strata(metadata), {"LOW": 1, "MEDIUM": 1, "HIGH": 1})


class TotalLockoutScenariosTests(unittest.TestCase):

    def test_lockout_scenario_with_rows_is_counted_separately_from_zero_row_lockout(self):

        metadata = [
            make_scenario_entry(scenario_id="scn-lockout-rows", exit_count=1, blocked_exit_count=1, contributed_rows=True),
            make_scenario_entry(scenario_id="scn-lockout-zero", exit_count=1, blocked_exit_count=1, contributed_rows=False),
            make_scenario_entry(scenario_id="scn-open", exit_count=1, blocked_exit_count=0, contributed_rows=True),
        ]

        report = total_lockout_scenarios(metadata)

        self.assertEqual(report["total_lockout_scenario_count"], 2)
        self.assertEqual(report["total_lockout_with_candidate_rows"], 1)
        self.assertEqual(report["total_lockout_zero_rows"], 1)
        self.assertEqual(report["sample_scenario_ids_with_rows"], ["scn-lockout-rows"])

    def test_no_lockouts_reports_zero(self):

        metadata = [make_scenario_entry(exit_count=1, blocked_exit_count=0)]

        report = total_lockout_scenarios(metadata)

        self.assertEqual(report["total_lockout_scenario_count"], 0)
        self.assertEqual(report["total_lockout_with_candidate_rows"], 0)


class StairFeatureRepairReportTests(unittest.TestCase):
    """The exact evidence this milestone stakes the Stair-blindness-fixed
    claim on -- must correctly distinguish V1's degenerate all-zero
    signature from V2's genuinely-varying one."""

    def test_v1_style_all_zero_stair_rows_are_reported_as_degenerate(self):

        rows = [
            make_stair_row(candidate_queue_length=0, candidate_approaching_count=0, candidate_walking_distance=0.0),
            make_stair_row(candidate_queue_length=0, candidate_approaching_count=0, candidate_walking_distance=0.0),
        ]

        report = stair_feature_repair_report(rows)

        self.assertTrue(report["walking_distance_is_zero_for_all_rows"])
        self.assertEqual(report["queue_length_nonzero_row_count"], 0)
        self.assertEqual(report["approaching_count_nonzero_row_count"], 0)

    def test_v2_style_varying_stair_rows_are_reported_as_repaired(self):

        rows = [
            make_stair_row(candidate_queue_length=0, candidate_approaching_count=0, candidate_walking_distance=4.5, target=False),
            make_stair_row(candidate_queue_length=3, candidate_approaching_count=2, candidate_walking_distance=4.5, target=True),
        ]

        report = stair_feature_repair_report(rows)

        self.assertFalse(report["walking_distance_is_zero_for_all_rows"])
        self.assertEqual(report["queue_length_nonzero_row_count"], 1)
        self.assertEqual(report["approaching_count_nonzero_row_count"], 1)
        self.assertEqual(report["positive_stair_row_count"], 1)
        self.assertEqual(report["trainable_stair_row_count"], 2)

    def test_no_stair_rows_reports_zero_count_without_error(self):

        self.assertEqual(stair_feature_repair_report([]), {"stair_row_count": 0})


class VerifyCoverageTargetsTests(unittest.TestCase):

    def test_target_met_when_actual_meets_or_exceeds_minimum(self):

        targets = {"foo": CoverageTarget("desc", 10)}
        report = verify_coverage_targets({"foo": 10}, targets)

        self.assertTrue(report["foo"]["met"])
        self.assertEqual(report["foo"]["actual"], 10)
        self.assertEqual(report["foo"]["minimum_required"], 10)

    def test_target_not_met_when_actual_below_minimum(self):

        targets = {"foo": CoverageTarget("desc", 10)}
        report = verify_coverage_targets({"foo": 9}, targets)

        self.assertFalse(report["foo"]["met"])

    def test_missing_actual_count_is_reported_as_not_met_not_an_error(self):

        targets = {"foo": CoverageTarget("desc", 10)}
        report = verify_coverage_targets({}, targets)

        self.assertFalse(report["foo"]["met"])
        self.assertIsNone(report["foo"]["actual"])


if __name__ == "__main__":
    unittest.main()
