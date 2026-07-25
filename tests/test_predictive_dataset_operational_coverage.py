import unittest

from predictive_dataset.operational_coverage import operational_coverage_report


def make_meta(**overrides):

    base = {
        "scenario_id": "scn-1", "total_occupants": 10, "fire_growth_time_seconds": 200.0,
        "blocked_door_count": 0, "blocked_exit_count": 0, "unavailable_stair_count": 0,
    }
    base.update(overrides)
    return base


class OperationalCoverageTests(unittest.TestCase):

    def test_fully_open_scenario_counted(self):

        metadata = [make_meta(scenario_id="scn-1")]
        rows = [{"scenario_id": "scn-1", "candidate_id": "exit-1", "currently_congested": False}]

        report = operational_coverage_report(rows, metadata, {"Exit": 1})

        self.assertEqual(report["fully_open_scenarios"], 1)
        self.assertEqual(report["blocked_exit_scenarios"], 0)

    def test_blocked_route_scenario_counted(self):

        metadata = [make_meta(scenario_id="scn-1", blocked_exit_count=1, blocked_door_count=1, unavailable_stair_count=1)]
        rows = [{"scenario_id": "scn-1", "candidate_id": "exit-1", "currently_congested": False}]

        report = operational_coverage_report(rows, metadata, {"Exit": 1})

        self.assertEqual(report["blocked_exit_scenarios"], 1)
        self.assertEqual(report["blocked_door_scenarios"], 1)
        self.assertEqual(report["blocked_stair_scenarios"], 1)
        self.assertEqual(report["fully_open_scenarios"], 0)

    def test_no_bottleneck_vs_single_vs_multiple(self):

        metadata = [make_meta(scenario_id="no-bn"), make_meta(scenario_id="one-bn"), make_meta(scenario_id="two-bn")]

        rows = [
            {"scenario_id": "no-bn", "candidate_id": "exit-1", "currently_congested": False},
            {"scenario_id": "one-bn", "candidate_id": "exit-1", "currently_congested": True},
            {"scenario_id": "one-bn", "candidate_id": "exit-1", "currently_congested": True},  # same candidate again -- still 1
            {"scenario_id": "two-bn", "candidate_id": "exit-1", "currently_congested": True},
            {"scenario_id": "two-bn", "candidate_id": "door-1", "currently_congested": True},
        ]

        report = operational_coverage_report(rows, metadata, {"Exit": 3, "Door": 1})

        self.assertEqual(report["no_bottleneck_scenarios"], 1)
        self.assertEqual(report["single_bottleneck_scenarios"], 1)
        self.assertEqual(report["multiple_bottleneck_scenarios"], 1)

    def test_scenario_with_no_rows_at_all_counts_as_no_bottleneck(self):

        metadata = [make_meta(scenario_id="scn-1"), make_meta(scenario_id="scn-2-no-rows")]
        rows = [{"scenario_id": "scn-1", "candidate_id": "exit-1", "currently_congested": False}]

        report = operational_coverage_report(rows, metadata, {"Exit": 1})

        self.assertEqual(report["no_bottleneck_scenarios"], 2)

    def test_occupancy_and_severity_buckets_present(self):

        metadata = [make_meta(scenario_id="scn-1", total_occupants=3, fire_growth_time_seconds=100.0)]
        rows = []

        report = operational_coverage_report(rows, metadata, {})

        self.assertEqual(report["occupancy_level_scenario_counts"], {"LOW": 1})
        self.assertEqual(report["fire_severity_scenario_counts"], {"FAST_GROWTH_MORE_SEVERE": 1})


if __name__ == "__main__":
    unittest.main()
