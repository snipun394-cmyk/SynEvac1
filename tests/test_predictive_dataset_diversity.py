import unittest

from predictive_dataset.diversity import candidate_utilization_report, scenario_diversity_report


def make_scenario_meta(**overrides):

    base = {
        "scenario_id": "scn-1", "total_occupants": 10, "ignition_zone_id": "zone-1",
        "fire_growth_time_seconds": 200.0, "blocked_door_count": 0, "blocked_exit_count": 0,
        "unavailable_stair_count": 0, "evacuation_duration": 100.0, "unreachable_occupant_count": 0,
    }
    base.update(overrides)
    return base


class ScenarioDiversityTests(unittest.TestCase):

    def test_empty_campaign_does_not_crash(self):

        self.assertEqual(scenario_diversity_report([]), {"scenario_count": 0})

    def test_reports_spread_across_scenarios(self):

        metadata = [
            make_scenario_meta(scenario_id="scn-1", total_occupants=5),
            make_scenario_meta(scenario_id="scn-2", total_occupants=25),
        ]
        report = scenario_diversity_report(metadata)

        self.assertEqual(report["occupant_count"]["min"], 5)
        self.assertEqual(report["occupant_count"]["max"], 25)
        self.assertEqual(report["scenario_count"], 2)

    def test_identical_scenarios_report_zero_spread(self):

        metadata = [make_scenario_meta(scenario_id=f"scn-{i}") for i in range(5)]
        report = scenario_diversity_report(metadata)

        self.assertEqual(report["occupant_count"]["stddev"], 0.0)
        self.assertEqual(report["occupant_count"]["distinct_values"], 1)

    def test_blocked_route_fraction(self):

        metadata = [
            make_scenario_meta(scenario_id="scn-1", blocked_exit_count=1),
            make_scenario_meta(scenario_id="scn-2", blocked_exit_count=0),
        ]
        report = scenario_diversity_report(metadata)

        self.assertAlmostEqual(report["fraction_scenarios_with_any_blocked_route"], 0.5)


class CandidateUtilizationTests(unittest.TestCase):

    def test_active_fraction_reflects_demand_presence(self):

        rows = [
            {"candidate_id": "exit-1", "candidate_type": "Exit", "candidate_queue_length": 0, "candidate_approaching_count": 0},
            {"candidate_id": "exit-1", "candidate_type": "Exit", "candidate_queue_length": 2, "candidate_approaching_count": 0},
            {"candidate_id": "exit-2", "candidate_type": "Exit", "candidate_queue_length": 0, "candidate_approaching_count": 0},
        ]
        report = candidate_utilization_report(rows)

        self.assertEqual(report["exit-1"]["active_observations"], 1)
        self.assertAlmostEqual(report["exit-1"]["active_fraction"], 0.5)
        self.assertEqual(report["exit-2"]["active_observations"], 0)
        self.assertEqual(report["exit-2"]["active_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
