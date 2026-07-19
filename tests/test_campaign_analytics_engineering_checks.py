import unittest

from training_dataset.loader import SimulationSample

from campaign_analytics.engineering_checks import run_checks


def _sample(
    scenario_id="scn-1",
    *,
    scenario_features=None,
    simulation_outcome=None,
    zone_results=None,
    ground_truth=None,
):

    return SimulationSample(
        scenario_id=scenario_id,
        scenario_features=scenario_features or {"scenario_id": scenario_id, "total_occupants": 2},
        simulation_outcome=simulation_outcome or {"scenario_id": scenario_id, "maximum_congestion": 1},
        zone_results=zone_results or [],
        timeline=[],
        ground_truth=ground_truth,
        decision_policy=None,
    )


def _ground_truth(
    *,
    zone_risk_scores=(),
    exit_risk_scores=(),
    stair_risk_scores=(),
    zone_route_stats=(),
    hazard_spread_order=(),
    first_hazardous_zone=None,
    maximum_hazard_zone=None,
    doors_that_became_bottlenecks=(),
):

    return {
        "zone_risk_scores": list(zone_risk_scores),
        "exit_risk_scores": list(exit_risk_scores),
        "stair_risk_scores": list(stair_risk_scores),
        "zone_route_stats": list(zone_route_stats),
        "hazard_spread_order": list(hazard_spread_order),
        "first_hazardous_zone": first_hazardous_zone,
        "maximum_hazard_zone": maximum_hazard_zone,
        "doors_that_became_bottlenecks": list(doors_that_became_bottlenecks),
    }


def _codes(findings):
    return {finding.code for finding in findings}


class ZonesNeverOnFireTests(unittest.TestCase):

    def test_a_zone_absent_from_every_hazard_signal_is_flagged(self):

        samples = [
            _sample(
                "scn-1",
                ground_truth=_ground_truth(
                    zone_risk_scores=[{"zone_id": "zone-1"}, {"zone_id": "zone-2"}],
                    hazard_spread_order=["zone-1"],
                    maximum_hazard_zone="zone-1",
                ),
            ),
        ]

        findings = run_checks(samples)

        self.assertIn("zones_never_on_fire", _codes(findings))
        finding = next(f for f in findings if f.code == "zones_never_on_fire")
        self.assertIn("zone-2", finding.message)

    def test_every_zone_catching_fire_across_the_campaign_produces_no_finding(self):

        samples = [
            _sample(
                "scn-1",
                ground_truth=_ground_truth(
                    zone_risk_scores=[{"zone_id": "zone-1"}, {"zone_id": "zone-2"}],
                    maximum_hazard_zone="zone-1",
                ),
            ),
            _sample(
                "scn-2",
                ground_truth=_ground_truth(
                    zone_risk_scores=[{"zone_id": "zone-1"}, {"zone_id": "zone-2"}],
                    maximum_hazard_zone="zone-2",
                ),
            ),
        ]

        findings = run_checks(samples)

        self.assertNotIn("zones_never_on_fire", _codes(findings))


class ExitsNeverUsedTests(unittest.TestCase):

    def test_an_exit_never_appearing_in_zone_results_is_flagged(self):

        samples = [
            _sample(
                "scn-1",
                ground_truth=_ground_truth(
                    exit_risk_scores=[{"exit_id": "exit-1"}, {"exit_id": "exit-2"}],
                ),
                zone_results=[{"scenario_id": "scn-1", "zone_id": "zone-1", "exit_used": "exit-1"}],
            ),
        ]

        findings = run_checks(samples)

        self.assertIn("exits_never_used", _codes(findings))
        finding = next(f for f in findings if f.code == "exits_never_used")
        self.assertIn("exit-2", finding.message)

    def test_every_exit_used_at_least_once_produces_no_finding(self):

        samples = [
            _sample(
                "scn-1",
                ground_truth=_ground_truth(exit_risk_scores=[{"exit_id": "exit-1"}]),
                zone_results=[{"scenario_id": "scn-1", "zone_id": "zone-1", "exit_used": "exit-1"}],
            ),
        ]

        findings = run_checks(samples)

        self.assertNotIn("exits_never_used", _codes(findings))


class StairsNeverUsedTests(unittest.TestCase):

    def test_a_stair_never_preferred_by_any_zone_is_flagged(self):

        samples = [
            _sample(
                "scn-1",
                ground_truth=_ground_truth(
                    stair_risk_scores=[{"stair_id": "stair-1"}, {"stair_id": "stair-2"}],
                    zone_route_stats=[{"zone_id": "zone-1", "preferred_stair": "stair-1"}],
                ),
            ),
        ]

        findings = run_checks(samples)

        self.assertIn("stairs_never_used", _codes(findings))


class DeviceAlwaysActiveTests(unittest.TestCase):

    def test_a_camera_that_never_fails_is_flagged(self):

        samples = [
            _sample(
                "scn-1",
                scenario_features={
                    "scenario_id": "scn-1", "total_occupants": 2, "Camera_1_State": "AVAILABLE",
                },
            ),
            _sample(
                "scn-2",
                scenario_features={
                    "scenario_id": "scn-2", "total_occupants": 2, "Camera_1_State": "AVAILABLE",
                },
            ),
        ]

        findings = run_checks(samples)

        self.assertIn("camera_always_active", _codes(findings))

    def test_a_detector_that_sometimes_fails_is_not_flagged(self):

        samples = [
            _sample(
                "scn-1",
                scenario_features={
                    "scenario_id": "scn-1", "total_occupants": 2, "Detector_1_State": "AVAILABLE",
                },
            ),
            _sample(
                "scn-2",
                scenario_features={
                    "scenario_id": "scn-2", "total_occupants": 2, "Detector_1_State": "FAILED",
                },
            ),
        ]

        findings = run_checks(samples)

        self.assertNotIn("detector_always_active", _codes(findings))


class CongestionChecksTests(unittest.TestCase):

    def test_zero_congestion_scenario_is_flagged(self):

        samples = [
            _sample("scn-1", simulation_outcome={"scenario_id": "scn-1", "maximum_congestion": 0}),
        ]

        findings = run_checks(samples)

        self.assertIn("zero_congestion_scenarios", _codes(findings))

    def test_congestion_exceeding_total_occupants_is_flagged_as_impossible(self):

        samples = [
            _sample(
                "scn-1",
                scenario_features={"scenario_id": "scn-1", "total_occupants": 2},
                simulation_outcome={"scenario_id": "scn-1", "maximum_congestion": 99},
            ),
        ]

        findings = run_checks(samples)

        impossible = [f for f in findings if f.code == "impossible_congestion"]
        self.assertEqual(len(impossible), 1)
        self.assertEqual(impossible[0].scenario_id, "scn-1")

    def test_plausible_congestion_is_not_flagged_as_impossible(self):

        samples = [
            _sample(
                "scn-1",
                scenario_features={"scenario_id": "scn-1", "total_occupants": 10},
                simulation_outcome={"scenario_id": "scn-1", "maximum_congestion": 3},
            ),
        ]

        findings = run_checks(samples)

        self.assertNotIn("impossible_congestion", _codes(findings))


class NoFindingsTests(unittest.TestCase):

    def test_an_empty_sample_list_produces_no_findings(self):

        self.assertEqual(run_checks([]), [])

    def test_every_finding_is_severity_warning(self):

        samples = [
            _sample("scn-1", simulation_outcome={"scenario_id": "scn-1", "maximum_congestion": 0}),
        ]

        for finding in run_checks(samples):
            self.assertEqual(finding.severity, "warning")


if __name__ == "__main__":
    unittest.main()
