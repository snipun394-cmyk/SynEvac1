import unittest

from training_dataset.loader import SimulationSample

from campaign_analytics.anomaly_detection import run_checks


def _sample(scenario_id, total_evacuation_time=None, total_occupants=2):

    return SimulationSample(
        scenario_id=scenario_id,
        scenario_features={"scenario_id": scenario_id, "total_occupants": total_occupants},
        simulation_outcome={
            "scenario_id": scenario_id, "total_evacuation_time": total_evacuation_time,
        },
        zone_results=[],
        timeline=[],
        ground_truth=None,
        decision_policy=None,
    )


def _codes(findings):
    return {finding.code for finding in findings}


class RepeatedEvacuationTimeTests(unittest.TestCase):

    def test_a_value_repeated_past_the_threshold_is_flagged(self):

        samples = [_sample(f"scn-{i}", total_evacuation_time=42.0) for i in range(10)]

        findings = run_checks(samples)

        self.assertIn("repeated_evacuation_time", _codes(findings))

    def test_all_distinct_values_produce_no_finding(self):

        samples = [_sample(f"scn-{i}", total_evacuation_time=float(i)) for i in range(10)]

        findings = run_checks(samples)

        self.assertNotIn("repeated_evacuation_time", _codes(findings))

    def test_a_small_number_of_ties_below_threshold_is_not_flagged(self):

        # 2 ties out of 10 -- below both the absolute (3) and fractional
        # (10%% -> 1, but min-count of 3 wins) thresholds.
        samples = [_sample(f"scn-{i}", total_evacuation_time=float(i)) for i in range(8)]
        samples.append(_sample("scn-8", total_evacuation_time=1.0))

        findings = run_checks(samples)

        self.assertNotIn("repeated_evacuation_time", _codes(findings))


class ConstantOccupantCountTests(unittest.TestCase):

    def test_identical_occupant_counts_across_the_campaign_are_flagged(self):

        samples = [_sample(f"scn-{i}", total_occupants=5) for i in range(5)]

        findings = run_checks(samples)

        self.assertIn("constant_occupant_count", _codes(findings))

    def test_varying_occupant_counts_are_not_flagged(self):

        samples = [_sample(f"scn-{i}", total_occupants=i + 1) for i in range(5)]

        findings = run_checks(samples)

        self.assertNotIn("constant_occupant_count", _codes(findings))

    def test_a_single_scenario_is_never_flagged_as_constant(self):

        findings = run_checks([_sample("scn-1", total_occupants=5)])

        self.assertNotIn("constant_occupant_count", _codes(findings))


class EvacuationTimeOutlierTests(unittest.TestCase):

    def test_a_far_outlier_is_flagged_with_its_scenario_id(self):

        samples = [_sample(f"scn-{i}", total_evacuation_time=50.0 + i) for i in range(10)]
        samples.append(_sample("scn-outlier", total_evacuation_time=100000.0))

        findings = run_checks(samples)

        outliers = [f for f in findings if f.code == "evacuation_time_outlier"]
        self.assertEqual(len(outliers), 1)
        self.assertEqual(outliers[0].scenario_id, "scn-outlier")

    def test_tightly_clustered_values_produce_no_outlier_finding(self):

        samples = [_sample(f"scn-{i}", total_evacuation_time=50.0 + i * 0.1) for i in range(10)]

        findings = run_checks(samples)

        self.assertNotIn("evacuation_time_outlier", _codes(findings))

    def test_too_few_samples_skips_the_outlier_check_entirely(self):

        samples = [_sample(f"scn-{i}", total_evacuation_time=float(i)) for i in range(3)]
        samples.append(_sample("scn-huge", total_evacuation_time=100000.0))

        findings = run_checks(samples)

        self.assertNotIn("evacuation_time_outlier", _codes(findings))


class NoFindingsTests(unittest.TestCase):

    def test_an_empty_sample_list_produces_no_findings(self):

        self.assertEqual(run_checks([]), [])

    def test_every_finding_is_severity_warning(self):

        samples = [_sample(f"scn-{i}", total_occupants=5) for i in range(5)]

        for finding in run_checks(samples):
            self.assertEqual(finding.severity, "warning")


if __name__ == "__main__":
    unittest.main()
