import json
import shutil
import tempfile
import unittest
from pathlib import Path

from training_dataset.loader import SampleLoadError, discover_scenario_ids

from campaign_analytics.analyzer import (
    CampaignAnalysis,
    Finding,
    analyze_campaign,
    compute_kpis,
    compute_overview,
)

from tests.training_dataset_fixtures import make_campaign


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="campaign_analytics_analyzer_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


class OverviewTests(unittest.TestCase):

    def test_total_scenarios_matches_accepted_count(self):

        with _TempOutputDir() as output_dir:

            summary = make_campaign(output_dir, count=6, master_seed=1)

            analysis = analyze_campaign(output_dir)

            self.assertEqual(analysis.overview.total_scenarios, summary.accepted)

    def test_requested_and_rejected_default_to_none_and_leave_ratio_none(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=3, master_seed=1)

            analysis = analyze_campaign(output_dir)

            self.assertIsNone(analysis.overview.total_requested)
            self.assertIsNone(analysis.overview.rejected)
            self.assertIsNone(analysis.overview.acceptance_ratio)

    def test_requested_and_rejected_are_used_when_supplied(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=4, master_seed=1)

            analysis = analyze_campaign(output_dir, total_requested=5, rejected=1)

            self.assertEqual(analysis.overview.total_requested, 5)
            self.assertEqual(analysis.overview.rejected, 1)
            self.assertAlmostEqual(analysis.overview.acceptance_ratio, 4 / 5)

    def test_evacuation_time_bounds_and_percentiles_are_ordered(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=10, master_seed=2)

            overview = compute_overview(list(analyze_campaign(output_dir).dataset))

            if overview.average_evacuation_time is not None:

                self.assertLessEqual(overview.minimum_evacuation_time, overview.average_evacuation_time)
                self.assertLessEqual(overview.average_evacuation_time, overview.maximum_evacuation_time)

                percentiles = overview.evacuation_time_percentiles
                self.assertLessEqual(percentiles["p50"], percentiles["p90"])
                self.assertLessEqual(percentiles["p90"], percentiles["p95"])
                self.assertLessEqual(percentiles["p95"], percentiles["p99"])

    def test_average_occupants_matches_manual_computation(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=5, master_seed=3)

            dataset = analyze_campaign(output_dir).dataset
            samples = list(dataset)

            expected = sum(s.scenario_features["total_occupants"] for s in samples) / len(samples)

            overview = compute_overview(samples)

            self.assertAlmostEqual(overview.average_occupants, expected)

    def test_device_state_frequencies_sum_to_device_count_times_scenarios(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=5, master_seed=3)

            dataset = analyze_campaign(output_dir).dataset
            overview = compute_overview(list(dataset))

            # Building fixture: 1 detector, 1 camera, 1 door, 1 exit.
            self.assertEqual(sum(overview.detector_failure_frequencies.values()), len(dataset))
            self.assertEqual(sum(overview.camera_failure_frequencies.values()), len(dataset))
            self.assertEqual(sum(overview.door_state_frequencies.values()), len(dataset))
            self.assertEqual(sum(overview.exit_state_frequencies.values()), len(dataset))


class KPITests(unittest.TestCase):

    def test_kpi_percentages_are_within_bounds(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=8, master_seed=4)

            dataset = analyze_campaign(output_dir).dataset
            kpis = compute_kpis(list(dataset))

            for value in (
                kpis.exit_utilization_percentage,
                kpis.stair_utilization_percentage,
                kpis.detector_activation_coverage,
                kpis.camera_coverage,
                kpis.evacuation_success_rate,
                kpis.unreachable_occupant_rate,
            ):
                if value is not None:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 100.0)

    def test_unreachable_occupant_rate_matches_manual_ratio(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=6, master_seed=5)

            samples = list(analyze_campaign(output_dir).dataset)

            total_occupants = sum(s.scenario_features["total_occupants"] for s in samples)
            total_unreachable = sum(s.simulation_outcome["unreachable_occupants"] for s in samples)

            kpis = compute_kpis(samples)

            if total_occupants:
                self.assertAlmostEqual(
                    kpis.unreachable_occupant_rate, total_unreachable / total_occupants * 100,
                )
            else:
                self.assertIsNone(kpis.unreachable_occupant_rate)

    def test_average_rset_equals_mean_evacuation_time(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=6, master_seed=6)

            samples = list(analyze_campaign(output_dir).dataset)
            finite_times = [
                s.simulation_outcome["total_evacuation_time"]
                for s in samples
                if isinstance(s.simulation_outcome.get("total_evacuation_time"), (int, float))
            ]

            kpis = compute_kpis(samples)

            if finite_times:
                self.assertAlmostEqual(kpis.average_rset, sum(finite_times) / len(finite_times))
            else:
                self.assertIsNone(kpis.average_rset)


class AnalyzeCampaignOrchestrationTests(unittest.TestCase):

    def test_returns_a_fully_populated_campaign_analysis(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=4, master_seed=7)

            analysis = analyze_campaign(output_dir)

            self.assertIsInstance(analysis, CampaignAnalysis)
            self.assertEqual(analysis.campaign_dir, output_dir)
            self.assertIsInstance(analysis.engineering_findings, tuple)
            self.assertIsInstance(analysis.anomaly_findings, tuple)
            self.assertEqual(
                set(analysis.findings), set(analysis.engineering_findings) | set(analysis.anomaly_findings),
            )

    def test_to_dict_excludes_dataset_and_is_json_serializable(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=3, master_seed=8)

            analysis = analyze_campaign(output_dir)
            data = analysis.to_dict()

            self.assertNotIn("dataset", data)
            json.dumps(data)  # must not raise

    def test_strict_true_raises_on_a_corrupted_scenario(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=2, master_seed=9)

            scenario_id = discover_scenario_ids(output_dir)[0]
            (Path(output_dir) / "ground_truth" / scenario_id / "ground_truth.json").unlink()

            with self.assertRaises(SampleLoadError):
                analyze_campaign(output_dir, strict=True)

    def test_default_is_lenient_and_skips_a_corrupted_scenario(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=3, master_seed=9)

            scenario_id = discover_scenario_ids(output_dir)[0]
            (Path(output_dir) / "ground_truth" / scenario_id / "ground_truth.json").unlink()

            analysis = analyze_campaign(output_dir)

            self.assertEqual(analysis.overview.total_scenarios, 2)

    def test_repeated_analysis_of_the_same_campaign_is_deterministic(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=6, master_seed=10)

            first = analyze_campaign(output_dir)
            second = analyze_campaign(output_dir)

            self.assertEqual(first.overview.to_dict(), second.overview.to_dict())
            self.assertEqual(first.kpis.to_dict(), second.kpis.to_dict())
            self.assertEqual(first.distributions, second.distributions)
            self.assertEqual(
                [f.to_dict() for f in first.findings], [f.to_dict() for f in second.findings],
            )


class FindingTests(unittest.TestCase):

    def test_to_dict_round_trips_all_fields(self):

        finding = Finding("warning", "some_code", "some message", scenario_id="scn-1")

        self.assertEqual(
            finding.to_dict(),
            {
                "severity": "warning",
                "code": "some_code",
                "message": "some message",
                "scenario_id": "scn-1",
            },
        )


if __name__ == "__main__":
    unittest.main()
