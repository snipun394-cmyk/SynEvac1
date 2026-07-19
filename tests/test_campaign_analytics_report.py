import shutil
import tempfile
import unittest
from pathlib import Path

from campaign_analytics.analyzer import (
    CampaignAnalysis,
    CampaignKPIs,
    CampaignOverview,
    Finding,
    analyze_campaign,
)
from campaign_analytics.report import generate_report, write_report

from tests.training_dataset_fixtures import make_campaign

FIXED_TIMESTAMP = "2026-01-01T00:00:00+00:00"


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="campaign_analytics_report_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


def _empty_overview(total_scenarios: int = 0) -> CampaignOverview:

    return CampaignOverview(
        total_scenarios=total_scenarios, total_requested=None, rejected=None, acceptance_ratio=None,
        average_evacuation_time=None, minimum_evacuation_time=None, maximum_evacuation_time=None,
        evacuation_time_percentiles={"p50": None, "p90": None, "p95": None, "p99": None},
        average_occupants=None, fire_profile_frequencies={}, ignition_zone_frequencies={},
        detector_failure_frequencies={}, camera_failure_frequencies={}, door_state_frequencies={},
        exit_state_frequencies={},
    )


def _empty_kpis() -> CampaignKPIs:

    return CampaignKPIs(
        average_rset=None, peak_evacuation_time=None, average_queue_time=None,
        average_congestion_duration=None, exit_utilization_percentage=None,
        stair_utilization_percentage=None, detector_activation_coverage=None,
        camera_coverage=None, average_hazard_growth_rate=None, evacuation_success_rate=None,
        unreachable_occupant_rate=None,
    )


def _clean_analysis() -> CampaignAnalysis:

    return CampaignAnalysis(
        campaign_dir="/fake/campaign",
        overview=_empty_overview(total_scenarios=1),
        kpis=_empty_kpis(),
        distributions={},
        engineering_findings=(),
        anomaly_findings=(),
        dataset=[],
    )


class ReportSectionTests(unittest.TestCase):

    def test_every_required_section_heading_is_present(self):

        report = generate_report(_clean_analysis(), generated_at=FIXED_TIMESTAMP)

        for heading in (
            "# Campaign Analytics Report",
            "## Campaign Summary",
            "## Dataset Quality",
            "## Engineering Observations",
            "## Potential Biases",
            "## Rare Events",
            "## Recommendations Before AI Training",
        ):
            self.assertIn(heading, report)

    def test_a_clean_analysis_reports_no_issues_in_every_findings_section(self):

        report = generate_report(_clean_analysis(), generated_at=FIXED_TIMESTAMP)

        self.assertIn("No engineering plausibility issues were detected.", report)
        self.assertIn("No coverage or distribution biases were detected.", report)
        self.assertIn("No rare or outlier scenarios were detected.", report)
        self.assertIn("No significant issues were detected", report)

    def test_findings_are_routed_to_the_correct_section(self):

        analysis = CampaignAnalysis(
            campaign_dir="/fake/campaign",
            overview=_empty_overview(),
            kpis=_empty_kpis(),
            distributions={},
            engineering_findings=(
                Finding("warning", "zones_never_on_fire", "zone-9 never caught fire"),
            ),
            anomaly_findings=(
                Finding("warning", "evacuation_time_outlier", "outlier scenario", scenario_id="scn-1"),
            ),
            dataset=[],
        )

        report = generate_report(analysis, generated_at=FIXED_TIMESTAMP)

        lines = report.splitlines()
        biases_section = _section(lines, "## Potential Biases")
        rare_events_section = _section(lines, "## Rare Events")
        engineering_section = _section(lines, "## Engineering Observations")

        self.assertIn("zone-9 never caught fire", engineering_section)
        self.assertIn("zone-9 never caught fire", biases_section)
        self.assertNotIn("zone-9 never caught fire", rare_events_section)

        self.assertIn("outlier scenario", rare_events_section)
        self.assertNotIn("outlier scenario", biases_section)

    def test_recommendations_mention_constant_occupant_count_when_present(self):

        analysis = CampaignAnalysis(
            campaign_dir="/fake/campaign",
            overview=_empty_overview(),
            kpis=_empty_kpis(),
            distributions={},
            engineering_findings=(),
            anomaly_findings=(
                Finding("warning", "constant_occupant_count", "always 5 occupants"),
            ),
            dataset=[],
        )

        report = generate_report(analysis, generated_at=FIXED_TIMESTAMP)
        recommendations_section = _section(report.splitlines(), "## Recommendations Before AI Training")

        self.assertIn("Vary the occupant count distribution", recommendations_section)


def _section(lines, heading):

    start = lines.index(heading)
    end = len(lines)

    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break

    return "\n".join(lines[start:end])


class ReportDeterminismTests(unittest.TestCase):

    def test_same_analysis_produces_byte_identical_report(self):

        first = generate_report(_clean_analysis(), generated_at=FIXED_TIMESTAMP)
        second = generate_report(_clean_analysis(), generated_at=FIXED_TIMESTAMP)

        self.assertEqual(first, second)


class ReportRealCampaignIntegrationTests(unittest.TestCase):

    def test_report_generated_from_a_real_campaign_is_well_formed(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=5, master_seed=31)

            analysis = analyze_campaign(output_dir)
            report = generate_report(analysis, generated_at=FIXED_TIMESTAMP)

            self.assertIn(f"Total scenarios (accepted): {analysis.overview.total_scenarios}", report)
            self.assertIn(output_dir, report)

    def test_write_report_writes_the_same_text_generate_report_returns(self):

        with _TempOutputDir() as output_dir, _TempOutputDir() as report_dir:

            make_campaign(output_dir, count=3, master_seed=32)

            analysis = analyze_campaign(output_dir)
            expected_text = generate_report(analysis, generated_at=FIXED_TIMESTAMP)

            file_path = Path(report_dir) / "report.md"
            returned_path = write_report(analysis, str(file_path), generated_at=FIXED_TIMESTAMP)

            self.assertEqual(returned_path, str(file_path))
            self.assertEqual(file_path.read_text(encoding="utf-8"), expected_text)


if __name__ == "__main__":
    unittest.main()
