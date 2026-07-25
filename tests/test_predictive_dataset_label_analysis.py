import unittest

from predictive_dataset.label_analysis import (
    occupancy_bucket, severity_bucket, label_bias_report, temporal_coverage_report,
)


class BucketBoundaryTests(unittest.TestCase):

    def test_occupancy_boundaries(self):

        self.assertEqual(occupancy_bucket(10), "LOW")
        self.assertEqual(occupancy_bucket(11), "MEDIUM")
        self.assertEqual(occupancy_bucket(19), "MEDIUM")
        self.assertEqual(occupancy_bucket(20), "HIGH")

    def test_severity_boundaries(self):

        self.assertEqual(severity_bucket(150.0), "FAST_GROWTH_MORE_SEVERE")
        self.assertEqual(severity_bucket(151.0), "MODERATE_GROWTH")
        self.assertEqual(severity_bucket(299.0), "MODERATE_GROWTH")
        self.assertEqual(severity_bucket(300.0), "SLOW_GROWTH_LESS_SEVERE")
        self.assertEqual(severity_bucket(None), "UNKNOWN")


def make_row(scenario_id, observation_time, target):

    return {"scenario_id": scenario_id, "observation_time": observation_time, "target": target}


class LabelBiasReportTests(unittest.TestCase):

    def test_rows_bucketed_by_scenario_occupancy(self):

        metadata = [
            {"scenario_id": "low-scn", "total_occupants": 3, "fire_growth_time_seconds": 200.0},
            {"scenario_id": "high-scn", "total_occupants": 30, "fire_growth_time_seconds": 200.0},
        ]
        rows = [
            make_row("low-scn", 0.0, False),
            make_row("high-scn", 0.0, True),
        ]

        report = label_bias_report(rows, metadata)

        self.assertEqual(report["by_building_occupancy_level"]["LOW"]["negative"], 1)
        self.assertEqual(report["by_building_occupancy_level"]["HIGH"]["positive"], 1)

    def test_currently_congested_rows_excluded(self):

        metadata = [{"scenario_id": "scn-1", "total_occupants": 5, "fire_growth_time_seconds": 200.0}]
        rows = [make_row("scn-1", 0.0, None)]

        report = label_bias_report(rows, metadata)

        totals = report["by_building_occupancy_level"]
        self.assertEqual(sum(v["positive"] + v["negative"] for v in totals.values()), 0)


class TemporalCoverageTests(unittest.TestCase):

    def test_phase_boundaries(self):

        metadata = [{"scenario_id": "scn-1", "evacuation_duration": 90.0}]

        rows = [
            make_row("scn-1", 10.0, True),   # 10/90 = 0.11 -> EARLY
            make_row("scn-1", 50.0, False),  # 50/90 = 0.56 -> MID
            make_row("scn-1", 80.0, True),   # 80/90 = 0.89 -> LATE
        ]

        report = temporal_coverage_report(rows, metadata)

        self.assertEqual(report["EARLY"]["positive"], 1)
        self.assertEqual(report["MID"]["negative"], 1)
        self.assertEqual(report["LATE"]["positive"], 1)

    def test_missing_duration_falls_back_to_unknown(self):

        metadata = [{"scenario_id": "scn-1", "evacuation_duration": None}]
        rows = [make_row("scn-1", 10.0, True)]

        report = temporal_coverage_report(rows, metadata)

        self.assertEqual(report["UNKNOWN"]["positive"], 1)


if __name__ == "__main__":
    unittest.main()
