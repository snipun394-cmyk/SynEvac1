import unittest

from predictive_dataset.analysis import class_balance_report, horizon_analysis, recommend_first_horizon


def make_row(candidate_type, horizon, observation_time, target, scenario_id="scn-1"):

    return {
        "scenario_id": scenario_id, "observation_time": observation_time,
        "candidate_type": candidate_type, "prediction_horizon": horizon, "target": target,
    }


class ClassBalanceReportTests(unittest.TestCase):

    def test_currently_congested_rows_are_excluded_from_trainable_and_positive_counts(self):

        rows = [
            make_row("Exit", 30.0, 5.0, target=True),
            make_row("Exit", 30.0, 5.0, target=None),  # currently congested -- not applicable
            make_row("Exit", 30.0, 5.0, target=False),
        ]

        report = class_balance_report(rows)

        self.assertEqual(report["total_rows"], 3)
        self.assertEqual(report["trainable_rows"], 2)
        self.assertEqual(report["excluded_currently_congested_rows"], 1)
        self.assertEqual(report["positive_count"], 1)
        self.assertEqual(report["negative_count"], 1)

    def test_scenario_count_reported_separately_from_row_count(self):

        rows = [
            make_row("Exit", 30.0, 5.0, target=True, scenario_id="scn-1"),
            make_row("Exit", 30.0, 5.0, target=True, scenario_id="scn-1"),
            make_row("Exit", 30.0, 5.0, target=False, scenario_id="scn-2"),
        ]

        report = class_balance_report(rows)

        self.assertEqual(report["total_rows"], 3)
        self.assertEqual(report["distinct_scenario_count"], 2)

    def test_breakdown_by_type_and_horizon(self):

        rows = [
            make_row("Door", 10.0, 0.0, target=True),
            make_row("Stair", 60.0, 0.0, target=False),
        ]

        report = class_balance_report(rows)

        self.assertEqual(report["by_candidate_type"]["Door"]["positive"], 1)
        self.assertEqual(report["by_horizon"][60.0]["negative"], 1)


class HorizonAnalysisTests(unittest.TestCase):

    def test_reports_per_horizon_stats_independently(self):

        rows = [
            make_row("Exit", 10.0, 0.0, target=True),
            make_row("Exit", 10.0, 0.0, target=False),
            make_row("Exit", 60.0, 0.0, target=True),
        ]

        report = horizon_analysis(rows)

        self.assertEqual(report[10.0]["trainable_rows"], 2)
        self.assertEqual(report[10.0]["positive_count"], 1)
        self.assertEqual(report[60.0]["trainable_rows"], 1)
        self.assertEqual(report[60.0]["positive_count"], 1)

    def test_already_congested_rows_counted_separately_per_horizon(self):

        rows = [
            make_row("Exit", 30.0, 0.0, target=None),
            make_row("Exit", 30.0, 0.0, target=True),
        ]

        report = horizon_analysis(rows)

        self.assertEqual(report[30.0]["already_congested_at_observation_rows"], 1)
        self.assertEqual(report[30.0]["trainable_rows"], 1)


class RecommendFirstHorizonTests(unittest.TestCase):

    def test_prefers_shortest_horizon_that_clears_both_advance_warning_and_statistical_floors(self):

        report = {
            10.0: {"positive_count": 500, "positive_rate": 0.5},  # plenty of data, but too short for genuine advance warning
            20.0: {"positive_count": 50, "positive_rate": 0.1},
            30.0: {"positive_count": 100, "positive_rate": 0.2},
        }

        self.assertEqual(recommend_first_horizon(report), 20.0)

    def test_skips_horizons_with_too_few_positives_even_if_long_enough(self):

        report = {
            20.0: {"positive_count": 1, "positive_rate": 0.5},  # too few positives despite high rate
            60.0: {"positive_count": 100, "positive_rate": 0.2},
        }

        self.assertEqual(recommend_first_horizon(report), 60.0)

    def test_falls_back_to_most_positives_when_nothing_clears_the_bar(self):

        report = {
            10.0: {"positive_count": 1, "positive_rate": 0.01},
            20.0: {"positive_count": 3, "positive_rate": 0.01},
        }

        self.assertEqual(recommend_first_horizon(report), 20.0)


if __name__ == "__main__":
    unittest.main()
