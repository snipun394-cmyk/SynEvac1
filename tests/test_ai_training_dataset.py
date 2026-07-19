import unittest

from ai_training.dataset import NON_FEATURE_SCENARIO_COLUMNS, ScenarioRecord, load_campaign_dataset

from tests.ai_training_fixtures import RealCampaignTestCase


class LoadCampaignDatasetTests(RealCampaignTestCase):

    def test_dataset_has_one_record_per_accepted_scenario(self):

        self.assertEqual(len(self.dataset), self.campaign_summary.accepted)
        self.assertEqual(len(self.dataset.scenario_ids), self.campaign_summary.accepted)

    def test_records_are_scenario_record_instances(self):

        for record in self.dataset:
            self.assertIsInstance(record, ScenarioRecord)

    def test_indexing_and_iteration_agree(self):

        self.assertEqual(self.dataset[0].scenario_id, list(self.dataset)[0].scenario_id)
        self.assertEqual(len(list(self.dataset)), len(self.dataset))

    def test_no_load_errors_for_a_cleanly_generated_campaign(self):

        self.assertEqual(self.dataset.errors, ())


class ScenarioFeatureRowsTests(RealCampaignTestCase):

    def test_returns_one_row_per_scenario_aligned_with_records(self):

        rows = self.dataset.scenario_feature_rows()

        self.assertEqual(len(rows), len(self.dataset))

        for row, record in zip(rows, self.dataset.records):
            self.assertEqual(row, record.features)

    def test_rows_are_copies_not_the_underlying_dicts(self):

        rows = self.dataset.scenario_feature_rows()
        rows[0]["injected"] = "mutated"

        self.assertNotIn("injected", self.dataset.records[0].features)

    def test_contains_expected_scenario_feature_columns(self):

        row = self.dataset.scenario_feature_rows()[0]

        for column in ("ignition_zone", "ignition_floor", "fire_profile", "growth_time", "total_occupants"):
            self.assertIn(column, row)

        for column in NON_FEATURE_SCENARIO_COLUMNS:
            self.assertIn(column, row)


class OutcomeRowsTests(RealCampaignTestCase):

    def test_returns_one_row_per_scenario_with_expected_columns(self):

        rows = self.dataset.outcome_rows()

        self.assertEqual(len(rows), len(self.dataset))

        for column in ("total_evacuation_time", "people_evacuated", "building_cleared"):
            self.assertIn(column, rows[0])

    def test_rows_are_copies(self):

        rows = self.dataset.outcome_rows()
        rows[0]["injected"] = "mutated"

        self.assertNotIn("injected", self.dataset.records[0].outcome)


class ZoneResultAndTimelineRowsTests(RealCampaignTestCase):

    def test_zone_result_rows_flattens_across_every_scenario(self):

        flattened = self.dataset.zone_result_rows()
        expected = sum(len(record.zone_results) for record in self.dataset.records)

        self.assertEqual(len(flattened), expected)
        self.assertGreater(len(flattened), 0)

        for row in flattened:
            self.assertIn("scenario_id", row)
            self.assertIn("zone_id", row)
            self.assertIn("exit_used", row)

    def test_timeline_rows_flattens_and_carries_scenario_id(self):

        flattened = self.dataset.timeline_rows()
        expected = sum(len(record.timeline) for record in self.dataset.records)

        self.assertEqual(len(flattened), expected)
        self.assertGreater(len(flattened), 0)

        scenario_ids = set(self.dataset.scenario_ids)

        for row in flattened:
            self.assertIn(row["scenario_id"], scenario_ids)
            self.assertIn("simulation_time", row)


class GroundTruthRowsTests(RealCampaignTestCase):

    def test_ground_truth_rows_are_parallel_to_records(self):

        rows = self.dataset.ground_truth_rows()

        self.assertEqual(len(rows), len(self.dataset))

        for row, record in zip(rows, self.dataset.records):

            if record.ground_truth is None:
                self.assertIsNone(row)
            else:
                self.assertEqual(row["scenario_id"], record.scenario_id)

    def test_ground_truth_present_for_every_scenario_by_default(self):

        # load_campaign_dataset() defaults require_ground_truth=True,
        # same as training_dataset.load_campaign() -- every scenario in
        # a cleanly generated campaign must have one.

        for row in self.dataset.ground_truth_rows():
            self.assertIsNotNone(row)


class LoadCampaignDatasetOptionsTests(unittest.TestCase):

    def test_strict_true_is_the_default(self):

        import inspect

        signature = inspect.signature(load_campaign_dataset)
        self.assertTrue(signature.parameters["strict"].default)
        self.assertTrue(signature.parameters["require_ground_truth"].default)


if __name__ == "__main__":
    unittest.main()
