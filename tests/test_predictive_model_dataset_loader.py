import json
import os
import shutil
import tempfile
import unittest

from predictive_dataset.dataset_builder import CSV_COLUMNS
from predictive_dataset.versioning import dataset_version
from predictive_model.dataset_loader import (
    DatasetRequirement,
    IncompatibleDatasetVersionError,
    load_dataset,
    load_dataset_manifest,
    select_horizon,
    assert_compatible,
)


def _write_report(path, dataset_version_dict):
    with open(path, "w", encoding="utf-8") as report_file:
        json.dump({"dataset_version": dataset_version_dict}, report_file)


def _write_csv(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        csv_file.write(",".join(columns) + "\n")
        for row in rows:
            csv_file.write(",".join(str(row.get(col, "")) for col in columns) + "\n")


class DatasetLoaderTests(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.valid_version = dataset_version(20.0).to_dict()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _sample_row(self, horizon=20.0, target="False"):
        row = {name: "" for name in CSV_COLUMNS}
        row.update({
            "scenario_id": "scn-1",
            "observation_time": 5.0,
            "candidate_id": "door-1",
            "candidate_type": "Door",
            "prediction_horizon": horizon,
            "total_active_occupant_count": 10,
            "candidate_capacity": 5,
            "candidate_walking_distance": 10.0,
            "candidate_traversable": "True",
            "candidate_adjacent_zone_occupancy": 3,
            "candidate_queue_length": 0,
            "candidate_approaching_count": 1,
            "candidate_congestion_level": "LOW",
            "currently_congested": "False",
            "had_any_activity_in_window": "True",
            "target": target,
        })
        return row

    def test_load_dataset_manifest_reads_version_fields(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        manifest = load_dataset_manifest(report_path)

        self.assertEqual(manifest.schema_version, self.valid_version["schema_version"])
        self.assertEqual(manifest.campaign_version, self.valid_version["campaign_version"])
        self.assertEqual(manifest.feature_version, self.valid_version["feature_version"])
        self.assertEqual(manifest.target_version, self.valid_version["target_version"])
        self.assertEqual(manifest.recommended_horizon_seconds, 20.0)

    def test_load_dataset_manifest_rejects_report_missing_dataset_version(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump({"something_else": True}, report_file)

        with self.assertRaises(IncompatibleDatasetVersionError):
            load_dataset_manifest(report_path)

    def test_assert_compatible_passes_for_matching_versions(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)
        manifest = load_dataset_manifest(report_path)

        assert_compatible(manifest, DatasetRequirement())  # must not raise

    def test_assert_compatible_rejects_mismatched_schema_version(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        mismatched = dict(self.valid_version)
        mismatched["schema_version"] = "0.9"
        _write_report(report_path, mismatched)
        manifest = load_dataset_manifest(report_path)

        with self.assertRaises(IncompatibleDatasetVersionError):
            assert_compatible(manifest, DatasetRequirement())

    def test_assert_compatible_rejects_mismatched_target_version(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        mismatched = dict(self.valid_version)
        mismatched["target_version"] = "some-other-target-def"
        _write_report(report_path, mismatched)
        manifest = load_dataset_manifest(report_path)

        with self.assertRaises(IncompatibleDatasetVersionError):
            assert_compatible(manifest, DatasetRequirement())

    def test_load_dataset_rejects_csv_missing_expected_columns(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        csv_path = os.path.join(self.tmp_dir, "data.csv")
        bad_columns = [c for c in CSV_COLUMNS if c != "candidate_queue_length"]
        _write_csv(csv_path, bad_columns, [{}])

        with self.assertRaises(IncompatibleDatasetVersionError):
            load_dataset(csv_path, report_path)

    def test_load_dataset_end_to_end_with_synthetic_fixture(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        csv_path = os.path.join(self.tmp_dir, "data.csv")
        _write_csv(csv_path, CSV_COLUMNS, [self._sample_row(horizon=10.0), self._sample_row(horizon=20.0)])

        dataset = load_dataset(csv_path, report_path)

        self.assertEqual(len(dataset.frame), 2)
        self.assertIn(10.0, dataset.available_horizons)
        self.assertIn(20.0, dataset.available_horizons)

    def test_select_horizon_rejects_unavailable_horizon(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        csv_path = os.path.join(self.tmp_dir, "data.csv")
        _write_csv(csv_path, CSV_COLUMNS, [self._sample_row(horizon=20.0)])

        dataset = load_dataset(csv_path, report_path)

        with self.assertRaises(IncompatibleDatasetVersionError):
            select_horizon(dataset, 999.0)

    def test_select_horizon_returns_only_matching_rows(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        csv_path = os.path.join(self.tmp_dir, "data.csv")
        _write_csv(csv_path, CSV_COLUMNS, [self._sample_row(horizon=10.0), self._sample_row(horizon=20.0)])

        dataset = load_dataset(csv_path, report_path)
        subset = select_horizon(dataset, 20.0)

        self.assertEqual(len(subset), 1)
        self.assertEqual(subset.iloc[0]["prediction_horizon"], 20.0)


if __name__ == "__main__":
    unittest.main()
