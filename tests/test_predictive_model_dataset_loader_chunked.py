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
    load_dataset_single_horizon_chunked,
    select_horizon,
)


# =====================================================
# Localized Predictive Model V2 milestone, Phase 2/21 -- the chunked,
# single-horizon loader must produce EXACTLY the same rows (modulo
# dtype) that load_dataset() + select_horizon() already produce for V1,
# regardless of chunksize, and must never silently include rows from
# another horizon or a differently-shaped CSV.
# =====================================================


def _write_report(path, dataset_version_dict):
    with open(path, "w", encoding="utf-8") as report_file:
        json.dump({"dataset_version": dataset_version_dict}, report_file)


def _write_csv(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        csv_file.write(",".join(columns) + "\n")
        for row in rows:
            csv_file.write(",".join(str(row.get(col, "")) for col in columns) + "\n")


class ChunkedDatasetLoaderTests(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.valid_version = dataset_version(20.0, campaign_version="predictive_dataset_campaign_v2").to_dict()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _sample_row(self, scenario_id="scn-1", horizon=20.0, target="False", candidate_id="door-1"):
        row = {name: "" for name in CSV_COLUMNS}
        row.update({
            "scenario_id": scenario_id,
            "observation_time": 5.0,
            "candidate_id": candidate_id,
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

    def _requirement(self):
        return DatasetRequirement(campaign_version="predictive_dataset_campaign_v2")

    def test_chunked_loader_matches_select_horizon_row_count(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        csv_path = os.path.join(self.tmp_dir, "data.csv")
        rows = [self._sample_row(horizon=h, scenario_id=f"scn-{i}") for i, h in
                enumerate([10.0, 20.0, 20.0, 30.0, 60.0, 20.0])]
        _write_csv(csv_path, CSV_COLUMNS, rows)

        whole = load_dataset(csv_path, report_path, requirement=self._requirement())
        via_select = select_horizon(whole, 20.0)

        chunked = load_dataset_single_horizon_chunked(
            csv_path, report_path, 20.0, requirement=self._requirement(), chunksize=2,
        )

        self.assertEqual(len(chunked.frame), len(via_select))
        self.assertEqual(len(chunked.frame), 3)
        self.assertTrue((chunked.frame["prediction_horizon"] == 20.0).all())

    def test_chunk_size_does_not_change_result(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        csv_path = os.path.join(self.tmp_dir, "data.csv")
        rows = [self._sample_row(horizon=20.0 if i % 2 == 0 else 10.0, scenario_id=f"scn-{i}") for i in range(20)]
        _write_csv(csv_path, CSV_COLUMNS, rows)

        small_chunks = load_dataset_single_horizon_chunked(
            csv_path, report_path, 20.0, requirement=self._requirement(), chunksize=3,
        )
        big_chunks = load_dataset_single_horizon_chunked(
            csv_path, report_path, 20.0, requirement=self._requirement(), chunksize=1000,
        )

        self.assertEqual(len(small_chunks.frame), len(big_chunks.frame))
        self.assertEqual(
            sorted(small_chunks.frame["scenario_id"].astype(str).tolist()),
            sorted(big_chunks.frame["scenario_id"].astype(str).tolist()),
        )

    def test_rejects_csv_missing_expected_columns(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        csv_path = os.path.join(self.tmp_dir, "data.csv")
        bad_columns = [c for c in CSV_COLUMNS if c != "candidate_queue_length"]
        _write_csv(csv_path, bad_columns, [{}])

        with self.assertRaises(IncompatibleDatasetVersionError):
            load_dataset_single_horizon_chunked(csv_path, report_path, 20.0, requirement=self._requirement())

    def test_rejects_horizon_absent_from_csv(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        csv_path = os.path.join(self.tmp_dir, "data.csv")
        _write_csv(csv_path, CSV_COLUMNS, [self._sample_row(horizon=10.0)])

        with self.assertRaises(IncompatibleDatasetVersionError):
            load_dataset_single_horizon_chunked(csv_path, report_path, 999.0, requirement=self._requirement())

    def test_rejects_incompatible_campaign_version(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        csv_path = os.path.join(self.tmp_dir, "data.csv")
        _write_csv(csv_path, CSV_COLUMNS, [self._sample_row(horizon=20.0)])

        wrong_requirement = DatasetRequirement(campaign_version="predictive_dataset_campaign_v1")
        with self.assertRaises(IncompatibleDatasetVersionError):
            load_dataset_single_horizon_chunked(csv_path, report_path, 20.0, requirement=wrong_requirement)

    def test_result_has_compact_dtypes_and_category_identity_columns(self):

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        csv_path = os.path.join(self.tmp_dir, "data.csv")
        _write_csv(csv_path, CSV_COLUMNS, [self._sample_row(horizon=20.0)])

        loaded = load_dataset_single_horizon_chunked(csv_path, report_path, 20.0, requirement=self._requirement())

        self.assertEqual(str(loaded.frame["scenario_id"].dtype), "category")
        self.assertEqual(str(loaded.frame["candidate_id"].dtype), "category")
        self.assertEqual(str(loaded.frame["candidate_type"].dtype), "category")
        self.assertEqual(str(loaded.frame["total_active_occupant_count"].dtype), "int32")
        self.assertEqual(str(loaded.frame["candidate_walking_distance"].dtype), "float32")

    def test_target_missingness_survives_chunking(self):
        """A row with target=='' (None) must still read as NaN/missing after
        chunked loading -- trainable_rows()'s notna() filter depends on
        this, exactly like it does for load_dataset()."""

        report_path = os.path.join(self.tmp_dir, "report.json")
        _write_report(report_path, self.valid_version)

        csv_path = os.path.join(self.tmp_dir, "data.csv")
        rows = [
            self._sample_row(horizon=20.0, scenario_id="scn-a", target=""),
            self._sample_row(horizon=20.0, scenario_id="scn-b", target="True"),
        ]
        _write_csv(csv_path, CSV_COLUMNS, rows)

        loaded = load_dataset_single_horizon_chunked(csv_path, report_path, 20.0, requirement=self._requirement())

        self.assertEqual(loaded.frame["target"].isna().sum(), 1)


if __name__ == "__main__":
    unittest.main()
