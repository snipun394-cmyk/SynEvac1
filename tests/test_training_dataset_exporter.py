import csv
import json
import shutil
import tempfile
import unittest
from unittest import mock

from training_dataset.loader import load_campaign
from training_dataset.splitter import split_dataset
from training_dataset.exporter import (
    ParquetUnavailableError,
    export_combined_csv,
    export_combined_parquet,
    export_dataset,
    export_manifest,
)
from training_dataset.manifest import SCHEMA_VERSION

from tests.training_dataset_fixtures import make_campaign


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="training_dataset_exporter_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


class CombinedCsvExportTests(unittest.TestCase):

    def test_combined_csv_has_one_row_per_scenario_with_a_split_column(self):

        with _TempOutputDir() as campaign_dir, _TempOutputDir() as export_dir:

            make_campaign(campaign_dir, count=6, master_seed=13)
            dataset = load_campaign(campaign_dir)
            split = split_dataset(dataset, master_seed=1)

            path = export_combined_csv(dataset, split, export_dir + "/combined.csv")

            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), len(dataset))
            self.assertEqual(
                {row["scenario_id"] for row in rows}, set(dataset.scenario_ids),
            )
            self.assertTrue(all(row["split"] in ("train", "validation", "test") for row in rows))

    def test_combined_csv_prefixes_outcome_columns_and_omits_duplicate_scenario_id(self):

        with _TempOutputDir() as campaign_dir, _TempOutputDir() as export_dir:

            make_campaign(campaign_dir, count=2, master_seed=13)
            dataset = load_campaign(campaign_dir)

            path = export_combined_csv(dataset, None, export_dir + "/combined.csv")

            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                rows = list(reader)

            self.assertEqual(fieldnames.count("scenario_id"), 1)
            self.assertIn("outcome_people_evacuated", fieldnames)
            self.assertNotIn("split", fieldnames)

    def test_combined_csv_without_a_split_has_no_split_column(self):

        with _TempOutputDir() as campaign_dir, _TempOutputDir() as export_dir:

            make_campaign(campaign_dir, count=1, master_seed=13)
            dataset = load_campaign(campaign_dir)

            path = export_combined_csv(dataset, None, export_dir + "/combined.csv")

            with open(path, newline="", encoding="utf-8") as handle:
                fieldnames = csv.DictReader(handle).fieldnames

            self.assertNotIn("split", fieldnames)


class ManifestExportTests(unittest.TestCase):

    def test_manifest_json_carries_required_fields(self):

        with _TempOutputDir() as campaign_dir, _TempOutputDir() as export_dir:

            make_campaign(campaign_dir, count=4, master_seed=13)
            dataset = load_campaign(campaign_dir)
            split = split_dataset(dataset, master_seed=1)

            path = export_manifest(
                dataset, split, export_dir + "/manifest.json",
                campaign_name="Test Campaign", master_seed=1,
            )

            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            self.assertIn("generated_at", manifest)
            self.assertEqual(manifest["campaign"]["name"], "Test Campaign")
            self.assertEqual(manifest["campaign"]["directory"], campaign_dir)
            self.assertEqual(manifest["campaign"]["master_seed"], 1)
            self.assertEqual(manifest["num_samples"], len(dataset))
            self.assertEqual(
                manifest["split_sizes"],
                {
                    "train": len(split.train_ids),
                    "validation": len(split.validation_ids),
                    "test": len(split.test_ids),
                },
            )
            self.assertIsInstance(manifest["feature_list"], list)
            self.assertIsInstance(manifest["label_list"], list)
            self.assertIn("total_occupants", manifest["feature_list"])
            self.assertIn("people_evacuated", manifest["label_list"])
            self.assertNotIn("scenario_id", manifest["feature_list"])
            self.assertNotIn("scenario_id", manifest["label_list"])

    def test_manifest_without_a_split_has_no_split_sizes(self):

        with _TempOutputDir() as campaign_dir, _TempOutputDir() as export_dir:

            make_campaign(campaign_dir, count=1, master_seed=13)
            dataset = load_campaign(campaign_dir)

            path = export_manifest(dataset, None, export_dir + "/manifest.json")

            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertIsNone(manifest["split_sizes"])


class ExportDatasetTests(unittest.TestCase):

    def test_export_dataset_writes_csv_and_manifest_by_default(self):

        with _TempOutputDir() as campaign_dir, _TempOutputDir() as export_dir:

            make_campaign(campaign_dir, count=3, master_seed=13)
            dataset = load_campaign(campaign_dir)
            split = split_dataset(dataset, master_seed=1)

            paths = export_dataset(
                dataset, split, export_dir, campaign_name="C", master_seed=1,
            )

            self.assertEqual(set(paths.keys()), {"combined_csv", "manifest"})

            import os
            self.assertTrue(os.path.isfile(paths["combined_csv"]))
            self.assertTrue(os.path.isfile(paths["manifest"]))

    def test_export_dataset_raises_when_parquet_is_requested_without_pandas(self):

        with _TempOutputDir() as campaign_dir, _TempOutputDir() as export_dir:

            make_campaign(campaign_dir, count=1, master_seed=13)
            dataset = load_campaign(campaign_dir)

            with mock.patch.dict("sys.modules", {"pandas": None}):
                with self.assertRaises(ParquetUnavailableError):
                    export_dataset(dataset, None, export_dir, include_parquet=True)

    def test_export_combined_parquet_raises_a_clear_error_without_pandas(self):

        with _TempOutputDir() as campaign_dir, _TempOutputDir() as export_dir:

            make_campaign(campaign_dir, count=1, master_seed=13)
            dataset = load_campaign(campaign_dir)

            with mock.patch.dict("sys.modules", {"pandas": None}):
                with self.assertRaises(ParquetUnavailableError):
                    export_combined_parquet(dataset, None, export_dir + "/combined.parquet")


if __name__ == "__main__":
    unittest.main()
