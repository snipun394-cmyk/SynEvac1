import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from training_dataset.loader import discover_scenario_ids
from training_dataset.validator import sample_content_hash, validate_campaign

from tests.training_dataset_fixtures import make_campaign


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="training_dataset_validator_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


# =====================================================


class ValidCampaignTests(unittest.TestCase):

    def test_a_freshly_generated_campaign_is_valid(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=3)

            report = validate_campaign(output_dir)

            self.assertTrue(report.is_valid, report.errors)
            self.assertEqual(report.scenarios_discovered, 3)
            self.assertEqual(report.samples_loaded, 3)
            self.assertEqual(report.errors, [])

    def test_report_to_dict_round_trips_counts(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=2)

            report = validate_campaign(output_dir)
            data = report.to_dict()

            self.assertEqual(data["scenarios_discovered"], 2)
            self.assertEqual(data["samples_loaded"], 2)
            self.assertEqual(data["error_count"], len(report.errors))
            self.assertEqual(data["warning_count"], len(report.warnings))


# =====================================================


class MissingFileValidationTests(unittest.TestCase):

    def test_missing_ground_truth_is_reported_as_an_error(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=2)

            scenario_id = discover_scenario_ids(output_dir)[0]
            (Path(output_dir) / "ground_truth" / scenario_id / "ground_truth.json").unlink()

            report = validate_campaign(output_dir)

            self.assertFalse(report.is_valid)
            self.assertTrue(
                any(
                    issue.code == "missing_file" and issue.scenario_id == scenario_id
                    for issue in report.errors
                )
            )
            # The other (untouched) scenario still loads successfully.
            self.assertEqual(report.samples_loaded, 1)


class BrokenJsonValidationTests(unittest.TestCase):

    def test_corrupted_json_is_reported_with_its_own_code(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            path = Path(output_dir) / "decision_policy" / scenario_id / "decision_policy.json"
            path.write_text("{ this is not json", encoding="utf-8")

            report = validate_campaign(output_dir)

            self.assertFalse(report.is_valid)
            self.assertTrue(any(issue.code == "broken_json" for issue in report.errors))


class MismatchedScenarioIdValidationTests(unittest.TestCase):

    def test_a_scenario_id_that_disagrees_with_its_directory_is_reported(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            path = Path(output_dir) / "ground_truth" / scenario_id / "ground_truth.json"

            data = json.loads(path.read_text(encoding="utf-8"))
            data["scenario_id"] = "scn-wrong"
            path.write_text(json.dumps(data), encoding="utf-8")

            report = validate_campaign(output_dir)

            self.assertFalse(report.is_valid)
            self.assertTrue(
                any(issue.code == "scenario_id_mismatch" for issue in report.errors)
            )


class EmptyCsvValidationTests(unittest.TestCase):

    def test_zone_results_with_no_rows_is_a_warning_not_an_error(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            path = Path(output_dir) / "datasets" / scenario_id / "zone_results.csv"

            header = path.read_text(encoding="utf-8").splitlines()[0]
            path.write_text(header + "\n", encoding="utf-8")

            report = validate_campaign(output_dir)

            self.assertTrue(any(issue.code == "empty_zone_results" for issue in report.warnings))


class DuplicateScenarioIdCatalogValidationTests(unittest.TestCase):

    def test_a_duplicated_catalog_row_is_reported_as_an_error(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=2)

            catalog_path = Path(output_dir) / "catalog.csv"

            with open(catalog_path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames

            duplicated_row = dict(rows[0])

            with open(catalog_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writerow(duplicated_row)

            report = validate_campaign(output_dir)

            self.assertFalse(report.is_valid)
            self.assertTrue(
                any(
                    issue.code == "duplicate_scenario_id" and issue.scenario_id == rows[0]["scenario_id"]
                    for issue in report.errors
                )
            )


class DuplicateContentHashValidationTests(unittest.TestCase):

    def test_two_samples_with_identical_content_are_flagged_as_a_warning(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=2)

            scenario_ids = discover_scenario_ids(output_dir)
            first_id, second_id = scenario_ids

            # Force the second scenario's own artifacts to declare the
            # same content as the first (except its own scenario_id/seed,
            # which must remain correct for the loader's own consistency
            # checks) -- proves the duplicate-hash detector actually
            # compares content, not just scenario_id equality.

            first_path = Path(output_dir) / "datasets" / first_id / "scenario_features.csv"
            second_path = Path(output_dir) / "datasets" / second_id / "scenario_features.csv"

            with open(first_path, encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                first_rows = list(reader)
                fieldnames = reader.fieldnames

            merged_row = dict(first_rows[0])
            merged_row["scenario_id"] = second_id

            with open(second_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(merged_row)

            first_outcome_path = Path(output_dir) / "datasets" / first_id / "simulation_outcomes.csv"
            second_outcome_path = Path(output_dir) / "datasets" / second_id / "simulation_outcomes.csv"

            with open(first_outcome_path, encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                first_outcome_rows = list(reader)
                outcome_fieldnames = reader.fieldnames

            merged_outcome_row = dict(first_outcome_rows[0])
            merged_outcome_row["scenario_id"] = second_id

            with open(second_outcome_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=outcome_fieldnames)
                writer.writeheader()
                writer.writerow(merged_outcome_row)

            report = validate_campaign(output_dir)

            duplicate_warnings = [
                issue for issue in report.warnings if issue.code == "duplicate_content_hash"
            ]
            self.assertEqual(len(duplicate_warnings), 1)
            self.assertIn(first_id, duplicate_warnings[0].message)
            self.assertIn(second_id, duplicate_warnings[0].message)


class CorruptedTimelineValidationTests(unittest.TestCase):

    def test_a_timeline_with_decreasing_simulation_time_is_reported(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            path = Path(output_dir) / "timelines" / scenario_id / "timeline.csv"

            with open(path, encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames

            self.assertGreaterEqual(len(rows), 2, "test needs at least 2 timeline rows")

            # Swap the first two rows' simulation_time values so the
            # sequence goes backwards.
            rows[0]["simulation_time"], rows[1]["simulation_time"] = (
                rows[1]["simulation_time"], rows[0]["simulation_time"],
            )

            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            report = validate_campaign(output_dir)

            self.assertFalse(report.is_valid)
            self.assertTrue(
                any(
                    issue.code == "corrupted_timeline" and issue.scenario_id == scenario_id
                    for issue in report.errors
                )
            )


class InvalidValueValidationTests(unittest.TestCase):

    def test_a_negative_people_evacuated_value_is_reported(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            path = Path(output_dir) / "datasets" / scenario_id / "simulation_outcomes.csv"

            with open(path, encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames

            rows[0]["people_evacuated"] = "-1"

            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            report = validate_campaign(output_dir)

            self.assertFalse(report.is_valid)
            self.assertTrue(
                any(
                    issue.code == "invalid_value" and issue.scenario_id == scenario_id
                    for issue in report.errors
                )
            )


class MissingRequiredColumnValidationTests(unittest.TestCase):

    def test_a_missing_required_column_is_reported(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            path = Path(output_dir) / "datasets" / scenario_id / "simulation_outcomes.csv"

            with open(path, encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = [name for name in reader.fieldnames if name != "people_trapped"]

            for row in rows:
                row.pop("people_trapped", None)

            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            report = validate_campaign(output_dir)

            self.assertFalse(report.is_valid)
            self.assertTrue(
                any(issue.code == "missing_required_columns" for issue in report.errors)
            )


class SampleContentHashTests(unittest.TestCase):

    def test_hash_ignores_scenario_id_and_seed(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            from training_dataset.loader import load_sample

            scenario_id = discover_scenario_ids(output_dir)[0]
            sample = load_sample(output_dir, scenario_id)

            import dataclasses

            renamed = dataclasses.replace(
                sample,
                scenario_id="scn-different",
                scenario_features={**sample.scenario_features, "scenario_id": "scn-different", "seed": 999},
                simulation_outcome={**sample.simulation_outcome, "scenario_id": "scn-different"},
            )

            self.assertEqual(sample_content_hash(sample), sample_content_hash(renamed))


if __name__ == "__main__":
    unittest.main()
