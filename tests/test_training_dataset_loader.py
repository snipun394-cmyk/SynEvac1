import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from scenario_storage import read_catalog_rows

from training_dataset.loader import (
    SampleLoadError,
    SimulationSample,
    discover_scenario_ids,
    load_campaign,
    load_sample,
    scenario_artifact_paths,
)

from tests.training_dataset_fixtures import make_campaign


class _TempOutputDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="training_dataset_loader_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


# =====================================================
# Complete campaigns.
# =====================================================


class CompleteCampaignLoadingTests(unittest.TestCase):

    def test_discover_scenario_ids_matches_catalog(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=3)

            discovered = discover_scenario_ids(output_dir)
            catalog_ids = sorted(row["scenario_id"] for row in read_catalog_rows(output_dir))

            self.assertEqual(discovered, catalog_ids)

    def test_load_campaign_loads_one_sample_per_accepted_scenario(self):

        with _TempOutputDir() as output_dir:

            summary = make_campaign(output_dir, count=3)

            dataset = load_campaign(output_dir)

            self.assertEqual(len(dataset), summary.accepted)
            self.assertEqual(dataset.errors, [])
            self.assertEqual(sorted(dataset.scenario_ids), discover_scenario_ids(output_dir))

    def test_each_sample_carries_every_artifact_with_a_consistent_scenario_id(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=2)

            dataset = load_campaign(output_dir)

            for sample in dataset:

                self.assertIsInstance(sample, SimulationSample)

                self.assertEqual(sample.scenario_features["scenario_id"], sample.scenario_id)
                self.assertEqual(sample.simulation_outcome["scenario_id"], sample.scenario_id)

                self.assertTrue(sample.zone_results)
                self.assertTrue(all(
                    row["scenario_id"] == sample.scenario_id for row in sample.zone_results
                ))

                self.assertTrue(sample.timeline)
                self.assertTrue(all(
                    row["scenario_id"] == sample.scenario_id for row in sample.timeline
                ))

                self.assertEqual(sample.ground_truth["scenario_id"], sample.scenario_id)
                self.assertEqual(sample.decision_policy["scenario_id"], sample.scenario_id)

    def test_csv_values_are_type_coerced(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            dataset = load_campaign(output_dir)
            sample = dataset[0]

            self.assertIsInstance(sample.scenario_features["total_occupants"], int)
            self.assertIsInstance(sample.simulation_outcome["people_evacuated"], int)
            self.assertIsInstance(sample.simulation_outcome["building_cleared"], bool)

    def test_load_sample_matches_load_campaign_for_a_single_scenario(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            sample = load_sample(output_dir, scenario_id)

            self.assertEqual(sample.scenario_id, scenario_id)

    def test_scenario_artifact_paths_point_at_real_files(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            paths = scenario_artifact_paths(output_dir, scenario_id)

            for path in paths.values():
                self.assertTrue(os.path.isfile(path), path)

    def test_empty_campaign_directory_discovers_nothing(self):

        with _TempOutputDir() as output_dir:

            dataset = load_campaign(output_dir)

            self.assertEqual(len(dataset), 0)
            self.assertEqual(discover_scenario_ids(output_dir), [])


# =====================================================
# Partial campaigns -- some scenarios are missing one or more
# artifacts (e.g. a cancelled or still-running campaign).
# =====================================================


class PartialCampaignLoadingTests(unittest.TestCase):

    def test_strict_load_raises_when_an_artifact_is_missing(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=2)

            scenario_id = discover_scenario_ids(output_dir)[0]
            os.remove(Path(output_dir) / "ground_truth" / scenario_id / "ground_truth.json")

            with self.assertRaises(SampleLoadError):
                load_campaign(output_dir, strict=True)

    def test_non_strict_load_skips_broken_scenarios_and_keeps_the_rest(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=3)

            scenario_ids = discover_scenario_ids(output_dir)
            broken_id = scenario_ids[0]

            os.remove(Path(output_dir) / "ground_truth" / broken_id / "ground_truth.json")

            dataset = load_campaign(output_dir, strict=False)

            self.assertEqual(len(dataset), 2)
            self.assertNotIn(broken_id, dataset.scenario_ids)

            self.assertEqual(len(dataset.errors), 1)
            self.assertEqual(dataset.errors[0].scenario_id, broken_id)

    def test_optional_artifacts_can_be_marked_not_required(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            os.remove(Path(output_dir) / "ground_truth" / scenario_id / "ground_truth.json")

            sample = load_sample(output_dir, scenario_id, require_ground_truth=False)

            self.assertIsNone(sample.ground_truth)
            self.assertIsNotNone(sample.decision_policy)


# =====================================================
# Corrupted datasets.
# =====================================================


class CorruptedDatasetLoadingTests(unittest.TestCase):

    def test_broken_json_raises_sample_load_error(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            path = Path(output_dir) / "ground_truth" / scenario_id / "ground_truth.json"
            path.write_text("{not valid json", encoding="utf-8")

            with self.assertRaises(SampleLoadError) as ctx:
                load_sample(output_dir, scenario_id)

            self.assertIn("broken JSON", str(ctx.exception))

    def test_scenario_id_mismatch_raises_sample_load_error(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            path = Path(output_dir) / "ground_truth" / scenario_id / "ground_truth.json"

            data = json.loads(path.read_text(encoding="utf-8"))
            data["scenario_id"] = "scn-not-the-real-one"
            path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaises(SampleLoadError) as ctx:
                load_sample(output_dir, scenario_id)

            self.assertIn("declares scenario_id", str(ctx.exception))

    def test_empty_required_csv_raises_sample_load_error(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            scenario_id = discover_scenario_ids(output_dir)[0]
            path = Path(output_dir) / "datasets" / scenario_id / "scenario_features.csv"
            path.write_text("scenario_id,definition_id\n", encoding="utf-8")

            with self.assertRaises(SampleLoadError) as ctx:
                load_sample(output_dir, scenario_id)

            self.assertIn("must contain exactly 1 row", str(ctx.exception))

    def test_missing_scenario_directory_raises_sample_load_error(self):

        with _TempOutputDir() as output_dir:

            make_campaign(output_dir, count=1)

            with self.assertRaises(SampleLoadError):
                load_sample(output_dir, "scn-does-not-exist")


if __name__ == "__main__":
    unittest.main()
