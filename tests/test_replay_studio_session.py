import json
import os
import shutil
import tempfile
import unittest

from replay_studio.session import discover_scenario_ids, resolve_scenario_artifacts


class _TempDir:

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="replay_studio_session_test_")
        return self.path

    def __exit__(self, *args):
        shutil.rmtree(self.path, ignore_errors=True)


def _touch(path, content="{}"):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _write_campaign_layout(output_dir, scenario_id, *, with_occupant_routes=True):

    # Mirrors designer/campaign/campaign_worker.py's own
    # _export_scenario_artifacts() directory layout exactly.

    _touch(os.path.join(output_dir, "building.syn"), "{}")
    _touch(os.path.join(output_dir, "datasets", scenario_id, "scenario_features.csv"), "")
    _touch(os.path.join(output_dir, "timelines", scenario_id, "timeline_rows.json"), "[]")
    _touch(os.path.join(output_dir, "ground_truth", scenario_id, "ground_truth.json"), "{}")
    _touch(os.path.join(output_dir, "decision_policy", scenario_id, "decision_policy.json"), "{}")

    if with_occupant_routes:
        _touch(os.path.join(output_dir, "occupant_routes", scenario_id, "occupant_routes.json"), "[]")
        _touch(os.path.join(output_dir, "decision_events", scenario_id, "decision_events.json"), "[]")


class DiscoverScenarioIdsTests(unittest.TestCase):

    def test_discovers_every_scenario_with_a_dataset_directory(self):

        with _TempDir() as output_dir:

            _write_campaign_layout(output_dir, "scn-0001")
            _write_campaign_layout(output_dir, "scn-0002")

            scenario_ids = discover_scenario_ids(output_dir)

        self.assertEqual(scenario_ids, ("scn-0001", "scn-0002"))

    # =====================================================

    def test_missing_datasets_directory_returns_empty(self):

        with _TempDir() as output_dir:
            scenario_ids = discover_scenario_ids(output_dir)

        self.assertEqual(scenario_ids, ())


class ResolveScenarioArtifactsTests(unittest.TestCase):

    def test_resolves_every_present_artifact(self):

        with _TempDir() as output_dir:

            _write_campaign_layout(output_dir, "scn-0001")
            artifacts = resolve_scenario_artifacts(output_dir, "scn-0001")

        self.assertEqual(artifacts["scenario_id"], "scn-0001")
        self.assertEqual(artifacts["scenario_storage_root"], output_dir)
        self.assertTrue(artifacts["project_path"].endswith("building.syn"))
        self.assertTrue(artifacts["ground_truth_path"].endswith("ground_truth.json"))
        self.assertTrue(artifacts["decision_policy_path"].endswith("decision_policy.json"))
        self.assertTrue(artifacts["timeline_rows_path"].endswith("timeline_rows.json"))
        self.assertTrue(artifacts["occupant_routes_path"].endswith("occupant_routes.json"))
        self.assertTrue(artifacts["decision_events_path"].endswith("decision_events.json"))

    # =====================================================

    def test_missing_optional_artifacts_resolve_to_none_not_a_fabricated_path(self):

        with _TempDir() as output_dir:

            _write_campaign_layout(output_dir, "scn-0001", with_occupant_routes=False)
            artifacts = resolve_scenario_artifacts(output_dir, "scn-0001")

        self.assertIsNone(artifacts["occupant_routes_path"])
        self.assertIsNone(artifacts["decision_events_path"])
        self.assertIsNotNone(artifacts["ground_truth_path"])


if __name__ == "__main__":
    unittest.main()
