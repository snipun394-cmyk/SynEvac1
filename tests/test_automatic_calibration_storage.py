import tempfile
import unittest
from pathlib import Path

from automatic_calibration.budget import AutoCalibrationBudget
from automatic_calibration.run import AutoCalibrationRun, AutoCalibrationRunStatus

import automatic_calibration.storage as storage


class AutomaticCalibrationStorageTestCase(unittest.TestCase):

    def setUp(self):

        self._tmpdir = tempfile.TemporaryDirectory()
        self.storage_root = Path(self._tmpdir.name)

    def tearDown(self):

        self._tmpdir.cleanup()


def _make_run(project_id="proj-1"):

    return AutoCalibrationRun(
        project_id=project_id, objective_description="test objective", objective_direction="minimize",
        strategy_description="test strategy", budget=AutoCalibrationBudget(max_evaluations=5),
        search_space_description=({"name": "x", "bounds": [0.0, 1.0]},),
    )


class SaveLoadRunRoundTripTests(AutomaticCalibrationStorageTestCase):

    def test_save_then_load_restores_the_run(self):

        run = _make_run()
        run.mark_running()
        run.record_evaluation("session-1", 5.0)
        run.mark_completed()

        storage.save_run(run, self.storage_root)
        reloaded = storage.load_run(run.run_id, self.storage_root)

        self.assertEqual(reloaded.run_id, run.run_id)
        self.assertEqual(reloaded.status, AutoCalibrationRunStatus.COMPLETED)
        self.assertEqual(reloaded.session_ids, ("session-1",))
        self.assertEqual(reloaded.best_session_id, "session-1")

    def test_load_unknown_run_id_raises_file_not_found(self):

        with self.assertRaises(FileNotFoundError):
            storage.load_run("does-not-exist", self.storage_root)

    def test_save_run_overwrites_in_place_no_duplicate_catalog_row(self):

        run = _make_run()
        storage.save_run(run, self.storage_root)

        run.mark_running()
        storage.save_run(run, self.storage_root)

        self.assertEqual(len(storage.list_runs(self.storage_root)), 1)

    def test_corrupted_json_file_raises(self):

        run = _make_run()
        storage.save_run(run, self.storage_root)

        from automatic_calibration.paths import run_json_path
        json_path = run_json_path(self.storage_root, run.run_id)
        json_path.write_text("not valid json {{{", encoding="utf-8")

        with self.assertRaises(storage.CorruptedRecordFileError):
            storage.load_run(run.run_id, self.storage_root)

    def test_incompatible_schema_version_raises(self):

        run = _make_run()
        storage.save_run(run, self.storage_root)

        from serialization.json_reader import JsonReader
        from serialization.json_writer import JsonWriter
        from automatic_calibration.paths import run_json_path

        json_path = run_json_path(self.storage_root, run.run_id)
        data = JsonReader.read(str(json_path))
        data["schema_version"] = "automatic_calibration_run/999"
        JsonWriter.write(str(json_path), data)

        with self.assertRaises(storage.IncompatibleSchemaVersionError):
            storage.load_run(run.run_id, self.storage_root)


class ListRunsTests(AutomaticCalibrationStorageTestCase):

    def test_lists_every_saved_run(self):

        run_a = _make_run()
        run_b = _make_run()

        storage.save_run(run_a, self.storage_root)
        storage.save_run(run_b, self.storage_root)

        runs = storage.list_runs(self.storage_root)

        self.assertEqual({r.run_id for r in runs}, {run_a.run_id, run_b.run_id})

    def test_filters_by_project_id(self):

        run_a = _make_run(project_id="proj-a")
        run_b = _make_run(project_id="proj-b")

        storage.save_run(run_a, self.storage_root)
        storage.save_run(run_b, self.storage_root)

        runs = storage.list_runs(self.storage_root, project_id="proj-a")

        self.assertEqual({r.run_id for r in runs}, {run_a.run_id})

    def test_empty_storage_root_returns_empty_tuple(self):

        self.assertEqual(storage.list_runs(self.storage_root), ())


if __name__ == "__main__":
    unittest.main()
