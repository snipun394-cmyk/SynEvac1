import json
import tempfile
import unittest
from pathlib import Path

from calibration_benchmark import WalkingSpeedCandidate

import calibration_studio.storage as storage
from calibration_studio.project import CalibrationProject
from calibration_studio.session import CalibrationSession


class CalibrationStudioStorageTestCase(unittest.TestCase):

    def setUp(self):

        self._tmpdir = tempfile.TemporaryDirectory()
        self.storage_root = Path(self._tmpdir.name)

    def tearDown(self):

        self._tmpdir.cleanup()


class SaveLoadProjectRoundTripTests(CalibrationStudioStorageTestCase):

    def test_save_then_load_restores_metadata(self):

        project = CalibrationProject(name="Q3 Stair Recalibration", description="desc", tags=("stair",))
        project.add_benchmark_id("nist-10story")

        storage.save_project(project, self.storage_root)
        reloaded = storage.load_project(project.project_id, self.storage_root)

        self.assertEqual(reloaded.name, "Q3 Stair Recalibration")
        self.assertEqual(reloaded.description, "desc")
        self.assertEqual(reloaded.tags, ("stair",))
        self.assertEqual(reloaded.benchmark_ids, ("nist-10story",))

    def test_load_unknown_project_id_raises_file_not_found(self):

        with self.assertRaises(FileNotFoundError):
            storage.load_project("does-not-exist", self.storage_root)

    def test_save_project_overwrites_in_place_no_duplicate_catalog_row(self):

        project = CalibrationProject(name="Original")
        storage.save_project(project, self.storage_root)

        project.rename("Renamed")
        storage.save_project(project, self.storage_root)

        reloaded = storage.load_project(project.project_id, self.storage_root)
        self.assertEqual(reloaded.name, "Renamed")

        # Exactly one project, not two -- an append-on-every-save
        # catalog (unlike this module's own append-if-new design) would
        # produce a duplicate row and two entries here.
        self.assertEqual(len(storage.list_projects(self.storage_root)), 1)


class ListProjectsTests(CalibrationStudioStorageTestCase):

    def test_lists_every_saved_project(self):

        project_a = CalibrationProject(name="A")
        project_b = CalibrationProject(name="B")

        storage.save_project(project_a, self.storage_root)
        storage.save_project(project_b, self.storage_root)

        names = {p.name for p in storage.list_projects(self.storage_root)}
        self.assertEqual(names, {"A", "B"})

    def test_empty_storage_root_returns_no_projects(self):

        self.assertEqual(storage.list_projects(self.storage_root), ())


class SessionResolutionOnProjectLoadTests(CalibrationStudioStorageTestCase):

    def test_saved_sessions_are_resolved_and_attached_on_project_load(self):

        project = CalibrationProject(name="P")
        session = project.create_session(
            benchmark_id="nist-10story",
            candidate=WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test"),
            master_seed=90210,
        )
        session.mark_running(n_scenarios_total=5)
        session.update_progress(2)

        storage.save_project(project, self.storage_root)
        storage.save_session(session, self.storage_root)

        reloaded_project = storage.load_project(project.project_id, self.storage_root)

        self.assertEqual(len(reloaded_project.sessions), 1)
        reloaded_session = reloaded_project.sessions[0]
        self.assertEqual(reloaded_session.session_id, session.session_id)
        self.assertEqual(reloaded_session.status.value, "RUNNING")
        self.assertAlmostEqual(reloaded_session.progress, 0.4)
        self.assertEqual(reloaded_session.master_seed, 90210)
        self.assertEqual(reloaded_session.git_commit_hash, session.git_commit_hash)
        self.assertEqual(reloaded_session.candidate_snapshot, session.candidate_snapshot)

    def test_project_referencing_an_unsaved_session_still_loads(self):

        # save_project()/save_session() are separate, explicit calls by
        # design (this milestone's own requirement list treats them as
        # distinct capabilities) -- a project can legitimately reference
        # a session id whose own file was never (or not yet) saved.
        project = CalibrationProject(name="P")
        project.create_session()  # never separately saved

        storage.save_project(project, self.storage_root)

        reloaded = storage.load_project(project.project_id, self.storage_root)

        self.assertEqual(len(reloaded.session_ids), 1)
        self.assertEqual(reloaded.sessions, ())

    def test_resolve_sessions_false_skips_session_loading(self):

        project = CalibrationProject(name="P")
        session = project.create_session()

        storage.save_project(project, self.storage_root)
        storage.save_session(session, self.storage_root)

        reloaded = storage.load_project(project.project_id, self.storage_root, resolve_sessions=False)

        self.assertEqual(reloaded.session_ids, (session.session_id,))
        self.assertEqual(reloaded.sessions, ())


class SaveLoadSessionTests(CalibrationStudioStorageTestCase):

    def test_save_then_load_restores_session(self):

        session = CalibrationSession(benchmark_id="nist-10story", master_seed=1)
        session.mark_running()
        session.mark_failed("boom")

        storage.save_session(session, self.storage_root)
        reloaded = storage.load_session(session.session_id, self.storage_root)

        self.assertEqual(reloaded.status.value, "FAILED")
        self.assertEqual(reloaded.failure_reason, "boom")

    def test_load_unknown_session_id_raises_file_not_found(self):

        with self.assertRaises(FileNotFoundError):
            storage.load_session("does-not-exist", self.storage_root)


class ListSessionsTests(CalibrationStudioStorageTestCase):

    def test_lists_every_saved_session(self):

        session_a = CalibrationSession(project_id="proj-1")
        session_b = CalibrationSession(project_id="proj-2")

        storage.save_session(session_a, self.storage_root)
        storage.save_session(session_b, self.storage_root)

        ids = {s.session_id for s in storage.list_sessions(self.storage_root)}
        self.assertEqual(ids, {session_a.session_id, session_b.session_id})

    def test_list_sessions_filters_by_project_id(self):

        session_a = CalibrationSession(project_id="proj-1")
        session_b = CalibrationSession(project_id="proj-2")

        storage.save_session(session_a, self.storage_root)
        storage.save_session(session_b, self.storage_root)

        filtered = storage.list_sessions(self.storage_root, project_id="proj-1")

        self.assertEqual([s.session_id for s in filtered], [session_a.session_id])


class CorruptionHandlingTests(CalibrationStudioStorageTestCase):

    def test_garbled_project_file_raises_corrupted_record_file_error(self):

        project = CalibrationProject(name="P")
        json_path = storage.save_project(project, self.storage_root)

        json_path.write_text("{ this is not valid json ]", encoding="utf-8")

        with self.assertRaises(storage.CorruptedRecordFileError):
            storage.load_project(project.project_id, self.storage_root)

    def test_garbled_session_file_raises_corrupted_record_file_error(self):

        session = CalibrationSession()
        json_path = storage.save_session(session, self.storage_root)

        json_path.write_text("not json at all {{{", encoding="utf-8")

        with self.assertRaises(storage.CorruptedRecordFileError):
            storage.load_session(session.session_id, self.storage_root)


class VersionMismatchTests(CalibrationStudioStorageTestCase):

    def test_incompatible_schema_version_raises_on_project_load(self):

        project = CalibrationProject(name="P")
        json_path = storage.save_project(project, self.storage_root)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        data["schema_version"] = "calibration_studio_project/99"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(storage.IncompatibleSchemaVersionError):
            storage.load_project(project.project_id, self.storage_root)

    def test_missing_schema_version_key_raises_incompatible_schema_version_error(self):

        project = CalibrationProject(name="P")
        json_path = storage.save_project(project, self.storage_root)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        del data["schema_version"]
        json_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(storage.IncompatibleSchemaVersionError):
            storage.load_project(project.project_id, self.storage_root)

    def test_incompatible_schema_version_raises_on_session_load(self):

        session = CalibrationSession()
        json_path = storage.save_session(session, self.storage_root)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        data["schema_version"] = "calibration_studio_session/99"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(storage.IncompatibleSchemaVersionError):
            storage.load_session(session.session_id, self.storage_root)


class MissingAndUnknownFieldsOnDiskTests(CalibrationStudioStorageTestCase):

    def test_project_file_missing_a_non_identity_field_still_loads(self):

        project = CalibrationProject(name="P", description="will be removed", tags=("x",))
        json_path = storage.save_project(project, self.storage_root)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        del data["description"]
        del data["tags"]
        json_path.write_text(json.dumps(data), encoding="utf-8")

        reloaded = storage.load_project(project.project_id, self.storage_root)

        self.assertEqual(reloaded.description, "")
        self.assertEqual(reloaded.tags, ())
        self.assertEqual(reloaded.name, "P")

    def test_project_file_with_an_unknown_future_field_still_loads(self):

        project = CalibrationProject(name="P")
        json_path = storage.save_project(project, self.storage_root)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        data["a_field_from_a_future_calibration_studio_version"] = {"whatever": True}
        json_path.write_text(json.dumps(data), encoding="utf-8")

        reloaded = storage.load_project(project.project_id, self.storage_root)

        self.assertEqual(
            reloaded.extra["a_field_from_a_future_calibration_studio_version"], {"whatever": True},
        )
        # And it survives a further save/load cycle -- not lost the
        # moment this version touches the file again.
        storage.save_project(reloaded, self.storage_root)
        reloaded_again = storage.load_project(project.project_id, self.storage_root)
        self.assertIn("a_field_from_a_future_calibration_studio_version", reloaded_again.extra)


class FullVerifyScenarioTests(CalibrationStudioStorageTestCase):

    def test_create_save_reopen_sessions_metadata_benchmarks_git_provenance_all_restored(self):

        # This milestone's own VERIFY section, as one end-to-end test.
        project = CalibrationProject(name="Q3 Stair Recalibration", description="quarterly re-validation")
        project.add_benchmark_id("nist-10story")
        project.add_benchmark_id("nist-18story")

        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "Julich test", "test rationale")
        session = project.create_session(benchmark_id="nist-10story", candidate=candidate, master_seed=90210)
        session.mark_running(n_scenarios_total=10)
        session.update_progress(4)

        original_commit_hash = session.git_commit_hash
        original_dirty = session.git_dirty

        storage.save_project(project, self.storage_root)
        storage.save_session(session, self.storage_root)

        reopened = storage.load_project(project.project_id, self.storage_root)

        # metadata restored
        self.assertEqual(reopened.name, "Q3 Stair Recalibration")
        self.assertEqual(reopened.description, "quarterly re-validation")

        # benchmark references restored
        self.assertEqual(reopened.benchmark_ids, ("nist-10story", "nist-18story"))

        # sessions restored
        self.assertEqual(len(reopened.sessions), 1)
        reopened_session = reopened.sessions[0]
        self.assertEqual(reopened_session.session_id, session.session_id)
        self.assertEqual(reopened_session.status.value, "RUNNING")
        self.assertAlmostEqual(reopened_session.progress, 0.4)

        # git provenance restored (not re-captured against whatever the
        # repo's current state happens to be at load time)
        self.assertEqual(reopened_session.git_commit_hash, original_commit_hash)
        self.assertEqual(reopened_session.git_dirty, original_dirty)


if __name__ == "__main__":
    unittest.main()
