import unittest

from calibration_studio.project import (
    SCHEMA_VERSION,
    CalibrationProject,
    CorruptedProjectRecordError,
    ProjectStatus,
)


class CalibrationProjectRoundTripTests(unittest.TestCase):

    def test_active_project_round_trips(self):

        project = CalibrationProject(name="Q3 Stair Recalibration", description="desc", tags=("stair", "nist"))
        project.add_benchmark_id("nist-10story")
        project.add_benchmark_id("nist-18story")
        session = project.create_session(benchmark_id="nist-10story")

        restored = CalibrationProject.from_dict(project.to_dict())

        self.assertEqual(restored.project_id, project.project_id)
        self.assertEqual(restored.name, "Q3 Stair Recalibration")
        self.assertEqual(restored.description, "desc")
        self.assertEqual(restored.tags, ("stair", "nist"))
        self.assertEqual(restored.status, ProjectStatus.ACTIVE)
        self.assertEqual(restored.created_at, project.created_at)
        self.assertEqual(restored.updated_at, project.updated_at)
        self.assertEqual(restored.version, project.version)
        self.assertEqual(restored.benchmark_ids, ("nist-10story", "nist-18story"))
        self.assertEqual(restored.session_ids, (session.session_id,))

    def test_from_dict_alone_does_not_resolve_session_objects(self):

        # Deliberate, documented limitation of the bare model-level
        # from_dict(): session_ids is restored, but the actual
        # CalibrationSession objects require the persistence layer
        # (calibration_studio/storage.py's own load_project(), which
        # calls _attach_loaded_sessions() after reading each session's
        # own file) -- from_dict() alone has no storage_root to resolve
        # them from.
        project = CalibrationProject(name="P")
        project.create_session()

        restored = CalibrationProject.from_dict(project.to_dict())

        self.assertEqual(len(restored.session_ids), 1)
        self.assertEqual(restored.sessions, ())

    def test_non_active_status_round_trips(self):

        project = CalibrationProject(name="P")
        project.set_status(ProjectStatus.ARCHIVED)

        restored = CalibrationProject.from_dict(project.to_dict())

        self.assertEqual(restored.status, ProjectStatus.ARCHIVED)

    def test_extra_round_trips(self):

        project = CalibrationProject(name="P", extra={"owner": "engineer-1"})

        restored = CalibrationProject.from_dict(project.to_dict())

        self.assertEqual(restored.extra, {"owner": "engineer-1"})


class CalibrationProjectForwardCompatibilityTests(unittest.TestCase):

    def test_missing_optional_fields_use_sensible_defaults(self):

        minimal = {"schema_version": SCHEMA_VERSION, "project_id": "proj-minimal", "name": "P"}

        restored = CalibrationProject.from_dict(minimal)

        self.assertEqual(restored.project_id, "proj-minimal")
        self.assertEqual(restored.description, "")
        self.assertEqual(restored.tags, ())
        self.assertEqual(restored.benchmark_ids, ())
        self.assertEqual(restored.session_ids, ())
        self.assertEqual(restored.status, ProjectStatus.ACTIVE)
        self.assertEqual(restored.version, 1)

    def test_unknown_top_level_field_is_preserved_in_extra_not_dropped(self):

        project = CalibrationProject(name="P")
        data = project.to_dict()
        data["a_future_field_this_version_has_never_heard_of"] = ["x", "y"]

        restored = CalibrationProject.from_dict(data)

        self.assertEqual(
            restored.extra["a_future_field_this_version_has_never_heard_of"], ["x", "y"],
        )

    def test_unrecognised_status_raises_corrupted_project_record_error(self):

        project = CalibrationProject(name="P")
        data = project.to_dict()
        data["status"] = "NOT_A_REAL_STATUS"

        with self.assertRaises(CorruptedProjectRecordError):
            CalibrationProject.from_dict(data)


if __name__ == "__main__":
    unittest.main()
