import unittest

from calibration_benchmark import WalkingSpeedCandidate

from calibration_studio.project import CalibrationProject, ProjectNotActiveError, ProjectStatus


class CalibrationProjectIdentityTests(unittest.TestCase):

    def test_project_id_is_generated_and_nonempty(self):

        self.assertTrue(CalibrationProject(name="P").project_id)

    def test_two_projects_get_distinct_ids(self):

        self.assertNotEqual(
            CalibrationProject(name="A").project_id,
            CalibrationProject(name="B").project_id,
        )

    def test_created_at_is_set(self):

        self.assertTrue(CalibrationProject(name="P").created_at)


class CalibrationProjectMetadataTests(unittest.TestCase):

    def test_defaults(self):

        project = CalibrationProject(name="Q3 Stair Recalibration")

        self.assertEqual(project.name, "Q3 Stair Recalibration")
        self.assertEqual(project.description, "")
        self.assertEqual(project.status, ProjectStatus.ACTIVE)
        self.assertEqual(project.tags, ())
        self.assertEqual(project.version, 1)

    def test_constructor_accepts_description_and_tags(self):

        project = CalibrationProject(name="P", description="desc", tags=("stair", "nist"))

        self.assertEqual(project.description, "desc")
        self.assertEqual(project.tags, ("stair", "nist"))

    def test_rename_updates_name_and_bumps_version(self):

        project = CalibrationProject(name="Old")
        version_before = project.version

        project.rename("New")

        self.assertEqual(project.name, "New")
        self.assertGreater(project.version, version_before)

    def test_set_description_bumps_version(self):

        project = CalibrationProject(name="P")
        version_before = project.version

        project.set_description("updated")

        self.assertEqual(project.description, "updated")
        self.assertGreater(project.version, version_before)

    def test_rename_bumps_updated_at(self):

        project = CalibrationProject(name="Old")
        updated_before = project.updated_at

        project.rename("New")

        self.assertGreaterEqual(project.updated_at, updated_before)

    def test_add_tag_is_idempotent(self):

        project = CalibrationProject(name="P")

        project.add_tag("stair")
        version_after_first_add = project.version
        project.add_tag("stair")

        self.assertEqual(project.tags, ("stair",))
        self.assertEqual(project.version, version_after_first_add)

    def test_remove_tag(self):

        project = CalibrationProject(name="P", tags=("stair", "nist"))

        project.remove_tag("stair")

        self.assertEqual(project.tags, ("nist",))

    def test_set_status_bumps_version(self):

        project = CalibrationProject(name="P")
        version_before = project.version

        project.set_status(ProjectStatus.PAUSED)

        self.assertEqual(project.status, ProjectStatus.PAUSED)
        self.assertGreater(project.version, version_before)


class CalibrationProjectBenchmarkReferenceTests(unittest.TestCase):

    def test_add_benchmark_id_is_append_only_and_idempotent(self):

        project = CalibrationProject(name="P")

        project.add_benchmark_id("nist-10story")
        project.add_benchmark_id("nist-18story")
        project.add_benchmark_id("nist-10story")

        self.assertEqual(project.benchmark_ids, ("nist-10story", "nist-18story"))


class CalibrationProjectSessionLifecycleTests(unittest.TestCase):

    def test_create_session_returns_a_session_bound_to_this_project(self):

        project = CalibrationProject(name="P")

        session = project.create_session(benchmark_id="nist-10story")

        self.assertEqual(session.project_id, project.project_id)
        self.assertEqual(session.benchmark_id, "nist-10story")

    def test_create_session_appends_to_sessions_and_session_ids(self):

        project = CalibrationProject(name="P")

        session = project.create_session()

        self.assertIn(session, project.sessions)
        self.assertIn(session.session_id, project.session_ids)

    def test_create_session_bumps_version(self):

        project = CalibrationProject(name="P")
        version_before = project.version

        project.create_session()

        self.assertGreater(project.version, version_before)

    def test_create_session_forwards_candidate(self):

        project = CalibrationProject(name="P")
        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")

        session = project.create_session(candidate=candidate)

        self.assertIs(session.candidate, candidate)

    def test_create_session_forwards_master_seed(self):

        project = CalibrationProject(name="P")

        session = project.create_session(master_seed=90210)

        self.assertEqual(session.master_seed, 90210)

    def test_create_session_on_a_paused_project_raises(self):

        project = CalibrationProject(name="P")
        project.set_status(ProjectStatus.PAUSED)

        with self.assertRaises(ProjectNotActiveError):
            project.create_session()

    def test_create_session_on_a_closed_project_raises(self):

        project = CalibrationProject(name="P")
        project.set_status(ProjectStatus.CLOSED)

        with self.assertRaises(ProjectNotActiveError):
            project.create_session()

    def test_create_session_on_an_archived_project_raises(self):

        project = CalibrationProject(name="P")
        project.set_status(ProjectStatus.ARCHIVED)

        with self.assertRaises(ProjectNotActiveError):
            project.create_session()

    def test_get_session_finds_an_existing_session(self):

        project = CalibrationProject(name="P")
        session = project.create_session()

        self.assertIs(project.get_session(session.session_id), session)

    def test_get_session_returns_none_for_unknown_id(self):

        project = CalibrationProject(name="P")

        self.assertIsNone(project.get_session("does-not-exist"))

    def test_multiple_sessions_are_all_tracked(self):

        project = CalibrationProject(name="P")

        session_a = project.create_session(benchmark_id="a")
        session_b = project.create_session(benchmark_id="b")

        self.assertEqual(set(project.session_ids), {session_a.session_id, session_b.session_id})


class CalibrationProjectToDictTests(unittest.TestCase):

    def test_to_dict_reflects_current_state(self):

        project = CalibrationProject(name="P", description="d", tags=("x",))
        project.add_benchmark_id("nist-10story")
        session = project.create_session()

        as_dict = project.to_dict()

        self.assertEqual(as_dict["project_id"], project.project_id)
        self.assertEqual(as_dict["name"], "P")
        self.assertEqual(as_dict["description"], "d")
        self.assertEqual(as_dict["tags"], ["x"])
        self.assertEqual(as_dict["benchmark_ids"], ["nist-10story"])
        self.assertEqual(as_dict["session_ids"], [session.session_id])
        self.assertEqual(as_dict["status"], "ACTIVE")


if __name__ == "__main__":
    unittest.main()
