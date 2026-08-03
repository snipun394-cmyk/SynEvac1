import unittest

from calibration_benchmark import WalkingSpeedCandidate

from calibration_studio.project import CalibrationProject
from calibration_studio.session import CalibrationSession
from calibration_studio.studio import CalibrationStudio


class CalibrationStudioProjectCreationTests(unittest.TestCase):

    def test_create_project_returns_a_project(self):

        studio = CalibrationStudio()

        project = studio.create_project(name="Q3 Stair Recalibration")

        self.assertIsInstance(project, CalibrationProject)
        self.assertEqual(project.name, "Q3 Stair Recalibration")

    def test_create_project_registers_it_for_retrieval(self):

        studio = CalibrationStudio()
        project = studio.create_project(name="P")

        self.assertIs(studio.get_project(project.project_id), project)

    def test_get_project_returns_none_for_unknown_id(self):

        studio = CalibrationStudio()

        self.assertIsNone(studio.get_project("does-not-exist"))

    def test_list_projects_returns_every_created_project(self):

        studio = CalibrationStudio()
        project_a = studio.create_project(name="A")
        project_b = studio.create_project(name="B")

        self.assertEqual(set(studio.list_projects()), {project_a, project_b})

    def test_list_projects_is_empty_on_a_fresh_studio(self):

        self.assertEqual(CalibrationStudio().list_projects(), ())


class CalibrationStudioSessionAccessTests(unittest.TestCase):

    def test_a_project_created_session_is_visible_via_studio_list_sessions(self):

        studio = CalibrationStudio()
        project = studio.create_project(name="P")
        session = project.create_session(
            benchmark_id="nist-10story",
            candidate=WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test"),
            master_seed=90210,
        )

        self.assertIn(session, studio.list_sessions())

    def test_studio_get_session_finds_a_session_created_via_its_project(self):

        studio = CalibrationStudio()
        project = studio.create_project(name="P")
        session = project.create_session()

        found = studio.get_session(session.session_id)

        self.assertIs(found, session)
        self.assertIsInstance(found, CalibrationSession)

    def test_studio_get_session_returns_none_for_unknown_id(self):

        studio = CalibrationStudio()
        studio.create_project(name="P")

        self.assertIsNone(studio.get_session("does-not-exist"))

    def test_sessions_across_multiple_projects_are_not_conflated(self):

        studio = CalibrationStudio()
        project_a = studio.create_project(name="A")
        project_b = studio.create_project(name="B")

        session_a = project_a.create_session(benchmark_id="a")
        session_b = project_b.create_session(benchmark_id="b")

        self.assertEqual(set(studio.list_sessions()), {session_a, session_b})
        self.assertIn(session_a, project_a.sessions)
        self.assertNotIn(session_a, project_b.sessions)

    def test_sessions_expose_the_approved_public_api(self):

        # Restates this milestone's own VERIFY requirement directly:
        # a session created through the full CalibrationStudio ->
        # CalibrationProject -> CalibrationSession chain exposes every
        # field/property the approved persistent data model named for
        # Phase 3 (benchmark reference, experiment identity,
        # reproducibility metadata, execution state, progress, status).
        studio = CalibrationStudio()
        project = studio.create_project(name="P")
        candidate = WalkingSpeedCandidate("Adult_Default", 0.65, "test", "test")
        session = project.create_session(benchmark_id="nist-10story", candidate=candidate, master_seed=90210)

        self.assertEqual(session.benchmark_id, "nist-10story")
        self.assertIs(session.candidate, candidate)
        self.assertEqual(session.master_seed, 90210)
        self.assertIsNotNone(session.git_commit_hash)
        self.assertEqual(session.simulator_id, "synevac")
        self.assertEqual(session.status.value, "PENDING")
        self.assertIsNone(session.progress)
        self.assertIsNone(session.reproducible)


class CalibrationStudioOrchestrationPlaceholderTests(unittest.TestCase):

    # These prove the public API SHAPE exists (per this milestone's own
    # "most methods may remain placeholders" allowance) while also
    # proving calling one does not silently run, or duplicate, any real
    # calibration_benchmark/research_framework/Replay Studio/Dataset
    # Builder logic.

    def test_run_published_benchmark_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            CalibrationStudio().run_published_benchmark()

    def test_run_parameter_sweep_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            CalibrationStudio().run_parameter_sweep()

    def test_open_in_replay_studio_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            CalibrationStudio().open_in_replay_studio()

    def test_generate_validation_dashboard_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            CalibrationStudio().generate_validation_dashboard()


if __name__ == "__main__":
    unittest.main()
