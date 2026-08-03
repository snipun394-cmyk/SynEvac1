import tempfile
import unittest
from pathlib import Path

from calibration_studio.studio import CalibrationStudio


class CalibrationStudioTestCase(unittest.TestCase):

    def setUp(self):

        self._tmpdir = tempfile.TemporaryDirectory()
        self.storage_root = Path(self._tmpdir.name)

    def tearDown(self):

        self._tmpdir.cleanup()


class StudioWithoutStorageRootTests(unittest.TestCase):

    def test_save_project_raises_without_storage_root(self):

        studio = CalibrationStudio()
        project = studio.create_project(name="P")

        with self.assertRaises(ValueError):
            studio.save_project(project)

    def test_load_project_raises_without_storage_root(self):

        with self.assertRaises(ValueError):
            CalibrationStudio().load_project("some-id")

    def test_save_session_raises_without_storage_root(self):

        studio = CalibrationStudio()
        project = studio.create_project(name="P")
        session = project.create_session()

        with self.assertRaises(ValueError):
            studio.save_session(session)


class StudioWithStorageRootTests(CalibrationStudioTestCase):

    def test_save_and_load_project_through_the_facade(self):

        studio = CalibrationStudio(storage_root=self.storage_root)
        project = studio.create_project(name="P", description="d")

        studio.save_project(project)

        studio2 = CalibrationStudio(storage_root=self.storage_root)
        reloaded = studio2.load_project(project.project_id)

        self.assertEqual(reloaded.name, "P")
        self.assertEqual(reloaded.description, "d")

    def test_load_project_registers_it_in_the_in_memory_registry(self):

        studio = CalibrationStudio(storage_root=self.storage_root)
        project = studio.create_project(name="P")
        studio.save_project(project)

        studio2 = CalibrationStudio(storage_root=self.storage_root)
        self.assertIsNone(studio2.get_project(project.project_id))

        studio2.load_project(project.project_id)

        self.assertIsNotNone(studio2.get_project(project.project_id))
        self.assertIn(project.project_id, [p.project_id for p in studio2.list_projects()])

    def test_list_persisted_projects_registers_all_in_memory(self):

        studio = CalibrationStudio(storage_root=self.storage_root)
        project_a = studio.create_project(name="A")
        project_b = studio.create_project(name="B")
        studio.save_project(project_a)
        studio.save_project(project_b)

        studio2 = CalibrationStudio(storage_root=self.storage_root)
        persisted = studio2.list_persisted_projects()

        self.assertEqual({p.name for p in persisted}, {"A", "B"})
        self.assertEqual(len(studio2.list_projects()), 2)

    def test_save_and_load_session_through_the_facade(self):

        studio = CalibrationStudio(storage_root=self.storage_root)
        project = studio.create_project(name="P")
        session = project.create_session(benchmark_id="nist-10story", master_seed=1)
        session.mark_running()

        studio.save_session(session)

        reloaded = CalibrationStudio(storage_root=self.storage_root).load_session(session.session_id)

        self.assertEqual(reloaded.benchmark_id, "nist-10story")
        self.assertEqual(reloaded.status.value, "RUNNING")


if __name__ == "__main__":
    unittest.main()
