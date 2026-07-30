import os
import shutil
import sys
import tempfile
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from builder.windows.builder_main_window import BuilderMainWindow

from models.project import Project
from models.zone import Zone

from serialization.serializer import Serializer


class BuilderNewProjectTests(unittest.TestCase):

    def setUp(self):

        self.window = BuilderMainWindow()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # =====================================================

    def test_window_starts_with_a_default_project(self):

        project = self.window.canvas.scene_obj.project

        self.assertIsInstance(project, Project)
        self.assertIsNotNone(project.building)
        self.assertEqual(project.building.floor_count, 1)

    # =====================================================

    def test_new_project_replaces_current_project_and_resets_dirty_flag(self):

        floor = self.window.canvas.scene_obj.current_floor
        floor.add_zone(Zone(name="Zone 1", x=0, y=0, width=2, height=2, floor_id=floor.id))

        self.window._mark_dirty()
        self.assertTrue(self.window._dirty)

        original_project_id = self.window.canvas.scene_obj.project.id

        self.window.new_project()

        self.assertNotEqual(original_project_id, self.window.canvas.scene_obj.project.id)
        self.assertFalse(self.window._dirty)
        self.assertEqual(self.window.canvas.scene_obj.project.building.floors[0].zone_count, 0)

    # =====================================================

    def test_new_project_rebinds_every_dependent_panel(self):

        self.window.new_project()

        project = self.window.canvas.scene_obj.project

        self.assertIs(self.window.project_tree.project, project)
        self.assertIs(self.window.floor_list.building, project.building)
        self.assertIs(self.window.property_panel.building, project.building)


class BuilderSaveOpenTests(unittest.TestCase):

    def setUp(self):

        self.window = BuilderMainWindow()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # =====================================================

    def test_save_to_writes_a_loadable_syn_file(self):

        floor = self.window.canvas.scene_obj.current_floor
        floor.add_zone(Zone(name="Zone 1", x=0, y=0, width=3, height=3, floor_id=floor.id))

        path = os.path.join(self.tmpdir, "project.syn")

        result = self.window._save_to(path)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(path))

        reloaded = Serializer.load(path)

        self.assertEqual(reloaded.building.ordered_floors()[0].zone_count, 1)

    # =====================================================

    def test_save_to_sets_current_filename_and_clears_dirty_flag(self):

        path = os.path.join(self.tmpdir, "project.syn")

        self.window._mark_dirty()
        self.window._save_to(path)

        self.assertEqual(self.window._current_filename, path)
        self.assertFalse(self.window._dirty)

    # =====================================================

    def test_open_project_file_replaces_current_project(self):

        path = os.path.join(self.tmpdir, "project.syn")

        original_project = Project.new_default()
        original_project.building.create_floor(name="Second Floor")

        Serializer.save(original_project, path)

        self.window._open_project_file(path)

        self.assertEqual(self.window.canvas.scene_obj.project.building.floor_count, 2)
        self.assertEqual(self.window._current_filename, path)
        self.assertFalse(self.window._dirty)

    # =====================================================

    def test_open_project_file_adds_to_recent_projects(self):

        path = os.path.join(self.tmpdir, "project.syn")

        Serializer.save(Project.new_default(), path)

        self.window._settings.remove("recent_projects")

        self.window._open_project_file(path)

        recent = self.window._settings.value("recent_projects", [], type=list)

        self.assertIn(path, recent)

    # =====================================================

    def test_save_project_as_appends_syn_extension(self):

        path_without_extension = os.path.join(self.tmpdir, "no_extension")

        from unittest.mock import patch

        with patch(
            "builder.windows.builder_main_window.QFileDialog.getSaveFileName",
            return_value=(path_without_extension, ""),
        ):
            result = self.window.save_project_as()

        self.assertTrue(result)
        self.assertTrue(os.path.exists(path_without_extension + ".syn"))


class BuilderStudioRoundTripTests(unittest.TestCase):

    # Confirms Builder-generated projects open directly inside SynEvac
    # Studio without modification -- the milestone's own explicit
    # acceptance requirement. Uses designer.windows.main_window.
    # MainWindow directly (the real Studio window), not a stand-in.

    def setUp(self):

        self.window = BuilderMainWindow()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # =====================================================

    def test_builder_project_opens_unmodified_in_studio_main_window(self):

        from models.door import Door
        from models.exit import Exit
        from models.staircase import Staircase

        scene = self.window.canvas.scene_obj
        floor = scene.current_floor

        second_floor = scene.project.building.create_floor(name="Second Floor")

        zone_a = Zone(name="Zone A", x=0, y=0, width=4, height=4, floor_id=floor.id)
        zone_b = Zone(name="Zone B", x=5, y=0, width=4, height=4, floor_id=floor.id)
        floor.add_zone(zone_a)
        floor.add_zone(zone_b)

        floor.add_door(Door(
            name="Door 1", start_point=(4, 2), end_point=(5, 2),
            floor_id=floor.id, zone_a_id=zone_a.id, zone_b_id=zone_b.id,
        ))

        floor.add_exit(Exit(
            name="Exit 1", start_point=(0, 0), end_point=(1, 0),
            floor_id=floor.id, zone_id=zone_a.id,
        ))

        zone_c = Zone(name="Zone C", x=0, y=0, width=4, height=4, floor_id=second_floor.id)
        second_floor.add_zone(zone_c)

        floor.add_stair(Staircase(
            name="Stair 1", from_position=(6, 6), to_position=(1, 1),
            from_floor_id=floor.id, to_floor_id=second_floor.id,
            from_zone_id=zone_b.id, to_zone_id=zone_c.id,
        ))

        path = os.path.join(self.tmpdir, "interop.syn")

        self.assertTrue(self.window._save_to(path))

        from designer.windows.main_window import MainWindow

        studio_window = MainWindow()

        loaded_project = Serializer.load(path, credential_store=studio_window._credential_store)

        studio_window.canvas.scene_obj.project = loaded_project
        studio_window.canvas.scene_obj.current_floor = loaded_project.building.ordered_floors()[0]
        studio_window.canvas.scene_obj.rebuild_scene()

        building = studio_window.canvas.scene_obj.project.building

        self.assertEqual(building.floor_count, 2)

        loaded_floor = building.get_floor(floor.id)

        self.assertEqual(loaded_floor.zone_count, 2)
        self.assertEqual(loaded_floor.door_count, 1)
        self.assertEqual(loaded_floor.exit_count, 1)
        self.assertEqual(loaded_floor.stair_count, 1)

        from designer.validation import validate_building_authoring

        report = validate_building_authoring(building)

        self.assertEqual(report.errors, [])
