import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from builder.widgets.builder_toolbar import BuilderToolbar
from builder.widgets.project_summary_panel import ProjectSummaryPanel
from builder.windows.builder_main_window import BuilderMainWindow

from models.building import Building
from models.camera import Camera
from models.door import Door
from models.exit import Exit
from models.obstacle import Obstacle
from models.smoke_detector import SmokeDetector
from models.zone import Zone


class BuilderToolbarScopeTests(unittest.TestCase):

    # The milestone brief explicitly forbids Simulation from Builder,
    # and scopes authoring to exactly nine asset types + Zone. This
    # confirms the new, Builder-only toolbar reflects that -- no
    # occupant/simulation action exists at all (not merely hidden),
    # and Undo/Redo are disabled with a tooltip rather than silently
    # wired to nothing, matching Studio's own current, honest state.

    def setUp(self):

        self.toolbar = BuilderToolbar()

    def test_no_simulation_or_occupant_actions_exist(self):

        self.assertFalse(hasattr(self.toolbar, "simulation_action"))
        self.assertFalse(hasattr(self.toolbar, "occupant_action"))

    def test_undo_redo_are_disabled(self):

        self.assertFalse(self.toolbar.undo_action.isEnabled())
        self.assertFalse(self.toolbar.redo_action.isEnabled())

    def test_every_spec_required_asset_tool_exists(self):

        for action_name in (
            "zone_action", "door_action", "exit_action", "stair_action",
            "obstacle_action", "camera_action", "smoke_detector_action",
            "heat_detector_action", "speaker_action",
        ):
            self.assertTrue(hasattr(self.toolbar, action_name), f"missing {action_name}")

    def test_calibrate_scale_action_exists(self):

        self.assertTrue(hasattr(self.toolbar, "calibrate_scale_action"))


class ProjectSummaryPanelTests(unittest.TestCase):

    def setUp(self):

        self.panel = ProjectSummaryPanel()

    def test_none_building_shows_zero_counts(self):

        self.panel.refresh(None)

        self.assertEqual(self.panel.zone_count.text(), "0")
        self.assertEqual(self.panel.scale_status.text(), "-")

    def test_counts_every_supported_asset_type(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        floor.add_zone(Zone(name="Z1", x=0, y=0, width=2, height=2, floor_id=floor.id))
        floor.add_door(Door(name="D1", floor_id=floor.id))
        floor.add_exit(Exit(name="E1", floor_id=floor.id))
        floor.add_camera(Camera(name="C1", floor_id=floor.id))
        floor.add_smoke_detector(SmokeDetector(name="SD1", floor_id=floor.id))
        floor.add_obstacle(Obstacle(name="O1", floor_id=floor.id))

        self.panel.refresh(building, validation_is_valid=True, validation_summary_text="0 errors")

        self.assertEqual(self.panel.zone_count.text(), "1")
        self.assertEqual(self.panel.door_count.text(), "1")
        self.assertEqual(self.panel.exit_count.text(), "1")
        self.assertEqual(self.panel.camera_count.text(), "1")
        self.assertEqual(self.panel.smoke_detector_count.text(), "1")
        self.assertEqual(self.panel.obstacle_count.text(), "1")

    def test_total_area_sums_zone_areas(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        floor.add_zone(Zone(name="Z1", x=0, y=0, width=2, height=3, floor_id=floor.id))
        floor.add_zone(Zone(name="Z2", x=5, y=0, width=4, height=2, floor_id=floor.id))

        self.panel.refresh(building)

        self.assertIn("14.00", self.panel.total_area.text())

    def test_scale_status_reflects_calibrated_floors(self):

        building = Building(name="B")
        floor_a = building.create_floor(name="Ground Floor")
        floor_b = building.create_floor(name="First Floor")

        floor_a.set_scale_calibration((0, 0), (100, 0), 5.0, 20.0)

        self.panel.refresh(building)

        self.assertEqual(self.panel.scale_status.text(), "1 / 2 floor(s) calibrated")

    def test_validation_status_reflects_validity(self):

        building = Building(name="B")
        building.create_floor(name="Ground Floor")

        self.panel.refresh(building, validation_is_valid=False, validation_summary_text="1 error(s)")

        self.assertTrue(self.panel.validation_status.text().startswith("INVALID"))


class BuilderAppEntryPointTests(unittest.TestCase):

    # Confirms Builder is launchable as a standalone application -- the
    # same shape as core.app.SynEvacApp/main.py, but importing nothing
    # from designer.windows.main_window or core.app.

    def test_builder_app_imports_no_studio_main_window(self):

        import ast

        import builder.app as builder_app_module

        source = open(builder_app_module.__file__, encoding="utf-8").read()

        imported_modules = set()

        for node in ast.walk(ast.parse(source)):

            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

            elif isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)

        self.assertNotIn("designer.windows.main_window", imported_modules)
        self.assertNotIn("core.app", imported_modules)

    def test_builder_main_window_constructs_without_error(self):

        window = BuilderMainWindow()

        self.assertEqual(window.windowTitle(), "SynEvac Builder")
