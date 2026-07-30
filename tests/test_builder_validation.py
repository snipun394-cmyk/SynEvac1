import os
import shutil
import sys
import tempfile
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from builder.validation_extras import INFO, validate_builder_extras
from builder.widgets.validation_panel import ValidationPanel
from builder.windows.builder_main_window import BuilderMainWindow

from models.building import Building
from models.floor import Floor
from models.zone import Zone
from models.door import Door

from navigation.validation import ValidationReport


class ValidationExtrasTests(unittest.TestCase):

    def test_empty_building_has_no_issues(self):

        building = Building(name="Empty")
        building.create_floor(name="Ground Floor")

        report = validate_builder_extras(building)

        self.assertEqual(report.errors, [])
        self.assertEqual(report.warnings, [])

    # =====================================================

    def test_unnamed_zone_produces_missing_name_warning(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        floor.add_zone(Zone(name="", x=0, y=0, width=2, height=2, floor_id=floor.id))

        report = validate_builder_extras(building)

        self.assertTrue(any(issue.code == "zone_missing_name" for issue in report))

    # =====================================================

    def test_unnamed_floor_produces_missing_name_warning(self):

        building = Building(name="B")
        building.create_floor(name="")

        report = validate_builder_extras(building)

        self.assertTrue(any(issue.code == "floor_missing_name" for issue in report))

    # =====================================================

    def test_overlapping_zones_are_detected(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        floor.add_zone(Zone(name="Zone A", x=0, y=0, width=5, height=5, floor_id=floor.id))
        floor.add_zone(Zone(name="Zone B", x=2, y=2, width=5, height=5, floor_id=floor.id))

        report = validate_builder_extras(building)

        overlap_issues = report.by_code("overlapping_zones")

        self.assertEqual(len(overlap_issues), 1)

    # =====================================================

    def test_adjacent_non_overlapping_zones_are_not_flagged(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        floor.add_zone(Zone(name="Zone A", x=0, y=0, width=5, height=5, floor_id=floor.id))
        floor.add_zone(Zone(name="Zone B", x=5, y=0, width=5, height=5, floor_id=floor.id))

        report = validate_builder_extras(building)

        self.assertEqual(report.by_code("overlapping_zones"), [])

    # =====================================================

    def test_floor_plan_without_calibration_produces_warning(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        floor.floor_plan = "some_plan.png"

        report = validate_builder_extras(building)

        self.assertTrue(any(issue.code == "floor_missing_scale_calibration" for issue in report))
        self.assertTrue(
            any(issue.code == "floor_missing_scale_calibration" and issue.severity == ValidationReport.WARNING
                for issue in report)
        )

    # =====================================================

    def test_calibrated_floor_plan_produces_no_calibration_warning(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        floor.floor_plan = "some_plan.png"
        floor.set_scale_calibration((0, 0), (100, 0), 5.0, 20.0)

        report = validate_builder_extras(building)

        self.assertFalse(any(issue.code == "floor_missing_scale_calibration" for issue in report))

    # =====================================================

    def test_floor_without_floor_plan_gets_an_info_notice_not_a_warning(self):

        building = Building(name="B")
        building.create_floor(name="Ground Floor")

        report = validate_builder_extras(building)

        notice = [issue for issue in report if issue.code == "floor_missing_floor_plan"]

        self.assertEqual(len(notice), 1)
        self.assertEqual(notice[0].severity, INFO)

    # =====================================================

    def test_none_building_returns_empty_report(self):

        report = validate_builder_extras(None)

        self.assertEqual(len(report), 0)


class ValidationPanelTests(unittest.TestCase):

    def setUp(self):

        self.panel = ValidationPanel()

    # =====================================================

    def test_none_building_is_valid_with_no_project_message(self):

        self.panel.refresh(None)

        self.assertTrue(self.panel.is_valid)
        self.assertIn("No project loaded", self.panel.summary_label.text())

    # =====================================================

    def test_door_missing_zone_produces_error_and_invalid_report(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        floor.add_door(Door(name="Door 1", start_point=(0, 0), end_point=(1, 0), floor_id=floor.id))

        self.panel.refresh(building)

        self.assertFalse(self.panel.is_valid)
        self.assertIn("error(s)", self.panel.summary_label.text())

    # =====================================================

    def test_clean_building_is_valid(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        floor.add_zone(Zone(name="Zone 1", x=0, y=0, width=3, height=3, floor_id=floor.id))

        self.panel.refresh(building)

        self.assertTrue(self.panel.is_valid)

    # =====================================================

    def test_combines_authoring_graph_and_extras_reports(self):

        # A single building triggering one issue from each of the three
        # underlying sources this panel merges: designer.validation
        # (door_missing_zone_a -- ERROR), navigation.graph_builder
        # (isolated_zone -- WARNING, once a Zone exists with no
        # connections at all), and builder.validation_extras
        # (zone_missing_name -- WARNING).

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        floor.add_zone(Zone(name="", x=0, y=0, width=3, height=3, floor_id=floor.id))
        floor.add_door(Door(name="Door 1", start_point=(0, 0), end_point=(1, 0), floor_id=floor.id))

        self.panel.refresh(building)

        codes = {
            self.panel.list_widget.item(i).text()
            for i in range(self.panel.list_widget.count())
        }

        self.assertTrue(any("Door" in text and "ERROR" in text for text in codes))
        self.assertTrue(any("no name" in text for text in codes))


class BuilderMainWindowValidationGatingTests(unittest.TestCase):

    # "Critical validation errors should prevent export" -- Save IS
    # export in Builder (no separate export action; see
    # docs/architecture/synevac_builder_feasibility_investigation.md).

    def setUp(self):

        self.window = BuilderMainWindow()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # =====================================================

    def test_save_blocked_when_project_has_critical_errors(self):

        from unittest.mock import patch

        floor = self.window.canvas.scene_obj.current_floor

        floor.add_door(Door(name="Bad Door", start_point=(0, 0), end_point=(1, 0), floor_id=floor.id))

        path = os.path.join(self.tmpdir, "invalid.syn")

        with patch("builder.windows.builder_main_window.QMessageBox.critical") as critical_mock:

            result = self.window._save_to(path)

        self.assertFalse(result)
        self.assertFalse(os.path.exists(path))
        critical_mock.assert_called_once()

    # =====================================================

    def test_save_succeeds_once_error_is_resolved(self):

        floor = self.window.canvas.scene_obj.current_floor

        zone = Zone(name="Zone 1", x=0, y=0, width=3, height=3, floor_id=floor.id)
        floor.add_zone(zone)

        door = Door(
            name="Door 1", start_point=(3, 1), end_point=(4, 1),
            floor_id=floor.id, zone_a_id=zone.id, zone_b_id="",
        )
        floor.add_door(door)

        path = os.path.join(self.tmpdir, "still_invalid.syn")

        self.assertFalse(self.window._save_to(path))

        # Resolve the error the same way the Property Panel would
        # (writing straight onto the model, see BuilderPropertyPanel.
        # _update_door_zones()).
        second_zone = Zone(name="Zone 2", x=5, y=0, width=3, height=3, floor_id=floor.id)
        floor.add_zone(second_zone)
        door.zone_b_id = second_zone.id

        self.assertTrue(self.window._save_to(path))
        self.assertTrue(os.path.exists(path))
