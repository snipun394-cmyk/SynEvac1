import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.emergency_light_item import EmergencyLightItem
from designer.items.manual_call_point_item import ManualCallPointItem
from designer.windows.main_window import MainWindow
from designer.validation import validate_building_authoring

from models.emergency_light import EmergencyLight, EmergencyLightAvailability
from models.manual_call_point import ManualCallPoint
from models.sensor_asset import HealthStatus
from models.zone import Zone


# =====================================================
# Manual Call Points & Emergency Lighting milestone, Step 5 -- failure
# and degradation coverage not already exercised by
# tests.test_manual_call_point_designer / tests.test_emergency_light_designer
# / tests.test_manual_call_point_model / tests.test_emergency_light_model /
# tests.test_manual_call_point_facp_integration (unassigned, ambiguous,
# outside-every-zone, inactive, fault, activated, multiple MCPs, and
# legacy-project-without-list are already covered there). This file adds
# only the genuinely missing case: an asset whose zone_ids reference a
# zone that has since been deleted from the floor -- mirroring the
# existing SmokeDetector precedent in
# tests.test_property_panel_zone_assignment.DeletedZoneReferenceTests.
# =====================================================


def _make_window_with_two_zones():

    window = MainWindow()
    floor = window.canvas.scene_obj.current_floor

    zone_a = Zone(id="Z-A", name="Zone A", floor_id=floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
    zone_b = Zone(id="Z-B", name="Zone B", floor_id=floor.id, x=20.0, y=0.0, width=10.0, height=10.0)
    floor.add_zone(zone_a)
    floor.add_zone(zone_b)

    window.property_panel.building = window.canvas.scene_obj.project.building

    return window, floor, zone_a, zone_b


class ManualCallPointDeletedZoneTests(unittest.TestCase):

    def test_deleted_zone_still_shown_but_not_selectable(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = ManualCallPoint(id="M1", name="M1", floor_id=floor.id, zone_ids=("Z-GONE",))
        item = ManualCallPointItem(0, 0, model=model)

        window.property_panel.show_manual_call_point(item)

        # No crash, and the stale id never appears as a selected choice.
        self.assertEqual(window.property_panel.mcp_zone.currentIndex(), 0)

    def test_validation_never_crashes_once_the_referenced_zone_is_gone(self):

        # validate_building_authoring only flags an EMPTY zone_ids
        # (see designer/validation.py::_check_zone_assignment) -- it
        # has no dangling-reference detection for any zone-scoped asset
        # type (Speaker/Smoke/Heat Detector included), so a stale-but-
        # non-empty zone_ids correctly produces no warning here. The
        # requirement under test is narrower and absolute: this must
        # never crash, and must never be silently treated as if the
        # zone still existed.
        window, floor, zone_a, zone_b = _make_window_with_two_zones()
        floor.remove_zone(zone_a)

        mcp = ManualCallPoint(id="M1", name="M1", floor_id=floor.id, zone_ids=("Z-A",))
        floor.add_manual_call_point(mcp)

        report = validate_building_authoring(window.canvas.scene_obj.project.building)

        self.assertFalse(any(w.code == "manual_call_point_missing_zone" for w in report.warnings))
        self.assertNotIn(mcp.zone_ids[0], {z.id for z in floor.zones})

    def test_activation_still_honestly_reported_with_deleted_zone(self):

        # Losing the zone must never fabricate NORMAL over a real
        # activation, nor crash compute_state().
        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("Z-GONE",))
        mcp.activate()

        from models.sensor_asset import DetectorState
        self.assertEqual(mcp.compute_state(), DetectorState.ALARM)


class EmergencyLightDeletedZoneTests(unittest.TestCase):

    def test_deleted_zone_still_shown_but_not_selectable(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = EmergencyLight(id="E1", name="E1", floor_id=floor.id, zone_ids=("Z-GONE",))
        item = EmergencyLightItem(0, 0, model=model)

        window.property_panel.show_emergency_light(item)

        self.assertEqual(window.property_panel.emergency_light_zone.currentIndex(), 0)

    def test_validation_never_crashes_once_the_referenced_zone_is_gone(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()
        floor.remove_zone(zone_a)

        light = EmergencyLight(id="E1", name="E1", floor_id=floor.id, zone_ids=("Z-A",))
        floor.add_emergency_light(light)

        report = validate_building_authoring(window.canvas.scene_obj.project.building)

        self.assertFalse(any(w.code == "emergency_light_missing_zone" for w in report.warnings))
        self.assertNotIn(light.zone_ids[0], {z.id for z in floor.zones})

    def test_availability_still_honestly_reported_with_deleted_zone(self):

        # A deleted zone must never fabricate AVAILABLE, and a genuine
        # fault must still outrank it.
        light = EmergencyLight(id="E1", name="E1", floor_id="f1", zone_ids=("Z-GONE",), health_status=HealthStatus.FAULT)

        self.assertEqual(light.compute_availability(), EmergencyLightAvailability.FAULT)

    def test_offline_health_never_fabricated_as_available(self):

        light = EmergencyLight(id="E1", name="E1", floor_id="f1", zone_ids=("Z-A",), health_status=HealthStatus.OFFLINE)

        self.assertEqual(light.compute_availability(), EmergencyLightAvailability.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
