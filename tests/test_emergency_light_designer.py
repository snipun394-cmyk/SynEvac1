import sys
import unittest

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.emergency_light_item import EmergencyLightItem
from designer.windows.main_window import MainWindow

from models.emergency_light import EmergencyLight
from models.zone import Zone


class _FakeSceneMouseEvent:

    def __init__(self, x, y):
        self._pos = QPointF(x, y)

    def scenePos(self):
        return self._pos


def _make_window_with_zone():

    window = MainWindow()
    floor = window.canvas.scene_obj.current_floor

    zone = Zone(id="Z-A", name="Zone A", floor_id=floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
    floor.add_zone(zone)

    window.property_panel.building = window.canvas.scene_obj.project.building

    return window, floor, zone


class ToolbarWiringTests(unittest.TestCase):

    def test_emergency_light_action_exists_and_connected(self):

        window = MainWindow()

        self.assertTrue(hasattr(window.toolbar, "emergency_light_action"))
        self.assertGreater(
            window.toolbar.emergency_light_action.receivers(window.toolbar.emergency_light_action.triggered), 0,
        )


class PlacementTests(unittest.TestCase):

    def test_click_to_place_creates_an_emergency_light(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.emergency_light_action.trigger()
        self.assertEqual(window.canvas.scene_obj.current_tool, "emergency_light")

        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.emergency_light_count, 1)
        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, EmergencyLightItem))
        self.assertIs(item.model, floor.emergency_lights[0])


class ZoneAutoAssignmentTests(unittest.TestCase):

    def test_auto_assigned_inside_a_single_zone(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.emergency_light_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.emergency_lights[0].zone_ids, ("Z-A",))

    def test_outside_every_zone_stays_unassigned(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.emergency_light_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(5000, 5000))

        self.assertEqual(floor.emergency_lights[0].zone_ids, ())


class PropertyPanelTests(unittest.TestCase):

    def test_shows_current_zone_and_availability(self):

        window, floor, zone = _make_window_with_zone()

        model = EmergencyLight(id="E1", name="E1", floor_id=floor.id, zone_ids=("Z-A",))
        item = EmergencyLightItem(250, 250, model=model)

        window.property_panel.show_emergency_light(item)

        self.assertEqual(
            window.property_panel.emergency_light_zone.itemData(
                window.property_panel.emergency_light_zone.currentIndex()
            ), "Z-A",
        )
        self.assertEqual(window.property_panel.emergency_light_availability.text(), "AVAILABLE")
        self.assertTrue(window.property_panel.emergency_light_zone_warning.isHidden())

    def test_deactivating_updates_availability(self):

        window, floor, zone = _make_window_with_zone()

        model = EmergencyLight(id="E1", name="E1", floor_id=floor.id, zone_ids=("Z-A",))
        item = EmergencyLightItem(250, 250, model=model)

        window.property_panel.show_emergency_light(item)
        window.property_panel.emergency_light_active.setChecked(False)

        self.assertFalse(model.active)
        self.assertEqual(window.property_panel.emergency_light_availability.text(), "UNAVAILABLE")

    def test_fault_health_reflected_in_availability(self):

        window, floor, zone = _make_window_with_zone()

        model = EmergencyLight(id="E1", name="E1", floor_id=floor.id, zone_ids=("Z-A",))
        item = EmergencyLightItem(250, 250, model=model)

        window.property_panel.show_emergency_light(item)

        index = window.property_panel.emergency_light_health.findText("Fault")
        window.property_panel.emergency_light_health.setCurrentIndex(index)

        self.assertEqual(window.property_panel.emergency_light_availability.text(), "FAULT")

    def test_light_type_editable(self):

        window, floor, zone = _make_window_with_zone()

        model = EmergencyLight(id="E1", name="E1", floor_id=floor.id)
        item = EmergencyLightItem(250, 250, model=model)

        window.property_panel.show_emergency_light(item)

        index = window.property_panel.emergency_light_type.findText("Ceiling Mounted")
        window.property_panel.emergency_light_type.setCurrentIndex(index)

        self.assertEqual(model.light_type, "Ceiling Mounted")

    def test_warning_visible_when_unassigned(self):

        window, floor, zone = _make_window_with_zone()

        model = EmergencyLight(id="E1", name="E1", floor_id=floor.id)
        item = EmergencyLightItem(250, 250, model=model)

        window.property_panel.show_emergency_light(item)

        self.assertFalse(window.property_panel.emergency_light_zone_warning.isHidden())


class SaveReloadTests(unittest.TestCase):

    def test_save_reload_preserves_emergency_light(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.emergency_light_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        from models.project import Project

        data = window.canvas.scene_obj.project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(floor.id)

        self.assertEqual(restored_floor.emergency_light_count, 1)
        self.assertEqual(restored_floor.emergency_lights[0].zone_ids, ("Z-A",))


if __name__ == "__main__":
    unittest.main()
