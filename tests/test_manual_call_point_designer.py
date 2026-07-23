import sys
import unittest

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.manual_call_point_item import ManualCallPointItem
from designer.windows.main_window import MainWindow

from models.manual_call_point import ManualCallPoint
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

    def test_manual_call_point_action_exists_and_connected(self):

        window = MainWindow()

        self.assertTrue(hasattr(window.toolbar, "manual_call_point_action"))
        self.assertGreater(
            window.toolbar.manual_call_point_action.receivers(window.toolbar.manual_call_point_action.triggered), 0,
        )


class PlacementTests(unittest.TestCase):

    def test_click_to_place_creates_a_manual_call_point(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.manual_call_point_action.trigger()
        self.assertEqual(window.canvas.scene_obj.current_tool, "manual_call_point")

        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.manual_call_point_count, 1)
        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, ManualCallPointItem))
        self.assertIs(item.model, floor.manual_call_points[0])

    def test_selectable_and_movable(self):

        window, floor, zone = _make_window_with_zone()

        model = ManualCallPoint(id="M1", name="M1", floor_id=floor.id, position=(1.0, 1.0))
        item = ManualCallPointItem(50, 50, model=model)
        window.canvas.scene_obj.addItem(item)

        from PyQt6.QtWidgets import QGraphicsItem
        self.assertTrue(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.assertTrue(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)

        item.setPos(500, 500)
        item.sync_to_model()
        self.assertEqual(model.position, (10.0, 10.0))


class ZoneAutoAssignmentTests(unittest.TestCase):

    def test_auto_assigned_inside_a_single_zone(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.manual_call_point_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.manual_call_points[0].zone_ids, ("Z-A",))

    def test_outside_every_zone_stays_unassigned(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.manual_call_point_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(5000, 5000))

        self.assertEqual(floor.manual_call_points[0].zone_ids, ())

    def test_ambiguous_overlapping_zones_stay_unassigned(self):

        window, floor, zone = _make_window_with_zone()
        floor.add_zone(Zone(id="Z-B", name="Zone B", floor_id=floor.id, x=0.0, y=0.0, width=10.0, height=10.0))

        window.toolbar.manual_call_point_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.manual_call_points[0].zone_ids, ())

    def test_manual_reassignment_still_possible(self):

        window, floor, zone = _make_window_with_zone()
        floor.add_zone(Zone(id="Z-B", name="Zone B", floor_id=floor.id, x=20.0, y=0.0, width=10.0, height=10.0))

        window.toolbar.manual_call_point_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, ManualCallPointItem))
        window.property_panel.show_manual_call_point(item)

        index = window.property_panel.mcp_zone.findData("Z-B")
        window.property_panel.mcp_zone.setCurrentIndex(index)

        self.assertEqual(item.model.zone_ids, ("Z-B",))


class PropertyPanelTests(unittest.TestCase):

    def test_shows_current_zone_and_state(self):

        window, floor, zone = _make_window_with_zone()

        model = ManualCallPoint(id="M1", name="M1", floor_id=floor.id, zone_ids=("Z-A",))
        item = ManualCallPointItem(250, 250, model=model)

        window.property_panel.show_manual_call_point(item)

        self.assertEqual(
            window.property_panel.mcp_zone.itemData(window.property_panel.mcp_zone.currentIndex()), "Z-A",
        )
        self.assertEqual(window.property_panel.mcp_state.text(), "NORMAL")
        self.assertTrue(window.property_panel.mcp_zone_warning.isHidden())

    def test_activated_checkbox_activates_the_device(self):

        window, floor, zone = _make_window_with_zone()

        model = ManualCallPoint(id="M1", name="M1", floor_id=floor.id, zone_ids=("Z-A",))
        item = ManualCallPointItem(250, 250, model=model)

        window.property_panel.show_manual_call_point(item)
        window.property_panel.mcp_activated.setChecked(True)

        self.assertTrue(model.activated)
        self.assertEqual(window.property_panel.mcp_state.text(), "ALARM")

    def test_unchecking_activated_restores_the_device(self):

        window, floor, zone = _make_window_with_zone()

        model = ManualCallPoint(id="M1", name="M1", floor_id=floor.id, zone_ids=("Z-A",), activated=True)
        item = ManualCallPointItem(250, 250, model=model)

        window.property_panel.show_manual_call_point(item)
        window.property_panel.mcp_activated.setChecked(False)

        self.assertFalse(model.activated)
        self.assertEqual(window.property_panel.mcp_state.text(), "NORMAL")

    def test_warning_visible_when_unassigned(self):

        window, floor, zone = _make_window_with_zone()

        model = ManualCallPoint(id="M1", name="M1", floor_id=floor.id)
        item = ManualCallPointItem(250, 250, model=model)

        window.property_panel.show_manual_call_point(item)

        self.assertFalse(window.property_panel.mcp_zone_warning.isHidden())

    def test_rename_via_shared_name_field(self):

        window, floor, zone = _make_window_with_zone()

        model = ManualCallPoint(id="M1", name="Old Name", floor_id=floor.id)
        item = ManualCallPointItem(250, 250, model=model)

        window.property_panel.show_manual_call_point(item)
        self.assertEqual(window.property_panel.object_name.text(), "Old Name")


class SaveReloadTests(unittest.TestCase):

    def test_save_reload_preserves_manual_call_point(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.manual_call_point_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        mcp_before = floor.manual_call_points[0]
        mcp_before.activate()

        from models.project import Project

        data = window.canvas.scene_obj.project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(floor.id)

        self.assertEqual(restored_floor.manual_call_point_count, 1)
        restored_mcp = restored_floor.manual_call_points[0]
        self.assertEqual(restored_mcp.id, mcp_before.id)
        self.assertEqual(restored_mcp.zone_ids, ("Z-A",))
        self.assertTrue(restored_mcp.activated)


if __name__ == "__main__":
    unittest.main()
