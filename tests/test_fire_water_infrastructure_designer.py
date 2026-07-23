import sys
import unittest

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.fire_pump_item import FirePumpItem
from designer.items.fire_service_inlet_item import FireServiceInletItem
from designer.items.fire_water_tank_item import FireWaterTankItem
from designer.items.jockey_pump_item import JockeyPumpItem
from designer.windows.main_window import MainWindow

from models.fire_pump import FirePump
from models.fire_service_inlet import FireServiceInlet
from models.fire_water_tank import FireWaterTank
from models.jockey_pump import JockeyPump
from models.project import Project
from models.zone import Zone


# =====================================================
# Fire Water Supply & Suppression Infrastructure milestone -- Designer
# placement/property-panel/zone-assignment/save-reload tests for
# FireWaterTank/FirePump/JockeyPump/FireServiceInlet, mirroring
# tests.test_fire_safety_asset_designer's own established pattern.
# =====================================================


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

    def test_all_four_actions_exist_and_connected(self):

        window = MainWindow()

        for action_name in (
            "fire_water_tank_action", "fire_pump_action", "jockey_pump_action", "fire_service_inlet_action",
        ):
            action = getattr(window.toolbar, action_name)
            self.assertGreater(action.receivers(action.triggered), 0)


class PlacementTests(unittest.TestCase):

    def test_click_to_place_creates_a_fire_water_tank(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_water_tank_action.trigger()
        self.assertEqual(window.canvas.scene_obj.current_tool, "fire_water_tank")

        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.fire_water_tank_count, 1)
        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, FireWaterTankItem))
        self.assertIs(item.model, floor.fire_water_tanks[0])

    def test_click_to_place_creates_a_fire_pump(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_pump_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.fire_pump_count, 1)
        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, FirePumpItem))
        self.assertIs(item.model, floor.fire_pumps[0])

    def test_click_to_place_creates_a_jockey_pump(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.jockey_pump_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.jockey_pump_count, 1)
        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, JockeyPumpItem))
        self.assertIs(item.model, floor.jockey_pumps[0])

    def test_click_to_place_creates_a_fire_service_inlet(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_service_inlet_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.fire_service_inlet_count, 1)
        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, FireServiceInletItem))
        self.assertIs(item.model, floor.fire_service_inlets[0])

    def test_jockey_pump_item_is_not_a_fire_pump_item(self):

        # Guards against the isinstance-collision hazard a subclass
        # relationship between the two graphics items would create
        # (see designer/items/jockey_pump_item.py's own docstring).
        window, floor, zone = _make_window_with_zone()

        window.toolbar.jockey_pump_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, JockeyPumpItem))
        self.assertNotIsInstance(item, FirePumpItem)


class ZoneAutoAssignmentTests(unittest.TestCase):

    def test_tank_auto_assigned_inside_a_single_zone(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_water_tank_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.fire_water_tanks[0].zone_ids, ("Z-A",))

    def test_tank_outside_every_zone_stays_unassigned(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_water_tank_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(5000, 5000))

        self.assertEqual(floor.fire_water_tanks[0].zone_ids, ())

    def test_pump_auto_assigned_inside_a_single_zone(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_pump_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.fire_pumps[0].zone_ids, ("Z-A",))

    def test_inlet_auto_assigned_inside_a_single_zone(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_service_inlet_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.fire_service_inlets[0].zone_ids, ("Z-A",))


class PropertyPanelTests(unittest.TestCase):

    def test_tank_shows_capacity_level_and_state(self):

        window, floor, zone = _make_window_with_zone()

        model = FireWaterTank(id="T1", name="T1", floor_id=floor.id, zone_ids=("Z-A",), capacity_liters=10000.0, current_level_liters=1000.0)
        item = FireWaterTankItem(250, 250, model=model)

        window.property_panel.show_fire_water_tank(item)

        self.assertEqual(window.property_panel.fire_water_tank_state.text(), "LOW_LEVEL")
        self.assertEqual(window.property_panel.fire_water_tank_capacity.text(), "10000.00")

    def test_tank_level_can_be_cleared_to_unmeasured(self):

        window, floor, zone = _make_window_with_zone()

        model = FireWaterTank(id="T1", name="T1", floor_id=floor.id, zone_ids=("Z-A",), current_level_liters=500.0)
        item = FireWaterTankItem(250, 250, model=model)

        window.property_panel.show_fire_water_tank(item)
        window.property_panel.fire_water_tank_level.setText("")
        window.property_panel.update_fire_water_tank_level()

        self.assertIsNone(model.current_level_liters)
        self.assertEqual(window.property_panel.fire_water_tank_state.text(), "AVAILABLE")

    def test_pump_running_checkbox_updates_state(self):

        window, floor, zone = _make_window_with_zone()

        model = FirePump(id="P1", name="P1", floor_id=floor.id, zone_ids=("Z-A",))
        item = FirePumpItem(250, 250, model=model)

        window.property_panel.show_fire_pump(item)
        window.property_panel.fire_pump_running.setChecked(True)

        self.assertTrue(model.running)
        self.assertEqual(window.property_panel.fire_pump_state.text(), "RUNNING")

    def test_pump_control_mode_editable(self):

        window, floor, zone = _make_window_with_zone()

        model = FirePump(id="P1", name="P1", floor_id=floor.id)
        item = FirePumpItem(250, 250, model=model)

        window.property_panel.show_fire_pump(item)

        index = window.property_panel.fire_pump_control_mode.findText("Manual")
        window.property_panel.fire_pump_control_mode.setCurrentIndex(index)

        self.assertEqual(model.control_mode, "Manual")

    def test_jockey_pump_running_checkbox_updates_state(self):

        window, floor, zone = _make_window_with_zone()

        model = JockeyPump(id="J1", name="J1", floor_id=floor.id, zone_ids=("Z-A",))
        item = JockeyPumpItem(250, 250, model=model)

        window.property_panel.show_jockey_pump(item)
        window.property_panel.jockey_pump_running.setChecked(True)

        self.assertTrue(model.running)
        self.assertEqual(window.property_panel.jockey_pump_state.text(), "RUNNING")

    def test_inlet_shows_availability_and_type(self):

        window, floor, zone = _make_window_with_zone()

        model = FireServiceInlet(id="I1", name="I1", floor_id=floor.id, zone_ids=("Z-A",))
        item = FireServiceInletItem(250, 250, model=model)

        window.property_panel.show_fire_service_inlet(item)

        self.assertEqual(window.property_panel.fire_service_inlet_availability.text(), "AVAILABLE")

        index = window.property_panel.fire_service_inlet_type.findText("Dry Riser Inlet")
        window.property_panel.fire_service_inlet_type.setCurrentIndex(index)

        self.assertEqual(model.inlet_type, "Dry Riser Inlet")


class FireWaterSystemAssignmentTests(unittest.TestCase):

    def test_assigning_tank_to_system_through_property_panel_combo(self):

        window, floor, zone = _make_window_with_zone()

        building = window.canvas.scene_obj.project.building
        system = building.create_fire_water_system("FW-1")

        model = FireWaterTank(id="T1", name="T1", floor_id=floor.id, zone_ids=("Z-A",))
        item = FireWaterTankItem(250, 250, model=model)

        window.property_panel.show_fire_water_tank(item)

        index = window.property_panel.fire_water_tank_fire_water_system.findData(system.id)
        window.property_panel.fire_water_tank_fire_water_system.setCurrentIndex(index)

        self.assertEqual(system.tank_ids, ("T1",))

    def test_reassigning_pump_moves_between_systems(self):

        window, floor, zone = _make_window_with_zone()

        building = window.canvas.scene_obj.project.building
        system_a = building.create_fire_water_system("FW-A")
        system_b = building.create_fire_water_system("FW-B")

        model = FirePump(id="P1", name="P1", floor_id=floor.id)
        item = FirePumpItem(250, 250, model=model)

        window.property_panel.show_fire_pump(item)

        index_a = window.property_panel.fire_pump_fire_water_system.findData(system_a.id)
        window.property_panel.fire_pump_fire_water_system.setCurrentIndex(index_a)
        self.assertEqual(system_a.pump_ids, ("P1",))

        index_b = window.property_panel.fire_pump_fire_water_system.findData(system_b.id)
        window.property_panel.fire_pump_fire_water_system.setCurrentIndex(index_b)

        self.assertEqual(system_a.pump_ids, ())
        self.assertEqual(system_b.pump_ids, ("P1",))

    def test_sprinkler_hydrant_hose_reel_can_also_be_assigned_to_a_system(self):

        from designer.items.sprinkler_item import SprinklerItem
        from models.sprinkler import Sprinkler

        window, floor, zone = _make_window_with_zone()

        building = window.canvas.scene_obj.project.building
        system = building.create_fire_water_system("FW-1")

        model = Sprinkler(id="S1", name="S1", floor_id=floor.id, zone_ids=("Z-A",))
        item = SprinklerItem(250, 250, model=model)

        window.property_panel.show_sprinkler(item)

        index = window.property_panel.sprinkler_fire_water_system.findData(system.id)
        window.property_panel.sprinkler_fire_water_system.setCurrentIndex(index)

        self.assertEqual(system.sprinkler_ids, ("S1",))


class FireWaterSystemListTests(unittest.TestCase):

    def test_create_and_refresh_shows_system(self):

        window = MainWindow()

        window.canvas.scene_obj.project.building.create_fire_water_system("FW-1")
        window.fire_water_system_list.refresh()

        self.assertEqual(window.fire_water_system_list.list_widget.count(), 1)

    def test_rename_and_delete_via_building_methods(self):

        window = MainWindow()

        building = window.canvas.scene_obj.project.building
        system = building.create_fire_water_system("FW-1")
        window.fire_water_system_list.refresh()

        building.rename_fire_water_system(system, "Renamed")
        window.fire_water_system_list.refresh()
        self.assertEqual(window.fire_water_system_list.list_widget.item(0).text(), "Renamed")

        building.remove_fire_water_system(system)
        window.fire_water_system_list.refresh()
        self.assertEqual(window.fire_water_system_list.list_widget.count(), 0)


class SaveReloadTests(unittest.TestCase):

    def test_save_reload_preserves_all_four_asset_types_and_system_membership(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_water_tank_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        window.toolbar.fire_pump_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(300, 300))

        window.toolbar.jockey_pump_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(350, 350))

        window.toolbar.fire_service_inlet_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(400, 400))

        tank_before = floor.fire_water_tanks[0]
        pump_before = floor.fire_pumps[0]

        building = window.canvas.scene_obj.project.building
        system = building.create_fire_water_system("FW-1")
        system.tank_ids = (tank_before.id,)
        system.pump_ids = (pump_before.id,)

        pump_before.running = True

        data = window.canvas.scene_obj.project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(floor.id)

        self.assertEqual(restored_floor.fire_water_tank_count, 1)
        self.assertEqual(restored_floor.fire_pump_count, 1)
        self.assertEqual(restored_floor.jockey_pump_count, 1)
        self.assertEqual(restored_floor.fire_service_inlet_count, 1)

        self.assertEqual(restored_floor.fire_water_tanks[0].zone_ids, ("Z-A",))

        restored_pump = restored_floor.fire_pumps[0]
        self.assertTrue(restored_pump.running)

        self.assertEqual(len(restored.building.fire_water_systems), 1)
        restored_system = restored.building.fire_water_systems[0]
        self.assertEqual(restored_system.tank_ids, (tank_before.id,))
        self.assertEqual(restored_system.pump_ids, (pump_before.id,))


if __name__ == "__main__":
    unittest.main()
