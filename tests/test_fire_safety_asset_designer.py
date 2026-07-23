import sys
import unittest

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.sprinkler_item import SprinklerItem
from designer.items.fire_extinguisher_item import FireExtinguisherItem
from designer.items.fire_hydrant_item import FireHydrantItem
from designer.items.hose_reel_item import HoseReelItem
from designer.windows.main_window import MainWindow

from models.fire_extinguisher import FireExtinguisher
from models.fire_hydrant import FireHydrant
from models.hose_reel import HoseReel
from models.sprinkler import Sprinkler
from models.zone import Zone
from models.project import Project


# =====================================================
# Fire Suppression & Water-Based Safety Asset Digital Twin milestone --
# Designer placement/property-panel/zone-assignment/save-reload tests
# for Sprinkler/FireExtinguisher/FireHydrant/HoseReel, mirroring the
# established tests.test_manual_call_point_designer /
# tests.test_emergency_light_designer pattern exactly.
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
            "sprinkler_action", "fire_extinguisher_action", "fire_hydrant_action", "hose_reel_action",
        ):
            action = getattr(window.toolbar, action_name)
            self.assertGreater(action.receivers(action.triggered), 0)


class PlacementTests(unittest.TestCase):

    def test_click_to_place_creates_a_sprinkler(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.sprinkler_action.trigger()
        self.assertEqual(window.canvas.scene_obj.current_tool, "sprinkler")

        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.sprinkler_count, 1)
        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, SprinklerItem))
        self.assertIs(item.model, floor.sprinklers[0])

    def test_click_to_place_creates_a_fire_extinguisher(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_extinguisher_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.fire_extinguisher_count, 1)
        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, FireExtinguisherItem))
        self.assertIs(item.model, floor.fire_extinguishers[0])

    def test_click_to_place_creates_a_fire_hydrant(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_hydrant_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.fire_hydrant_count, 1)
        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, FireHydrantItem))
        self.assertIs(item.model, floor.fire_hydrants[0])

    def test_click_to_place_creates_a_hose_reel(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.hose_reel_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.hose_reel_count, 1)
        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, HoseReelItem))
        self.assertIs(item.model, floor.hose_reels[0])


class ZoneAutoAssignmentTests(unittest.TestCase):

    def test_sprinkler_auto_assigned_inside_a_single_zone(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.sprinkler_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.sprinklers[0].zone_ids, ("Z-A",))

    def test_sprinkler_outside_every_zone_stays_unassigned(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.sprinkler_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(5000, 5000))

        self.assertEqual(floor.sprinklers[0].zone_ids, ())

    def test_sprinkler_ambiguous_overlapping_zones_stay_unassigned(self):

        window, floor, zone = _make_window_with_zone()
        floor.add_zone(Zone(id="Z-B", name="Zone B", floor_id=floor.id, x=0.0, y=0.0, width=10.0, height=10.0))

        window.toolbar.sprinkler_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.sprinklers[0].zone_ids, ())

    def test_fire_extinguisher_auto_assigned_inside_a_single_zone(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_extinguisher_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.fire_extinguishers[0].zone_ids, ("Z-A",))

    def test_fire_hydrant_auto_assigned_inside_a_single_zone(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.fire_hydrant_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.fire_hydrants[0].zone_ids, ("Z-A",))

    def test_hose_reel_auto_assigned_inside_a_single_zone(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.hose_reel_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.hose_reels[0].zone_ids, ("Z-A",))

    def test_manual_reassignment_still_possible(self):

        window, floor, zone = _make_window_with_zone()
        floor.add_zone(Zone(id="Z-B", name="Zone B", floor_id=floor.id, x=20.0, y=0.0, width=10.0, height=10.0))

        window.toolbar.sprinkler_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        item = next(i for i in window.canvas.scene_obj.items() if isinstance(i, SprinklerItem))
        window.property_panel.show_sprinkler(item)

        index = window.property_panel.sprinkler_zone.findData("Z-B")
        window.property_panel.sprinkler_zone.setCurrentIndex(index)

        self.assertEqual(item.model.zone_ids, ("Z-B",))


class SprinklerPropertyPanelTests(unittest.TestCase):

    def test_shows_current_zone_and_state(self):

        window, floor, zone = _make_window_with_zone()

        model = Sprinkler(id="S1", name="S1", floor_id=floor.id, zone_ids=("Z-A",))
        item = SprinklerItem(250, 250, model=model)

        window.property_panel.show_sprinkler(item)

        self.assertEqual(
            window.property_panel.sprinkler_zone.itemData(window.property_panel.sprinkler_zone.currentIndex()), "Z-A",
        )
        self.assertEqual(window.property_panel.sprinkler_state.text(), "NORMAL")
        self.assertTrue(window.property_panel.sprinkler_zone_warning.isHidden())

    def test_test_temperature_below_threshold_shows_normal(self):

        window, floor, zone = _make_window_with_zone()

        model = Sprinkler(id="S1", name="S1", floor_id=floor.id, zone_ids=("Z-A",))
        item = SprinklerItem(250, 250, model=model)

        window.property_panel.show_sprinkler(item)
        window.property_panel.sprinkler_test_temperature.setText("40")
        window.property_panel.update_sprinkler_test_reading()

        self.assertEqual(window.property_panel.sprinkler_state.text(), "NORMAL")

    def test_test_temperature_above_threshold_shows_activated(self):

        window, floor, zone = _make_window_with_zone()

        model = Sprinkler(id="S1", name="S1", floor_id=floor.id, zone_ids=("Z-A",))
        item = SprinklerItem(250, 250, model=model)

        window.property_panel.show_sprinkler(item)
        window.property_panel.sprinkler_test_temperature.setText("90")
        window.property_panel.update_sprinkler_test_reading()

        self.assertEqual(window.property_panel.sprinkler_state.text(), "ACTIVATED")

    def test_fault_health_shows_fault_regardless_of_temperature(self):

        window, floor, zone = _make_window_with_zone()

        model = Sprinkler(id="S1", name="S1", floor_id=floor.id, zone_ids=("Z-A",))
        item = SprinklerItem(250, 250, model=model)

        window.property_panel.show_sprinkler(item)

        index = window.property_panel.sprinkler_health.findText("Fault")
        window.property_panel.sprinkler_health.setCurrentIndex(index)

        window.property_panel.sprinkler_test_temperature.setText("90")
        window.property_panel.update_sprinkler_test_reading()

        self.assertEqual(window.property_panel.sprinkler_state.text(), "FAULT")

    def test_warning_visible_when_unassigned(self):

        window, floor, zone = _make_window_with_zone()

        model = Sprinkler(id="S1", name="S1", floor_id=floor.id)
        item = SprinklerItem(250, 250, model=model)

        window.property_panel.show_sprinkler(item)

        self.assertFalse(window.property_panel.sprinkler_zone_warning.isHidden())


class PassiveAssetPropertyPanelTests(unittest.TestCase):

    def test_fire_extinguisher_shows_availability_and_type(self):

        window, floor, zone = _make_window_with_zone()

        model = FireExtinguisher(id="E1", name="E1", floor_id=floor.id, zone_ids=("Z-A",))
        item = FireExtinguisherItem(250, 250, model=model)

        window.property_panel.show_fire_extinguisher(item)

        self.assertEqual(window.property_panel.fire_extinguisher_availability.text(), "AVAILABLE")

        index = window.property_panel.fire_extinguisher_type.findText("CO2")
        window.property_panel.fire_extinguisher_type.setCurrentIndex(index)

        self.assertEqual(model.extinguisher_type, "CO2")

    def test_fire_extinguisher_deactivating_updates_availability(self):

        window, floor, zone = _make_window_with_zone()

        model = FireExtinguisher(id="E1", name="E1", floor_id=floor.id, zone_ids=("Z-A",))
        item = FireExtinguisherItem(250, 250, model=model)

        window.property_panel.show_fire_extinguisher(item)
        window.property_panel.fire_extinguisher_active.setChecked(False)

        self.assertFalse(model.active)
        self.assertEqual(window.property_panel.fire_extinguisher_availability.text(), "UNAVAILABLE")

    def test_fire_hydrant_shows_availability_and_type(self):

        window, floor, zone = _make_window_with_zone()

        model = FireHydrant(id="H1", name="H1", floor_id=floor.id, zone_ids=("Z-A",))
        item = FireHydrantItem(250, 250, model=model)

        window.property_panel.show_fire_hydrant(item)

        self.assertEqual(window.property_panel.fire_hydrant_availability.text(), "AVAILABLE")

        index = window.property_panel.fire_hydrant_type.findText("External Hydrant")
        window.property_panel.fire_hydrant_type.setCurrentIndex(index)

        self.assertEqual(model.hydrant_type, "External Hydrant")

    def test_hose_reel_shows_availability(self):

        window, floor, zone = _make_window_with_zone()

        model = HoseReel(id="HR1", name="HR1", floor_id=floor.id, zone_ids=("Z-A",))
        item = HoseReelItem(250, 250, model=model)

        window.property_panel.show_hose_reel(item)

        self.assertEqual(window.property_panel.hose_reel_availability.text(), "AVAILABLE")

    def test_warning_visible_when_unassigned_for_all_three(self):

        window, floor, zone = _make_window_with_zone()

        extinguisher_item = FireExtinguisherItem(250, 250, model=FireExtinguisher(id="E1", name="E1", floor_id=floor.id))
        window.property_panel.show_fire_extinguisher(extinguisher_item)
        self.assertFalse(window.property_panel.fire_extinguisher_zone_warning.isHidden())

        hydrant_item = FireHydrantItem(250, 250, model=FireHydrant(id="H1", name="H1", floor_id=floor.id))
        window.property_panel.show_fire_hydrant(hydrant_item)
        self.assertFalse(window.property_panel.fire_hydrant_zone_warning.isHidden())

        hose_reel_item = HoseReelItem(250, 250, model=HoseReel(id="HR1", name="HR1", floor_id=floor.id))
        window.property_panel.show_hose_reel(hose_reel_item)
        self.assertFalse(window.property_panel.hose_reel_zone_warning.isHidden())


class SaveReloadTests(unittest.TestCase):

    def test_save_reload_preserves_all_four_asset_types(self):

        window, floor, zone = _make_window_with_zone()

        window.toolbar.sprinkler_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        window.toolbar.fire_extinguisher_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(300, 300))

        window.toolbar.fire_hydrant_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(350, 350))

        window.toolbar.hose_reel_action.trigger()
        window.canvas.scene_obj.mousePressEvent(_FakeSceneMouseEvent(400, 400))

        sprinkler_before = floor.sprinklers[0]
        sprinkler_before.activation_temperature = 74.0

        data = window.canvas.scene_obj.project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(floor.id)

        self.assertEqual(restored_floor.sprinkler_count, 1)
        self.assertEqual(restored_floor.fire_extinguisher_count, 1)
        self.assertEqual(restored_floor.fire_hydrant_count, 1)
        self.assertEqual(restored_floor.hose_reel_count, 1)

        restored_sprinkler = restored_floor.sprinklers[0]
        self.assertEqual(restored_sprinkler.id, sprinkler_before.id)
        self.assertEqual(restored_sprinkler.zone_ids, ("Z-A",))
        self.assertEqual(restored_sprinkler.activation_temperature, 74.0)

        self.assertEqual(restored_floor.fire_extinguishers[0].zone_ids, ("Z-A",))
        self.assertEqual(restored_floor.fire_hydrants[0].zone_ids, ("Z-A",))
        self.assertEqual(restored_floor.hose_reels[0].zone_ids, ("Z-A",))


if __name__ == "__main__":
    unittest.main()
