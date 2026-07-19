import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.heat_detector_item import HeatDetectorItem
from designer.items.smoke_detector_item import SmokeDetectorItem
from designer.scene.graphics_scene import GraphicsScene
from designer.widgets.property_panel import PropertyPanel

from models.building import Building
from models.engineering_asset import DeviceMode
from models.heat_detector import HeatDetector
from models.sensor_asset import DetectorState, HealthStatus
from models.smoke_detector import SmokeDetector


class SmokeDetectorToolTests(unittest.TestCase):

    def setUp(self):

        self.scene = GraphicsScene()
        self.floor = self.scene.current_floor

    def test_placing_a_smoke_detector_adds_it_to_the_floor_and_scene(self):

        self.scene.set_tool("smoke_detector")

        from PyQt6.QtCore import QPointF
        from PyQt6.QtWidgets import QGraphicsSceneMouseEvent

        # GraphicsScene's own mousePressEvent only reads event.scenePos(),
        # so a plain object with that one attribute is enough here --
        # matches this test module's own minimal-fixture convention.
        class FakeEvent:
            def scenePos(self):
                return QPointF(100.0, 150.0)

        self.scene.mousePressEvent(FakeEvent())

        self.assertEqual(self.floor.smoke_detector_count, 1)

        items = [item for item in self.scene.items() if isinstance(item, SmokeDetectorItem)]
        self.assertEqual(len(items), 1)
        self.assertIs(items[0].model, self.floor.smoke_detectors[0])

    def test_placing_a_heat_detector_adds_it_to_the_floor_and_scene(self):

        self.scene.set_tool("heat_detector")

        from PyQt6.QtCore import QPointF

        class FakeEvent:
            def scenePos(self):
                return QPointF(200.0, 250.0)

        self.scene.mousePressEvent(FakeEvent())

        self.assertEqual(self.floor.heat_detector_count, 1)

        items = [item for item in self.scene.items() if isinstance(item, HeatDetectorItem)]
        self.assertEqual(len(items), 1)
        self.assertIs(items[0].model, self.floor.heat_detectors[0])

    def test_rebuild_scene_renders_previously_placed_sensors(self):

        smoke = SmokeDetector(name="Smoke", floor_id=self.floor.id, position=(1.0, 1.0))
        heat = HeatDetector(name="Heat", floor_id=self.floor.id, position=(2.0, 2.0))

        self.floor.add_smoke_detector(smoke)
        self.floor.add_heat_detector(heat)

        self.scene.rebuild_scene()

        smoke_items = [item for item in self.scene.items() if isinstance(item, SmokeDetectorItem)]
        heat_items = [item for item in self.scene.items() if isinstance(item, HeatDetectorItem)]

        self.assertEqual(len(smoke_items), 1)
        self.assertEqual(len(heat_items), 1)
        self.assertIs(smoke_items[0].model, smoke)
        self.assertIs(heat_items[0].model, heat)

    def test_clear_graphics_items_removes_both_sensor_item_types(self):

        smoke = SmokeDetector(name="Smoke", floor_id=self.floor.id)
        self.floor.add_smoke_detector(smoke)
        self.scene.rebuild_scene()

        self.assertTrue(any(isinstance(i, SmokeDetectorItem) for i in self.scene.items()))

        self.scene.clear_graphics_items()

        self.assertFalse(any(isinstance(i, SmokeDetectorItem) for i in self.scene.items()))


class SmokeDetectorPropertyPanelTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.sensor = SmokeDetector(
            name="Smoke 1", floor_id=self.floor.id, position=(1.0, 2.0),
            activation_threshold=0.25,
        )
        self.floor.add_smoke_detector(self.sensor)

        self.item = SmokeDetectorItem(1.0 * 50, 2.0 * 50, model=self.sensor)

        self.panel = PropertyPanel()
        self.panel.set_building(self.building)

    def test_show_smoke_detector_populates_fields(self):

        self.panel.show_smoke_detector(self.item)

        self.assertEqual(self.panel.smoke_detector_x.text(), "1.00")
        self.assertEqual(self.panel.smoke_detector_y.text(), "2.00")
        self.assertTrue(self.panel.smoke_detector_active.isChecked())
        self.assertEqual(self.panel.smoke_detector_health.currentText(), HealthStatus.OK)
        self.assertEqual(self.panel.smoke_detector_mode.currentText(), DeviceMode.SIMULATION)
        self.assertEqual(self.panel.smoke_detector_threshold.text(), "0.25")

    def test_current_state_reflects_test_smoke_level_against_threshold(self):

        self.panel.show_smoke_detector(self.item)

        self.panel.smoke_detector_test_level.setText("0.5")
        self.panel.smoke_detector_test_level.editingFinished.emit()

        self.assertEqual(self.panel.smoke_detector_state.text(), DetectorState.ALARM.name)
        self.assertEqual(self.item.current_state, DetectorState.ALARM)

        self.panel.smoke_detector_test_level.setText("0.0")
        self.panel.smoke_detector_test_level.editingFinished.emit()

        self.assertEqual(self.panel.smoke_detector_state.text(), DetectorState.NORMAL.name)

    def test_editing_health_status_updates_the_model_and_state(self):

        self.panel.show_smoke_detector(self.item)

        fault_index = self.panel.smoke_detector_health.findText(HealthStatus.FAULT)
        self.panel.smoke_detector_health.setCurrentIndex(fault_index)

        self.assertEqual(self.sensor.health_status, HealthStatus.FAULT)
        self.assertEqual(self.panel.smoke_detector_state.text(), DetectorState.FAULT.name)

    def test_editing_mode_updates_the_model(self):

        self.panel.show_smoke_detector(self.item)

        live_index = self.panel.smoke_detector_mode.findText(DeviceMode.LIVE)
        self.panel.smoke_detector_mode.setCurrentIndex(live_index)

        self.assertEqual(self.sensor.mode, DeviceMode.LIVE)

    def test_editing_threshold_updates_the_model(self):

        self.panel.show_smoke_detector(self.item)

        self.panel.smoke_detector_threshold.setText("0.6")
        self.panel.smoke_detector_threshold.editingFinished.emit()

        self.assertEqual(self.sensor.activation_threshold, 0.6)

    def test_editing_installation_date_updates_the_model(self):

        self.panel.show_smoke_detector(self.item)

        self.panel.smoke_detector_installation_date.setText("2026-05-01")
        self.panel.smoke_detector_installation_date.editingFinished.emit()

        self.assertEqual(self.sensor.installation_date, "2026-05-01")

    def test_toggling_active_off_forces_normal_state_regardless_of_reading(self):

        self.panel.show_smoke_detector(self.item)

        self.panel.smoke_detector_test_level.setText("0.9")
        self.panel.smoke_detector_test_level.editingFinished.emit()
        self.assertEqual(self.panel.smoke_detector_state.text(), DetectorState.ALARM.name)

        self.panel.smoke_detector_active.setChecked(False)

        self.assertFalse(self.sensor.active)
        self.assertEqual(self.panel.smoke_detector_state.text(), DetectorState.NORMAL.name)

    def test_clear_resets_every_field(self):

        self.panel.show_smoke_detector(self.item)
        self.panel.clear()

        self.assertEqual(self.panel.smoke_detector_x.text(), "")
        self.assertEqual(self.panel.smoke_detector_threshold.text(), "")
        self.assertEqual(self.panel.smoke_detector_state.text(), "-")


class HeatDetectorPropertyPanelTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.sensor = HeatDetector(
            name="Heat 1", floor_id=self.floor.id, activation_threshold=60.0,
        )
        self.floor.add_heat_detector(self.sensor)

        self.item = HeatDetectorItem(0.0, 0.0, model=self.sensor)

        self.panel = PropertyPanel()
        self.panel.set_building(self.building)

    def test_show_heat_detector_populates_fields(self):

        self.panel.show_heat_detector(self.item)

        self.assertEqual(self.panel.heat_detector_threshold.text(), "60.00")

    def test_current_state_reflects_test_temperature_against_threshold(self):

        self.panel.show_heat_detector(self.item)

        self.panel.heat_detector_test_temperature.setText("75")
        self.panel.heat_detector_test_temperature.editingFinished.emit()

        self.assertEqual(self.panel.heat_detector_state.text(), DetectorState.ALARM.name)
        self.assertEqual(self.sensor.last_activation_time, None)  # no `time` supplied by the panel

    def test_clear_resets_every_field(self):

        self.panel.show_heat_detector(self.item)
        self.panel.clear()

        self.assertEqual(self.panel.heat_detector_threshold.text(), "")
        self.assertEqual(self.panel.heat_detector_state.text(), "-")


if __name__ == "__main__":
    unittest.main()
