import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from builder.windows.builder_main_window import BuilderMainWindow

from designer.items.camera_item import CameraItem
from designer.items.door_item import DoorItem
from designer.items.exit_item import ExitItem
from designer.items.heat_detector_item import HeatDetectorItem
from designer.items.obstacle_item import ObstacleItem
from designer.items.smoke_detector_item import SmokeDetectorItem
from designer.items.speaker_item import SpeakerItem
from designer.items.stair_item import StairItem
from designer.items.zone_rectangle import ZoneRectangle

from models.camera import Camera
from models.door import Door
from models.exit import Exit
from models.heat_detector import HeatDetector
from models.obstacle import Obstacle
from models.smoke_detector import SmokeDetector
from models.speaker import Speaker
from models.staircase import Staircase
from models.zone import Zone


def _make_window_with_two_zones():

    window = BuilderMainWindow()
    floor = window.canvas.scene_obj.current_floor

    zone_a = Zone(id="Z-A", name="Zone A", floor_id=floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
    zone_b = Zone(id="Z-B", name="Zone B", floor_id=floor.id, x=20.0, y=0.0, width=10.0, height=10.0)

    floor.add_zone(zone_a)
    floor.add_zone(zone_b)

    window.property_panel.building = window.canvas.scene_obj.project.building

    return window, floor, zone_a, zone_b


class ZonePropertyTests(unittest.TestCase):

    def test_geometry_edit_resizes_item_and_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        item = ZoneRectangle(0, 0, 10 * 50, 10 * 50, model=zone_a)

        window.property_panel.show_zone(item)

        window.property_panel.zone_length.setText("15.00")
        window.property_panel.zone_length.editingFinished.emit()

        self.assertAlmostEqual(zone_a.width, 15.0)

    def test_zone_type_combo_updates_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        item = ZoneRectangle(0, 0, 10 * 50, 10 * 50, model=zone_a)

        window.property_panel.show_zone(item)

        index = window.property_panel.zone_type.findText("Corridor")
        window.property_panel.zone_type.setCurrentIndex(index)

        self.assertEqual(zone_a.zone_type, "Corridor")


class DoorPropertyTests(unittest.TestCase):

    def test_zone_assignment_persists_to_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Door(id="D-1", name="Door 1", floor_id=floor.id, start_point=(10, 5), end_point=(20, 5))
        item = DoorItem(500, 250, 1000, 250, model=model)

        window.property_panel.show_door(item)

        index_a = window.property_panel.door_zone_a.findData("Z-A")
        window.property_panel.door_zone_a.setCurrentIndex(index_a)

        index_b = window.property_panel.door_zone_b.findData("Z-B")
        window.property_panel.door_zone_b.setCurrentIndex(index_b)

        self.assertEqual(model.zone_a_id, "Z-A")
        self.assertEqual(model.zone_b_id, "Z-B")

    def test_width_edit_updates_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Door(id="D-1", name="Door 1", floor_id=floor.id, start_point=(10, 5), end_point=(20, 5))
        item = DoorItem(500, 250, 1000, 250, model=model)

        window.property_panel.show_door(item)

        window.property_panel.door_width.setText("1.50")
        window.property_panel.door_width.editingFinished.emit()

        self.assertAlmostEqual(model.width, 1.5)


class ExitPropertyTests(unittest.TestCase):

    def test_zone_assignment_persists_to_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Exit(id="E-1", name="Exit 1", floor_id=floor.id, start_point=(0, 0), end_point=(1, 0))
        item = ExitItem(0, 0, 50, 0, model=model)

        window.property_panel.show_exit(item)

        index = window.property_panel.exit_zone.findData("Z-A")
        window.property_panel.exit_zone.setCurrentIndex(index)

        self.assertEqual(model.zone_id, "Z-A")

    def test_capacity_edit_updates_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Exit(id="E-1", name="Exit 1", floor_id=floor.id, start_point=(0, 0), end_point=(1, 0))
        item = ExitItem(0, 0, 50, 0, model=model)

        window.property_panel.show_exit(item)

        window.property_panel.exit_capacity.setText("120")
        window.property_panel.exit_capacity.editingFinished.emit()

        self.assertEqual(model.capacity, 120)


class StairPropertyTests(unittest.TestCase):

    def test_from_and_to_zone_assignment_persists(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        second_floor = window.canvas.scene_obj.project.building.create_floor(name="Second Floor")
        zone_c = Zone(id="Z-C", name="Zone C", floor_id=second_floor.id, x=0, y=0, width=5, height=5)
        second_floor.add_zone(zone_c)

        model = Staircase(
            id="S-1", name="Stair 1", from_position=(1, 1), to_position=(2, 2),
            from_floor_id=floor.id, to_floor_id=second_floor.id,
        )

        item = StairItem(50, 50, 1.5, "from", model=model)

        window.property_panel.show_stair(item)

        from_index = window.property_panel.stair_from_zone.findData("Z-A")
        window.property_panel.stair_from_zone.setCurrentIndex(from_index)

        to_index = window.property_panel.stair_to_zone.findData("Z-C")
        window.property_panel.stair_to_zone.setCurrentIndex(to_index)

        self.assertEqual(model.from_zone_id, "Z-A")
        self.assertEqual(model.to_zone_id, "Z-C")

    def test_changing_destination_floor_clears_to_zone(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        second_floor = window.canvas.scene_obj.project.building.create_floor(name="Second Floor")
        third_floor = window.canvas.scene_obj.project.building.create_floor(name="Third Floor")

        model = Staircase(
            id="S-1", name="Stair 1", from_position=(1, 1), to_position=(2, 2),
            from_floor_id=floor.id, to_floor_id=second_floor.id, to_zone_id="stale-id",
        )

        item = StairItem(50, 50, 1.5, "from", model=model)

        window.property_panel.show_stair(item)

        new_index = window.property_panel.stair_to_floor.findData(third_floor.id)
        window.property_panel.stair_to_floor.setCurrentIndex(new_index)

        self.assertEqual(model.to_floor_id, third_floor.id)
        self.assertEqual(model.to_zone_id, "")


class CameraPropertyTests(unittest.TestCase):

    def test_position_edit_moves_item_and_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Camera(id="C-1", name="Camera 1", floor_id=floor.id, position=(0, 0))
        item = CameraItem(0, 0, model=model)

        window.property_panel.show_camera(item)

        window.property_panel.camera_x.setText("3.00")
        window.property_panel.camera_y.setText("4.00")
        window.property_panel.camera_x.editingFinished.emit()

        self.assertAlmostEqual(model.position[0], 3.0)
        self.assertAlmostEqual(model.position[1], 4.0)

    def test_fov_and_range_edit_updates_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Camera(id="C-1", name="Camera 1", floor_id=floor.id, position=(0, 0))
        item = CameraItem(0, 0, model=model)

        window.property_panel.show_camera(item)

        window.property_panel.camera_fov.setText("120.0")
        window.property_panel.camera_range.setText("30.0")
        window.property_panel.camera_fov.editingFinished.emit()

        self.assertAlmostEqual(model.horizontal_fov, 120.0)
        self.assertAlmostEqual(model.max_range, 30.0)

    def test_zone_assignment(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Camera(id="C-1", name="Camera 1", floor_id=floor.id, position=(0, 0))
        item = CameraItem(0, 0, model=model)

        window.property_panel.show_camera(item)

        index = window.property_panel.camera_zone.findData("Z-B")
        window.property_panel.camera_zone.setCurrentIndex(index)

        self.assertEqual(model.zone_ids, ("Z-B",))


class SmokeDetectorPropertyTests(unittest.TestCase):

    def test_threshold_and_health_edit_updates_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = SmokeDetector(id="SD-1", name="SD-1", floor_id=floor.id, position=(0, 0))
        item = SmokeDetectorItem(0, 0, model=model)

        window.property_panel.show_smoke_detector(item)

        window.property_panel.smoke_threshold.setText("0.5")
        window.property_panel.smoke_threshold.editingFinished.emit()

        self.assertAlmostEqual(model.activation_threshold, 0.5)


class HeatDetectorPropertyTests(unittest.TestCase):

    def test_threshold_edit_updates_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = HeatDetector(id="HD-1", name="HD-1", floor_id=floor.id, position=(0, 0))
        item = HeatDetectorItem(0, 0, model=model)

        window.property_panel.show_heat_detector(item)

        window.property_panel.heat_threshold.setText("65.0")
        window.property_panel.heat_threshold.editingFinished.emit()

        self.assertAlmostEqual(model.activation_threshold, 65.0)


class SpeakerPropertyTests(unittest.TestCase):

    def test_multi_zone_checklist_persists_to_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Speaker(id="SP-1", name="Speaker 1", floor_id=floor.id, position=(0, 0))
        item = SpeakerItem(0, 0, model=model)

        window.property_panel.show_speaker(item)

        from PyQt6.QtCore import Qt

        for row in range(window.property_panel.speaker_zones.count()):

            list_item = window.property_panel.speaker_zones.item(row)
            list_item.setCheckState(Qt.CheckState.Checked)

        window.property_panel._update_speaker_zones()

        self.assertEqual(set(model.zone_ids), {"Z-A", "Z-B"})


class ObstaclePropertyTests(unittest.TestCase):

    def test_traversability_and_geometry_edit_updates_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Obstacle(id="O-1", name="Obstacle 1", floor_id=floor.id, x=0, y=0, length=2, width=2)
        item = ObstacleItem(0, 0, 2, 2, model=model)

        window.property_panel.show_obstacle(item)

        window.property_panel.obstacle_length.setText("4.00")
        window.property_panel.obstacle_length.editingFinished.emit()

        index = window.property_panel.obstacle_traversability.findText("Passable")
        window.property_panel.obstacle_traversability.setCurrentIndex(index)

        self.assertAlmostEqual(model.length, 4.0)
        self.assertEqual(model.traversability, "Passable")


class FloorPropertyTests(unittest.TestCase):

    def test_height_edit_updates_model_and_notifies_callback(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        window.property_panel.show_floor(floor)

        window.property_panel.floor_height.setText("4.20")
        window.property_panel.floor_height.editingFinished.emit()

        self.assertAlmostEqual(floor.height, 4.2)

    def test_uncalibrated_floor_shows_not_calibrated(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        window.property_panel.show_floor(floor)

        self.assertEqual(window.property_panel.floor_scale.text(), "Not Calibrated")

    def test_calibrated_floor_shows_scale(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        floor.set_scale_calibration((0, 0), (100, 0), 5.0, 20.0)

        window.property_panel.show_floor(floor)

        self.assertIn("20.00", window.property_panel.floor_scale.text())


class SelectionRoutingTests(unittest.TestCase):

    # BuilderMainWindow.on_selection_changed() must route each of the
    # nine supported item types (plus Zone) to the matching
    # BuilderPropertyPanel.show_*() method.

    def test_each_supported_item_type_routes_to_the_matching_show_method(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        cases = (
            (ZoneRectangle(0, 0, 100, 100, model=zone_a), "Zone"),
            (DoorItem(0, 0, 50, 0, model=Door(id="D", name="D", floor_id=floor.id)), "Door"),
            (ExitItem(0, 0, 50, 0, model=Exit(id="E", name="E", floor_id=floor.id)), "Exit"),
            (CameraItem(0, 0, model=Camera(id="C", name="C", floor_id=floor.id)), "Camera"),
            (SmokeDetectorItem(0, 0, model=SmokeDetector(id="SD", name="SD", floor_id=floor.id)), "Smoke Detector"),
            (HeatDetectorItem(0, 0, model=HeatDetector(id="HD", name="HD", floor_id=floor.id)), "Heat Detector"),
            (SpeakerItem(0, 0, model=Speaker(id="SP", name="SP", floor_id=floor.id)), "Speaker"),
            (ObstacleItem(0, 0, 2, 2, model=Obstacle(id="O", name="O", floor_id=floor.id)), "Obstacle"),
        )

        for item, expected_type_label in cases:

            window.on_selection_changed(item)

            self.assertEqual(window.property_panel.object_type.text(), expected_type_label)

    def test_none_selection_clears_panel(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        window.on_selection_changed(ZoneRectangle(0, 0, 100, 100, model=zone_a))
        window.on_selection_changed(None)

        self.assertEqual(window.property_panel.object_type.text(), "-")
