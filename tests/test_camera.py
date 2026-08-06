import sys
import unittest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

# Module-level QApplication singleton -- same convention every other
# PyQt6-backed test module in this repo already establishes (see
# tests/test_designer_main_window.py, tests/training_dataset_fixtures.py).
_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.camera_item import CameraItem
from designer.widgets.property_panel import PropertyPanel

from models.building import Building
from models.camera import Camera
from models.engineering_asset import ConnectionInfo, DeviceMode
from models.floor import Floor
from models.zone import Zone


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


class CameraModelTests(unittest.TestCase):

    def test_defaults_match_the_pre_framework_shape(self):

        camera = Camera()

        self.assertEqual(camera.object_type, "Camera")
        self.assertEqual(camera.position, (0.0, 0.0))
        self.assertEqual(camera.rotation, 0.0)
        self.assertEqual(camera.horizontal_fov, 90.0)
        self.assertEqual(camera.max_range, 25.0)
        self.assertEqual(camera.mount_height, 3.0)
        self.assertTrue(camera.active)

    def test_new_logical_fields_have_sensible_defaults(self):

        camera = Camera()

        self.assertEqual(camera.zone_ids, ())
        self.assertEqual(camera.resolution, "1920x1080")
        self.assertEqual(camera.fps, 15)
        self.assertEqual(camera.mode, DeviceMode.SIMULATION)
        self.assertEqual(camera.connection, ConnectionInfo())

    def test_coverage_polygon_unchanged_by_the_refactor(self):

        # A camera facing along +x (rotation=0) with a 90-degree FOV
        # and 10m range -- the arc should span from -45 to +45 degrees
        # around the position, exactly as it did before Camera was
        # rebased onto EngineeringAsset.
        camera = Camera(position=(0.0, 0.0), rotation=0.0, horizontal_fov=90.0, max_range=10.0)

        polygon = camera.coverage_polygon(segments=4)

        self.assertEqual(polygon[0], (0.0, 0.0))
        self.assertEqual(len(polygon), 6)

        last_x, last_y = polygon[-1]
        self.assertAlmostEqual(last_x, 10.0 * 0.7071067811865476, places=6)
        self.assertAlmostEqual(last_y, 10.0 * 0.7071067811865476, places=6)

    def test_to_dict_from_dict_round_trip_including_new_fields(self):

        camera = Camera(
            name="Lobby Cam",
            position=(4.0, 5.0),
            floor_id="floor-1",
            zone_ids=("zone-a",),
            rotation=30.0,
            horizontal_fov=110.0,
            max_range=40.0,
            mount_height=2.4,
            active=False,
            mode=DeviceMode.LIVE,
            resolution="1280x720",
            fps=30,
            connection=ConnectionInfo(
                rtsp_address="rtsp://10.0.0.9/stream",
                ip_address="10.0.0.9",
                username="admin",
                password="secret",
            ),
        )

        restored = Camera.from_dict(camera.to_dict())

        self.assertEqual(restored.name, "Lobby Cam")
        self.assertEqual(restored.position, (4.0, 5.0))
        self.assertEqual(restored.floor_id, "floor-1")
        self.assertEqual(restored.zone_ids, ("zone-a",))
        self.assertEqual(restored.rotation, 30.0)
        self.assertEqual(restored.horizontal_fov, 110.0)
        self.assertEqual(restored.max_range, 40.0)
        self.assertEqual(restored.mount_height, 2.4)
        self.assertFalse(restored.active)
        self.assertEqual(restored.mode, DeviceMode.LIVE)
        self.assertEqual(restored.resolution, "1280x720")
        self.assertEqual(restored.fps, 30)
        self.assertEqual(restored.connection.rtsp_address, "rtsp://10.0.0.9/stream")
        self.assertEqual(restored.connection.username, "admin")

    def test_loading_a_pre_framework_camera_dict_still_works(self):

        # Exact shape Camera.to_dict() produced before this milestone --
        # no zone_ids/mode/connection/resolution/fps keys at all. A
        # project saved by an older build of SynEvac must still load.
        legacy_dict = {
            "id": "legacy-cam-1",
            "name": "Old Camera",
            "object_type": "Camera",
            "properties": {},
            "created_at": "2025-01-01T00:00:00",
            "modified_at": "2025-01-01T00:00:00",
            "position": (7.0, 8.0),
            "floor_id": "floor-2",
            "rotation": 15.0,
            "horizontal_fov": 75.0,
            "max_range": 20.0,
            "mount_height": 3.5,
            "active": True,
        }

        camera = Camera.from_dict(legacy_dict)

        self.assertEqual(camera.id, "legacy-cam-1")
        self.assertEqual(camera.position, (7.0, 8.0))
        self.assertEqual(camera.floor_id, "floor-2")
        self.assertEqual(camera.rotation, 15.0)
        self.assertEqual(camera.horizontal_fov, 75.0)
        self.assertEqual(camera.max_range, 20.0)

        # New fields fall back to their documented defaults rather than
        # raising or leaving the attribute unset.
        self.assertEqual(camera.zone_ids, ())
        self.assertEqual(camera.resolution, "1920x1080")
        self.assertEqual(camera.fps, 15)
        self.assertEqual(camera.mode, DeviceMode.SIMULATION)
        self.assertEqual(camera.connection, ConnectionInfo())


class FloorCameraSerializationTests(unittest.TestCase):

    def test_floor_round_trips_a_camera_with_every_new_field_set(self):

        floor = Floor(name="Ground Floor")

        camera = Camera(
            name="Entrance Cam",
            floor_id=floor.id,
            zone_ids=("zone-x", "zone-y"),
            mode=DeviceMode.REPLAY,
            resolution="3840x2160",
            fps=25,
            connection=ConnectionInfo(ip_address="192.168.0.50"),
        )

        floor.add_camera(camera)

        restored_floor = Floor.from_dict(floor.to_dict())

        self.assertEqual(len(restored_floor.cameras), 1)

        restored_camera = restored_floor.cameras[0]

        self.assertEqual(restored_camera.zone_ids, ("zone-x", "zone-y"))
        self.assertEqual(restored_camera.mode, DeviceMode.REPLAY)
        self.assertEqual(restored_camera.resolution, "3840x2160")
        self.assertEqual(restored_camera.fps, 25)
        self.assertEqual(restored_camera.connection.ip_address, "192.168.0.50")

    def test_floor_with_a_legacy_shaped_camera_dict_still_loads(self):

        # An entire pre-milestone Floor.to_dict() output, with one
        # camera in the old shape -- proves an existing saved project
        # containing cameras still opens without error.
        floor_data = {
            "id": "floor-legacy",
            "name": "Ground Floor",
            "display_order": 0,
            "height": 3.0,
            "floor_plan": "",
            "visible": True,
            "locked": False,
            "zones": [],
            "exits": [],
            "stairs": [],
            "elevators": [],
            "cameras": [
                {
                    "id": "cam-legacy",
                    "name": "Legacy Cam",
                    "object_type": "Camera",
                    "properties": {},
                    "created_at": "",
                    "modified_at": "",
                    "position": (1.0, 1.0),
                    "floor_id": "floor-legacy",
                    "rotation": 0.0,
                    "horizontal_fov": 90.0,
                    "max_range": 25.0,
                    "mount_height": 3.0,
                    "active": True,
                }
            ],
            "detectors": [],
            "assembly_points": [],
            "obstacles": [],
            "doors": [],
        }

        floor = Floor.from_dict(floor_data)

        self.assertEqual(len(floor.cameras), 1)
        self.assertEqual(floor.cameras[0].id, "cam-legacy")
        self.assertEqual(floor.cameras[0].zone_ids, ())


class CameraPropertyPanelTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Lobby Zone", x=0.0, y=0.0)
        self.floor.add_zone(self.zone)

        self.camera = Camera(
            name="Cam 1", position=(2.0, 3.0), floor_id=self.floor.id, rotation=10.0,
        )
        self.floor.add_camera(self.camera)

        self.camera_item = CameraItem(2.0 * 50, 3.0 * 50, model=self.camera)

        self.panel = PropertyPanel()
        self.panel.set_building(self.building)

    def test_show_camera_populates_every_new_field(self):

        self.camera.zone_ids = (self.zone.id,)
        self.camera.mode = DeviceMode.LIVE
        self.camera.resolution = "1280x720"
        self.camera.fps = 24
        self.camera.connection = ConnectionInfo(
            rtsp_address="rtsp://cam/1", ip_address="1.2.3.4", username="u", password="p",
        )

        self.panel.show_camera(self.camera_item)

        checked = {
            self.panel.camera_zone.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.panel.camera_zone.count())
            if self.panel.camera_zone.item(row).checkState() == Qt.CheckState.Checked
        }
        self.assertEqual(checked, {self.zone.id})
        self.assertEqual(self.panel.camera_resolution.text(), "1280x720")
        self.assertEqual(self.panel.camera_fps.text(), "24")
        self.assertEqual(self.panel.camera_mode.currentText(), DeviceMode.LIVE)
        self.assertEqual(self.panel.camera_rtsp.text(), "rtsp://cam/1")
        self.assertEqual(self.panel.camera_ip.text(), "1.2.3.4")
        self.assertEqual(self.panel.camera_username.text(), "u")
        self.assertEqual(self.panel.camera_password.text(), "p")

    def test_checking_the_zone_checklist_assigns_the_zone_to_the_model(self):

        self.panel.show_camera(self.camera_item)

        list_widget = self.panel.camera_zone
        matching_rows = [
            row for row in range(list_widget.count())
            if list_widget.item(row).data(Qt.ItemDataRole.UserRole) == self.zone.id
        ]
        self.assertEqual(len(matching_rows), 1)

        list_widget.item(matching_rows[0]).setCheckState(Qt.CheckState.Checked)

        self.assertEqual(self.camera.zone_ids, (self.zone.id,))

    def test_editing_metadata_fields_updates_the_model(self):

        self.panel.show_camera(self.camera_item)

        self.panel.camera_resolution.setText("640x480")
        self.panel.camera_resolution.editingFinished.emit()

        self.panel.camera_fps.setText("60")
        self.panel.camera_fps.editingFinished.emit()

        self.assertEqual(self.camera.resolution, "640x480")
        self.assertEqual(self.camera.fps, 60)

    def test_selecting_a_mode_updates_the_model(self):

        self.panel.show_camera(self.camera_item)

        index = self.panel.camera_mode.findText(DeviceMode.REPLAY)
        self.panel.camera_mode.setCurrentIndex(index)

        self.assertEqual(self.camera.mode, DeviceMode.REPLAY)

    def test_editing_connection_fields_updates_the_model(self):

        self.panel.show_camera(self.camera_item)

        self.panel.camera_rtsp.setText("rtsp://new")
        self.panel.camera_rtsp.editingFinished.emit()

        self.panel.camera_ip.setText("9.9.9.9")
        self.panel.camera_ip.editingFinished.emit()

        self.panel.camera_username.setText("root")
        self.panel.camera_username.editingFinished.emit()

        self.panel.camera_password.setText("toor")
        self.panel.camera_password.editingFinished.emit()

        self.assertEqual(self.camera.connection.rtsp_address, "rtsp://new")
        self.assertEqual(self.camera.connection.ip_address, "9.9.9.9")
        self.assertEqual(self.camera.connection.username, "root")
        self.assertEqual(self.camera.connection.password, "toor")

    def test_password_field_uses_password_echo_mode(self):

        from PyQt6.QtWidgets import QLineEdit

        self.assertEqual(
            self.panel.camera_password.echoMode(),
            QLineEdit.EchoMode.Password,
        )

    def test_clear_resets_every_new_field(self):

        self.panel.show_camera(self.camera_item)
        self.panel.clear()

        self.assertEqual(self.panel.camera_resolution.text(), "")
        self.assertEqual(self.panel.camera_fps.text(), "")
        self.assertEqual(self.panel.camera_rtsp.text(), "")
        self.assertEqual(self.panel.camera_ip.text(), "")
        self.assertEqual(self.panel.camera_username.text(), "")
        self.assertEqual(self.panel.camera_password.text(), "")
        self.assertEqual(self.panel.camera_visible_zones.text(), "-")
        self.assertEqual(self.panel.camera_partial_zones.text(), "-")
        self.assertEqual(self.panel.camera_hidden_zones.text(), "-")
        self.assertEqual(self.panel.camera_max_visible_distance.text(), "-")


class CameraVisibilityStatsTests(unittest.TestCase):

    # Camera Coverage & Visibility Engine -- the Property Panel's
    # read-only stats fields (see visibility/engine.py::
    # VisibilityEngine). Uses a camera placed squarely inside its own
    # zone, facing into it, so the exact counts are unambiguous.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Room", x=0.0, y=0.0, width=10.0, height=10.0)
        self.floor.add_zone(self.zone)

        self.camera = Camera(
            name="Cam", position=(5.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=25.0,
        )
        self.floor.add_camera(self.camera)

        self.camera_item = CameraItem(5.0 * 50, 5.0 * 50, model=self.camera)

        self.panel = PropertyPanel()

    def test_stats_show_placeholder_without_a_building(self):

        self.panel.show_camera(self.camera_item)

        self.assertEqual(self.panel.camera_visible_zones.text(), "-")
        self.assertEqual(self.panel.camera_partial_zones.text(), "-")
        self.assertEqual(self.panel.camera_hidden_zones.text(), "-")
        self.assertEqual(self.panel.camera_max_visible_distance.text(), "-")

    def test_stats_are_computed_once_a_building_is_set(self):

        self.panel.set_building(self.building)
        self.panel.show_camera(self.camera_item)

        self.assertEqual(self.panel.camera_visible_zones.text(), "1")
        self.assertEqual(self.panel.camera_partial_zones.text(), "0")
        self.assertEqual(self.panel.camera_hidden_zones.text(), "0")
        self.assertNotEqual(self.panel.camera_max_visible_distance.text(), "-")

    def test_on_visual_change_fires_after_a_geometry_edit(self):

        self.panel.set_building(self.building)
        self.panel.show_camera(self.camera_item)

        calls = []
        self.panel.on_visual_change = lambda: calls.append(True)

        self.panel.camera_fov.setText("45.0")
        self.panel.update_camera_geometry()

        self.assertEqual(len(calls), 1)

    def test_on_visual_change_fires_after_toggling_active(self):

        self.panel.set_building(self.building)
        self.panel.show_camera(self.camera_item)

        calls = []
        self.panel.on_visual_change = lambda: calls.append(True)

        # setChecked() alone already fires the connected `toggled`
        # signal straight to update_camera_active() -- no separate
        # manual call needed (unlike update_camera_geometry(), which
        # editingFinished doesn't auto-trigger in this headless test
        # since no real keystroke/focus-out event occurs).
        self.panel.camera_active.setChecked(False)

        self.assertEqual(len(calls), 1)

    def test_on_visual_change_is_optional(self):

        # None (the default) must never crash a call site -- every
        # handler that fires it guards for it.
        self.panel.set_building(self.building)
        self.panel.show_camera(self.camera_item)

        self.panel.camera_fov.setText("45.0")
        self.panel.update_camera_geometry()


if __name__ == "__main__":
    unittest.main()
