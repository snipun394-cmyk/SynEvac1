import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.widgets.camera_manager_panel import CameraManagerPanel

from camera_manager.connection_status import CameraConnectionState

from models.building import Building
from models.camera import Camera
from models.engineering_asset import ConnectionInfo, DeviceMode
from models.zone import Zone


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=5.0, height=5.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


class CameraManagerPanelTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor_1 = self.building.create_floor(name="Ground Floor")
        self.floor_2 = self.building.create_floor(name="Floor 1", height=3.0)

        self.zone_a = make_zone("Zone A", floor_id=self.floor_1.id)
        self.zone_b = make_zone("Zone B", x=10.0, floor_id=self.floor_1.id)
        self.floor_1.add_zone(self.zone_a)
        self.floor_1.add_zone(self.zone_b)

        self.camera_1 = Camera(
            name="Cam 1", floor_id=self.floor_1.id, zone_ids=(self.zone_a.id,),
        )
        self.camera_2 = Camera(
            name="Cam 2", floor_id=self.floor_1.id, zone_ids=(self.zone_b.id,),
        )
        self.camera_3 = Camera(name="Cam 3", floor_id=self.floor_2.id)

        self.floor_1.add_camera(self.camera_1)
        self.floor_1.add_camera(self.camera_2)
        self.floor_2.add_camera(self.camera_3)

        self.panel = CameraManagerPanel()

    def test_refresh_lists_every_camera_across_every_floor(self):

        self.panel.refresh(self.building)

        self.assertEqual(self.panel.camera_table.rowCount(), 3)
        self.assertEqual(len(self.panel.manager.all_cameras()), 3)

    def test_floor_filter_is_populated_with_all_floors_option_plus_each_floor(self):

        self.panel.refresh(self.building)

        floor_names = [
            self.panel.floor_filter.itemText(i) for i in range(self.panel.floor_filter.count())
        ]
        self.assertEqual(floor_names, ["All Floors", "Ground Floor", "Floor 1"])

    def test_filtering_by_floor_narrows_the_table(self):

        self.panel.refresh(self.building)

        index = self.panel.floor_filter.findData(self.floor_2.id)
        self.panel.floor_filter.setCurrentIndex(index)

        self.assertEqual(self.panel.camera_table.rowCount(), 1)
        self.assertEqual(self.panel.camera_table.item(0, 0).text(), "Cam 3")

    def test_filtering_by_zone_narrows_the_table(self):

        self.panel.refresh(self.building)

        index = self.panel.zone_filter.findData(self.zone_b.id)
        self.panel.zone_filter.setCurrentIndex(index)

        self.assertEqual(self.panel.camera_table.rowCount(), 1)
        self.assertEqual(self.panel.camera_table.item(0, 0).text(), "Cam 2")

    def test_selecting_all_floors_resets_the_zone_filter_to_every_zone(self):

        self.panel.refresh(self.building)

        floor_index = self.panel.floor_filter.findData(self.floor_1.id)
        self.panel.floor_filter.setCurrentIndex(floor_index)

        zone_names = [
            self.panel.zone_filter.itemText(i) for i in range(self.panel.zone_filter.count())
        ]
        self.assertEqual(zone_names, ["All Zones", "Zone A", "Zone B"])

        all_floors_index = self.panel.floor_filter.findData(None)
        self.panel.floor_filter.setCurrentIndex(all_floors_index)

        self.assertEqual(self.panel.camera_table.rowCount(), 3)

    def test_toggling_the_active_checkbox_disables_the_camera(self):

        self.panel.refresh(self.building)

        checkbox = self.panel.camera_table.cellWidget(0, 4)
        self.assertTrue(checkbox.isChecked())

        checkbox.setChecked(False)

        self.assertFalse(self.camera_1.active)

    def test_on_camera_changed_fires_after_toggling_active(self):

        self.panel.refresh(self.building)

        calls = []
        self.panel.on_camera_changed = lambda: calls.append(True)

        checkbox = self.panel.camera_table.cellWidget(0, 4)
        checkbox.setChecked(False)

        self.assertEqual(len(calls), 1)

    def test_changing_the_mode_combo_updates_the_camera(self):

        self.panel.refresh(self.building)

        mode_combo = self.panel.camera_table.cellWidget(0, 5)
        live_index = mode_combo.findText(DeviceMode.LIVE)
        mode_combo.setCurrentIndex(live_index)

        self.assertEqual(self.camera_1.mode, DeviceMode.LIVE)

    def test_on_camera_changed_fires_after_a_mode_change(self):

        self.panel.refresh(self.building)

        calls = []
        self.panel.on_camera_changed = lambda: calls.append(True)

        mode_combo = self.panel.camera_table.cellWidget(0, 5)
        replay_index = mode_combo.findText(DeviceMode.REPLAY)
        mode_combo.setCurrentIndex(replay_index)

        self.assertEqual(len(calls), 1)

    def test_status_column_reflects_active_mode_and_provider_state(self):

        self.panel.refresh(self.building)

        status_text = self.panel.camera_table.item(0, 6).text()

        self.assertIn("Active", status_text)
        self.assertIn(DeviceMode.SIMULATION, status_text)
        self.assertIn("no provider", status_text)

    def test_refresh_with_no_building_clears_the_table(self):

        self.panel.refresh(self.building)
        self.assertEqual(self.panel.camera_table.rowCount(), 3)

        self.panel.refresh(None)

        self.assertEqual(self.panel.camera_table.rowCount(), 0)

    def test_refresh_button_rerenders_using_the_last_building(self):

        self.panel.refresh(self.building)

        camera_4 = Camera(name="Cam 4", floor_id=self.floor_1.id)
        self.floor_1.add_camera(camera_4)

        self.panel._rerun_last()

        self.assertEqual(self.panel.camera_table.rowCount(), 4)


class CameraManagerPanelModeDetailStatusTests(unittest.TestCase):

    # CCTV Pipeline End-to-End Offline Validation milestone, Phase 8:
    # the status column's appended mode-detail segment must be
    # truthful and derived only from information the panel already has
    # -- never a fabricated "Connected" for Live mode.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.camera = Camera(name="Cam", floor_id=self.floor.id)
        self.floor.add_camera(self.camera)

        self.panel = CameraManagerPanel()
        self.panel.refresh(self.building)

    def _status_text(self):

        self.panel._rerun_last()
        return self.panel.camera_table.item(0, 6).text()

    def test_simulation_with_no_provider_is_not_configured(self):

        self.assertIn("Not Configured", self._status_text())

    def test_simulation_with_provider_is_ready(self):

        self.panel.manager.register_detection_provider(DeviceMode.SIMULATION, object())

        self.assertIn("Ready", self._status_text())

    def test_replay_with_no_provider_is_no_source(self):

        self.panel.manager.set_camera_mode(self.camera.id, DeviceMode.REPLAY)

        self.assertIn("No Source", self._status_text())

    def test_replay_with_stream_unavailable_is_source_missing(self):

        self.panel.manager.register_detection_provider(DeviceMode.REPLAY, object())
        self.panel.manager.set_camera_mode(self.camera.id, DeviceMode.REPLAY)
        self.panel.manager.set_connection_status(
            self.camera.id, CameraConnectionState.STREAM_UNAVAILABLE,
        )

        self.assertIn("Source Missing", self._status_text())

    def test_replay_with_provider_and_no_stream_issue_is_source_loaded(self):

        self.panel.manager.register_detection_provider(DeviceMode.REPLAY, object())
        self.panel.manager.set_camera_mode(self.camera.id, DeviceMode.REPLAY)

        self.assertIn("Source Loaded", self._status_text())

    def test_live_with_no_credentials_is_credentials_missing(self):

        self.panel.manager.set_camera_mode(self.camera.id, DeviceMode.LIVE)

        self.assertIn("Credentials Missing", self._status_text())

    def test_live_with_credential_ref_but_not_online_is_not_connected(self):

        self.camera.connection = ConnectionInfo(credential_ref="CAM-CRED-1")
        self.panel.manager.set_camera_mode(self.camera.id, DeviceMode.LIVE)

        status_text = self._status_text()

        self.assertIn("Not Connected", status_text)
        self.assertNotIn("Credentials Missing", status_text)

    def test_live_never_fabricates_online_without_an_explicit_connection_status(self):

        self.camera.connection = ConnectionInfo(credential_ref="CAM-CRED-1")
        self.panel.manager.set_camera_mode(self.camera.id, DeviceMode.LIVE)

        self.assertNotIn("Online", self._status_text())

    def test_live_reports_online_only_when_connection_status_is_explicitly_online(self):

        self.camera.connection = ConnectionInfo(credential_ref="CAM-CRED-1")
        self.panel.manager.set_camera_mode(self.camera.id, DeviceMode.LIVE)
        self.panel.manager.set_connection_status(self.camera.id, CameraConnectionState.ONLINE)

        self.assertIn("Online", self._status_text())


if __name__ == "__main__":
    unittest.main()
