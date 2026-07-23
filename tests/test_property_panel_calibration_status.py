import sys
import unittest
from datetime import datetime, timezone

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.camera_item import CameraItem
from designer.windows.main_window import MainWindow

from models.camera import Camera

from camera_calibration.camera_model import CalibrationProfile, CalibrationQuality, CameraExtrinsics, CameraIntrinsics


# =====================================================
# Real Camera Calibration & World-Coordinate Validation milestone,
# Phase 13 -- Property Panel calibration status display. Exactly the
# three states Phase 13 names: NOT CONFIGURED / CONFIGURED --
# UNVALIDATED / VALIDATED -- RMSE: X m.
# =====================================================


def _make_window_with_camera():

    window = MainWindow()
    floor = window.canvas.scene_obj.current_floor

    window.property_panel.building = window.canvas.scene_obj.project.building

    model = Camera(id="CAM-STATUS-TEST", name="Test Camera", floor_id=floor.id)
    item = CameraItem(0, 0, model=model)

    return window, model, item


class CalibrationStatusDisplayTests(unittest.TestCase):

    def test_not_configured_is_the_default(self):

        window, model, item = _make_window_with_camera()

        window.property_panel.show_camera(item)

        self.assertEqual(window.property_panel.camera_calibration_status.text(), "CALIBRATION: NOT CONFIGURED")

    def test_configured_unvalidated_once_a_profile_is_registered(self):

        window, model, item = _make_window_with_camera()

        intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
        extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=45.0)
        profile = CalibrationProfile(camera_id=model.id, floor_id=model.floor_id, intrinsics=intrinsics, extrinsics=extrinsics)

        window.property_panel.calibration_registry.set(profile)
        window.property_panel.show_camera(item)

        self.assertEqual(window.property_panel.camera_calibration_status.text(), "CALIBRATION: CONFIGURED -- UNVALIDATED")

    def test_validated_shows_rmse(self):

        window, model, item = _make_window_with_camera()

        intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
        extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=45.0)
        quality = CalibrationQuality(
            reference_point_count=4, validated_point_count=4,
            mean_error_m=0.05, median_error_m=0.04, max_error_m=0.09, rmse_m=0.061,
            validation_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        profile = CalibrationProfile(
            camera_id=model.id, floor_id=model.floor_id, intrinsics=intrinsics, extrinsics=extrinsics, quality=quality,
        )

        window.property_panel.calibration_registry.set(profile)
        window.property_panel.show_camera(item)

        self.assertEqual(window.property_panel.camera_calibration_status.text(), "CALIBRATION: VALIDATED -- RMSE: 0.061 m")

    def test_switching_to_a_different_camera_shows_that_cameras_own_status(self):

        window, model, item = _make_window_with_camera()

        intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
        extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=45.0)
        profile = CalibrationProfile(camera_id=model.id, floor_id=model.floor_id, intrinsics=intrinsics, extrinsics=extrinsics)
        window.property_panel.calibration_registry.set(profile)

        other_model = Camera(id="CAM-OTHER", name="Other Camera", floor_id=model.floor_id)
        other_item = CameraItem(0, 0, model=other_model)

        window.property_panel.show_camera(other_item)
        self.assertEqual(window.property_panel.camera_calibration_status.text(), "CALIBRATION: NOT CONFIGURED")

        window.property_panel.show_camera(item)
        self.assertEqual(window.property_panel.camera_calibration_status.text(), "CALIBRATION: CONFIGURED -- UNVALIDATED")


if __name__ == "__main__":
    unittest.main()
